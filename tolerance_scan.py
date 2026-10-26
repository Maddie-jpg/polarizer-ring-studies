"""
tolerance_scan.py
=================

How big can the multipole field errors be before the ring stops working?

The answer is a single number: a scale factor on the borrowed ILC damping-ring
error table.  Scale 1 means "exactly the ILC magnets"; scale 10 means "ten times
worse".  We try a list of scales, and for each one we build several random
machines and ask whether they still pass.

Two stages:

  Stage 1 - how much room the beam needs
      The acceptance the ring has to provide is STATED here, not measured from
      the injected distribution: a nominal emittance, a number of sigma, and a
      momentum deviation.  A measured file would set the requirement from its
      far tail, which is not what a field-quality tolerance should hang on, and
      it would tie this study to one particular delivery.  From
          W_max = REQUIRED_SIGMA^2 * REQUIRED_EMIT
      the beam-stay-clear around the ring is:

          R_x(s) = sqrt(beta_x W_max_x) + |D_x| delta_max + ORBIT_ALLOWANCE
          R_y(s) = sqrt(beta_y W_max_y) + |D_y| delta_max + ORBIT_ALLOWANCE

      Round that up to a chamber half-aperture per magnet type, add the wall to
      get the pole radius, and set R_ref = R_REF_FRACTION * pole radius.
      (Antoniou, CLIC-Note-989 Eq. 6.6, writes sqrt(2 beta eps_max); the two
      agree for W_max = 2 eps_max.)

  Stage 2 - the scan
      For every (scale, seed):
          misalignments + main-field errors   (same draw at every scale)
          multipole errors x scale            (same unit draws at every scale)
          orbit correction
          dynamic aperture and momentum acceptance, tracked with the apertures

      Because the draws are fixed per seed, scale 10 is the *same machine* as
      scale 1 with ten times the harmonic content, not a different random
      machine.  That is what makes this a controlled ramp rather than a
      measurement of seed-to-seed scatter.

      A machine passes when
          DA / required acceptance >= DA_REQUIRED      (in every direction)
      and momentum acceptance covers delta_max on both signs of delta.
      The tolerance is the last scale before the first failure, where "failure"
      means fewer than PASS_TARGET of the seeds pass.

Results are appended to runs.csv after every machine, so an interrupted scan
picks up where it stopped when you run it again.

Multipole errors are thin kicks (half at magnet entry, half at exit) that
inherit their magnet's shift and roll, so feed-down from the misalignments is
automatic.  Their integrated strength follows the magnet's own strength:

    knl[n-1] = K_{m-1}L * b_n * (n-1)!/(m-1)! * R_ref^(m-n)

Run it with:    python tolerance_scan.py
Edit SETTINGS below to change anything.
"""
import os
import sys
import json
import time
from math import factorial

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import xtrack as xt
import xobjects as xo

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import LatticeBuild.misalignments_corrections as mc
import paths
from paths import results_dir

xo.context_cpu.allow_no_prebuilt_kernel = True


# ============================================================================
#  SETTINGS  -  edit these, then run the file
# ============================================================================

# Which lattice.  None takes the value from config.DEFAULT_STUDY.
DESIGN  = None
CONFIG  = None
PHASE   = None
CHANGES = None

# The scan itself
SCALES      = [0, 1, 2, 5, 10, 20]   # multipliers on the error table
N_SEEDS     = 10                     # random machines per scale
FIRST_SEED  = 1000
ERROR_TABLE = 'ilc_dr'
ONLY        = None                   # e.g. ['Quadrupole:b6'] to screen one harmonic

# Error sizes.  These match the misalignment model in section 2.5.1.
SIGMA_SHIFT  = 0.25e-3   # m,   magnet and corrector displacements
SIGMA_ROT    = 0.25e-3   # rad, magnet rotations
SIGMA_FIELD  = 1e-3      # relative error on the magnet's own strength
TRUNCATE_AT  = 2.5       # Gaussian truncated at this many sigma, by redrawing

# The acceptance the ring must provide - a nominal beam, stated not measured.
# W_max is the Courant-Snyder invariant at REQUIRED_SIGMA sigma, so
#     W_max = REQUIRED_SIGMA**2 * REQUIRED_EMIT
REQUIRED_EMIT_X = 1.3e-6   # m rad, geometric
REQUIRED_EMIT_Y = 1.3e-6   # m rad, geometric
REQUIRED_SIGMA  = 10       # the ring must hold particles out to this amplitude
REQUIRED_DELTA  = 0.03     # ...and this momentum deviation, both signs

# Chamber and pole (stage 1)
ORBIT_ALLOWANCE  = 1e-3    # m, residual closed orbit + alignment
WALL_THICKNESS   = 1.5e-3  # m, chamber wall + clearance to the pole
R_REF_FRACTION   = 2 / 3   # R_ref / pole radius
FIXED_APERTURE   = None    # e.g. {'Bend': (30e-3, 15e-3), ...} to override

# Tracking
TURNS      = 1000   # turns for both the DA and the MA scan
N_ANGLES   = 13     # amplitude ratios in the DA scan
N_RADII    = 30
MAX_RADIUS = 3.0    # DA scan limit, in units of the required acceptance
DELTA_SCAN = 0.04   # MA scan goes to +-this
N_DELTA    = 20     # steps either side of zero
RADIATION  = 'mean'   # 'mean' or 'none'
THREADS    = 'auto'   # OpenMP threads for tracking; 0 = serial

# What counts as a pass
DA_REQUIRED = 1.0    # DA / required acceptance must reach this
PASS_TARGET = 0.9    # ...for at least this fraction of seeds

APERTURE_ONLY = False   # True = stop after stage 1
START_ELEMENT = None    # cycle the ring to start here, e.g. the injection point


# ============================================================================
#  The borrowed error table
#  {magnet type: [(component, n, systematic, random rms), ...]}
#  Plain fractions of the main field at R_ref, NOT 1e-4 "units".
# ============================================================================
ERROR_TABLES = {
    # ILC damping ring: Shanks, Rubin & Sagan, arXiv:1309.2248, Table 8,
    # itself from PEP-II magnet measurements by Y. Cai.
    # Source reference radii: 30 / 50 / 32 mm for bend / quad / sextupole.
    'ilc_dr': {
        'Bend': [
            ('b', 3, 1.6e-4, 8e-5),
            ('b', 4, -1.6e-5, 8e-6),
            ('b', 5, 7.6e-5, 3.8e-5),
        ],
        'Quadrupole': [
            ('a', 3, -1.15e-5, 7.25e-5), ('a', 4, 1.41e-5, 1.27e-4),
            ('a', 5, 6.2e-7, 1.62e-5),   ('a', 6, -4.93e-5, 3.63e-4),
            ('a', 7, -1.02e-6, 6.6e-6),  ('a', 8, 3.8e-7, 6.6e-6),
            ('a', 9, -2.8e-7, 4.9e-6),   ('a', 10, -5.77e-5, 2.33e-4),
            ('a', 11, -3.8e-7, 3.5e-6),  ('a', 12, -6.53e-6, 3.66e-5),
            ('a', 13, 1.2e-6, 8.6e-6),   ('a', 14, -7.4e-7, 4.46e-5),
            ('b', 3, -1.24e-5, 7.61e-5), ('b', 4, 2.3e-6, 1.32e-4),
            ('b', 5, -4.3e-6, 1.5e-5),   ('b', 6, 3.4e-4, 1.65e-4),
            ('b', 7, 3e-7, 6.7e-6),      ('b', 8, 6e-7, 8.9e-6),
            ('b', 9, 6e-7, 4.6e-6),      ('b', 10, -6.17e-5, 2.46e-4),
            ('b', 11, -2e-7, 4.2e-6),    ('b', 12, 3.6e-6, 3.48e-5),
            ('b', 13, 6e-7, 9.2e-6),     ('b', 14, 1e-6, 4.76e-5),
        ],
        'Sextupole': [
            ('b', 4, 1e-4, 1e-4),    ('b', 5, 5e-5, 3e-5),
            ('b', 6, 3.5e-4, 1e-4),  ('b', 7, 5e-5, 3e-5),
            ('b', 8, 5e-5, 3e-5),    ('b', 9, 5e-5, 3e-5),
            ('b', 10, 5e-5, 3e-5),   ('b', 11, 5e-5, 3e-5),
            ('b', 12, 1.6e-3, 1e-4), ('b', 13, 5e-5, 3e-5),
            ('b', 14, 5e-5, 3e-5),
        ],
    },
}

MAGNET_TYPES = ('Bend', 'Quadrupole', 'Sextupole')
MAIN_ORDER = {'Bend': 1, 'Quadrupole': 2, 'Sextupole': 3}
N_KNL = 14   # thin error kicks hold up to b14 (28-pole)

# Every row written to runs.csv has exactly these columns, in this order.
# Without a fixed list, a row from a failed machine has different keys from a
# successful one, and appending it without a header shifts every later value
# into the wrong column.
RESULT_COLUMNS = [
    'scale', 'seed', 'ok', 'delta_max',
    'qx', 'qy', 'orbit_rms_x', 'orbit_rms_y',
    'DA_ratio_min', 'DA_x_pos', 'DA_x_neg', 'DA_y', 'DA_boundary',
    'MA_pos', 'MA_neg', 'runtime_s', 'error',
]


# ============================================================================
#  Small helpers
# ============================================================================
def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def truncnorm(rng, size=None, cut=TRUNCATE_AT):
    """A standard normal with anything beyond +-cut redrawn, not clipped.

    Clipping would pile the whole tail onto the boundary and make the most
    extreme value the most common one; redrawing removes the tail and leaves
    the core untouched.  Same convention as mc.misalignments().
    """
    if size is None:
        v = rng.normal()
        while abs(v) > cut:
            v = rng.normal()
        return v
    out = rng.normal(size=size)
    bad = np.abs(out) > cut
    while np.any(bad):
        out[bad] = rng.normal(size=bad.sum())
        bad = np.abs(out) > cut
    return out


def magnet_names(line):
    """{'Bend': [...], 'Quadrupole': [...], 'Sextupole': [...]}"""
    tt = line.get_table()
    if any(t in ('BFieldExpansion', 'ThickSliceBFieldExpansion') for t in tt.element_type):
        raise NotImplementedError(
            'This script puts its error kicks on hard-edge magnets. '
            'Build the lattice with field_expansion_bends=False.')
    return {t: list(tt.rows[tt.element_type == t].name) for t in MAGNET_TYPES}


# ============================================================================
#  Stage 1 - how much room the beam needs
# ============================================================================
def required_acceptance():
    """The acceptance the ring must provide, from the settings above.

    No distribution file is read.  The requirement is a nominal beam, so the
    tolerance this scan returns is a property of the lattice and the stated
    acceptance, not of one particular delivered distribution.
    """
    return {'source': f'nominal: {REQUIRED_SIGMA} sigma, delta {REQUIRED_DELTA}',
            'eps_x': REQUIRED_EMIT_X, 'eps_y': REQUIRED_EMIT_Y,
            'W_max_x': REQUIRED_SIGMA ** 2 * REQUIRED_EMIT_X,
            'W_max_y': REQUIRED_SIGMA ** 2 * REQUIRED_EMIT_Y,
            'delta_max': float(REQUIRED_DELTA)}


def beam_stay_clear(line, mags, beam, ds=0.05):
    """Beam-stay-clear sampled every ds metres, worst value inside each magnet."""
    s_grid = np.arange(ds, line.get_length() - ds / 2, ds)
    sliced = line.copy(shallow=True)
    sliced.cut_at_s(s_grid)
    tw = sliced.twiss4d()
    s = np.asarray(tw.s)

    Rx = np.sqrt(tw.betx * beam['W_max_x']) + np.abs(tw.dx) * beam['delta_max'] + ORBIT_ALLOWANCE
    Ry = np.sqrt(tw.bety * beam['W_max_y']) + np.abs(tw.dy) * beam['delta_max'] + ORBIT_ALLOWANCE

    s_of = dict(zip(line.get_table().name, line.get_table().s))
    rows = []
    for magnet_type, names in mags.items():
        for m in names:
            s0 = s_of[m]
            s1 = s0 + line[m].length
            i0 = np.searchsorted(s, s0 - 1e-9)
            i1 = np.searchsorted(s, s1 + 1e-9)
            inside = slice(i0, max(i1, i0 + 1))
            rows.append({'name': m, 'type': magnet_type, 's_start': s0, 's_end': s1,
                         'betx_max': float(np.max(tw.betx[inside])),
                         'bety_max': float(np.max(tw.bety[inside])),
                         'dx_max': float(np.max(np.abs(tw.dx[inside]))),
                         'R_req_x': float(np.max(Rx[inside])),
                         'R_req_y': float(np.max(Ry[inside]))})
    return pd.DataFrame(rows)


def choose_apertures(df_req, round_to=1e-3):
    """Chamber half-apertures, pole radius and R_ref, one set per magnet type."""
    apertures = {}
    for magnet_type in MAGNET_TYPES:
        sub = df_req[df_req.type == magnet_type]
        if len(sub) == 0:
            continue
        rx, ry = sub.R_req_x.max(), sub.R_req_y.max()

        if FIXED_APERTURE is not None:
            a, b = FIXED_APERTURE[magnet_type]
        elif magnet_type == 'Bend':        # wide horizontally, narrow gap vertically
            a = np.ceil(rx / round_to) * round_to
            b = np.ceil(ry / round_to) * round_to
        else:                              # round bore
            a = b = np.ceil(max(rx, ry) / round_to) * round_to

        # in a bend the field quality is set by the vertical half gap
        pole = (b if magnet_type == 'Bend' else max(a, b)) + WALL_THICKNESS
        apertures[magnet_type] = {
            'chamber_a': float(a), 'chamber_b': float(b),
            'R_req_x_max': float(rx), 'R_req_y_max': float(ry),
            'pole_radius': float(pole), 'R_ref': float(R_REF_FRACTION * pole),
            'margin_x': float(a - rx), 'margin_y': float(b - ry)}
    return apertures


def insert_apertures(pdr, line, mags, apertures):
    """A LimitEllipse at the entry and exit of every magnet."""
    places = []
    for magnet_type, names in mags.items():
        a = apertures[magnet_type]['chamber_a']
        b = apertures[magnet_type]['chamber_b']
        for m in names:
            for end in ('start', 'end'):
                name = f'AP{end[0]}_{m}'
                pdr.new(name, xt.LimitEllipse, a=a, b=b)
                places.append(pdr.place(name, at=f'{m}@{end}'))
    line.insert(places)


def insert_error_kicks(pdr, line, mags):
    """Empty thin multipoles at the entry and exit of every magnet."""
    places = []
    for names in mags.values():
        for m in names:
            for end in ('start', 'end'):
                name = f'MPE{end[0]}_{m}'
                pdr.new(name, xt.Multipole, knl=np.zeros(N_KNL), ksl=np.zeros(N_KNL))
                places.append(pdr.place(name, at=f'{m}@{end}'))
    line.insert(places)


# ============================================================================
#  Stage 2 - putting the errors in
# ============================================================================
def apply_alignment_and_field_errors(line, mags, seed):
    """Six displacements/rotations plus a main-field error, per magnet.

    Same distribution as mc.misalignments(), but the field error acts on
    k0/k1/k2 rather than knl, which is zero on a thick magnet.
    """
    rng = np.random.default_rng([seed, 0])
    refs = line.element_refs

    for magnet_type in MAGNET_TYPES:
        for m in mags[magnet_type]:
            ref = refs[m]
            ref.shift_x = SIGMA_SHIFT * truncnorm(rng)
            ref.shift_y = SIGMA_SHIFT * truncnorm(rng)
            ref.shift_s = SIGMA_SHIFT * truncnorm(rng)
            ref.rot_s_rad = SIGMA_ROT * truncnorm(rng)
            ref.rot_x_rad = SIGMA_ROT * truncnorm(rng)
            ref.rot_y_rad = SIGMA_ROT * truncnorm(rng)

            error = SIGMA_FIELD * truncnorm(rng)
            if magnet_type == 'Quadrupole':
                ref.k1 = ref.k1 * (1 + error)
            elif magnet_type == 'Sextupole':
                ref.k2 = ref.k2 * (1 + error)
            else:
                line[m].k0_from_h = False      # hold the geometry, perturb the field
                ref.k0 = ref.angle / ref.length * (1 + error)

    # the error kicks follow their magnet, so feed-down happens by itself
    for names in mags.values():
        for m in names:
            for end in 'se':
                kick = line[f'MPE{end}_{m}']
                kick.shift_x = line[m].shift_x
                kick.shift_y = line[m].shift_y
                kick.rot_s_rad = line[m].rot_s_rad


def misalign_correctors(line, seed):
    """Displace the steering correctors too - correcting an orbit with magnets
    that are themselves off-centre is the realistic case."""
    rng = np.random.default_rng([seed, 2])
    names = [n for n in line.get_table().name if n.startswith(('Mx_', 'My_'))]
    for n in names:
        ref = line.element_refs[n]
        ref.shift_x = SIGMA_SHIFT * truncnorm(rng)
        ref.shift_y = SIGMA_SHIFT * truncnorm(rng)
        ref.rot_s_rad = SIGMA_ROT * truncnorm(rng)
        error = 1 + SIGMA_FIELD * truncnorm(rng)
        ref.knl = ref.knl * error
        ref.ksl = ref.ksl * error


def apply_multipoles(line, mags, table, apertures, seed, scale):
    """Systematic + random harmonics, the whole table multiplied by `scale`.

    The unit random draws depend only on (seed, magnet, harmonic) - never on
    the scale - so the same seed gives the same machine at every scale.
    """
    rng = np.random.default_rng([seed, 1])
    refs = line.element_refs

    for magnet_type in MAGNET_TYPES:
        harmonics = table.get(magnet_type, [])
        if not harmonics or magnet_type not in apertures:
            continue
        m_order = MAIN_ORDER[magnet_type]
        R_ref = apertures[magnet_type]['R_ref']

        for m in mags[magnet_type]:
            # drawn for every harmonic even if ONLY skips it, to keep the
            # random stream identical between runs
            draws = truncnorm(rng, size=len(harmonics))

            if magnet_type == 'Quadrupole':
                KL = refs[m].k1 * refs[m].length
            elif magnet_type == 'Sextupole':
                KL = refs[m].k2 * refs[m].length
            else:
                KL = refs[m].k0 * refs[m].length

            for (component, n, systematic, random_rms), z in zip(harmonics, draws):
                if ONLY and f'{magnet_type}:{component}{n}' not in ONLY:
                    continue
                bn = scale * (systematic + random_rms * z)
                strength = (0.5 * bn * factorial(n - 1) / factorial(m_order - 1)
                            * R_ref ** (m_order - n))
                for end in 'se':
                    kick = refs[f'MPE{end}_{m}']
                    if component == 'b':
                        kick.knl[n - 1] = KL * strength
                    else:
                        kick.ksl[n - 1] = KL * strength


# ============================================================================
#  Stage 2 - the two measurements
# ============================================================================
def track_dynamic_aperture(line, tw, beam, gamma_beta):
    """DA in units of the acceptance the beam needs: a ratio of 1 means it fits.

    x_norm = 1 corresponds to W = W_max, so the required region is the square
    |x_norm| <= 1, |y_norm| <= 1, and the ratio reported is the smallest over
    all the amplitude ratios sampled.
    """
    theta = np.linspace(0, np.pi, N_ANGLES)
    radii = np.linspace(MAX_RADIUS / N_RADII, MAX_RADIUS, N_RADII)
    TT, RR = np.meshgrid(theta, radii, indexing='ij')

    particles = line.build_particles(
        x_norm=(RR * np.cos(TT)).ravel(), px_norm=0,
        y_norm=(RR * np.sin(TT)).ravel(), py_norm=0, delta=0,
        nemitt_x=beam['W_max_x'] * gamma_beta,
        nemitt_y=beam['W_max_y'] * gamma_beta,
        W_matrix=tw.W_matrix[0], particle_on_co=tw.particle_on_co.copy())

    line.track(particles, num_turns=TURNS)
    particles.sort(interleave_lost_particles=True)
    alive = (particles.state > 0).reshape(N_ANGLES, N_RADII)

    da = np.zeros(N_ANGLES)
    for i in range(N_ANGLES):
        lost = np.where(~alive[i])[0]
        first_lost = lost[0] if len(lost) else N_RADII
        da[i] = radii[first_lost - 1] if first_lost > 0 else 0.0

    # distance to the edge of the required square, along each ray
    r_required = 1.0 / np.maximum(np.abs(np.cos(theta)), np.abs(np.sin(theta)))
    i_vertical = np.argmin(np.abs(theta - np.pi / 2))
    return {'DA_ratio_min': float(np.min(da / r_required)),
            'DA_x_pos': float(da[0]), 'DA_x_neg': float(da[-1]),
            'DA_y': float(da[i_vertical]),
            'DA_boundary': json.dumps(da.tolist())}


def track_momentum_acceptance(line, tw, beam, gamma_beta, amplitude=0.05):
    """Largest |delta| that survives, found separately for each sign.

    The two signs are kept apart because the degradation is asymmetric in
    momentum - a symmetric +-delta would hide that.
    """
    deltas = np.linspace(-DELTA_SCAN, DELTA_SCAN, 2 * N_DELTA + 1)
    particles = line.build_particles(
        x_norm=amplitude, px_norm=0, y_norm=amplitude, py_norm=0, delta=deltas,
        nemitt_x=beam['W_max_x'] * gamma_beta,
        nemitt_y=beam['W_max_y'] * gamma_beta,
        W_matrix=tw.W_matrix[0], particle_on_co=tw.particle_on_co.copy())

    line.track(particles, num_turns=TURNS)
    particles.sort(interleave_lost_particles=True)
    alive = particles.state > 0

    centre = N_DELTA
    result = {}
    for sign, walk in (('pos', range(centre, len(deltas))),
                       ('neg', range(centre, -1, -1))):
        last_surviving = 0.0
        for i in walk:
            if not alive[i]:
                break
            last_surviving = deltas[i]
        result[f'MA_{sign}'] = float(last_surviving)
    return result


# ============================================================================
#  Summary
# ============================================================================
def summarise(runs, out_dir):
    """Work out the tolerance and draw the two summary plots."""
    df = runs.copy()
    # 'ok' comes back from CSV as object dtype if any row failed
    df['ok'] = df['ok'].fillna(False).astype(bool)

    df['pass'] = (df['ok']
                  & (df['DA_ratio_min'] >= DA_REQUIRED)
                  & (df['MA_pos'] >= df['delta_max'])
                  & (-df['MA_neg'] >= df['delta_max']))

    by_scale = df.groupby('scale')
    summary = pd.DataFrame({
        'n_seeds': by_scale.size(),
        'pass_fraction': by_scale['pass'].mean(),
        'DA_ratio_p10': by_scale['DA_ratio_min'].quantile(0.1),
        'DA_ratio_median': by_scale['DA_ratio_min'].median(),
        'MA_pos_median': by_scale['MA_pos'].median(),
        'MA_neg_median': by_scale['MA_neg'].median(),
        'orbit_rms_x_mm': by_scale['orbit_rms_x'].median() * 1e3,
        'orbit_rms_y_mm': by_scale['orbit_rms_y'].median() * 1e3,
    }).reset_index()
    summary.to_csv(f'{out_dir}/summary.csv', index=False)

    # the tolerance: walk up the scales, stop at the first failure
    tolerance = None
    for _, row in summary.sort_values('scale').iterrows():
        if row.pass_fraction >= PASS_TARGET:
            tolerance = row.scale
        else:
            break

    delta_max = df['delta_max'].median()
    scales = summary.scale.values
    fig, (ax_pass, ax_da, ax_ma) = plt.subplots(1, 3, figsize=(14, 4.2))

    ax_pass.plot(scales, summary.pass_fraction, 'o-')
    ax_pass.axhline(PASS_TARGET, color='r', ls='--', lw=1)
    ax_pass.set_ylabel('fraction of seeds passing')

    ax_da.scatter(df.scale, df.DA_ratio_min, s=10, alpha=0.4)
    ax_da.plot(scales, summary.DA_ratio_median, 'k-', label='median')
    ax_da.plot(scales, summary.DA_ratio_p10, 'k--', label='10th percentile')
    ax_da.axhline(DA_REQUIRED, color='r', ls='--', lw=1)
    ax_da.set_ylabel('DA / required acceptance')
    ax_da.legend(fontsize='small')

    ax_ma.scatter(df.scale, df.MA_pos * 100, s=10, alpha=0.4, label=r'$+\delta$')
    ax_ma.scatter(df.scale, -df.MA_neg * 100, s=10, alpha=0.4, label=r'$-\delta$')
    ax_ma.axhline(delta_max * 100, color='r', ls='--', lw=1)
    ax_ma.set_ylabel('momentum acceptance [%]')
    ax_ma.legend(fontsize='small')

    for ax in (ax_pass, ax_da, ax_ma):
        ax.set_xlabel('multipole error scale')
        ax.grid(alpha=0.3)
    fig.suptitle(f'{ERROR_TABLE}: tolerance ('
                 f'{PASS_TARGET:.0%} of seeds pass) = '
                 f'{"none" if tolerance is None else f"{tolerance:g}x"}')
    fig.tight_layout()
    fig.savefig(f'{out_dir}/tolerance_scan.png', dpi=200)
    plt.close(fig)

    log(f'Summary:\n{summary.to_string(index=False)}')
    log(f'Tolerance (last scale before the first failure): '
        f'{"none" if tolerance is None else f"{tolerance:g}x"}')
    if tolerance is None:
        log('  Scale 0 already fails, so there is no tolerance to quote. '
            'The alignment errors alone are enough to break the pass criteria.')


def save_run(row, csv_path):
    """Append one machine to runs.csv, always with the same columns."""
    pd.DataFrame([row], columns=RESULT_COLUMNS).to_csv(
        csv_path, mode='a', index=False, header=not os.path.exists(csv_path))


# ============================================================================
#  Main
# ============================================================================
def main():
    study = paths.study_from_env()
    design = DESIGN if DESIGN is not None else study['design']
    config = CONFIG if CONFIG is not None else study['config']
    phase = PHASE if PHASE is not None else study['phase']
    changes = CHANGES if CHANGES is not None else study['changes']

    table = ERROR_TABLES[ERROR_TABLE]
    sub = ERROR_TABLE + (f'_only_{"_".join(ONLY)}' if ONLY else '')
    out_dir = results_dir(design, config, phase, changes=changes,
                          metric='ToleranceScan', sub=sub)
    aperture_dir = results_dir(design, config, phase, changes=changes,
                               metric='ToleranceScan', sub='BeamStayClear')

    # ---------------- the lattice ----------------
    log(f'Loading {paths.lattice_json_path(design, config, "perfect", phase, changes)}')
    pdr = paths.load_lattice(design, config, 'perfect', phase, changes)
    ring = pdr.lines['ring']
    ring.configure_radiation(model=None)
    mc.insert_bpms_and_correctors(pdr, design, config)
    if START_ELEMENT:
        ring.cycle(name_first_element=START_ELEMENT, inplace=True)

    mags = magnet_names(ring)
    log('Magnets: ' + ', '.join(f'{t}: {len(v)}' for t, v in mags.items()))

    p0c = ring.particle_ref.p0c[0]
    gamma_beta = ring.particle_ref.gamma0[0] * ring.particle_ref.beta0[0]

    # ---------------- stage 1 ----------------
    beam = required_acceptance()
    log(f'Required acceptance: {REQUIRED_SIGMA} sigma on '
        f'{REQUIRED_EMIT_X*1e6:.2f} / {REQUIRED_EMIT_Y*1e6:.2f} um, so '
        f'W_max = {beam["W_max_x"]*1e6:.1f} / {beam["W_max_y"]*1e6:.1f} um rad, '
        f'delta_max = +-{beam["delta_max"]*100:.2f} %')

    # Worth checking before spending hours of tracking.
    if beam['delta_max'] > DELTA_SCAN:
        log(f'  WARNING: the MA scan only reaches +-{DELTA_SCAN*100:.1f} %, so no '
            f'machine can ever meet the +-{beam["delta_max"]*100:.1f} % requirement. '
            f'Raise DELTA_SCAN or lower REQUIRED_DELTA.')

    df_req = beam_stay_clear(ring, mags, beam)
    apertures = choose_apertures(df_req)

    df_req.to_csv(f'{aperture_dir}/required_aperture_per_magnet.csv', index=False)
    with open(f'{aperture_dir}/apertures.json', 'w') as f:
        json.dump({'required_acceptance': beam,
                   'orbit_allowance': ORBIT_ALLOWANCE, 'wall': WALL_THICKNESS,
                   'apertures': apertures}, f, indent=2)

    for t, a in apertures.items():
        log(f'{t:10s} chamber a/b = {a["chamber_a"]*1e3:.1f}/{a["chamber_b"]*1e3:.1f} mm '
            f'(needs {a["R_req_x_max"]*1e3:.1f}/{a["R_req_y_max"]*1e3:.1f} mm), '
            f'pole {a["pole_radius"]*1e3:.1f} mm, R_ref {a["R_ref"]*1e3:.1f} mm')
        if a['margin_x'] < 0 or a['margin_y'] < 0:
            log(f'  WARNING: the {t} chamber is smaller than the beam needs somewhere')
    log(f'Stage 1 written to {aperture_dir}')
    if APERTURE_ONLY:
        return

    # ---------------- stage 2 ----------------
    insert_apertures(pdr, ring, mags, apertures)
    insert_error_kicks(pdr, ring, mags)
    perfect_ring = ring.copy()

    csv_path = f'{out_dir}/runs.csv'
    already_done = set()
    if os.path.exists(csv_path):
        previous = pd.read_csv(csv_path)
        if list(previous.columns) != RESULT_COLUMNS:
            sys.exit(
                f'\n{csv_path} was written by an older version of this script and '
                f'has different columns:\n'
                f'  found:    {list(previous.columns)}\n'
                f'  expected: {RESULT_COLUMNS}\n'
                f'Appending to it would put the new values under the old headings. '
                f'Convert it first with\n'
                f'    python repair_runs_csv.py {csv_path}\n'
                f'or move it aside to start the scan from scratch.\n')
        already_done = set(zip(previous.scale, previous.seed))
        log(f'Resuming: {len(already_done)} machines already in {csv_path}')

    with open(f'{out_dir}/settings.json', 'w') as f:
        json.dump({k: v for k, v in globals().items()
                   if k.isupper() and isinstance(v, (int, float, str, list, type(None)))},
                  f, indent=2, default=str)

    seeds = [FIRST_SEED + i for i in range(N_SEEDS)]
    serial = xo.ContextCpu()
    threads = int(THREADS) if str(THREADS).isdigit() else THREADS
    tracking_context = serial if threads == 0 else xo.ContextCpu(omp_num_threads=threads)

    # orbit-correction response matrix, from the perfect machine, computed once
    reference = perfect_ring.copy()
    reference.build_tracker(_context=serial)
    if RADIATION == 'mean':
        reference.configure_radiation(model='mean')
    tw_ref = reference.twiss(method='6d' if RADIATION == 'mean' else '4d')

    for scale in SCALES:
        for seed in seeds:
            if (float(scale), seed) in already_done or (scale, seed) in already_done:
                continue
            start = time.time()
            row = {c: np.nan for c in RESULT_COLUMNS}
            row.update({'scale': scale, 'seed': seed, 'ok': False,
                        'delta_max': beam['delta_max']})
            try:
                line = perfect_ring.copy()
                # twiss with radiation needs the serial context
                line.build_tracker(_context=serial)
                apply_alignment_and_field_errors(line, mags, seed)
                apply_multipoles(line, mags, table, apertures, seed, scale)
                if RADIATION == 'mean':
                    line.configure_radiation(model='mean')
                misalign_correctors(line, seed)
                mc.correct_orbit_with_fallback(line, tw_ref, seed)

                tw = line.twiss(method='6d' if RADIATION == 'mean' else '4d')

                # everything twiss-dependent is in `tw`; now switch to OpenMP
                if tracking_context is not serial:
                    line.discard_tracker()
                    line.build_tracker(_context=tracking_context)
                    if RADIATION == 'mean':
                        line.configure_radiation(model='mean')

                row.update({'qx': tw.qx, 'qy': tw.qy,
                            'orbit_rms_x': float(np.std(tw.x)),
                            'orbit_rms_y': float(np.std(tw.y))})
                row.update(track_dynamic_aperture(line, tw, beam, gamma_beta))
                row.update(track_momentum_acceptance(line, tw, beam, gamma_beta))
                row['ok'] = True
            except Exception as e:
                log(f'  scale {scale:g} seed {seed} FAILED ({type(e).__name__}: {e})')
                row.update({'DA_ratio_min': 0.0, 'MA_pos': 0.0, 'MA_neg': 0.0,
                            'error': str(e)[:200]})

            row['runtime_s'] = time.time() - start
            save_run(row, csv_path)
            log(f'scale {scale:g} seed {seed}: DA/required = {row["DA_ratio_min"]:.2f}, '
                f'MA = [{row["MA_neg"]*100:.2f}, {row["MA_pos"]*100:.2f}] % '
                f'(needs +-{beam["delta_max"]*100:.2f} %)  [{row["runtime_s"]:.0f} s]')

    summarise(pd.read_csv(csv_path), out_dir)
    log(f'Results in {out_dir}')


if __name__ == '__main__':
    main()
