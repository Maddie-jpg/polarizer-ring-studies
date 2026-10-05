"""
tolerance_scan.py
=================

Two-stage study for one lattice configuration (D{design}/C{config}/{phase}deg):

Stage 1 -- physical aperture
    * Takes the injected positron distribution (or CLIC-PDR fallback numbers),
      finds the largest single-particle Courant-Snyder invariant W_max and
      momentum deviation delta_max that contain a fraction `--containment` of
      the beam.
    * Computes the beam-stay-clear around the ring, sampled inside every magnet
          R_x(s) = sqrt(beta_x W_max,x) + |D_x| delta_max + d_co
          R_y(s) = sqrt(beta_y W_max,y) + |D_y| delta_max + d_co
      (Antoniou, CLIC-Note-989, Eq. 6.6. NOTE: W is the Courant-Snyder
      invariant, so the amplitude is sqrt(beta W). Antoniou writes
      sqrt(2 beta eps_max); the two agree for W_max = 2 eps_max.)
    * Chooses a vacuum-chamber aperture per magnet type ('required': rounded-up
      requirement, or 'fixed': user values, e.g. the PDR's 30 mm), the pole
      radius (chamber + wall) and the multipole reference radius
      R_ref = ref_fraction * pole radius.
    * Saves a per-magnet table, a plot and an aperture JSON.

Stage 2 -- error tolerance scan
    For every scale factor and seed:
      misalignments + main-field errors (same draw for all scales)
      -> systematic + random multipole errors x scale (same unit draws for all scales)
      -> orbit correction (your mc.orbit_correction)
      -> DA, MA and injection survival of the real linac distribution, tracked
         WITH the physical apertures from stage 1.
    The DA is expressed in units of the required acceptance (x_norm = 1 <=>
    W = W_max), so DA_ratio >= 1 means the injected beam fits.
    Results are appended to a CSV after every run, so an interrupted scan can
    be resumed by re-running the same command.

Multipole errors are modelled as thin kicks (half at magnet entry, half at
exit) that inherit the magnet's shift_x/shift_y/rot_s_rad, so feed-down from
misalignments is included. Their integrated strength follows the magnet's main
strength (including its main-field error):
    knl[n-1] = K_{m-1}L * b_n * (n-1)!/(m-1)! * R_ref^(m-n)

Usage (from the repo root):
    python tolerance_scan.py --design 1 --config 9 --phase 90
    python tolerance_scan.py --aperture-only                # stage 1 only
    python tolerance_scan.py --scales 0,1,5,10,20 --n-seeds 20
    python tolerance_scan.py --only Quadrupole:b6,Quadrupole:b10   # screen harmonics
    python tolerance_scan.py --aperture-mode fixed \
        --fixed-aperture '{"Bend":[30e-3,15e-3],"Quadrupole":[30e-3,30e-3],"Sextupole":[30e-3,30e-3]}'
"""
#%%
import os
import sys
import json
import time
import argparse
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

xo.context_cpu.allow_no_prebuilt_kernel = True

MAGNET_TYPES = ('Bend', 'Quadrupole', 'Sextupole')
MAIN_ORDER = {'Bend': 1, 'Quadrupole': 2, 'Sextupole': 3}
N_KNL = 14   # thin error kicks hold up to 28-pole (b14)

# =============================================================================
# Error tables: {magnet type: [(component, n, systematic, random_rms), ...]}
# Plain fractions of the main field at R_ref (NOT 1e-4 "units").
# Assumed to be quoted at ~2/3 of the source magnet's aperture, and therefore
# applied at R_ref = ref_fraction * (your pole radius). See README discussion.
# =============================================================================
ERROR_TABLES = {
    # ILC DR, Shanks, Rubin & Sagan, arXiv:1309.2248, Table 8 (PEP-II
    # measurements by Y. Cai). Source radii: 30 / 50 / 32 mm.
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
            ('b', 4, 1e-4, 1e-4),   ('b', 5, 5e-5, 3e-5),
            ('b', 6, 3.5e-4, 1e-4), ('b', 7, 5e-5, 3e-5),
            ('b', 8, 5e-5, 3e-5),   ('b', 9, 5e-5, 3e-5),
            ('b', 10, 5e-5, 3e-5),  ('b', 11, 5e-5, 3e-5),
            ('b', 12, 1.6e-3, 1e-4), ('b', 13, 5e-5, 3e-5),
            ('b', 14, 5e-5, 3e-5),
        ],
    },
    # Placeholder: fill in from Etisken et al., PRAB 26, 081601 (2023), Table VI
    # (systematic: quad orders 3-5, sextupole orders 4-8) once you have the PDF.
    'a_pbr': {'Bend': [], 'Quadrupole': [], 'Sextupole': []},
}


# =============================================================================
# Small helpers
# =============================================================================
def truncnorm(rng, cut, size=None):
    """Standard normal truncated at +-cut (resampling, like your misalignments())."""
    if size is None:
        while True:
            v = rng.normal()
            if abs(v) <= cut:
                return v
    out = rng.normal(size=size)
    bad = np.abs(out) > cut
    while np.any(bad):
        out[bad] = rng.normal(size=bad.sum())
        bad = np.abs(out) > cut
    return out


def results_dir(design, config, phase, changes=None, metric=None, sub=None):
    """Same layout as my_functions.results_dir (inlined to avoid its cpymad import)."""
    base = f'Results/D{design}/C{config}/{phase}deg' + (f'_{changes}' if changes else '')
    for part in (metric, sub):
        if part:
            base = f'{base}/{part}'
    os.makedirs(base, exist_ok=True)
    return base


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def magnet_names(line):
    tt = line.get_table()
    out = {}
    for t in MAGNET_TYPES:
        out[t] = [n for n in tt.rows[tt.element_type == t].name]
    return out


# =============================================================================
# Lattice preparation
# =============================================================================
def load_environment(design, config, phase, changes):
    tag = f'_{changes}' if changes else ''
    path = f'JSON_Files/D{design}/C{config}/pdr_perfect_{phase}{tag}.json'
    log(f'Loading {path}')
    return xt.Environment.from_json(path)


def insert_bpms_and_correctors(pdr, design, config):
    """Same choice as analysis.py / spin_tracking.py."""
    mc.insert_BPMs_all_as_markers(pdr)
    if design == 1 and config == 1:
        mc.insert_correctors_var2(pdr)
    else:
        mc.insert_correctors(pdr)


def insert_apertures(pdr, line, mags, apertures):
    """LimitEllipse at entry and exit of every magnet."""
    places = []
    for t, names in mags.items():
        a, b = apertures[t]['chamber_a'], apertures[t]['chamber_b']
        for m in names:
            for end in ('start', 'end'):
                nm = f'AP{end[0]}_{m}'
                pdr.new(nm, xt.LimitEllipse, a=a, b=b)
                places.append(pdr.place(nm, at=f'{m}@{end}'))
    line.insert(places)


def insert_error_kicks(pdr, line, mags):
    """Thin multipoles (zero strength) at entry and exit of every magnet."""
    places = []
    for names in mags.values():
        for m in names:
            for end in ('start', 'end'):
                nm = f'MPE{end[0]}_{m}'
                pdr.new(nm, xt.Multipole, knl=np.zeros(N_KNL), ksl=np.zeros(N_KNL))
                places.append(pdr.place(nm, at=f'{m}@{end}'))
    line.insert(places)


# =============================================================================
# Stage 1: injected beam and physical aperture
# =============================================================================
def injected_beam_requirements(dist_file, p0c, gamma_beta, containment,
                               fallback_nemitt=7e-3, fallback_delta=0.03):
    """W_max (Courant-Snyder invariant, m) per plane and delta_max."""
    if dist_file and os.path.exists(dist_file):
        df = pd.read_csv(dist_file, sep=r'\s+')
        delta = (df['p[MeV/c]'].values * 1e6 - p0c) / p0c
        dd = delta - delta.mean()
        res = {'source': dist_file, 'n_particles': len(df)}
        for plane, (pc, ac) in {'x': ('x[mm]', 'xp[mrad]'), 'y': ('y[mm]', 'yp[mrad]')}.items():
            u = df[pc].values * 1e-3
            up = df[ac].values * 1e-3
            # remove the beam's own dispersive correlation before computing invariants
            D = np.cov(u, dd)[0, 1] / np.var(dd)
            Dp = np.cov(up, dd)[0, 1] / np.var(dd)
            u = u - u.mean() - D * dd
            up = up - up.mean() - Dp * dd
            s11, s22, s12 = np.var(u), np.var(up), np.cov(u, up)[0, 1]
            eps = np.sqrt(s11 * s22 - s12**2)
            beta, alpha = s11 / eps, -s12 / eps
            gamma = (1 + alpha**2) / beta
            W = gamma * u**2 + 2 * alpha * u * up + beta * up**2
            res[f'eps_rms_{plane}'] = eps
            res[f'W_max_{plane}'] = float(np.quantile(W, containment))
        res['delta_mean'] = float(delta.mean())
        res['delta_max'] = float(np.quantile(np.abs(delta), containment))
        return res, df
    log(f'Distribution file not found ({dist_file}); using CLIC PDR fallback: '
        f'eps_rms,n = {fallback_nemitt} m, W_max = 20 eps_rms, delta_max = {fallback_delta}')
    eps = fallback_nemitt / gamma_beta
    return {'source': 'CLIC PDR fallback (Antoniou, CLIC-Note-989)',
            'eps_rms_x': eps, 'eps_rms_y': eps,
            'W_max_x': 20 * eps, 'W_max_y': 20 * eps,
            'delta_mean': 0.0, 'delta_max': fallback_delta}, None


def required_aperture(line, mags, beam, d_co, ds=0.05):
    """Beam-stay-clear sampled every ds metres; max taken inside each magnet."""
    L = line.get_length()
    s_grid = np.arange(ds, L - ds / 2, ds)
    lc = line.copy(shallow=True)
    lc.cut_at_s(s_grid)
    tw = lc.twiss4d()
    s = np.asarray(tw.s)
    Rx = np.sqrt(tw.betx * beam['W_max_x']) + np.abs(tw.dx) * beam['delta_max'] + d_co
    Ry = np.sqrt(tw.bety * beam['W_max_y']) + np.abs(tw.dy) * beam['delta_max'] + d_co

    tt = line.get_table()
    s_of = dict(zip(tt.name, tt.s))
    rows = []
    for t, names in mags.items():
        for m in names:
            s0 = s_of[m]
            s1 = s0 + line[m].length
            i0, i1 = np.searchsorted(s, s0 - 1e-9), np.searchsorted(s, s1 + 1e-9)
            sl = slice(i0, max(i1, i0 + 1))
            rows.append({'name': m, 'type': t, 's_start': s0, 's_end': s1,
                         'betx_max': float(np.max(tw.betx[sl])),
                         'bety_max': float(np.max(tw.bety[sl])),
                         'dx_max': float(np.max(np.abs(tw.dx[sl]))),
                         'R_req_x': float(np.max(Rx[sl])),
                         'R_req_y': float(np.max(Ry[sl]))})
    return pd.DataFrame(rows), (s, Rx, Ry)


def choose_apertures(df_req, mode, fixed, wall, ref_fraction, round_to=1e-3):
    """Chamber half-apertures, pole radius and R_ref per magnet type."""
    ap = {}
    for t in MAGNET_TYPES:
        sub = df_req[df_req.type == t]
        if len(sub) == 0:
            continue
        rx, ry = sub.R_req_x.max(), sub.R_req_y.max()
        if mode == 'fixed':
            a, b = fixed[t]
        elif t == 'Bend':     # elliptical: wide horizontally, gap vertically
            a = np.ceil(rx / round_to) * round_to
            b = np.ceil(ry / round_to) * round_to
        else:                 # round bore
            a = b = np.ceil(max(rx, ry) / round_to) * round_to
        # Bend: field quality is set by the vertical half gap
        pole = (b if t == 'Bend' else max(a, b)) + wall
        ap[t] = {'chamber_a': float(a), 'chamber_b': float(b),
                 'R_req_x_max': float(rx), 'R_req_y_max': float(ry),
                 'pole_radius': float(pole), 'R_ref': float(ref_fraction * pole),
                 'margin_x': float(a - rx), 'margin_y': float(b - ry)}
    return ap


def plot_aperture(curves, df_req, apertures, out_png):
    s, Rx, Ry = curves
    fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, R, key, lab in [(axs[0], Rx, 'chamber_a', 'x'), (axs[1], Ry, 'chamber_b', 'y')]:
        ax.plot(s, R * 1e3, 'k-', lw=0.8, label=f'required $R_{lab}$')
        colors = {'Bend': 'tab:blue', 'Quadrupole': 'tab:red', 'Sextupole': 'tab:green'}
        for t, c in colors.items():
            if t not in apertures:
                continue
            sub = df_req[df_req.type == t]
            for _, r in sub.iterrows():
                ax.plot([r.s_start, r.s_end], [apertures[t][key] * 1e3] * 2, color=c, lw=3)
            ax.plot([], [], color=c, lw=3, label=f'{t} chamber')
        ax.set_ylabel(f'half-aperture {lab} [mm]')
        ax.grid(alpha=0.3)
        ax.set_ylim(0, None)
        ax.legend(fontsize='small', ncol=4, loc='lower left', bbox_to_anchor=(0, 1.0),
                  frameon=False)
    axs[1].set_xlabel('s [m]')
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# =============================================================================
# Stage 2: errors
# =============================================================================
def apply_alignment_and_field_errors(line, mags, seed, sig_shift, sig_rot,
                                     sig_field, cut):
    """Same distribution as mc.misalignments(), but the main-field error acts on
    k0/k1/k2 (mc.misalignments scales knl, which is zero on thick magnets)."""
    rng = np.random.default_rng([seed, 0])
    r = line.element_refs
    for t in MAGNET_TYPES:
        for m in mags[t]:
            ref = r[m]
            ref.shift_x = sig_shift * truncnorm(rng, cut)
            ref.shift_y = sig_shift * truncnorm(rng, cut)
            ref.shift_s = sig_shift * truncnorm(rng, cut)
            ref.rot_s_rad = sig_rot * truncnorm(rng, cut)
            ref.rot_x_rad = sig_rot * truncnorm(rng, cut)
            ref.rot_y_rad = sig_rot * truncnorm(rng, cut)
            e = sig_field * truncnorm(rng, cut)
            if t == 'Quadrupole':
                ref.k1 = ref.k1 * (1 + e)
            elif t == 'Sextupole':
                ref.k2 = ref.k2 * (1 + e)
            else:
                line[m].k0_from_h = False
                ref.k0 = ref.angle / ref.length * (1 + e)
    # error kicks follow their magnet (feed-down)
    for names in mags.values():
        for m in names:
            for end in 'se':
                k = line[f'MPE{end}_{m}']
                k.shift_x, k.shift_y = line[m].shift_x, line[m].shift_y
                k.rot_s_rad = line[m].rot_s_rad


def misalign_correctors(line, seed, sig_shift, cut):
    """Like mc.misalignments_correctors but only touches the steering
    correctors (Mx_/My_), not the thin multipole-error kicks."""
    rng = np.random.default_rng([seed, 2])
    tt = line.get_table()
    names = [n for n in tt.name if n.startswith(('Mx_', 'My_'))]
    for n in names:
        ref = line.element_refs[n]
        ref.shift_x = sig_shift * truncnorm(rng, cut)
        ref.shift_y = sig_shift * truncnorm(rng, cut)
        ref.rot_s_rad = sig_shift * truncnorm(rng, cut)
        e = 1 + 1e-3 * truncnorm(rng, cut)
        ref.knl = ref.knl * e
        ref.ksl = ref.ksl * e


def apply_multipoles(line, mags, table, apertures, seed, scale, only, cut):
    """Systematic + random multipoles, both multiplied by `scale`. The unit
    random draws depend only on (seed, magnet, harmonic), never on scale."""
    rng = np.random.default_rng([seed, 1])
    r = line.element_refs
    for t in MAGNET_TYPES:
        entries = table.get(t, [])
        if not entries or t not in apertures:
            continue
        m_ord, R = MAIN_ORDER[t], apertures[t]['R_ref']
        for m in mags[t]:
            z = truncnorm(rng, cut, size=len(entries))   # always drawn -> same stream
            if t == 'Quadrupole':
                KL = r[m].k1 * r[m].length
            elif t == 'Sextupole':
                KL = r[m].k2 * r[m].length
            else:
                KL = r[m].k0 * r[m].length
            for (comp, n, sys_, rnd), zi in zip(entries, z):
                if only and f'{t}:{comp}{n}' not in only:
                    continue
                bn = scale * (sys_ + rnd * zi)
                fac = 0.5 * bn * factorial(n - 1) / factorial(m_ord - 1) * R**(m_ord - n)
                for end in 'se':
                    k = r[f'MPE{end}_{m}']
                    if comp == 'b':
                        k.knl[n - 1] = KL * fac
                    else:
                        k.ksl[n - 1] = KL * fac


def correct_orbit(line, seed):
    tw = line.twiss(method='6d')
    try:
        mc.orbit_correction(line, tw, threading=False, seed=seed)
    except Exception as e:
        log(f'  seed {seed}: closed-orbit correction failed ({e}); trying threading')
        mc.orbit_correction(line, tw, threading=True, seed=seed)


# =============================================================================
# Stage 2: metrics
# =============================================================================
def track_DA(line, tw, beam, gb, n_turns, n_angles, n_r, r_max):
    """DA in units of the required acceptance: x_norm = 1 <=> W = W_max."""
    theta = np.linspace(0, np.pi, n_angles)
    r = np.linspace(r_max / n_r, r_max, n_r)
    TT, RR = np.meshgrid(theta, r, indexing='ij')
    xn, yn = (RR * np.cos(TT)).ravel(), (RR * np.sin(TT)).ravel()
    p = line.build_particles(x_norm=xn, px_norm=0, y_norm=yn, py_norm=0, delta=0,
                             nemitt_x=beam['W_max_x'] * gb, nemitt_y=beam['W_max_y'] * gb,
                             W_matrix=tw.W_matrix[0], particle_on_co=tw.particle_on_co.copy())
    line.track(p, num_turns=n_turns)
    p.sort(interleave_lost_particles=True)
    alive = (p.state > 0).reshape(n_angles, n_r)
    da = np.zeros(n_angles)
    for i in range(n_angles):
        lost = np.where(~alive[i])[0]
        k = lost[0] if len(lost) else n_r
        da[i] = r[k - 1] if k > 0 else 0.0
    # required region: |x_n| <= 1 and |y_n| <= 1 (per-plane requirement)
    r_req = 1.0 / np.maximum(np.abs(np.cos(theta)), np.abs(np.sin(theta)))
    i_y = np.argmin(np.abs(theta - np.pi / 2))
    return {'DA_ratio_min': float(np.min(da / r_req)),
            'DA_x_pos': float(da[0]), 'DA_x_neg': float(da[-1]), 'DA_y': float(da[i_y]),
            'DA_boundary': da.tolist()}


def track_MA(line, tw, beam, gb, n_turns, delta_scan, n_delta, amp=0.05):
    """Momentum acceptance with a small betatron amplitude (amp, in units of x_norm)."""
    deltas = np.linspace(-delta_scan, delta_scan, 2 * n_delta + 1)
    p = line.build_particles(x_norm=amp, px_norm=0, y_norm=amp, py_norm=0, delta=deltas,
                             nemitt_x=beam['W_max_x'] * gb, nemitt_y=beam['W_max_y'] * gb,
                             W_matrix=tw.W_matrix[0], particle_on_co=tw.particle_on_co.copy())
    line.track(p, num_turns=n_turns)
    p.sort(interleave_lost_particles=True)
    alive = p.state > 0
    i0 = n_delta
    ma = {}
    for sign, rng_ in (('pos', range(i0, len(deltas))), ('neg', range(i0, -1, -1))):
        last = 0.0
        for i in rng_:
            if not alive[i]:
                break
            last = deltas[i]
        ma[f'MA_{sign}'] = float(last)
    return ma


def injection_particles(line, tw, df, n_inj, seed, disp_match=True):
    sub = df.sample(n=min(n_inj, len(df)), random_state=seed)
    p0c = line.particle_ref.p0c[0]
    beta0 = line.particle_ref.beta0[0]
    delta = (sub['p[MeV/c]'].values * 1e6 - p0c) / p0c
    x = sub['x[mm]'].values * 1e-3
    y = sub['y[mm]'].values * 1e-3
    px = sub['xp[mrad]'].values * 1e-3 * (1 + delta)
    py = sub['yp[mrad]'].values * 1e-3 * (1 + delta)
    t_mm = sub['t[mm/c]'].values
    zeta = (np.mean(t_mm) - t_mm) * 1e-3 * beta0      # as in macroparticles.py
    x, px, y, py = x - x.mean(), px - px.mean(), y - y.mean(), py - py.mean()
    if disp_match:
        x = x + tw.dx[0] * delta
        px = px + tw.dpx[0] * delta
        y = y + tw.dy[0] * delta
        py = py + tw.dpy[0] * delta
    ref = line.particle_ref
    return xt.Particles(_context=line._context,
                        mass0=ref.mass0, q0=ref.q0, p0c=ref.p0c[0],
                        x=x + tw.x[0], px=px + tw.px[0],
                        y=y + tw.y[0], py=py + tw.py[0],
                        zeta=zeta + tw.zeta[0], delta=delta)


def track_injection(line, tw, df, n_inj, n_turns, seed):
    p = injection_particles(line, tw, df, n_inj, seed)
    line.track(p, num_turns=n_turns)
    return {'inj_survival': float(np.mean(p.state > 0))}


# =============================================================================
# Summary
# =============================================================================
def summarise(df_res, args, out_dir):
    df = df_res.copy()
    df['pass'] = df['ok'] & (df['DA_ratio_min'] >= args.da_required)
    if 'inj_survival' in df and df['inj_survival'].notna().any():
        df['pass'] &= df['inj_survival'] >= args.inj_required
    if args.require_ma:
        df['pass'] &= (df['MA_pos'] >= df['delta_max']) & (-df['MA_neg'] >= df['delta_max'])
    g = df.groupby('scale')
    summ = pd.DataFrame({
        'n_seeds': g.size(),
        'pass_fraction': g['pass'].mean(),
        'DA_ratio_p10': g['DA_ratio_min'].quantile(0.1),
        'DA_ratio_median': g['DA_ratio_min'].median(),
        'MA_pos_median': g['MA_pos'].median(),
        'MA_neg_median': g['MA_neg'].median(),
        'inj_p10': g['inj_survival'].quantile(0.1),
        'inj_median': g['inj_survival'].median(),
        'orbit_rms_x_mm': g['orbit_rms_x'].median() * 1e3,
        'orbit_rms_y_mm': g['orbit_rms_y'].median() * 1e3,
    }).reset_index()
    tol = None
    for _, row in summ.sort_values('scale').iterrows():
        if row.pass_fraction >= args.pass_target:
            tol = row.scale
        else:
            break
    summ.to_csv(f'{out_dir}/summary.csv', index=False)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    sc = summ.scale.values
    axs[0].plot(sc, summ.pass_fraction, 'o-')
    axs[0].axhline(args.pass_target, color='r', ls='--', lw=1)
    axs[0].set_ylabel('fraction of seeds passing')
    axs[1].scatter(df.scale, df.DA_ratio_min, s=10, alpha=0.4)
    axs[1].plot(sc, summ.DA_ratio_median, 'k-', label='median')
    axs[1].plot(sc, summ.DA_ratio_p10, 'k--', label='10th pct')
    axs[1].axhline(args.da_required, color='r', ls='--', lw=1)
    axs[1].set_ylabel('DA / required acceptance')
    axs[1].legend(fontsize='small')
    if df['inj_survival'].notna().any():
        axs[2].scatter(df.scale, df.inj_survival * 100, s=10, alpha=0.4)
        axs[2].plot(sc, summ.inj_median * 100, 'k-')
        axs[2].plot(sc, summ.inj_p10 * 100, 'k--')
        axs[2].axhline(args.inj_required * 100, color='r', ls='--', lw=1)
    axs[2].set_ylabel('injection survival [%]')
    for ax in axs:
        ax.set_xlabel('multipole error scale')
        ax.grid(alpha=0.3)
    fig.suptitle(f'{args.table}: tolerance (>= {args.pass_target:.0%} seeds pass) = '
                 f'{"none" if tol is None else f"{tol:g}x"}')
    fig.tight_layout()
    fig.savefig(f'{out_dir}/tolerance_scan.png', dpi=200)
    plt.close(fig)
    log(f'Summary:\n{summ.to_string(index=False)}')
    log(f'Tolerance (largest scale with all smaller scales passing): '
        f'{"none" if tol is None else f"{tol:g}x"}')


# =============================================================================
# Main
# =============================================================================
def parse_args():
    env = os.environ.get
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--design', type=int, default=int(env('DESIGN', 1)))
    ap.add_argument('--config', type=int, default=int(env('CONFIG', 9)))
    ap.add_argument('--phase', type=int, default=int(env('PHASE', 90)))
    ap.add_argument('--changes', default=env('CHANGES', None))
    # beam / aperture
    ap.add_argument('--dist', default='PositronBeam_2p86GeV_PolarizedEbeam/beam_ECS_04092026.dat')
    ap.add_argument('--containment', type=float, default=0.999,
                    help='beam fraction defining W_max and delta_max')
    ap.add_argument('--d-co', type=float, default=1e-3, help='orbit + alignment allowance [m]')
    ap.add_argument('--wall', type=float, default=1.5e-3, help='chamber wall + clearance to pole [m]')
    ap.add_argument('--ref-fraction', type=float, default=2 / 3, help='R_ref / pole radius')
    ap.add_argument('--aperture-mode', choices=['required', 'fixed'], default='required')
    ap.add_argument('--fixed-aperture', default=None,
                    help='JSON {type: [a, b]} chamber half-apertures [m] for --aperture-mode fixed')
    ap.add_argument('--aperture-only', action='store_true')
    ap.add_argument('--start-element', default=None,
                    help='cycle the ring so tracking starts here (e.g. injection point)')
    # errors
    ap.add_argument('--table', default='ilc_dr', choices=list(ERROR_TABLES))
    ap.add_argument('--scales', default='0,1,2,5,10,20')
    ap.add_argument('--n-seeds', type=int, default=10)
    ap.add_argument('--seed0', type=int, default=1000)
    ap.add_argument('--only', default=None, help='e.g. Quadrupole:b6,Bend:b3')
    ap.add_argument('--sig-shift', type=float, default=0.25e-3)
    ap.add_argument('--sig-rot', type=float, default=0.25e-3)
    ap.add_argument('--sig-field', type=float, default=1e-3)
    ap.add_argument('--cut', type=float, default=2.5)
    ap.add_argument('--no-correction', action='store_true')
    # tracking
    ap.add_argument('--turns-da', type=int, default=1000)
    ap.add_argument('--turns-ma', type=int, default=1000)
    ap.add_argument('--turns-inj', type=int, default=2000)
    ap.add_argument('--n-inj', type=int, default=2000)
    ap.add_argument('--n-angles', type=int, default=13)
    ap.add_argument('--n-r', type=int, default=30)
    ap.add_argument('--r-max', type=float, default=3.0, help='DA scan limit in units of required acceptance')
    ap.add_argument('--delta-scan', type=float, default=0.04)
    ap.add_argument('--n-delta', type=int, default=20)
    ap.add_argument('--radiation', default='mean', choices=['mean', 'none'])
    ap.add_argument('--threads', default='auto',
                    help="OpenMP threads for tracking ('auto', N, or 0 = serial)")
    # pass criteria
    ap.add_argument('--da-required', type=float, default=1.0)
    ap.add_argument('--inj-required', type=float, default=0.95)
    ap.add_argument('--require-ma', action='store_true')
    ap.add_argument('--pass-target', type=float, default=0.9)
    return ap.parse_args()


def main():
    args = parse_args()
    only = set(args.only.split(',')) if args.only else None
    table = ERROR_TABLES[args.table]
    out_dir = results_dir(args.design, args.config, args.phase, changes=args.changes,
                             metric='ToleranceScan', sub=args.table + (f'_only_{args.only}' if only else ''))
    ap_dir = results_dir(args.design, args.config, args.phase, changes=args.changes,
                            metric='PhysicalAperture')

    # ---------------- lattice ----------------
    pdr = load_environment(args.design, args.config, args.phase, args.changes)
    ring = pdr.lines['ring']
    ring.configure_radiation(model=None)
    insert_bpms_and_correctors(pdr, args.design, args.config)
    if args.start_element:
        ring.cycle(name_first_element=args.start_element, inplace=True)
    mags = magnet_names(ring)
    log('Magnets: ' + ', '.join(f'{t}: {len(v)}' for t, v in mags.items()))

    p0c = ring.particle_ref.p0c[0]
    gb = ring.particle_ref.gamma0[0] * ring.particle_ref.beta0[0]

    # ---------------- stage 1 ----------------
    beam, dist_df = injected_beam_requirements(args.dist, p0c, gb, args.containment)
    log('Injected beam: ' + ', '.join(f'{k}={v:.3g}' if isinstance(v, float) else f'{k}={v}'
                                       for k, v in beam.items()))
    df_req, curves = required_aperture(ring, mags, beam, args.d_co)
    fixed = None
    if args.aperture_mode == 'fixed':
        if not args.fixed_aperture:
            sys.exit('--aperture-mode fixed needs --fixed-aperture')
        fixed = {k: tuple(v) for k, v in json.loads(args.fixed_aperture).items()}
    apertures = choose_apertures(df_req, args.aperture_mode, fixed, args.wall, args.ref_fraction)

    df_req.to_csv(f'{ap_dir}/required_aperture_per_magnet.csv', index=False)
    with open(f'{ap_dir}/apertures.json', 'w') as f:
        json.dump({'beam': beam, 'd_co': args.d_co, 'wall': args.wall,
                   'containment': args.containment, 'mode': args.aperture_mode,
                   'apertures': apertures}, f, indent=2)
    plot_aperture(curves, df_req, apertures, f'{ap_dir}/required_aperture.png')
    for t, a in apertures.items():
        log(f'{t:10s} chamber a/b = {a["chamber_a"]*1e3:.1f}/{a["chamber_b"]*1e3:.1f} mm '
            f'(required {a["R_req_x_max"]*1e3:.1f}/{a["R_req_y_max"]*1e3:.1f} mm), '
            f'pole {a["pole_radius"]*1e3:.1f} mm, R_ref {a["R_ref"]*1e3:.1f} mm')
        if a['margin_x'] < 0 or a['margin_y'] < 0:
            log(f'  WARNING: {t} chamber is smaller than the requirement somewhere')
    log(f'Stage 1 written to {ap_dir}')
    if args.aperture_only:
        return

    # ---------------- stage 2 ----------------
    insert_apertures(pdr, ring, mags, apertures)
    insert_error_kicks(pdr, ring, mags)
    base = ring.copy()

    csv_path = f'{out_dir}/runs.csv'
    done = set()
    if os.path.exists(csv_path):
        prev = pd.read_csv(csv_path)
        done = set(zip(prev.scale, prev.seed))
        log(f'Resuming: {len(done)} runs already in {csv_path}')
    with open(f'{out_dir}/settings.json', 'w') as f:
        json.dump({**vars(args), 'apertures': apertures, 'beam': beam}, f, indent=2, default=str)

    scales = [float(s) for s in args.scales.split(',')]
    seeds = [args.seed0 + i for i in range(args.n_seeds)]
    serial_context = xo.ContextCpu()
    threads = int(args.threads) if str(args.threads).isdigit() else args.threads
    context = serial_context if threads == 0 else xo.ContextCpu(omp_num_threads=threads)

    for scale in scales:
        for seed in seeds:
            if (scale, seed) in done:
                continue
            t0 = time.time()
            row = {'scale': scale, 'seed': seed, 'ok': False,
                   'delta_max': beam['delta_max'], 'inj_survival': np.nan}
            try:
                line = base.copy()
                # Serial tracker for twiss / orbit correction: Xsuite does not
                # support twiss with radiation on an OpenMP context.
                line.build_tracker(_context=serial_context)
                apply_alignment_and_field_errors(line, mags, seed, args.sig_shift,
                                                 args.sig_rot, args.sig_field, args.cut)
                apply_multipoles(line, mags, table, apertures, seed, scale, only, args.cut)
                if args.radiation == 'mean':
                    line.configure_radiation(model='mean')
                if not args.no_correction:
                    misalign_correctors(line, seed, args.sig_shift, args.cut)
                    correct_orbit(line, seed)
                tw = line.twiss(method='6d' if args.radiation == 'mean' else '4d')
                # Everything twiss-dependent is now in `tw`; switch to the
                # tracking context (OpenMP if --threads != 0).
                if context is not serial_context:
                    line.discard_tracker()
                    line.build_tracker(_context=context)
                    if args.radiation == 'mean':
                        line.configure_radiation(model='mean')
                row.update({'qx': tw.qx, 'qy': tw.qy,
                            'orbit_rms_x': float(np.std(tw.x)), 'orbit_rms_y': float(np.std(tw.y))})
                da = track_DA(line, tw, beam, gb, args.turns_da, args.n_angles, args.n_r, args.r_max)
                row.update({k: v for k, v in da.items() if k != 'DA_boundary'})
                row['DA_boundary'] = json.dumps(da['DA_boundary'])
                row.update(track_MA(line, tw, beam, gb, args.turns_ma, args.delta_scan, args.n_delta))
                if dist_df is not None:
                    row.update(track_injection(line, tw, dist_df, args.n_inj, args.turns_inj, seed))
                row['ok'] = True
            except Exception as e:
                log(f'  scale {scale:g} seed {seed}: FAILED ({type(e).__name__}: {e})')
                row.update({'DA_ratio_min': 0.0, 'MA_pos': 0.0, 'MA_neg': 0.0,
                            'inj_survival': 0.0 if dist_df is not None else np.nan,
                            'error': str(e)[:200]})
            row['runtime_s'] = time.time() - t0
            pd.DataFrame([row]).to_csv(csv_path, mode='a', index=False,
                                       header=not os.path.exists(csv_path))
            log(f'scale {scale:g} seed {seed}: DA/req = {row.get("DA_ratio_min", 0):.2f}, '
                f'MA = [{row.get("MA_neg", 0)*100:.2f}, {row.get("MA_pos", 0)*100:.2f}] %, '
                f'inj = {row.get("inj_survival", np.nan):.3f} ({row["runtime_s"]:.0f} s)')

    summarise(pd.read_csv(csv_path), args, out_dir)
    log(f'Results in {out_dir}')


if __name__ == '__main__':
    main()