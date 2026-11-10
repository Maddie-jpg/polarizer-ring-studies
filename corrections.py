#%%
import numpy as np
import LatticeBuild.misalignments_corrections as mc
import xtrack as xt
import matplotlib.pyplot as plt

pdr=xt.Environment.from_json('/home/mwatson/Documents/laughing-octo-bassoon/JSON_Files/D1/C9/pdr_perfect_90_FDF.json')
pdr_copy = pdr.copy()
mc.insert_bpms_and_correctors(pdr_copy, 5, 9)

ring = pdr_copy.lines['ring']          # BPMs + correctors already inserted

# ---- Twiss of the ring with BPMs and correctors, as a MAD-X style TFS file ----
import paths
twiss_dir = paths.results_dir(1, 9, 90, changes='FDF', metric='Twiss')
tw_ring = ring.twiss(method='6d')
tw_ring['keyword'] = np.array(ring.get_table().element_type, dtype=object)
twiss_file = f'{twiss_dir}/twiss_ring_bpms_correctors.tfs'
tw_ring.to_tfs(twiss_file, missing='ignore', include=[
    'name', 'keyword', 's', 'betx', 'alfx', 'mux', 'dx', 'dpx', 'x', 'px',
    'bety', 'alfy', 'muy', 'dy', 'dpy', 'y', 'py',
    'circumference', 'qx', 'qy', 'dqx', 'dqy', 'qs', 'momentum_compaction_factor'])
print(f'Twiss written to {twiss_file}')

brho = float(ring.particle_ref.p0c[0]) / 299792458.0
fractions = [1.0, 0.75, 0.50, 0.25, 0.10]
seeds = range(20)

results = {}                      # frac -> dict of strengths / orbits

for frac in fractions:
    sx, sy, orbits = {}, {}, {}
    for seed in seeds:
        line = mc.prepare_seed_line(
            ring, seed=seed, correct=True,
            n_eig_x=None, n_eig_y=None,      # must be None for eig_fraction to apply
            eig_fraction=frac)
        for name in line.get_table().name:
            el = line.element_dict.get(name)
            if name.startswith('Mx_'):
                th = float(np.atleast_1d(el.knl)[0])
                sx[(seed, name)] = brho * th / el.length          # B [T]
            elif name.startswith('My_'):
                th = float(np.atleast_1d(el.ksl)[0])
                sy[(seed, name)] = brho * th / el.length
        tw = line.twiss(method='6d')
        orbits[seed] = (np.array(tw.s), np.array(tw.x), np.array(tw.y))
    results[frac] = dict(x=sx, y=sy, orbits=orbits)
    print(f'--- {frac*100:.0f}% of eigenvectors done ---')

# ---- strongest corrector per plane per fraction ----
worst = {}
for frac in fractions:
    for plane in ('x', 'y'):
        (seed, name), B = max(results[frac][plane].items(),
                              key=lambda kv: abs(kv[1]))
        worst[(frac, plane)] = (name, seed, abs(B))
        print(f'{frac*100:5.0f}%  {plane}: {name:24s} seed {seed:3d}  '
              f'B = {abs(B)*1e3:7.2f} mT')

# ---- orbits, one column per fraction ----
fig, axs = plt.subplots(2, len(fractions), figsize=(4*len(fractions), 7),
                        sharex=True, sharey='row')
for j, frac in enumerate(fractions):
    for i, plane in enumerate(('x', 'y')):
        ax = axs[i, j]
        for seed in seeds:
            s, x, y = results[frac]['orbits'][seed]
            ax.plot(s, (x if plane == 'x' else y) * 1e3, lw=0.6)
        ax.axhline(0, color='0.8', lw=0.8)
        ax.grid(alpha=0.25)
        name, seed, B = worst[(frac, plane)]
        ax.text(0.5, -0.19, f'{name}\nseed {seed},  B = {B*1e3:.2f} mT',
                transform=ax.transAxes, ha='center', va='top', fontsize=8)
        if j == 0:
            ax.set_ylabel(f'{plane} [mm]')
        if i == 0:
            ax.set_title(f'{frac*100:.0f}% of eigenvectors')
        if i == 1:
            ax.set_xlabel('s [m]')
fig.tight_layout()
fig.savefig('orbits_vs_eigenvectors.png', dpi=150)

# ---- the trade-off: corrector strength vs orbit quality ----
fig2, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()
pc = [f*100 for f in fractions]
for plane, c in (('x', 'tab:blue'), ('y', 'tab:red')):
    ax1.plot(pc, [worst[(f, plane)][2]*1e3 for f in fractions],
             'o-', color=c, label=f'max corrector B, {plane}')
    ax2.plot(pc, [np.mean([np.std(results[f]['orbits'][s][1 if plane=='x' else 2])
                           for s in seeds])*1e3 for f in fractions],
             's--', color=c, alpha=0.5, label=f'mean orbit rms, {plane}')
ax1.set_xlabel('eigenvectors used [%]')
ax1.set_ylabel('strongest corrector B [mT]')
ax2.set_ylabel('mean orbit rms [mm]')
ax1.grid(alpha=0.25)
ax1.legend(loc='upper left', fontsize=9)
ax2.legend(loc='upper right', fontsize=9)
fig2.tight_layout()
fig2.savefig('corrector_strength_vs_eigenvectors.png', dpi=150)
plt.show()
# %% Lattice sketches with the injected beam's sizes from macroparticles.py
import pandas as pd
import config as cfg
import paths
import lattice_Sketch as ls

# Must match the lattice loaded at the top of this file.
design, config, phase, changes = 1, 9, 90, 'FDF'
show_beam = True      # beam sizes above the cell and arc plots; False = layouts only
beam_n_sigma = (1, 3, 5)   # beam size drawn, in rms units: 1, 3, ... or several
sketch_sextant = '1R'


def main_bunch_sigma_delta(beam_file, t_col='t[mm/c]', mad_k=10):
    """rms momentum spread of the main bunch, selected exactly as in
    macroparticles.select_main_bunch (|t - median| < mad_k * 1.4826 * MAD)."""
    df = pd.read_csv(beam_file, sep=r'\s+')
    t = df[t_col].values.astype(float)
    med = np.median(t)
    mad = np.median(np.abs(t - med))
    if mad > 0:
        df = df[np.abs(t - med) < mad_k * mad * 1.4826]
    return float(np.std(df['p[MeV/c]']) / np.mean(df['p[MeV/c]']))


def ring_beam_sizes(ring, beam, ds=0.02):
    """rms beam sizes around the ring, as in macroparticles.physical_aperture_study:

        sigma_x(s) = sqrt(beta_x eps_x + (D_x sigma_delta)^2)
        sigma_y(s) = sqrt(beta_y eps_y + (D_y sigma_delta)^2)

    eps_x, eps_y are the core emittances and sigma_delta the main-bunch
    momentum spread, both from TwissResults.json (beam = paths.load_injected_beam).
    Older TwissResults.json files have no main-bunch spread; it is then
    recomputed from config.INJECTED_BEAM_FILE (without the energy compressor).
    The optics are sampled every ds metres on a copy of the ring, radiation off.
    Returns a dict of arrays in metres plus the inputs used."""
    eps_x = beam['x']['emittance_geo'] * 1e-6
    eps_y = beam['y']['emittance_geo'] * 1e-6
    sigma_delta = beam.get('longitudinal', {}).get('delta_rms_main_bunch')
    source = 'TwissResults.json'
    if sigma_delta is None:
        sigma_delta = main_bunch_sigma_delta(cfg.INJECTED_BEAM_FILE)
        source = f'{cfg.INJECTED_BEAM_FILE} (rerun macroparticles.py to store it)'
    print(f'Beam sizes: eps_x = {eps_x*1e6:.3f} um, eps_y = {eps_y*1e6:.3f} um, '
          f'sigma_delta = {sigma_delta*1e3:.2f} per mil (main bunch, from {source})')

    ring_bs = ring.copy()
    ring_bs.configure_radiation(model=None)
    ring_bs.cut_at_s(np.arange(ds, ring_bs.get_length() - ds / 2, ds))
    tw = ring_bs.twiss4d()
    return dict(s=np.asarray(tw.s),
                sig_x=np.sqrt(tw.betx * eps_x + (tw.dx * sigma_delta) ** 2),
                sig_y=np.sqrt(tw.bety * eps_y + (tw.dy * sigma_delta) ** 2),
                eps_x=eps_x, eps_y=eps_y, sigma_delta=sigma_delta)


sketch_dir = paths.results_dir(design, config, phase, changes=changes,
                               metric='LatticeSketch')
beam = ring_beam_sizes(pdr.lines['ring'],
                       paths.load_injected_beam(design, config, phase, changes))

pdr_sketch = pdr.copy()
mc.insert_bpms_and_correctors(pdr_sketch, design, config)

sketches = ls.sketch_all(pdr_sketch, sextant=sketch_sextant, outdir=sketch_dir,
                         beam=beam if show_beam else None, n_sigma=beam_n_sigma)
sketches['period'] = ls.sketch_period_beam_size(
    pdr_sketch, beam, sextants=('3R', '1L'), n_sigma=beam_n_sigma, outdir=sketch_dir)

# Floor plots with the FODO cells, matching cell, dispersion suppressor, doublet
# and triplet labelled: one sextant, one period and the whole ring.
for survey_line in ('arc1R', 'period', 'ring'):
    if survey_line in pdr.lines:
        sketches[f'survey_{survey_line}'] = ls.survey_sections(
            pdr.lines[survey_line], outdir=sketch_dir,
            fname=f'survey_sections_{survey_line}.png',
            a4=(survey_line == 'arc1R'))     # arc: A4 text width, PNG + PDF
plt.show()


# %% Injection: septum + kicker
#
# Single-turn off-axis injection. The septum sits in one long straight and the
# fast kicker in another, so the kicker launches a one-turn betatron
# oscillation in the stored beam -- a local closed bump would need three or
# four kickers in the SAME straight, which is not what is installed here.
#
# The kick is the free parameter. Linear theory says the stored beam arrives
# at the septum displaced by
#
#     x_sep = -theta_k * sqrt(beta_k * beta_sep) * sin(dmu_k->sep)
#
# (negative because an xtrack Multipole applies px -= knl[0]), and injection
# needs that to open up
#
#     n_sigma * sigma_circulating  +  blade  +  n_sigma * sigma_incoming
#
# at the septum ENTRANCE. Everything is also tracked, because the linear
# formula is only trustworthy while the oscillation stays small: the same
# kick that displaces the beam at the septum displaces it everywhere, by
# sqrt(beta/beta_k) more wherever beta is larger, and once that exceeds the
# chamber the beam is simply lost. That excursion, not the septum clearance,
# is usually what limits a single-kicker scheme.

import LatticeBuild.misalignments_corrections as mc

INJ_N_SIGMA = 3            # beam half-width counted each side of the blade
INJ_SEPTUM_SLOT = 0        # straight (s order, long ones only) for the septum
INJ_KICKER_SLOT = 1        # straight for the kicker
INJ_APERTURE_X = 0.0312    # [m] chamber half-aperture from the aperture study


def injection_optics(ring, info):
    """Optics the injection geometry depends on, at the septum and kicker."""
    tw = ring.twiss(method='4d')
    g = lambda r, k: float(np.atleast_1d(getattr(r, k))[0])
    sep, kick = tw.rows[info['septum_entry']], tw.rows[info['kicker']]
    # Forward kicker -> septum: if the septum is upstream in s the beam gets
    # there by going on round, so the phase advance wraps by the tune.
    dmu = (g(sep, 'mux') - g(kick, 'mux')) % float(tw.qx)
    bk, bs = g(kick, 'betx'), g(sep, 'betx')
    return dict(tw=tw, beta_kick=bk, beta_sep=bs, dmu=dmu,
                alfx_sep=g(sep, 'alfx'), dx_sep=g(sep, 'dx'),
                bety_sep=g(sep, 'bety'), dy_sep=g(sep, 'dy'),
                s_sep=g(sep, 's'), s_kick=g(kick, 's'), qx=float(tw.qx),
                betx_max=float(np.max(tw.betx)),
                lever=-np.sqrt(bk * bs) * np.sin(2 * np.pi * dmu))


def injection_beam_sizes(opt, beam, eq_gemitt_x=None, eq_gemitt_y=None):
    """Beam sizes at the septum entrance, incoming and circulating.

    The incoming beam is the injected one measured in macroparticles.py, so
    its emittance and momentum spread come straight from `beam` (as built by
    ring_beam_sizes). The circulating beam is whatever is already stored:
    pass eq_gemitt_* for the damped equilibrium, or leave them None to
    assume it is still as large as the injected beam -- the worst case, and
    the right one for the first turns of a fill.
    """
    eps_x, eps_y, sd = beam['eps_x'], beam['eps_y'], beam['sigma_delta']
    cx = eps_x if eq_gemitt_x is None else eq_gemitt_x
    cy = eps_y if eq_gemitt_y is None else eq_gemitt_y
    q = lambda b, e, d: float(np.sqrt(b * e + (d * sd) ** 2))
    return dict(sig_x_in=q(opt['beta_sep'], eps_x, opt['dx_sep']),
                sig_y_in=q(opt['bety_sep'], eps_y, opt['dy_sep']),
                sig_x_circ=q(opt['beta_sep'], cx, opt['dx_sep']),
                sig_y_circ=q(opt['bety_sep'], cy, opt['dy_sep']),
                circ_is_injected=(eq_gemitt_x is None))


def required_separation(sz, blade, n_sigma=INJ_N_SIGMA):
    """Room needed at the septum entrance: both beams plus the blade."""
    return n_sigma * sz['sig_x_circ'] + blade + n_sigma * sz['sig_x_in']


def kick_scan(ring, info, thetas):
    """Track one turn from the kicker for each kick.

    Returns x at the septum entrance and the largest |x| anywhere on that
    turn, both in metres. Tracking rather than twiss, because at the kicks
    this geometry needs there is no stable closed orbit to twiss -- a
    permanently powered kicker that strong drives the orbit into the
    sextupoles and the optics stop existing.
    """
    rc = ring.cycle(name_first_element=info['kicker'])
    if rc.tracker is None:
        rc.build_tracker()
    rc.configure_radiation(model=None)
    i_sep = list(rc.element_names).index(info['septum_entry'])
    knob = info['kick_knob']
    was = float(rc.varval[knob])
    x_sep, x_max = [], []
    for th in thetas:
        rc.vars[knob] = float(th)
        p = rc.build_particles(x=0, px=0, y=0, py=0, zeta=0, delta=0)
        rc.track(p, num_turns=1, turn_by_turn_monitor='ONE_TURN_EBE')
        x = np.asarray(rc.record_last_track.x[0], dtype=float)
        x_sep.append(float(x[i_sep]))
        x_max.append(float(np.nanmax(np.abs(x))))
    rc.vars[knob] = was
    return np.asarray(x_sep), np.asarray(x_max)


def first_turn_trajectory(ring, info, theta):
    """x(s) over the turn following the kick, as tracked."""
    rc = ring.cycle(name_first_element=info['kicker'])
    if rc.tracker is None:
        rc.build_tracker()
    rc.configure_radiation(model=None)
    rc.vars[info['kick_knob']] = float(theta)
    p = rc.build_particles(x=0, px=0, y=0, py=0, zeta=0, delta=0)
    rc.track(p, num_turns=1, turn_by_turn_monitor='ONE_TURN_EBE')
    x = np.asarray(rc.record_last_track.x[0], dtype=float)
    rc.vars[info['kick_knob']] = 0.0
    tab = rc.get_table()
    s = np.asarray(tab.s, dtype=float)[:len(x)]
    tw = rc.twiss(method='4d')
    return s, x, rc, tw


def plot_injection(opt, sz, info, thetas, x_sep, x_max, need, th_need,
                   n_sigma=INJ_N_SIGMA, aperture_x=INJ_APERTURE_X):
    """Kicker scan, septum entrance cross-section, first-turn trajectory."""
    figs = {}
    blade = info['septum_blade']

    # ---- 1. the free parameter, and what it costs elsewhere --------------
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 7.5), sharex=True)
    # Only show the septum reading where the beam is still in the chamber:
    # past that the particle has hit the wall and the number is fiction.
    alive = (x_max <= aperture_x) if aperture_x else np.ones_like(x_max, bool)
    ax.plot(thetas * 1e3, np.where(alive, np.abs(x_sep), np.nan) * 1e3,
            'o-', ms=3, lw=1.4, label='tracked |x| at septum entrance')
    if not alive.all():
        ax.axvspan(thetas[int(np.argmax(~alive))] * 1e3, thetas[-1] * 1e3,
                   color='0.85', alpha=0.6, zorder=0,
                   label='beam lost on the chamber')
    ax.plot(thetas * 1e3, np.abs(thetas * opt['lever']) * 1e3, ls='--', lw=1.2,
            color='0.45', label=r'linear $\theta\sqrt{\beta_k\beta_s}\sin\Delta\mu$')
    ax.axhline(need * 1e3, color='crimson', ls=':', lw=1.6,
               label=rf'needed: ${n_sigma}\sigma$+blade+${n_sigma}\sigma$'
                     rf' = {need*1e3:.2f} mm')
    if th_need is not None:
        ax.axvline(th_need * 1e3, color='crimson', ls=':', lw=1.6)
    ax.set_ylabel('|x| at septum entrance  [mm]')
    ax.set_title('Injection: kicker strength is the free parameter')
    ax.grid(alpha=0.3); ax.legend(fontsize=8.5)

    ax2.semilogy(thetas * 1e3, x_max * 1e3, 'o-', ms=3, lw=1.4,
                 color='tab:purple',
                 label='tracked max |x| anywhere on the turn')
    if aperture_x:
        ax2.axhline(aperture_x * 1e3, color='k', ls='-', lw=1.6,
                    label=f'chamber half-aperture ({aperture_x*1e3:.1f} mm)')
        ok = x_max <= aperture_x
        bad = int(np.argmax(~ok)) if not ok.all() else len(ok)
        if bad:
            ax2.axvspan(0, thetas[bad - 1] * 1e3, color='tab:green',
                        alpha=0.12, label='kick the chamber survives')
    ax2.set_xlabel(r'kicker strength $\theta_k$  [mrad]')
    ax2.set_ylabel('max |x| on the turn  [mm]')
    ax2.grid(alpha=0.3); ax2.legend(fontsize=8.5)
    fig.tight_layout(); figs['kick_scan'] = fig

    # ---- 2. the septum entrance ------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    sc, si = sz['sig_x_circ'], sz['sig_x_in']
    ax.add_patch(plt.Rectangle((-n_sigma * sc * 1e3, -1), 2 * n_sigma * sc * 1e3,
                               2, color='#1f77b4', alpha=0.35,
                               label=rf'circulating, $\pm{n_sigma}\sigma$'
                                     rf' ({sc*1e3:.3f} mm rms)'))
    ax.add_patch(plt.Rectangle((n_sigma * sc * 1e3, -1), blade * 1e3, 2,
                               color='0.35',
                               label=f'septum blade ({blade*1e3:.1f} mm)'))
    ax.add_patch(plt.Rectangle(((n_sigma * sc + blade) * 1e3, -1),
                               2 * n_sigma * si * 1e3, 2, color='#d62728',
                               alpha=0.35,
                               label=rf'incoming, $\pm{n_sigma}\sigma$'
                                     rf' ({si*1e3:.3f} mm rms)'))
    ax.annotate('', xy=(0, 1.25), xytext=(need * 1e3, 1.25),
                arrowprops=dict(arrowstyle='<->', color='crimson', lw=1.4))
    ax.text(need * 1e3 / 2, 1.38,
            f'centre-to-centre separation needed = {need*1e3:.2f} mm',
            ha='center', color='crimson', fontsize=9)
    ax.set_xlim(-n_sigma * sc * 1e3 * 1.7, need * 1e3 * 1.18)
    ax.set_ylim(-1.6, 1.95); ax.set_yticks([])
    ax.set_xlabel('horizontal position at septum entrance  [mm]')
    ax.set_title(rf'Injection: entrance into the septum ($\beta_x$ = '
                 rf'{opt["beta_sep"]:.2f} m, $s$ = {opt["s_sep"]:.1f} m)')
    ax.legend(fontsize=8.5, loc='upper left', framealpha=0.95)
    fig.tight_layout(); figs['septum_entrance'] = fig
    return figs


def plot_first_turn(ring, info, beam, theta, n_sigma=INJ_N_SIGMA,
                    aperture_x=INJ_APERTURE_X, viable=True):
    """Where the kicked beam actually goes on the turn after the kick."""
    s, x, rc, tw = first_turn_trajectory(ring, info, theta)
    sig = np.sqrt(np.asarray(tw.betx)[:len(x)] * beam['eps_x']
                  + (np.asarray(tw.dx)[:len(x)] * beam['sigma_delta']) ** 2)
    i_sep = list(rc.element_names).index(info['septum_entry'])

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.fill_between(s, -n_sigma * sig * 1e3, n_sigma * sig * 1e3,
                    color='#1f77b4', alpha=0.22,
                    label=rf'stored beam $\pm{n_sigma}\sigma_x$ (un-kicked)')
    ax.plot(s, x * 1e3, color='#d62728', lw=1.2,
            label=rf'kicked trajectory, $\theta_k$ = {theta*1e3:.3f} mrad')
    if aperture_x:
        for sgn in (1, -1):
            ax.axhline(sgn * aperture_x * 1e3, color='k', lw=1.4,
                       label='chamber' if sgn == 1 else None)
    ax.axvline(s[i_sep], color='crimson', ls='--', lw=1.3,
               label=f'septum entrance (x = {x[i_sep]*1e3:+.2f} mm)')
    ax.axvline(s[0], color='tab:green', ls='--', lw=1.3, label='kicker')
    ax.set_xlabel('s from the kicker  [m]')
    ax.set_ylabel('horizontal position  [mm]')
    ax.set_title('Injection: first turn after the kick' if viable else
                 'Injection: first turn after the kick '
                 '(linear-estimate kick -- no viable kick exists)')
    ax.grid(alpha=0.3); ax.legend(fontsize=8.5, ncol=2); ax.set_xlim(0, s.max())
    fig.tight_layout()
    return fig


def injection_report(ring, info, beam, n_sigma=INJ_N_SIGMA, n_theta=40,
                     eq_gemitt_x=None, aperture_x=INJ_APERTURE_X, outdir=None):
    """Run the injection analysis, print the numbers and draw the figures."""
    opt = injection_optics(ring, info)
    sz = injection_beam_sizes(opt, beam, eq_gemitt_x=eq_gemitt_x)
    need = required_separation(sz, info['septum_blade'], n_sigma)
    th_lin = need / abs(opt['lever'])          # linear first guess

    thetas = np.linspace(0, 1.5 * th_lin, n_theta)
    x_sep, x_max = kick_scan(ring, info, thetas)

    # The kick that really opens the separation, from the tracked curve --
    # but only among kicks the beam survives. Past the chamber the particle
    # has hit the wall, so its "position at the septum" is fiction.
    survived = (x_max <= aperture_x) if aperture_x else np.ones_like(x_max, bool)
    hit = np.where((np.abs(x_sep) >= need) & survived)[0]
    th_need = float(thetas[hit[0]]) if len(hit) else None
    brho = float(ring.particle_ref.p0c[0]) / 299792458.0

    print('\n=== Injection ===')
    print(f'septum entrance  s = {opt["s_sep"]:8.3f} m  betx = {opt["beta_sep"]:6.3f} m'
          f'  dx = {opt["dx_sep"]:+.4f} m')
    print(f'kicker           s = {opt["s_kick"]:8.3f} m  betx = {opt["beta_kick"]:6.3f} m')
    print(f'phase advance kicker->septum = {opt["dmu"] % 1:.4f} x 2pi '
          f'({(opt["dmu"] % 1)*360:.1f} deg),  lever arm = {abs(opt["lever"]):.3f} m')
    print(f'incoming beam    sigma_x = {sz["sig_x_in"]*1e3:.4f} mm, '
          f'sigma_y = {sz["sig_y_in"]*1e3:.4f} mm')
    print(f'circulating beam sigma_x = {sz["sig_x_circ"]*1e3:.4f} mm'
          + ('  (assumed still as large as the injected beam)'
             if sz['circ_is_injected'] else '  (damped equilibrium)'))
    print(f'needed at septum = {n_sigma}*{sz["sig_x_circ"]*1e3:.3f} + '
          f'{info["septum_blade"]*1e3:.1f} + {n_sigma}*{sz["sig_x_in"]*1e3:.3f}'
          f' = {need*1e3:.3f} mm')
    print(f'linear estimate of the kick needed = {th_lin*1e3:.4f} mrad')
    if th_need is None:
        reach = np.abs(x_sep) >= need
        print(f'TRACKED: NO viable kick up to {thetas[-1]*1e3:.3f} mrad.')
        if reach.any():
            i = int(np.argmax(reach))
            print(f'   the separation is first reached at '
                  f'{thetas[i]*1e3:.3f} mrad, but that kick throws the beam to '
                  f'{x_max[i]*1e3:.1f} mm, past the '
                  f'{aperture_x*1e3:.1f} mm chamber -- it is lost before it '
                  f'gets back to the septum.')
        else:
            print(f'   the separation is never reached at all; the largest '
                  f'displacement seen at the septum is '
                  f'{np.nanmax(np.abs(x_sep))*1e3:.2f} mm.')
    else:
        print(f'TRACKED kick needed = {th_need*1e3:.4f} mrad '
              f'= {th_need*brho*1e3:.3f} mT.m')
        print(f'   max |x| on that turn = '
              f'{x_max[hit[0]]*1e3:.2f} mm (betx_max = {opt["betx_max"]:.1f} m)')
    if aperture_x:
        ok = x_max <= aperture_x
        if not ok.all():
            # First violation, not thetas[ok].max(): x_max is not monotonic in
            # theta once the motion is nonlinear, so the largest passing theta
            # can sit above kicks that already hit the wall.
            bad = int(np.argmax(~ok))
            th_ok = float(thetas[bad - 1]) if bad else 0.0
            print(f'APERTURE: the chamber ({aperture_x*1e3:.1f} mm) is exceeded '
                  f'above {th_ok*1e3:.3f} mrad')
            if th_need is not None and th_need > th_ok:
                print(f'   -> the kick injection needs ({th_need*1e3:.3f} mrad) '
                      f'is LARGER than the chamber allows ({th_ok*1e3:.3f} mrad): '
                      f'a single kicker in a betx = {opt["beta_kick"]:.1f} m '
                      f'straight cannot do this. Raise betx at the septum, or '
                      f'use a local closed bump of 3-4 kickers.')

    figs = plot_injection(opt, sz, info, thetas, x_sep, x_max, need,
                          th_need if th_need is not None else th_lin,
                          n_sigma=n_sigma, aperture_x=aperture_x)
    figs['first_turn'] = plot_first_turn(
        ring, info, beam, th_need if th_need is not None else th_lin,
        n_sigma=n_sigma, aperture_x=aperture_x, viable=th_need is not None)
    if outdir:
        for k, f in figs.items():
            f.savefig(f'{outdir}/injection_{k}.png', dpi=200, bbox_inches='tight')
        print(f'figures written to {outdir}/injection_*.png')
    return dict(opt=opt, sizes=sz, need=need, theta=th_need,
                thetas=thetas, x_sep=x_sep, x_max=x_max, figs=figs)


# ---- run it ---------------------------------------------------------------
pdr_inj = pdr.copy()
inj_info = mc.insert_injection(pdr_inj, septum_slot=INJ_SEPTUM_SLOT,
                               kicker_slot=INJ_KICKER_SLOT)
ring_inj = pdr_inj.lines['ring']
ring_inj.configure_radiation(model=None)

inj = injection_report(ring_inj, inj_info, beam, n_sigma=INJ_N_SIGMA,
                       outdir=sketch_dir)
plt.show()
