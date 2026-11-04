import numpy as np
import LatticeBuild.misalignments_corrections as mc
import xtrack as xt
import matplotlib.pyplot as plt

pdr=xt.Environment.from_json('/home/mwatson/Documents/laughing-octo-bassoon/JSON_Files/D1/C9/pdr_perfect_90_FDF.json')
pdr_copy = pdr.copy()
mc.insert_bpms_and_correctors(pdr_copy, 5, 9)

ring = pdr_copy.lines['ring']          # BPMs + correctors already inserted
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
# %% Whole-ring sketch with the injected beam's sizes from macroparticles.py
import re
import pandas as pd
import config as cfg
import paths
import lattice_Sketch as ls

# Must match the lattice loaded at the top of this file.
design, config, phase, changes = 1, 9, 90, 'FDF'


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


def injection_s(ring, drift_name='DrTripl', occurrence=3):
    """s of the injection point used by macroparticles.py: the centre of the
    `occurrence`-th (0-indexed) plain `drift_name` drift, as in
    macroparticles.insert_marker_in_drift."""
    tab = ring.get_table()
    pattern = re.compile(rf'^{re.escape(drift_name)}(::\d+)?$')
    hits = [i for i, n in enumerate(tab.name) if pattern.match(n)]
    i = hits[occurrence]
    return float(0.5 * (tab.s[i] + tab.s[i + 1]))


beam = ring_beam_sizes(pdr.lines['ring'],
                       paths.load_injected_beam(design, config, phase, changes))
fig_bs = ls.sketch_ring_beam_size(
    pdr, beam, inject_s=injection_s(pdr.lines['ring']),
    outdir=paths.results_dir(design, config, phase, changes=changes,
                             metric='LatticeSketch'))
plt.show()
