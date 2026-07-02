# %%
import sys
import os

# Adds the parent directory to the search path
parent_dir = os.path.abspath('..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import xtrack as xt
import xpart as xp
import xfields as xf
import xobjects as xo
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
import LatticeBuild.misalignments_corrections as mc


#%%
# ===========================================================================
# SINGLE-SEED run: deep track (10k-turn decay + exponential fit) AND invariant
# spin vector, for ONE hardcoded seed, in BOTH the misaligned and corrected
# lattice. No scan, no cross-seed comparison plots.
# ===========================================================================

# ---- EDIT THIS: the seed to run ------------------------------------------
SEED = 585968990
# --------------------------------------------------------------------------

design = int(os.environ.get('DESIGN', 1))
config = int(os.environ.get('CONFIG', 1))

long_scan_turns = 10000   # turns for the deep polarization-decay track

# %%
# Start from the PERFECT base lattice; misalignments (and optionally correction)
# are applied per branch to a copy of it, using the SAME seed for both branches
# so misaligned vs corrected is a fair paired comparison.
pdr = xt.Environment.from_json(f'JSON Files/D{design}/C{config}/pdr_perfect.json')
pdr.lines['ring'].particle_ref.anomalous_magnetic_moment = 0.001159652181
pdr.lines['ring'].particle_ref.kinetic_energy0 = 2.86e9

if design == 1 and config == 1:
    mc.insert_BPMs_all_as_markers(pdr)
    mc.insert_correctors_var2(pdr)
else:
    mc.insert_BPMs_all_as_markers(pdr)
    mc.insert_correctors(pdr)

line = pdr.lines['ring']
line.configure_spin('auto')

base_line = line.copy()

results_dir = f'Results/D{design}/C{config}/SingleSeed_{SEED}'
os.makedirs(results_dir, exist_ok=True)


#%%
# ===========================================================================
# DEEP TRACK — 10k-turn polarization decay + exponential fit for one seed
# ===========================================================================

def deep_track_single(seed_val, apply_correction):
    
    """Deep-track a single seed (misaligned-only or misaligned+corrected).
    Returns a dict of decay curve + fitted quantities for plotting."""
    branch_label = 'corrected' if apply_correction else 'misaligned'
    print(f"Running deep track for Seed {seed_val} ({branch_label})...")

    line = base_line.copy()
    # Set mean radiation before building the tracker (twiss can't run under
    # quantum). Build tracker, then misalign onto the live tracker (element_refs);
    # never discard afterward or the misalignments are lost.
    line.configure_radiation('mean')
    line.build_tracker(_context=xo.ContextCpu(omp_num_threads=0))
    line = mc.misalignments(line, 0.2e-3, seed=seed_val)

    tw = line.twiss(method='6d', radiation_integrals=True, eneloss_and_damping=True,
                    spin=True, polarization=True)

    if apply_correction:
        orbit_x_rms_before = np.std(tw.x)
        orbit_y_rms_before = np.std(tw.y)
        try:
            mc.orbit_correction(line, tw, threading=False)
        except Exception as e:
            print(f"  [seed {seed_val}] orbit_correction(threading=False) raised: "
                  f"{type(e).__name__}: {e} -- retrying with threading=True")
            mc.orbit_correction(line, tw, threading=True)
        # Re-twiss after correction so tw reflects the corrected lattice.
        tw = line.twiss(method='6d', radiation_integrals=True, eneloss_and_damping=True,
                        spin=True, polarization=True)
        orbit_x_rms_after = np.std(tw.x)
        orbit_y_rms_after = np.std(tw.y)
        print(f"  [seed {seed_val}] orbit RMS x: {orbit_x_rms_before:.3e} -> {orbit_x_rms_after:.3e}, "
              f"y: {orbit_y_rms_before:.3e} -> {orbit_y_rms_after:.3e}")

    particles = xp.generate_matched_gaussian_bunch(
        line=line,
        nemitt_x=tw.eq_nemitt_x,
        nemitt_y=tw.eq_nemitt_y,
        sigma_z=np.sqrt(tw.eq_gemitt_zeta * tw.bets0),
        num_particles=300)

    particles.zeta += tw.zeta[0]
    particles.delta += tw.delta[0]
    particles.spin_x = tw.spin_x[0]
    particles.spin_y = tw.spin_y[0]
    particles.spin_z = tw.spin_z[0]

    # Switch to quantum radiation for tracking WITHOUT discarding the tracker
    # (which would wipe misalignments + correction).
    line.configure_radiation('quantum')
    line.build_tracker(_context=xo.ContextCpu(omp_num_threads='auto'))

    line.track(particles, num_turns=long_scan_turns, turn_by_turn_monitor=True, with_progress=10)
    mon = line.record_last_track

    # Polarization decay curve
    mask_alive = mon.state > 0
    pol_x = mon.spin_x.sum(axis=0) / mask_alive.sum(axis=0)
    pol_y = mon.spin_y.sum(axis=0) / mask_alive.sum(axis=0)
    pol_z = mon.spin_z.sum(axis=0) / mask_alive.sum(axis=0)
    pol = np.sqrt(pol_x**2 + pol_y**2 + pol_z**2)

    pol_to_fit = pol[3:] / pol[3]
    turns = np.arange(len(pol_to_fit))

    def exp_decay(t, tau):
        return np.exp(-t / tau)

    # No scan CSV to draw an initial guess from; use half the tracking window.
    tau0_guess = max(len(turns) / 2, 10)
    try:
        popt, pcov = curve_fit(exp_decay, turns, pol_to_fit, p0=[tau0_guess], maxfev=10000)
        t_dep_turns_long = popt[0]
        fit_curve = exp_decay(turns, t_dep_turns_long)
        fit_ok = True
    except (RuntimeError, ValueError) as e:
        print(f"  Exponential fit failed for seed {seed_val}: {e}")
        t_dep_turns_long = np.nan
        fit_curve = np.full_like(pol_to_fit, np.nan)
        fit_ok = False

    n_over_tau = len(turns) / t_dep_turns_long if fit_ok and t_dep_turns_long > 0 else np.nan
    print(f"  Seed {seed_val}: tau_fit = {t_dep_turns_long:.1f} turns, N/tau = {n_over_tau:.2f}")

    p_bks = tw.spin_polarization_inf_no_depol
    t_bks = tw.spin_t_pol_component_s
    t_pol = t_bks   # polarization buildup time in seconds
    t_pol_turns = t_bks / tw.T_rev0
    p_eq_long = (p_bks * 1 / (1 + t_pol_turns / t_dep_turns_long)) * 100

    return {
        'seed': seed_val,
        'turns': turns,
        'pol': pol_to_fit,
        'fit': fit_curve,
        'p_eq_long': p_eq_long,
        'p_bks': p_bks * 100,
        't_pol': t_pol,
        't_dep_turns_long': t_dep_turns_long,
    }


def plot_seed_with_textbox(data, title_prefix, out_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle(f"{title_prefix} - Seed {data['seed']}", fontsize=14, fontweight='bold')

    ax.plot(data['turns'], data['pol'], label='Tracking Data', color='blue', alpha=0.7)
    ax.plot(data['turns'], data['fit'], label='Exponential Fit', color='red', linestyle='--')
    ax.set_ylabel('$P(t)/P(0)$')
    ax.set_xlabel('Turns')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()

    info_text = (
        f"$P_{{BKS}}$ = {data['p_bks']:.2f}%   "
        f"$P_{{eq}}$ (long track) = {data['p_eq_long']:.2f}%\n"
        f"$\\tau_{{depol}}$ (fitted) = {data['t_dep_turns_long']:.1f} turns   "
        f"$t_{{pol}}$ = {data['t_pol']:.4e} s\n"
    )
    fig.text(0.5, 0.02, info_text, ha='center', va='bottom', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='whitesmoke', edgecolor='gray', alpha=0.9))

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved to {out_path}")


data_misaligned = deep_track_single(SEED, apply_correction=False)
data_corrected = deep_track_single(SEED, apply_correction=True)

plot_seed_with_textbox(
    data_misaligned, 'Misaligned',
    f'{results_dir}/Polarization_Misaligned.png')
plot_seed_with_textbox(
    data_corrected, 'Corrected',
    f'{results_dir}/Polarization_Corrected.png')


#%%
# ===========================================================================
# INVARIANT SPIN VECTOR along the ring, for the same seed
# ===========================================================================

def plot_invariant_spin_vector(seed_val, apply_correction):
    branch_label = 'corrected' if apply_correction else 'misaligned'
    print(f"Plotting invariant spin vector for Seed {seed_val} ({branch_label})...")

    line = base_line.copy()
    # Set mean radiation before building the tracker (twiss can't run under
    # quantum). Build tracker, then misalign onto the live tracker; do not
    # discard afterward or the misalignments are lost.
    line.configure_radiation('mean')
    line.build_tracker(_context=xo.ContextCpu(omp_num_threads=0))
    line = mc.misalignments(line, 0.2e-3, seed=seed_val)

    tw = line.twiss(method='6d', radiation_integrals=True, eneloss_and_damping=True,
                    spin=True, polarization=True)

    if apply_correction:
        try:
            mc.orbit_correction(line, tw, threading=False)
        except Exception as e:
            print(f"  [seed {seed_val}] orbit_correction(threading=False) raised: "
                  f"{type(e).__name__}: {e} -- retrying with threading=True")
            mc.orbit_correction(line, tw, threading=True)
        # Orbit correction changes the closed orbit/optics, so re-twiss to get
        # the n0 vector consistent with the corrected lattice.
        tw = line.twiss(method='6d', radiation_integrals=True, eneloss_and_damping=True,
                        spin=True, polarization=True)

    s = tw.s
    sx = tw.spin_x
    sy = tw.spin_y
    sz = tw.spin_z

    # Sanity check: |n0| should be 1 everywhere (it's a unit vector by construction).
    n0_mag = np.sqrt(sx**2 + sy**2 + sz**2)
    print(f"  |n0| range: [{n0_mag.min():.6f}, {n0_mag.max():.6f}] (should be ~1.0)")
    print(f"  spin tune = {tw.spin_tune_fractional:.6f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(s, sx, label='$n_{0,x}$', color='tab:blue')
    ax.plot(s, sy, label='$n_{0,y}$', color='tab:green')
    ax.plot(s, sz, label='$n_{0,z}$', color='tab:red')

    ax.set_xlabel('s (m)')
    ax.set_ylabel('Invariant spin vector $n_0$ component')
    branch_tag = 'Corrected' if apply_correction else 'Misaligned'
    ax.set_title(f'Invariant Spin Vector Along the Ring — Seed {seed_val} ({branch_tag})')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()

    plt.tight_layout()
    out_path = f'{results_dir}/InvariantSpinVector_{branch_tag}.png'
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved to {out_path}")


plot_invariant_spin_vector(SEED, apply_correction=False)
plot_invariant_spin_vector(SEED, apply_correction=True)
# %%