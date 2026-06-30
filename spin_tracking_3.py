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
from scipy.stats import linregress
import matplotlib.pyplot as plt
import json
import os
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import LatticeBuild.misalignments_corrections as mc


#%%
design=int(os.environ.get('DESIGN',1))
config=int(os.environ.get('CONFIG',1))

# %%
# Always start from the PERFECT base lattice. Both the "misaligned" and
# "corrected" branches below apply the SAME random misalignment seed to a
# copy of this lattice -- one branch is left as-is, the other additionally
# gets orbit_correction run on it. This guarantees the two branches are a
# fair paired comparison (same misalignment pattern, with vs without
# correction) rather than loading two separately-prepared JSON files that
# might not even share the same misalignment realization.
pdr = xt.Environment.from_json(f'JSON Files/D{design}/C{config}/pdr_perfect.json')
if design == 1 and config == 1:
    mc.insert_BPMs_all_as_markers(pdr)
    mc.insert_correctors_var2(pdr)
else:
    mc.insert_BPMs_all_as_markers(pdr)
    mc.insert_correctors(pdr)
pdr.particle_ref.anomalous_magnetic_moment=0.001159652181

# ===========================================================================
# PART 1 — SCAN over many random misalignment seeds (1000-turn screening)
# Each seed is run through BOTH the misaligned-only and corrected lattice,
# using the identical misalignment seed for both, so P_eq can be compared
# directly seed-by-seed.
# ===========================================================================

line=pdr.lines['ring']

# Simulate bunch evolution with stochastic photon emission
line.configure_spin('auto')

max_seed_value = np.iinfo(np.uint32).max  
num_seeds=50 
seeds = np.random.randint(0, max_seed_value, size=num_seeds)
scan_turns=1000

base_line = line.copy()

# Single shared OpenMP context, built ONCE. xobjects ContextCpu objects own a
# `kernels` cache keyed by compiled kernel signature -- a fresh
# xo.ContextCpu(omp_num_threads=0) instance starts with an EMPTY cache, so if
# you instantiate a new context object every time build_tracker() is called
# (as the per-seed loops below used to), every call pays a fresh compile even
# though the underlying lattice structure (and therefore kernel source) is
# identical across seeds -- only the misalignment numbers differ, not the
# code. Reusing this SAME context object across every build_tracker() call
# lets later calls hit the cache instead of recompiling from scratch.
# twiss() still needs the line on its OWN separate (default/serial) tracker,
# since twiss cannot run while the OpenMP-parallel context is active -- so
# this shared context is only ever attached right before tracking, never
# before a twiss() call.
omp_context = xo.ContextCpu(omp_num_threads=0)

results_dir = f'Results/D{design}/C{config}/Comparison'
os.makedirs(results_dir, exist_ok=True)
results_path = f'{results_dir}/SpinTrackingResults_MisalignedVsCorrected.dat'


def run_scan_pass(seed_list, apply_correction):
    """Run the full 1000-turn screen for ONE branch (misaligned or corrected)
    over seed_list, returning a DataFrame of results for just that branch.

    Builds ONE persistent tracked line for this branch, ONCE, before the seed
    loop -- mc.misalignments() mutates element offsets in place (it does not
    rebuild line structure), so the same already-tracker-built line can be
    re-misaligned seed-to-seed without ever calling discard_tracker()/
    build_tracker() again. twiss() can run on a line that already has the
    OpenMP tracker built (only the *tracking* step is restricted while
    OpenMP is active, not twiss), so a single tracker build covers both
    the twiss calls and the tracking calls for every seed in this branch.
    """
    branch_label = 'corrected' if apply_correction else 'misaligned'

    # Build the persistent line for this branch ONCE. Correction structurally
    # depends on this exact line's corrector elements, so each branch
    # (misaligned vs corrected) keeps its own persistent line rather than
    # sharing one across apply_correction values.
    persistent_line = base_line.copy()
    persistent_line.discard_tracker()
    persistent_line.build_tracker(_context=omp_context)

    P_BKS, tau_BKS, P_DKM, tau_DKM, tau_depol, tau_pol, P_eq = [], [], [], [], [], [], []
    tune_x, tune_y, spin_tune = [], [], []
    t_dep_turns_list = []
    result_seeds = []
    failed_seeds = []

    for seed in seed_list:
        try:
            # Overwrites this line's element offsets with the new seed's
            # random values in place -- no copy, no tracker rebuild.
            mc.misalignments(persistent_line, 0.2e-3, seed=seed)

            persistent_line.configure_radiation('mean')
            tw = persistent_line.twiss(method='6d', radiation_integrals=True,
                            eneloss_and_damping=True, spin=True, polarization=True)

            if apply_correction:
                try:
                    mc.orbit_correction(pdr, tw, threading=False)
                except:
                    mc.orbit_correction(pdr, tw, threading=True)
                # Re-twiss after correction so tw reflects the corrected
                # orbit/optics (not the pre-correction twiss the orbit
                # correction was based on). Tracker is still untouched.
                tw = persistent_line.twiss(method='6d', radiation_integrals=True,
                                eneloss_and_damping=True, spin=True, polarization=True)

            particles = xp.generate_matched_gaussian_bunch(
                line=persistent_line,
                nemitt_x=tw.eq_nemitt_x,
                nemitt_y=tw.eq_nemitt_y,
                sigma_z=np.sqrt(tw.eq_gemitt_zeta * tw.bets0),
                num_particles=300)
            particles.zeta += tw.zeta[0]
            particles.delta += tw.delta[0]
            particles.spin_x = tw.spin_x[0]
            particles.spin_y = tw.spin_y[0]
            particles.spin_z = tw.spin_z[0]
            persistent_line.configure_radiation(model='quantum')

            persistent_line.track(particles, num_turns=scan_turns, turn_by_turn_monitor=True,
                    with_progress=10)
            mon = persistent_line.record_last_track

            # Fit depolarization time
            mask_alive = mon.state > 0
            pol_x = mon.spin_x.sum(axis=0)/mask_alive.sum(axis=0)
            pol_y = mon.spin_y.sum(axis=0)/mask_alive.sum(axis=0)
            pol_z = mon.spin_z.sum(axis=0)/mask_alive.sum(axis=0)
            pol = np.sqrt(pol_x**2 + pol_y**2 + pol_z**2)

            pol_to_fit = pol[3:] / pol[3]
            turns = np.arange(len(pol_to_fit))
            slope, intercept, r_value, p_value, std_err = linregress(turns, pol_to_fit)
            t_dep_turns = -1 / slope

            p_bks=tw.spin_polarization_inf_no_depol
            t_bks=tw.spin_t_pol_component_s
            p_dkm=tw.spin_polarization_eq
            t_dkm=tw.spin_t_pol_buildup_s
            t_depol=tw.spin_t_depol_component_s
            t_pol=t_bks/(1+t_bks/tw.T_rev0)

            t_pol_turns = t_bks/tw.T_rev0
            p_eq = p_bks * 1 / (1 + t_pol_turns/t_dep_turns)

            P_BKS.append(p_bks*100)
            tau_BKS.append(t_bks)
            P_DKM.append(p_dkm*100)
            tau_DKM.append(t_dkm)
            tau_depol.append(t_depol)
            tau_pol.append(t_pol)
            P_eq.append(p_eq*100)
            t_dep_turns_list.append(t_dep_turns)
            tune_x.append(tw.qx)
            tune_y.append(tw.qy)
            spin_tune.append(tw.spin_tune_fractional)
            result_seeds.append(seed)

        except (RuntimeError, np.linalg.LinAlgError, ValueError) as e:
            print(e)
            failed_seeds.append(seed)

    print(f"[{branch_label}] failed seeds: {len(failed_seeds)} / {len(seed_list)}")

    branch_results = {
        'Seed': result_seeds,
        'Mode': [branch_label] * len(result_seeds),
        'P_BKS': P_BKS,
        't_BKS': tau_BKS,
        'P_DKM': P_DKM,
        't_DKM': tau_DKM,
        't_depol':tau_depol,
        't_pol':tau_pol,
        'P_eq': P_eq,
        't_dep_turns': t_dep_turns_list,
        'N_over_tau': [scan_turns / t for t in t_dep_turns_list],
        'qx': tune_x,
        'qy': tune_y,
        'spin_tune': spin_tune
    }
    return pd.DataFrame(branch_results)


# ---------------------------------------------------------------------------
# PASS 1 — misaligned-only scan, run first and saved to disk on its own.
# ---------------------------------------------------------------------------
#%%

df_misaligned = run_scan_pass(seeds, apply_correction=False)
df_misaligned.to_csv(results_path)
print(f"Pass 1 (misaligned) complete. Saved {len(df_misaligned)} rows to {results_path}")

#%%

# ---------------------------------------------------------------------------
# PASS 2 — corrected scan, using the EXACT seeds read back from the file
# Pass 1 just wrote (not the in-memory `seeds` array), so the seed list this
# pass uses is guaranteed to come from disk rather than from whatever is
# still sitting in memory -- this is the actual point of writing to disk
# in between passes rather than just reusing `seeds` directly.
# ---------------------------------------------------------------------------

df_misaligned_from_disk = pd.read_csv(results_path, index_col=0)
seeds_from_file = df_misaligned_from_disk['Seed'].to_numpy()

df_corrected = run_scan_pass(seeds_from_file, apply_correction=True)

# Append the corrected results onto the same file that already holds the
# misaligned pass, so the final file contains both branches for every seed.
df_scan_full = pd.concat([df_misaligned_from_disk, df_corrected], ignore_index=True)
df_scan_full.to_csv(results_path)
print(f"Pass 2 (corrected) complete. {results_path} now holds "
      f"{len(df_scan_full)} rows ({len(df_misaligned_from_disk)} misaligned + "
      f"{len(df_corrected)} corrected).")

# df used by the legacy plotting cells below
df = df_scan_full.copy()

#%%
'''pdf_run=False

if pdf_run is True:
    pdf = PdfPages(f"Results/D{design}/C{config}/Comparison/SpinTrackingResults.pdf")

    _old_savefig = plt.savefig

    def _new_savefig(*args, **kwargs):
        pdf.savefig(plt.gcf())
        _old_savefig(*args, **kwargs)

    plt.savefig = _new_savefig'''
# %%


#%%

# Polarization vs depol time, split by branch mode so misaligned and
# corrected points are visually distinguishable on the same axes.
df['t_depol_fitted_seconds'] = df['t_pol'] / ((df['P_BKS'] / df['P_eq']) - 1)

fig, ax = plt.subplots()
for branch_mode, color in [('misaligned', 'tab:red'), ('corrected', 'tab:blue')]:
    sub = df[df['Mode'] == branch_mode]
    ax.scatter(sub['P_eq'], sub['t_depol_fitted_seconds']/60**2, label=branch_mode,
               color=color, alpha=0.7)
ax.set_xlabel('Equilibrium Polarization (%)')
ax.set_ylabel('Depolarization Time (hours)')
ax.legend()
plt.savefig(f'Results/D{design}/C{config}/Comparison/EquilibriumPol_v_DepolTime.png')


#%%
df['t_depol_fitted_seconds'] = df['t_pol'] / ((df['P_BKS'] / df['P_eq']) - 1)

plt.figure(figsize=(8, 5))

df=df[df['P_eq']<=100]
# Histogram, split by branch mode (overlapping, semi-transparent)
for branch_mode, color in [('misaligned', 'tab:red'), ('corrected', 'tab:blue')]:
    sub = df[df['Mode'] == branch_mode]
    plt.hist(sub['P_eq'], bins=30, color=color, edgecolor='black', alpha=0.5, label=branch_mode)
    plt.axvline(sub['P_eq'].mean(), color=color, linestyle='--', linewidth=1.5,
                label=f'{branch_mode} mean: {sub["P_eq"].mean():.2f}%')

plt.title('Distribution of Equilibrium Polarization across Seeds', fontsize=13, fontweight='bold')
plt.xlabel('Equilibrium Polarization (%)', fontsize=11)
plt.ylabel('Number of Seeds (Frequency)', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(f'Results/D{design}/C{config}/Comparison/EqPolDist.png')


#%%

# ---------------------------------------------------------------------------
# Paired misaligned-vs-corrected P_eq comparison (same misalignment seed for
# both branches, so each point is a fair before/after-correction pair).
# ---------------------------------------------------------------------------

df_mis = df_scan_full[df_scan_full['Mode'] == 'misaligned'].set_index('Seed')
df_cor = df_scan_full[df_scan_full['Mode'] == 'corrected'].set_index('Seed')
# Only seeds present in both branches (should be all of them, since prep_branch
# either succeeds for both or the seed is skipped entirely -- see Part 1).
paired_seeds = df_mis.index.intersection(df_cor.index)

p_eq_mis = df_mis.loc[paired_seeds, 'P_eq']
p_eq_cor = df_cor.loc[paired_seeds, 'P_eq']

# Scatter: misaligned P_eq vs corrected P_eq, one point per seed
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(p_eq_mis, p_eq_cor, color='purple', alpha=0.7)
lims = [min(p_eq_mis.min(), p_eq_cor.min()), max(p_eq_mis.max(), p_eq_cor.max())]
ax.plot(lims, lims, color='gray', linestyle=':', label='No improvement (y=x)')
ax.set_xlabel('Misaligned $P_{eq}$ (%)')
ax.set_ylabel('Corrected $P_{eq}$ (%)')
ax.set_title('Equilibrium Polarization: Misaligned vs Corrected\n(same misalignment seed per point)')
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend()
plt.tight_layout()
plt.savefig(f'Results/D{design}/C{config}/Comparison/PEq_Misaligned_vs_Corrected_Scatter.png', dpi=300)
plt.close()

# Separate histograms, side by side, for direct distribution comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
axes[0].hist(p_eq_mis, bins=30, color='tab:red', edgecolor='black', alpha=0.8)
axes[0].axvline(p_eq_mis.mean(), color='black', linestyle='--',
                 label=f'Mean: {p_eq_mis.mean():.2f}%')
axes[0].set_title('Misaligned')
axes[0].set_xlabel('Equilibrium Polarization (%)')
axes[0].set_ylabel('Number of Seeds')
axes[0].legend()
axes[0].grid(True, linestyle=':', alpha=0.6)

axes[1].hist(p_eq_cor, bins=30, color='tab:blue', edgecolor='black', alpha=0.8)
axes[1].axvline(p_eq_cor.mean(), color='black', linestyle='--',
                 label=f'Mean: {p_eq_cor.mean():.2f}%')
axes[1].set_title('Corrected')
axes[1].set_xlabel('Equilibrium Polarization (%)')
axes[1].legend()
axes[1].grid(True, linestyle=':', alpha=0.6)

fig.suptitle('Equilibrium Polarization Distributions — Misaligned vs Corrected', fontweight='bold')
plt.tight_layout()
plt.savefig(f'Results/D{design}/C{config}/Comparison/PEq_Misaligned_vs_Corrected_Histograms.png', dpi=300)
plt.close()

# %%


# ===========================================================================
# PART 2 — DEEP TRACK on the single best and single worst seed from the scan
# ===========================================================================
# Uses df_scan_full (in memory from Part 1) instead of re-reading
# SpinTrackingResults.dat from disk -- this guarantees the deep track is
# always working from the scan that was JUST run, never a stale prior file.
#%%

from scipy.optimize import curve_fit

long_scan_turns=10000

# df_scan_full contains BOTH branches (misaligned and corrected). Deep-track
# the best/worst seed of EACH branch separately, since the best/worst seed
# under correction isn't necessarily the same seed as best/worst uncorrected.
df_corrected_only = df_scan_full[df_scan_full['Mode'] == 'corrected']
df_misaligned_only = df_scan_full[df_scan_full['Mode'] == 'misaligned']

# base_line already exists from Part 1 -- reuse it as the clean reference so
# each deep-track seed starts from the same untouched lattice.

def deep_track_branch(df_branch, apply_correction):
    """Deep-track the best (top_1) and worst (bottom_1) seed within df_branch.
    Returns a dict {'top_1': [...], 'bottom_1': [...]} matching the original
    plot_data structure, scoped to just this one branch.

    Builds ONE persistent tracked line, ONCE, reused for both the top_1 and
    bottom_1 seed -- same reasoning as run_scan_pass: mc.misalignments and
    mc.orbit_correction both fully overwrite from scratch each call, so the
    tracker never needs rebuilding between the two seeds tracked here.
    """
    top_1 = df_branch.nlargest(1, 'P_eq')
    bottom_1 = df_branch.nsmallest(1, 'P_eq')
    branch_plot_data = {'top_1': [], 'bottom_1': []}

    line = base_line.copy()
    line.discard_tracker()
    line.build_tracker(_context=omp_context)

    for group_name, df_group in [('top_1', top_1), ('bottom_1', bottom_1)]:
        for idx, row in df_group.iterrows():
            seed_val = int(row['Seed'])
            p_eq_scan = row['P_eq']
            print(f"Running deep track for {group_name} - Seed {seed_val} "
                  f"({'corrected' if apply_correction else 'misaligned'})...")

            mc.misalignments(line, 0.2e-3, seed=seed_val)

            line.configure_radiation('mean')
            tw = line.twiss(method='6d', radiation_integrals=True, eneloss_and_damping=True,
                            spin=True, polarization=True)

            if apply_correction:
                try:
                    mc.orbit_correction(pdr, tw, threading=False)
                except:
                    mc.orbit_correction(pdr, tw, threading=True)
                # Re-twiss after correction so tw reflects the corrected lattice.
                tw = line.twiss(method='6d', radiation_integrals=True, eneloss_and_damping=True,
                                spin=True, polarization=True)

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
            line.configure_radiation(model='quantum')

            # Track for full long duration
            line.track(particles, num_turns=long_scan_turns, turn_by_turn_monitor=True, with_progress=10)
            mon = line.record_last_track

            # Compute full polarization curve
            mask_alive = mon.state > 0
            pol_x = mon.spin_x.sum(axis=0) / mask_alive.sum(axis=0)
            pol_y = mon.spin_y.sum(axis=0) / mask_alive.sum(axis=0)
            pol_z = mon.spin_z.sum(axis=0) / mask_alive.sum(axis=0)
            pol = np.sqrt(pol_x**2 + pol_y**2 + pol_z**2)

            pol_to_fit = pol[3:] / pol[3]
            turns = np.arange(len(pol_to_fit))

            def exp_decay(t, tau):
                return np.exp(-t / tau)

            if 't_depol' in row and row['t_depol'] > 0:
                tau0_guess = row['t_depol'] / tw.T_rev0
            else:
                tau0_guess = max(len(turns) / 2, 10)
            try:
                popt, pcov = curve_fit(exp_decay, turns, pol_to_fit, p0=[tau0_guess],
                                        maxfev=10000)
                t_dep_turns_long = popt[0]
                fit_curve = exp_decay(turns, t_dep_turns_long)
                fit_ok = True
            except (RuntimeError, ValueError) as e:
                print(f"  Exponential fit failed for seed {seed_val}: {e}")
                t_dep_turns_long = np.nan
                fit_curve = np.full_like(pol_to_fit, np.nan)
                fit_ok = False

            n_over_tau = len(turns) / t_dep_turns_long if fit_ok and t_dep_turns_long > 0 else np.nan
            print(f"  Seed {seed_val}: tau_fit = {t_dep_turns_long:.1f} turns, "
                  f"N/tau = {n_over_tau:.2f}")

            p_bks=tw.spin_polarization_inf_no_depol
            t_bks=tw.spin_t_pol_component_s
            p_dkm=tw.spin_polarization_eq
            t_dkm=tw.spin_t_pol_buildup_s
            t_depol=tw.spin_t_depol_component_s
            t_pol=t_bks/(1+t_bks/tw.T_rev0)

            t_pol_turns = t_bks/tw.T_rev0
            p_eq = p_bks * 1 / (1 + t_pol_turns/t_dep_turns_long)

            # Calculate the revised equilibrium polarization from the deep track fit
            p_eq_long = (p_bks * 1 / (1 + t_pol_turns / t_dep_turns_long)) * 100

            branch_plot_data[group_name].append({
                'seed': seed_val,
                'turns': turns,
                'pol': pol_to_fit,
                'fit': fit_curve,
                'p_eq_long': p_eq_long,
                'p_bks': p_bks * 100,
                't_pol': t_pol,
                't_dep_turns_long': t_dep_turns_long
            })

    return branch_plot_data


plot_data_corrected = deep_track_branch(df_corrected_only, apply_correction=True)
plot_data_misaligned = deep_track_branch(df_misaligned_only, apply_correction=False)


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


plot_seed_with_textbox(
    plot_data_corrected['top_1'][0], 'Best Seed (Corrected)',
    f'Results/D{design}/C{config}/Comparison/TopSeed_Polarization_Corrected.png'
)

plot_seed_with_textbox(
    plot_data_corrected['bottom_1'][0], 'Worst Seed (Corrected)',
    f'Results/D{design}/C{config}/Comparison/BottomSeed_Polarization_Corrected.png'
)

plot_seed_with_textbox(
    plot_data_misaligned['top_1'][0], 'Best Seed (Misaligned)',
    f'Results/D{design}/C{config}/Comparison/TopSeed_Polarization_Misaligned.png'
)

plot_seed_with_textbox(
    plot_data_misaligned['bottom_1'][0], 'Worst Seed (Misaligned)',
    f'Results/D{design}/C{config}/Comparison/BottomSeed_Polarization_Misaligned.png'
)



# ===========================================================================
# PART 3 — INVARIANT SPIN VECTOR along the ring, for the same best/worst seeds
# ===========================================================================
# Reuses df_scan_full (Part 1) for seed selection again -- no disk read needed.
#%%

def plot_invariant_spin_vector(seed_val, seed_label, apply_correction):
    print(f"Plotting invariant spin vector for {seed_label} (Seed {seed_val}, "
          f"{'corrected' if apply_correction else 'misaligned'})...")

    line = base_line.copy()
    line = mc.misalignments(line, 0.2e-3, seed=seed_val)

    line.configure_radiation('mean')
    tw = line.twiss(method='6d', radiation_integrals=True, eneloss_and_damping=True,
                    spin=True, polarization=True)

    if apply_correction:
        try:
            line.discard_tracker()
            mc.orbit_correction(pdr, tw, threading=False)
        except:
            mc.orbit_correction(pdr, tw, threading=True)
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
    ax.set_title(f'Invariant Spin Vector Along the Ring — {seed_label} (Seed {seed_val})')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()

    plt.tight_layout()
    branch_tag = 'Corrected' if apply_correction else 'Misaligned'
    out_path = (f'Results/D{design}/C{config}/Comparison/'
                f'InvariantSpinVector_{seed_label.replace(" ", "")}_{branch_tag}.png')
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"  Saved to {out_path}")


# Best/worst seeds for each branch, selected within that branch's own results
# (a seed that's "best" when corrected isn't necessarily the same seed that's
# "best" when misaligned, so each branch gets its own top/bottom pick).
df_mis_full = df_scan_full[df_scan_full['Mode'] == 'misaligned']
df_cor_full = df_scan_full[df_scan_full['Mode'] == 'corrected']

top_row_mis = df_mis_full.nlargest(1, 'P_eq').iloc[0]
bottom_row_mis = df_mis_full.nsmallest(1, 'P_eq').iloc[0]
top_row_cor = df_cor_full.nlargest(1, 'P_eq').iloc[0]
bottom_row_cor = df_cor_full.nsmallest(1, 'P_eq').iloc[0]

plot_invariant_spin_vector(int(top_row_mis['Seed']), 'Best Seed', apply_correction=False)
plot_invariant_spin_vector(int(bottom_row_mis['Seed']), 'Worst Seed', apply_correction=False)
plot_invariant_spin_vector(int(top_row_cor['Seed']), 'Best Seed', apply_correction=True)
plot_invariant_spin_vector(int(bottom_row_cor['Seed']), 'Worst Seed', apply_correction=True)
# %%