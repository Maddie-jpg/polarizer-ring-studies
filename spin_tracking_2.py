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
pdr.lines['ring'].particle_ref.anomalous_magnetic_moment=0.001159652181
pdr.lines['ring'].particle_ref.kinetic_energy0=2.86e9

if design == 1 and config == 1:
    mc.insert_BPMs_all_as_markers(pdr)
    mc.insert_correctors_var2(pdr)
else:
    mc.insert_BPMs_all_as_markers(pdr)
    mc.insert_correctors(pdr)

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
num_seeds=20
seeds = np.random.randint(0, max_seed_value, size=num_seeds)
scan_turns=1000

base_line = line.copy()

results_dir = f'Results/D{design}/C{config}/Comparison'
os.makedirs(results_dir, exist_ok=True)
results_path = f'{results_dir}/SpinTrackingResults_MisalignedVsCorrected.dat'


def prep_branch(seed, apply_correction):
    """Build one branch (misaligned-only, or misaligned+corrected) for a seed.
    Returns (particles, tw, seed_line) for this branch, raises on failure."""
    seed_line = base_line.copy()

    # Set radiation to 'mean' BEFORE building the tracker: twiss cannot run under
    # the 'quantum' model, and the tracker captures the radiation mode at build
    # time, so configure_radiation must precede build_tracker. Then build the
    # tracker, then misalign onto the live tracker (element_refs), and never
    # discard the tracker afterward (that would wipe the misalignments).
    seed_line.configure_radiation('mean')
    seed_line.build_tracker(_context=xo.ContextCpu(omp_num_threads=0))
    seed_line = mc.misalignments(seed_line, 0.2e-3, seed=seed)

    tw = seed_line.twiss(method='6d', radiation_integrals=True, eneloss_and_damping=True,
                    spin=True, polarization=True)

    if apply_correction:
        orbit_x_rms_before = np.std(tw.x)
        orbit_y_rms_before = np.std(tw.y)
        try:
            mc.orbit_correction(seed_line, tw, threading=False)
        except Exception as e:
            print(f"  [seed {seed}] orbit_correction(threading=False) raised: "
                  f"{type(e).__name__}: {e} -- retrying with threading=True")
            mc.orbit_correction(seed_line, tw, threading=True)
        # Re-twiss after correction so tw reflects the corrected orbit/optics.
        tw = seed_line.twiss(method='6d', radiation_integrals=True, eneloss_and_damping=True,
                        spin=True, polarization=True)
        orbit_x_rms_after = np.std(tw.x)
        orbit_y_rms_after = np.std(tw.y)
        print(f"  [seed {seed}] orbit RMS x: {orbit_x_rms_before:.3e} -> {orbit_x_rms_after:.3e}, "
              f"y: {orbit_y_rms_before:.3e} -> {orbit_y_rms_after:.3e}")

    # Generate the matched bunch while still in 'mean' mode -- matched bunch
    # generation can twiss internally, which fails under 'quantum'.
    particles = xp.generate_matched_gaussian_bunch(
        line=seed_line,
        nemitt_x=tw.eq_nemitt_x,
        nemitt_y=tw.eq_nemitt_y,
        sigma_z=np.sqrt(tw.eq_gemitt_zeta * tw.bets0),
        num_particles=300)
    # Add stable phase
    particles.zeta += tw.zeta[0]
    particles.delta += tw.delta[0]

    # Initialize spin of all particles along n0
    particles.spin_x = tw.spin_x[0]
    particles.spin_y = tw.spin_y[0]
    particles.spin_z = tw.spin_z[0]

    # NOW switch to quantum radiation for tracking. configure_radiation acts on
    # the already-built tracker and does NOT discard it or reset element_refs,
    # so misalignments/correction remain in place. No twiss happens after this.
    seed_line.configure_radiation(model='quantum')

    return particles, tw, seed_line


def run_scan_pass(seed_list, apply_correction):
    """Run the full 1000-turn screen for ONE branch (misaligned or corrected)
    over seed_list, returning a DataFrame of results for just that branch."""
    branch_label = 'corrected' if apply_correction else 'misaligned'

    prepped_particles, prepped_twiss, prepped_lines = [], [], []
    failed_seeds, successful_seeds_local = [], []

    for seed in seed_list:
        try:
            particles, tw, seed_line = prep_branch(seed, apply_correction=apply_correction)
            prepped_particles.append(particles)
            prepped_twiss.append(tw)
            prepped_lines.append(seed_line)
            successful_seeds_local.append(seed)
        except (RuntimeError, np.linalg.LinAlgError, ValueError) as e:
            print(e)
            failed_seeds.append(seed)

    print(f"[{branch_label}] failed seeds: {len(failed_seeds)} / {len(seed_list)}")

    P_BKS, tau_BKS, P_DKM, tau_DKM, tau_depol, tau_pol, P_eq = [], [], [], [], [], [], []
    tune_x, tune_y, spin_tune = [], [], []
    t_dep_turns_list = []
    fit_reliable_list = []
    result_seeds = []

    for seed, particles, tw, seed_line in zip(
            successful_seeds_local, prepped_particles, prepped_twiss, prepped_lines):

        # NOTE: do NOT discard_tracker here -- that would wipe the element_refs
        # misalignments (and correction) applied in prep_branch, reverting the
        # line to a perfect lattice. The tracker was already built (serial
        # context) with radiation set to 'quantum' inside prep_branch.
        seed_line.track(particles, num_turns=scan_turns, turn_by_turn_monitor=True,
                with_progress=10)
        mon = seed_line.record_last_track

        # Fit depolarization time
        mask_alive = mon.state > 0
        pol_x = mon.spin_x.sum(axis=0)/mask_alive.sum(axis=0)
        pol_y = mon.spin_y.sum(axis=0)/mask_alive.sum(axis=0)
        pol_z = mon.spin_z.sum(axis=0)/mask_alive.sum(axis=0)
        pol = np.sqrt(pol_x**2 + pol_y**2 + pol_z**2)

        n_turns_rec = len(pol)
        turns = np.arange(n_turns_rec)

        # Two-parameter exponential: free amplitude 'a' and decay rate 'tauinv'.
        def exp_decay(t, a, tauinv):
            return a * np.exp(-tauinv * t)

        # Two-step fit: lstsq gives robust starting values, curve_fit refines.
        def two_step_exp_fit(t, p):
            t = np.asarray(t, dtype=float)
            p = np.asarray(p, dtype=float)
            A = np.vstack([t, np.ones_like(t)]).T
            lin_slope, intercept = np.linalg.lstsq(A, p, rcond=None)[0]
            p0 = [intercept, -lin_slope / intercept]
            popt, pcov = curve_fit(exp_decay, t, p, p0=p0, maxfev=10000)
            return popt, pcov

        icTrns = int(np.clip(scan_turns // 5, 0, n_turns_rec - 2))

        try:
            popt_all, _ = two_step_exp_fit(turns, pol)
            popt_cut, _ = two_step_exp_fit(turns[icTrns:], pol[icTrns:])
            _, tauinv_all = popt_all
            _, tauinv_cut = popt_cut
            t_dep_turns_all = 1.0 / tauinv_all if tauinv_all > 0 else np.nan
            t_dep_turns   = 1.0 / tauinv_cut if tauinv_cut > 0 else np.nan
            fit_reliable = np.isfinite(t_dep_turns) and t_dep_turns < 100 * scan_turns
        except (RuntimeError, ValueError) as e:
            print(f"  [seed {seed}] two-step fit failed: {e} -- using analytic t_depol")
            t_dep_turns_all = np.nan
            t_dep_turns = np.nan
            fit_reliable = False

        p_bks=tw.spin_polarization_inf_no_depol
        t_bks=tw.spin_t_pol_component_s
        p_dkm=tw.spin_polarization_eq
        t_dkm=tw.spin_t_pol_buildup_s
        t_depol=tw.spin_t_depol_component_s

        # t_pol is the polarization BUILDUP time (Sokolov-Ternov / BKS), in seconds.
        # It is simply tw.spin_t_pol_component_s.
        t_pol = t_bks

        t_pol_turns = t_bks/tw.T_rev0   # polarization time in TURNS (for the ratio below)

        # Fall back to analytic depol time if the fit was unreliable.
        if not fit_reliable or not np.isfinite(t_dep_turns):
            t_dep_turns = t_depol / tw.T_rev0
            fit_reliable = False
            print(f"  [seed {seed}] fit unreliable -- "
                  f"using analytic t_depol = {t_dep_turns:.3e} turns")

        # P_eq = P_inf / (1 + tau_pol/tau_dep), both times in turns. Guaranteed in
        # (0, p_bks] now that t_dep_turns > 0.
        p_eq = p_bks * 1 / (1 + t_pol_turns/t_dep_turns)

        if p_eq > p_bks or p_eq < 0:
            print(f"  [seed {seed}] WARNING: p_eq={p_eq*100:.3f}% outside (0, P_BKS={p_bks*100:.3f}%)")

        P_BKS.append(p_bks*100)
        tau_BKS.append(t_bks)
        P_DKM.append(p_dkm*100)
        tau_DKM.append(t_dkm)
        tau_depol.append(t_depol)
        tau_pol.append(t_pol)
        P_eq.append(p_eq*100)
        t_dep_turns_list.append(t_dep_turns)
        fit_reliable_list.append(fit_reliable)
        tune_x.append(tw.qx)
        tune_y.append(tw.qy)
        spin_tune.append(tw.spin_tune_fractional)
        result_seeds.append(seed)

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
        'fit_reliable': fit_reliable_list,
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
    pdf = PdfPages(f"{results_dir}/SpinTrackingResults.pdf")

    _old_savefig = plt.savefig

    def _new_savefig(*args, **kwargs):
        pdf.savefig(plt.gcf())
        _old_savefig(*args, **kwargs)

    plt.savefig = _new_savefig'''
# %%


#%%

# Polarization vs depol time, split by branch mode so misaligned and
# corrected points are visually distinguishable on the same axes.
# t_depol = t_pol / (P_BKS/P_eq - 1), the inverse of P_eq = P_BKS/(1 + t_pol/t_depol).
# This is now correct because t_pol holds the real buildup time in SECONDS
# (previously it was the collapsed ~T_rev0 value, making this ~1e9x too small).
df['t_depol_fitted_seconds'] = df['t_pol'] / ((df['P_BKS'] / df['P_eq']) - 1)

fig, ax = plt.subplots()
for branch_mode, color in [('misaligned', 'tab:red'), ('corrected', 'tab:blue')]:
    sub = df[df['Mode'] == branch_mode]
    ax.scatter(sub['P_eq'], sub['t_depol_fitted_seconds']/60**2, label=branch_mode,
               color=color, alpha=0.7)
ax.set_xlabel('Equilibrium Polarization (%)')
ax.set_ylabel('Depolarization Time (hours)')
ax.legend()
plt.savefig(f'{results_dir}/EquilibriumPol_v_DepolTime.png')


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
plt.savefig(f'{results_dir}/EqPolDist.png')


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
plt.savefig(f'{results_dir}/PEq_Misaligned_vs_Corrected_Scatter.png', dpi=300)
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
plt.savefig(f'{results_dir}/PEq_Misaligned_vs_Corrected_Histograms.png', dpi=300)
plt.close()

# %%


# ===========================================================================
# PART 2 — DEEP TRACK on the single best and single worst seed from the scan
# ===========================================================================
# Uses df_scan_full (in memory from Part 1) instead of re-reading
# SpinTrackingResults.dat from disk -- this guarantees the deep track is
# always working from the scan that was JUST run, never a stale prior file.
#%%

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
    plot_data structure, scoped to just this one branch."""
    top_1 = df_branch.nlargest(1, 'P_eq')
    bottom_1 = df_branch.nsmallest(1, 'P_eq')
    branch_plot_data = {'top_1': [], 'bottom_1': []}

    for group_name, df_group in [('top_1', top_1), ('bottom_1', bottom_1)]:
        for idx, row in df_group.iterrows():
            seed_val = int(row['Seed'])
            p_eq_scan = row['P_eq']
            print(f"Running deep track for {group_name} - Seed {seed_val} "
                  f"({'corrected' if apply_correction else 'misaligned'})...")

            line = base_line.copy()
            # Set mean radiation before building the tracker (twiss can't run
            # under quantum; base_line may carry quantum state from a prior copy).
            # Build tracker, then misalign onto the live tracker (element_refs);
            # never discard afterward or the misalignments are lost.
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

            # Switch to quantum radiation for tracking WITHOUT discarding the
            # tracker (which would wipe misalignments + correction).
            line.configure_radiation('quantum')

            line.build_tracker(_context=xo.ContextCpu(omp_num_threads='auto'))
            # Track for full long duration
            line.track(particles, num_turns=long_scan_turns, turn_by_turn_monitor=True, with_progress=10)
            mon = line.record_last_track

            # Compute full polarization curve
            mask_alive = mon.state > 0
            pol_x = mon.spin_x.sum(axis=0) / mask_alive.sum(axis=0)
            pol_y = mon.spin_y.sum(axis=0) / mask_alive.sum(axis=0)
            pol_z = mon.spin_z.sum(axis=0) / mask_alive.sum(axis=0)
            pol = np.sqrt(pol_x**2 + pol_y**2 + pol_z**2)

            n_turns_rec = len(pol)
            turns = np.arange(n_turns_rec)
            
            line.build_tracker(_context=xo.ContextCpu(omp_num_threads=0))
            # Two-parameter exponential: free amplitude 'a' and decay rate 'tauinv'.
            def exp_decay(t, a, tauinv):
                return a * np.exp(-tauinv * t)

            # Two-step fit: lstsq gives robust starting values, curve_fit refines.
            def two_step_exp_fit(t, p):
                t = np.asarray(t, dtype=float)
                p = np.asarray(p, dtype=float)
                A = np.vstack([t, np.ones_like(t)]).T
                lin_slope, intercept = np.linalg.lstsq(A, p, rcond=None)[0]
                p0 = [intercept, -lin_slope / intercept]
                popt, pcov = curve_fit(exp_decay, t, p, p0=p0, maxfev=10000)
                return popt, pcov

            icTrns = int(np.clip(long_scan_turns // 5, 0, n_turns_rec - 2))

            try:
                popt_all, _ = two_step_exp_fit(turns, pol)
                popt_cut, _ = two_step_exp_fit(turns[icTrns:], pol[icTrns:])
                amp_all, tauinv_all = popt_all
                amp_cut, tauinv_cut = popt_cut
                t_dep_turns_all  = 1.0 / tauinv_all if tauinv_all > 0 else np.nan
                t_dep_turns_long = 1.0 / tauinv_cut if tauinv_cut > 0 else np.nan
                fit_curve     = exp_decay(turns, amp_cut, tauinv_cut)
                fit_curve_all = exp_decay(turns, amp_all, tauinv_all)
                fit_ok = True
            except (RuntimeError, ValueError) as e:
                print(f"  Exponential fit failed for seed {seed_val}: {e}")
                t_dep_turns_all = np.nan
                t_dep_turns_long = np.nan
                fit_curve = np.full_like(pol, np.nan)
                fit_curve_all = np.full_like(pol, np.nan)
                fit_ok = False

            n_over_tau = n_turns_rec / t_dep_turns_long if fit_ok and t_dep_turns_long > 0 else np.nan
            print(f"  Seed {seed_val}: tau_fit(all) = {t_dep_turns_all:.1f} turns, "
                  f"tau_fit(cut@{icTrns}) = {t_dep_turns_long:.1f} turns, "
                  f"N/tau = {n_over_tau:.2f}")

            p_bks=tw.spin_polarization_inf_no_depol
            t_bks=tw.spin_t_pol_component_s
            p_dkm=tw.spin_polarization_eq
            t_dkm=tw.spin_t_pol_buildup_s
            t_depol=tw.spin_t_depol_component_s
            t_pol=t_bks   # polarization buildup time in seconds (see Part 1 note on the old broken formula)

            t_pol_turns = t_bks/tw.T_rev0
            p_eq = p_bks * 1 / (1 + t_pol_turns/t_dep_turns_long)

            # Calculate the revised equilibrium polarization from the deep track fit
            p_eq_long = (p_bks * 1 / (1 + t_pol_turns / t_dep_turns_long)) * 100

            branch_plot_data[group_name].append({
                'seed': seed_val,
                'turns': turns,
                'pol': pol,
                'fit': fit_curve,
                'fit_all': fit_curve_all,
                'icTrns': icTrns,
                'p_eq_long': p_eq_long,
                'p_bks': p_bks * 100,
                't_pol': t_pol,
                't_dep_turns_long': t_dep_turns_long,
                't_dep_turns_all': t_dep_turns_all,
            })

    return branch_plot_data


plot_data_corrected = deep_track_branch(df_corrected_only, apply_correction=True)
plot_data_misaligned = deep_track_branch(df_misaligned_only, apply_correction=False)


def plot_seed_with_textbox(data, title_prefix, out_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle(f"{title_prefix} - Seed {data['seed']}", fontsize=14, fontweight='bold')

    ax.plot(data['turns'], data['pol'], label='Tracking Data', color='blue', alpha=0.7)
    ax.plot(data['turns'], data['fit'], color='red', linestyle='--')
    if 'fit_all' in data:
        ax.plot(data['turns'], data['fit_all'], color='orange', linestyle=':',
                label='Exp. fit')
   
    ax.set_ylabel('$P(t)$')
    ax.set_xlabel('Turns')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()

    info_text = (
        f"$P_{{BKS}}$ = {data['p_bks']:.2f}%   "
        f"$P_{{eq}}$ (long track) = {data['p_eq_long']:.2f}%\n"
        f"$\\tau_{{depol}}$ (all turns) = {data['t_dep_turns_all']:.1f} turns   "
        f"$\\tau_{{depol}}$ (transient removed) = {data['t_dep_turns_long']:.1f} turns\n"
        f"$t_{{pol}}$ = {data['t_pol']:.4e} s\n"
    )
    fig.text(0.5, 0.02, info_text, ha='center', va='bottom', fontsize=10,
              bbox=dict(boxstyle='round', facecolor='whitesmoke', edgecolor='gray', alpha=0.9))

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig(out_path, dpi=300)
    plt.close()


plot_seed_with_textbox(
    plot_data_corrected['top_1'][0], 'Best Seed (Corrected)',
    f'{results_dir}/TopSeed_Polarization_Corrected.png'
)

plot_seed_with_textbox(
    plot_data_corrected['bottom_1'][0], 'Worst Seed (Corrected)',
    f'{results_dir}/BottomSeed_Polarization_Corrected.png'
)

plot_seed_with_textbox(
    plot_data_misaligned['top_1'][0], 'Best Seed (Misaligned)',
    f'{results_dir}/TopSeed_Polarization_Misaligned.png'
)

plot_seed_with_textbox(
    plot_data_misaligned['bottom_1'][0], 'Worst Seed (Misaligned)',
    f'{results_dir}/BottomSeed_Polarization_Misaligned.png'
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
    # Set mean radiation before building the tracker (twiss can't run under
    # quantum). Build tracker, then misalign onto the live tracker; do not
    # discard afterward (that wiped the misalignments and produced the flat
    # n0_y=1.0 perfect-lattice plot).
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
    ax.plot(s, sx*100, label='$100(n_{0,x})$', color='tab:blue')
    ax.plot(s, sy, label='$n_{0,y}$', color='tab:green')
    ax.plot(s, sz*100 +0.2, label='$100(n_{0,z})+0.2$', color='tab:red')

    ax.set_xlabel('s (m)')
    ax.set_ylabel('Invariant spin vector $n_0$ component')
    ax.set_title(f'Invariant Spin Vector Along the Ring — {seed_label} (Seed {seed_val})')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()

    plt.tight_layout()
    branch_tag = 'Corrected' if apply_correction else 'Misaligned'
    out_path = (f'{results_dir}/'
                f'InvariantSpinVector_{seed_label.replace(" ", "")}_{branch_tag}.png')
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"  Saved to {out_path}")



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