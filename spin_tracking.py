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
import my_functions as mf


# %%
design=int(os.environ.get('DESIGN',1))
config=int(os.environ.get('CONFIG',9))
phase=int(os.environ.get('PHASE',90))
changes=os.environ.get('CHANGES',None)

# %%
if changes is not None:
    pdr= xt.Environment.from_json(f"JSON Files/D{design}/C{config}/pdr_perfect_{phase}_{changes}.json")
else:
    pdr= xt.Environment.from_json(f"JSON Files/D{design}/C{config}/pdr_perfect_{phase}.json")

pdr.lines['ring'].particle_ref.anomalous_magnetic_moment=0.001159652181
pdr.lines['ring'].particle_ref.kinetic_energy0=2.86e9

if design == 1 and config == 1:
    mc.insert_BPMs_all_as_markers(pdr)
    mc.insert_correctors_var2(pdr)
else:
    mc.insert_BPMs_all_as_markers(pdr)
    mc.insert_correctors(pdr,debug_check=True)


line=pdr.lines['ring']

# Simulate bunch evolution with stochastic photon emission
line.configure_spin('auto')

max_seed_value = np.iinfo(np.uint32).max  
num_seeds=20
seeds = np.random.randint(0, max_seed_value, size=num_seeds)
scan_turns=20000

misalignment_val=0.25e-3

base_line = line.copy()

results_dir = f'Results/D{design}/C{config}/Comparison'
os.makedirs(results_dir, exist_ok=True)
results_path = f'{results_dir}/SpinTrackingResults_MisalignedVsCorrected.dat'


def prep_branch(seed, apply_correction):
    
    seed_line = base_line.copy()

    seed_line.configure_radiation('mean')
    seed_line.build_tracker(_context=xo.ContextCpu(omp_num_threads=0))
    seed_line = mc.misalignments(seed_line, misalignment_val, seed=seed)

    tw = seed_line.twiss(method='6d', radiation_integrals=True, eneloss_and_damping=True,
                    spin=True, polarization=True)

    if apply_correction:
        orbit_x_rms_before = np.std(tw.x)
        orbit_y_rms_before = np.std(tw.y)
        mc.misalignments_correctors(seed_line,0.25e-3,seed+1)
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

  
    seed_line.configure_radiation(model='quantum')

    return particles, tw, seed_line


def run_scan_pass(seed_list, apply_correction):
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

        seed_line.build_tracker(_context=xo.ContextCpu(omp_num_threads='auto'))
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

        seed_line.build_tracker(_context=xo.ContextCpu(omp_num_threads=0))

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

#%%
df_misaligned = run_scan_pass(seeds, apply_correction=False)
df_misaligned.to_csv(results_path)
print(f"Pass 1 (misaligned) complete. Saved {len(df_misaligned)} rows to {results_path}")
#%%
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
# PART 2 — DEEP TRACK on the single best and single worst seed from the scan

long_scan_turns=20000

# each deep-track seed starts from the same untouched lattice.
df_mis_full = df_scan_full[df_scan_full['Mode'] == 'misaligned']
df_cor_full = df_scan_full[df_scan_full['Mode'] == 'corrected']

top_row_mis = df_mis_full.nlargest(1, 'P_eq').iloc[0]
bottom_row_mis = df_mis_full.nsmallest(1, 'P_eq').iloc[0]
top_row_cor = df_cor_full.nlargest(1, 'P_eq').iloc[0]
bottom_row_cor = df_cor_full.nsmallest(1, 'P_eq').iloc[0]


plot_data_misaligned_top=mf.deep_track_single(base_line, top_row_mis['Seed'],long_scan_turns, apply_correction=False)
plot_data_misaligned_bottom=mf.deep_track_single(base_line, bottom_row_mis['Seed'],long_scan_turns, apply_correction=False)
plot_data_corrected_top=mf.deep_track_single(base_line, top_row_cor['Seed'],long_scan_turns, apply_correction=True)
plot_data_corrected_bottom=mf.deep_track_single(base_line, bottom_row_cor['Seed'],long_scan_turns, apply_correction=True)



mf.plot_seed_with_textbox(
    plot_data_corrected_top, 'Best Seed (Corrected)',
    f'{results_dir}/TopSeed_Polarization_Corrected.png'
)

mf.plot_seed_with_textbox(
    plot_data_corrected_bottom, 'Worst Seed (Corrected)',
    f'{results_dir}/BottomSeed_Polarization_Corrected.png'
)

mf.plot_seed_with_textbox(
    plot_data_misaligned_top, 'Best Seed (Misaligned)',
    f'{results_dir}/TopSeed_Polarization_Misaligned.png'
)

mf.plot_seed_with_textbox(
    plot_data_misaligned_bottom, 'Worst Seed (Misaligned)',
    f'{results_dir}/BottomSeed_Polarization_Misaligned.png'
)



#%%
# INVARIANT SPIN VECTOR along the ring, for the same best/worst seeds

mf.plot_invariant_spin_vector(base_line,int(top_row_mis['Seed']), apply_correction=False, out_path=f'{results_dir}/InvariantSpinVector_BestSeed_Misaligned.png')
mf.plot_invariant_spin_vector(base_line,int(bottom_row_mis['Seed']), 'Worst Seed', apply_correction=False, out_path=f'{results_dir}/InvariantSpinVector_WorstSeed_Misaligned.png')
mf.plot_invariant_spin_vector(base_line,int(top_row_cor['Seed']), 'Best Seed', apply_correction=True, out_path=f'{results_dir}/InvariantSpinVector_BestSeed_Corrected.png')
mf.plot_invariant_spin_vector(base_line,int(bottom_row_cor['Seed']), 'Worst Seed', apply_correction=True, out_path=f'{results_dir}/InvariantSpinVector_WorstSeed_Corrected.png')
