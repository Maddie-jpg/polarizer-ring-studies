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
mode=os.environ.get('MODE','corrected')

# %%
if mode == 'corrected':
    pdr = xt.Environment.from_json(f'JSON Files/D{design}/C{config}/pdr_perfect.json')
    if design ==1 and config==1:
        mc.insert_BPMs_all_as_markers(pdr)
        mc.insert_correctors_var2(pdr)
    else:
        mc.insert_BPMs_all_as_markers(pdr)
        mc.insert_correctors(pdr)
else:
    pdr = xt.Environment.from_json(f'JSON Files/D{design}/C{config}/pdr_{mode}.json')
pdr.particle_ref.anomalous_magnetic_moment=0.001159652181

line=pdr.lines['ring']

# Simulate bunch evolution with stochastic photon emission
line.configure_spin('auto')

max_seed_value = np.iinfo(np.uint32).max  
num_seeds=50 
seeds = np.random.randint(0, max_seed_value, size=num_seeds)
scan_turns=1000


P_BKS, tau_BKS, P_DKM, tau_DKM, tau_depol, tau_pol, P_eq = [], [], [], [], [], [], []
tune_x, tune_y, tune_s = [], [], []
t_dep_turns_list = []

prepped_particles, prepped_twiss, prepped_lines = [], [], []
failed_misalignments=[]
successful_seeds=[]

base_line = line.copy()


for seed in seeds:
    try:
        seed_line = base_line.copy()
        seed_line = mc.misalignments(seed_line, 0.2e-3, seed=seed)

        seed_line.configure_radiation('mean')
        tw = seed_line.twiss(method='6d', radiation_integrals=True, eneloss_and_damping=True,
                        spin=True, polarization=True)

        if mode=='corrected':
            try:
                seed_line.discard_tracker()
                mc.orbit_correction(pdr, tw, threading=False, rcond_x=1e-4, rcond_y=1e-2)
                
            except:
                mc.orbit_correction(pdr, tw, threading=False, rcond_x=1e-4, rcond_y=1e-2)
                

        seed_line.discard_tracker()
        seed_line.build_tracker()
        
        print(tw.cols)
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

        prepped_particles.append(particles)
        prepped_twiss.append(tw)
        prepped_lines.append(seed_line)
        successful_seeds.append(seed)


    except (RuntimeError, np.linalg.LinAlgError,ValueError) as e:
        print(e)
        failed_misalignments.append(seed)

print(len(failed_misalignments))

#%%

for seed, particles, tw, seed_line in zip(successful_seeds, prepped_particles, prepped_twiss, prepped_lines):

        
        seed_line.discard_tracker()
        seed_line.build_tracker(_context=xo.ContextCpu(omp_num_threads=0))

        seed_line.track(particles, num_turns=scan_turns, turn_by_turn_monitor=True,
                with_progress=10)
        mon = seed_line.record_last_track

        # Fit depolarization time
        mask_alive = mon.state > 0
        pol_x = mon.spin_x.sum(axis=0)/mask_alive.sum(axis=0)
        pol_y = mon.spin_y.sum(axis=0)/mask_alive.sum(axis=0)
        pol_z = mon.spin_z.sum(axis=0)/mask_alive.sum(axis=0)
        pol = np.sqrt(pol_x**2 + pol_y**2 + pol_z**2)

        pol_to_fit = pol[3:] / pol[3]
        turns = np.arange(len(pol_to_fit))
        slope, intercept, r_value, p_value, std_err = linregress(turns, pol_to_fit)
        # Calculate depolarization time
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
        tune_s.append(tw.qs)

scan_results={
    'Seed': successful_seeds,
    'P_BKS': P_BKS,
    't_BKS': tau_BKS,
    'P_DKM': P_DKM,
    't_DKM': tau_DKM,
    't_depol':tau_depol,
    't_pol':tau_pol,
    'P_eq': P_eq,
    't_dep_turns': t_dep_turns_list,          # raw linear-fit depolarization time, in TURNS
    'N_over_tau': [scan_turns / t for t in t_dep_turns_list],  # scan_turns/tau diagnostic
    'qx': tune_x,
    'qy': tune_y,
    'qs': tune_s

}


df=pd.DataFrame(scan_results)
df.to_csv(f'Results/D{design}/C{config}/{mode}/SpinTrackingResults.dat')

#%%
'''pdf_run=False

if pdf_run is True:
    pdf = PdfPages(f"Results/D{design}/C{config}/{mode}/SpinTrackingResults.pdf")

    _old_savefig = plt.savefig

    def _new_savefig(*args, **kwargs):
        pdf.savefig(plt.gcf())
        _old_savefig(*args, **kwargs)

    plt.savefig = _new_savefig'''
# %%


#%%

# Polarization vs depol time
df['t_depol_fitted_seconds'] = df['t_pol'] / ((df['P_BKS'] / df['P_eq']) - 1)

plt.scatter(df['P_eq'],df['t_depol_fitted_seconds']/60**2)
plt.xlabel('Equilibrium Polarization (%)')
plt.ylabel('Depolarization Time (hours)')
plt.savefig(f'Results/D{design}/C{config}/{mode}/EquilibriumPol_v_DepolTime.png')


#%%
df['t_depol_fitted_seconds'] = df['t_pol'] / ((df['P_BKS'] / df['P_eq']) - 1)

plt.figure(figsize=(8, 5))

df=df[df['P_eq']<=100]
# Plot histogram
counts, bins, patches = plt.hist(
    df['P_eq'], 
    bins=30, 
    color='skyblue', 
    edgecolor='black', 
    alpha=0.8
)
plt.title('Distribution of Equilibrium Polarization across Seeds', fontsize=13, fontweight='bold')
plt.xlabel('Equilibrium Polarization (%)', fontsize=11)
plt.ylabel('Number of Seeds (Frequency)', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)

# Add a vertical line for the mean value
mean_depol = df['P_eq'].mean()
plt.axvline(mean_depol, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_depol:.2e} s')
plt.legend()
plt.tight_layout()
plt.savefig(f'Results/D{design}/C{config}/{mode}/EqPolDist.png')


'''if pdf_run is True:
    pdf.close()'''
# %%
