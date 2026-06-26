#%%
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
mode=os.environ.get('MODE','corrected')

long_scan_turns=10000
#%%

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

#%%

df=pd.read_csv(f'Results/D{design}/C{config}/{mode}/SpinTrackingResults.dat')
top_1=df.nlargest(1,'P_eq')
bottom_1=df.nsmallest(1,'P_eq')
plot_data = {'top_1': [], 'bottom_1': []}

# Keep a clean, untouched reference line. Each seed below copies from THIS,
# not from whatever the previous seed's misaligned line ended up being -- otherwise
# misalignments compound across iterations instead of each seed being independent.
base_line = pdr.lines['ring'].copy()

for group_name, df_group in [('top_1', top_1), ('bottom_1', bottom_1)]:
    for idx, row in df_group.iterrows():
        seed_val = int(row['Seed'])
        p_eq_scan = row['P_eq']
        print(f"Running deep track for {group_name} - Seed {seed_val}...")

        line = base_line.copy()
        line.discard_tracker()
        line.build_tracker(_context=xo.ContextCpu(omp_num_threads=0))
        
        line=mc.misalignments(line,0.2e-3,seed=seed_val)

        line.configure_radiation('mean')
        tw = line.twiss(method='6d', radiation_integrals=True, eneloss_and_damping=True,
                        spin=True, polarization=True)

        if mode=='corrected':
            try:
                line.discard_tracker()
                mc.orbit_correction(pdr, tw, threading=False, rcond_x=1e-4, rcond_y=1e-2)
                
            except:
                mc.orbit_correction(pdr, tw, threading=False, rcond_x=1e-4, rcond_y=1e-2)
        
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

        line.configure_radiation('quantum')
        line.discard_tracker()
        line.build_tracker(_context=xo.ContextCpu(omp_num_threads=0))

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

        # Exponential fit: P(t)/P(0) = exp(-t/tau)
        # A linear fit (-1/slope) is only valid when turns << tau; once the
        # tracked window is a sizeable fraction of tau, the curve visibly bends
        # and a straight-line fit biases tau low (apparent depolarization too fast).
        # Fitting the actual exponential form removes that bias regardless of
        # how long N is relative to tau.
        def exp_decay(t, tau):
            return np.exp(-t / tau)

        # Use the short-scan's fitted depolarization time as the initial guess for
        # the exponential fit, converting seconds -> turns via T_rev0. This is a much
        # better starting point than guessing blind, and helps curve_fit converge
        # even for seeds where tau is short relative to the long tracking window.
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

        # Diagnostic: how much of one depolarization time-constant did we actually track?
        # N/tau >> 1 means the seed depolarizes fast relative to the tracked window;
        # worth a closer look (more turns or tighter fit window) if this is large.
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

        plot_data[group_name].append({
            'seed': seed_val,
            'turns': turns,
            'pol': pol_to_fit,
            'fit': fit_curve,
            'p_eq_long': p_eq_long,
            'p_bks': p_bks * 100,
            't_pol': t_pol,
            't_dep_turns_long': t_dep_turns_long
        })

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
    plot_data['top_1'][0], 'Best Seed',
    f'Results/D{design}/C{config}/{mode}/TopSeed_Polarization.png'
)

plot_seed_with_textbox(
    plot_data['bottom_1'][0], 'Worst Seed',
    f'Results/D{design}/C{config}/{mode}/BottomSeed_Polarization.png'
)