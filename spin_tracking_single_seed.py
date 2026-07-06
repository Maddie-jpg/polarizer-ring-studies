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

SEED = 4

design = int(os.environ.get('DESIGN', 1))
config = int(os.environ.get('CONFIG', 1))

long_scan_turns = 20000  

# %%
'''
pdr = xt.Environment.from_json(f'JSON Files/D{design}/C{config}/pdr_perfect_CC.json')
energy=2.86e9
pdr.lines['ring'].particle_ref.anomalous_magnetic_moment = 0.001159652181
pdr.lines['ring'].particle_ref.kinetic_energy0 = energy

if design == 1 and config == 1:
    mc.insert_BPMs_all_as_markers(pdr)
    mc.insert_correctors_var2(pdr)
else:
    mc.insert_BPMs_all_as_markers(pdr)
    mc.insert_correctors(pdr)'''

pdr = xt.Environment.from_json(f'JSON Files/D{design}/C{config}/pdr_misaligned_CC.json')


line = pdr.lines['ring']
line.configure_spin('auto')

base_line = line.copy()

results_dir = f'Results/D{design}/C{config}/SingleSeed_{SEED}_CC'
os.makedirs(results_dir, exist_ok=True)


#%%


def deep_track_single(seed_val, apply_correction, transient_turns=8000):
    

    branch_label = 'corrected' if apply_correction else 'misaligned'
    print(f"Running deep track for Seed {seed_val} ({branch_label})...")

    line = base_line.copy()

    line.configure_radiation('mean')
    line.build_tracker(_context=xo.ContextCpu(omp_num_threads=0))
    #line = mc.misalignments(line, 0.2e-3, seed=seed_val)

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
        
    num_particles=300

    '''particles = xp.generate_matched_gaussian_bunch(
        line=line,
        nemitt_x=tw.eq_nemitt_x,
        nemitt_y=tw.eq_nemitt_y,
        sigma_z=np.sqrt(tw.eq_gemitt_zeta * tw.bets0),
        num_particles=num_particles)

    particles.zeta += tw.zeta[0]
    particles.delta += tw.delta[0]
    particles.spin_x = tw.spin_x[0]
    particles.spin_y = tw.spin_y[0]
    particles.spin_z = tw.spin_z[0]'''

    rng = np.random.default_rng(seed_val)
    fct=0
 
    epsx  = tw.eq_gemitt_x
    epsy  = tw.eq_gemitt_y
    epsl  = tw.eq_gemitt_zeta
 
    betxin, alfxin = tw.betx[0], tw.alfx[0]
    betyin, alfyin = tw.bety[0], tw.alfy[0]
    dxin, dpxin    = tw.dx[0], tw.dpx[0]
    betlin         = tw.bets0
 
    xin, pxin      = tw.x[0], tw.px[0]
    yin, pyin      = tw.y[0], tw.py[0]
    zetain, deltain = tw.zeta[0], tw.delta[0]
 
    sPxin, sPyin, sPzin = tw.spin_x[0], tw.spin_y[0], tw.spin_z[0]
    dsPddx = tw.spin_dn_ddelta_x[0]
    dsPddy = tw.spin_dn_ddelta_y[0]
    dsPddz = tw.spin_dn_ddelta_z[0]
 
    
 
    normat = np.array([
        [(epsx*betxin)**.5,          0,                       0, 0, 0, dxin*(epsl/betlin)**.5],
        [-alfxin*(epsx/betxin)**.5,  (epsx/betxin)**.5,        0, 0, 0, dpxin*(epsl/betlin)**.5],
        [0, 0,  (epsx*betyin)**.5,          0,                       0, 0],
        [0, 0,  -alfyin*(epsx/betyin)**.5,  (epsx/betyin)**.5,  0, 0],
        [0, 0, 0, 0, (epsl*betlin)**.5, 0],
        [0, 0, 0, 0, 0,                 (epsl/betlin)**.5],
    ])
 
    def make_macro_part():
        x, px, y, py, zeta, delta = normat @ rng.standard_normal(6)
        sPx = sPxin + fct*(6.5*px*sPzin + delta*dsPddx)
        sPy = sPyin + fct*(6.5*py*sPzin + delta*dsPddy)
        sPz = sPzin + fct*(-6.5*(px*sPxin + py*sPyin) + delta*dsPddz)
        return (x + xin, px + pxin, y + yin, py + pyin,
                zeta + zetain, delta + deltain, sPx, sPy, sPz)
 
    parts = np.array([make_macro_part() for _ in range(num_particles)]).T
 
    particles = xt.Particles(
        p0c=pdr.lines['ring'].particle_ref.p0c[0],
        mass0=pdr.lines['ring'].particle_ref.mass0,
        q0=pdr.lines['ring'].particle_ref.q0,
        anomalous_magnetic_moment=pdr.lines['ring'].particle_ref.anomalous_magnetic_moment[0],
        x=parts[0], px=parts[1],
        y=parts[2], py=parts[3],
        zeta=parts[4], delta=parts[5],
        spin_x=parts[6], spin_y=parts[7], spin_z=parts[8],
    )
    

    line.discard_tracker()
   
    line.configure_radiation('quantum')
    line.build_tracker(_context=xo.ContextCpu(omp_num_threads='auto'))

    line.configure_spin('auto')

    line.track(particles, num_turns=long_scan_turns, turn_by_turn_monitor=True, with_progress=10)
    mon = line.record_last_track

    # --- Turn-by-turn polarization ---
    mask_alive = mon.state > 0
    n_alive = mask_alive.sum(axis=0)
    pol_x = mon.spin_x.sum(axis=0) / n_alive
    pol_y = mon.spin_y.sum(axis=0) / n_alive
    pol_z = mon.spin_z.sum(axis=0) / n_alive
    pol = np.sqrt(pol_x**2 + pol_y**2 + pol_z**2)

    n_turns_rec = len(pol)
    turns = np.arange(n_turns_rec)

    # Two-parameter exponential: free amplitude 'a' and decay rate 'tauinv' (= 1/tau).
    # Letting the amplitude float means we no longer have to force the curve through
    # the first point (as pol[3:]/pol[3] did), so an initial transient does not bias
    # the whole fit.
    def exp_decay(t, a, tauinv):
        return a * np.exp(-tauinv * t)

    # Two-step fit: (1) a linear least-squares fit provides robust starting values
    # for the amplitude and slope, then (2) curve_fit refines the exponential.
    def two_step_exp_fit(t, p):
        t = np.asarray(t, dtype=float)
        p = np.asarray(p, dtype=float)
        # Step 1: straight line P(t) ~ intercept + slope * t  ->  initial guesses.
        A = np.vstack([t, np.ones_like(t)]).T
        slope, intercept = np.linalg.lstsq(A, p, rcond=None)[0]
        # a0 = value extrapolated to t=0; tauinv0 = -slope/a0 (positive for decay).
        p0 = [intercept, -slope / intercept]
        popt, pcov = curve_fit(exp_decay, t, p, p0=p0, maxfev=10000)
        return popt, pcov

    # Turn index from which to fit, to drop the initial transient before the spin
    # distribution settles. Tune via the transient_turns argument.
    icTrns = int(np.clip(transient_turns, 0, n_turns_rec - 2))

    # Stage A: fit through *all* turns.
    # Stage B: fit from turn icTrns to the end (transient removed) -> primary result.
    try:
        popt_all, _ = two_step_exp_fit(turns, pol)
        popt_cut, _ = two_step_exp_fit(turns[icTrns:], pol[icTrns:])
        amp_all, tauinv_all = popt_all
        amp_cut, tauinv_cut = popt_cut
        t_dep_turns_all = 1.0 / tauinv_all if tauinv_all > 0 else np.nan
        t_dep_turns_long = 1.0 / tauinv_cut if tauinv_cut > 0 else np.nan
        fit_curve = exp_decay(turns, amp_cut, tauinv_cut)      # primary fit, full range
        fit_curve_all = exp_decay(turns, amp_all, tauinv_all)  # all-turns fit, full range
        fit_ok = True
    except (RuntimeError, ValueError) as e:
        print(f"  Exponential fit failed for seed {seed_val}: {e}")
        amp_cut = np.nan
        t_dep_turns_all = np.nan
        t_dep_turns_long = np.nan
        fit_curve = np.full_like(pol, np.nan)
        fit_curve_all = np.full_like(pol, np.nan)
        fit_ok = False

    n_over_tau = n_turns_rec / t_dep_turns_long if fit_ok and t_dep_turns_long > 0 else np.nan
    print(f"  Seed {seed_val}: tau_fit = {t_dep_turns_all:.1f} turns, "
          f"tau_fit(cut@{icTrns}) = {t_dep_turns_long:.1f} turns, N/tau = {n_over_tau:.2f}")

    p_bks = tw.spin_polarization_inf_no_depol
    t_bks = tw.spin_t_pol_component_s
    t_pol = t_bks   # polarization buildup time in seconds
    t_pol_turns = t_bks / tw.T_rev0
    p_eq_long = (p_bks * 1 / (1 + t_pol_turns / t_dep_turns_long)) * 100

    return {
        'seed': seed_val,
        'turns': turns,
        'pol': pol,
        'fit': fit_curve,
        'fit_all': fit_curve_all,
        'amp': amp_cut,
        'icTrns': icTrns,
        'p_eq_long': p_eq_long,
        'p_bks': p_bks * 100,
        't_pol': t_pol,
        't_dep_turns_long': t_dep_turns_long,
        't_dep_turns_all': t_dep_turns_all,
    }


def plot_seed_with_textbox(data, title_prefix, out_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle(f"{title_prefix} - Seed {data['seed']}", fontsize=14, fontweight='bold')

    ax.plot(data['turns'], data['pol'], label='Tracking Data', color='blue', alpha=0.7)
    ax.plot(data['turns'], data['fit'], color='red', linestyle='--',
            label=f"Exp. fit")
  

    ax.set_ylabel('$P(t)$')
    ax.set_xlabel('Turns')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()

    info_text = (
        f"$P_{{BKS}}$ = {data['p_bks']:.2f}%   "
        f"$P_{{eq}}$ (long track) = {data['p_eq_long']:.2f}%\n"
        f"$\\tau_{{depol}}$ = {data['t_dep_turns_long']:.1f} turns   "
        
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


def plot_invariant_spin_vector(seed_val, apply_correction):
    branch_label = 'corrected' if apply_correction else 'misaligned'
    print(f"Plotting invariant spin vector for Seed {seed_val} ({branch_label})...")

    line = base_line.copy()

    line.configure_radiation('mean')
    line.build_tracker(_context=xo.ContextCpu(omp_num_threads=0))
    line = mc.misalignments(line, 0.26e-3, seed=seed_val)

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

    n0_mag = np.sqrt(sx**2 + sy**2 + sz**2)
    print(f"  |n0| range: [{n0_mag.min():.6f}, {n0_mag.max():.6f}] (should be ~1.0)")
    print(f"  spin tune = {tw.spin_tune_fractional:.6f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(s, sx * 100, label='$100(n_{0,x})$', color='tab:blue')
    ax.plot(s, sy, label='$n_{0,y}$', color='tab:green')
    ax.plot(s, (sz* 100)+0.2, label='$100(n_{0,z})+0.2$', color='tab:red')

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
#plot_invariant_spin_vector(SEED, apply_correction=True)

#%%


def track_single_particle_nx1(seed_val, apply_correction):
    branch_label = 'corrected' if apply_correction else 'misaligned'
    print(f"Plotting spin vector (nx=1 init) for Seed {seed_val} ({branch_label})...")
    line = base_line.copy()

    tw_0 = line.twiss(method='6d', radiation_integrals=True, eneloss_and_damping=True,
                    spin=True, polarization=True)

    
    line.configure_radiation('mean')
    line.build_tracker(_context=xo.ContextCpu(omp_num_threads=0))
    line = mc.misalignments(line, 0.26e-3, seed=seed_val)
 
    tw = line.twiss(method='6d', radiation_integrals=True, eneloss_and_damping=True,
                    spin=True, polarization=True)
 
    if apply_correction:
        try:
            mc.orbit_correction(line, tw, threading=False)
        except Exception as e:
            print(f"  [seed {seed_val}] orbit_correction(threading=False) raised: "
                  f"{type(e).__name__}: {e} -- retrying with threading=True")
            mc.orbit_correction(line, tw, threading=True)
 
    tw_nx1 = line.twiss(start=tw_0.name[0], end=tw_0.name[-2], init_at=tw_0.name[0],
                        x=tw_0.x[0],       px=tw_0.px[0],
                        y=tw_0.y[0],       py=tw_0.py[0],
                        zeta=tw_0.zeta[0], delta=tw_0.delta[0],
                        alfx=tw_0.alfx[0], betx=tw_0.betx[0],
                        alfy=tw_0.alfy[0], bety=tw_0.bety[0],
                        dx=tw_0.dx[0],     dpx=tw_0.dpx[0],
                        spin=True, spin_x=1, spin_y=0, spin_z=0,
                        _continue_if_lost=True)



 
    s  = tw_nx1.s
    sx = tw_nx1.spin_x
    sy = tw_nx1.spin_y
    sz = tw_nx1.spin_z
 
    n0_mag = np.sqrt(sx**2 + sy**2 + sz**2)
    print(f"  |s| range: [{n0_mag.min():.6f}, {n0_mag.max():.6f}]")
 
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(s, sx , label='$s_x$',        color='tab:blue')
    ax.plot(s, sy,       label='$s_y$',              color='tab:green')
    ax.plot(s, sz , label='$s_z$', color='tab:red')
 
    ax.set_xlabel('s (m)')
    ax.set_ylabel('Spin vector component')
    branch_tag = 'Corrected' if apply_correction else 'Misaligned'
    ax.set_title(f'Spin Vector Along the Ring $(s_x(0)=1)$ — Seed {seed_val} ({branch_tag})')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
 
    plt.tight_layout()
    out_path = f'{results_dir}/SpinVector_nx1_{branch_tag}.png'
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved to {out_path}")


track_single_particle_nx1(SEED, apply_correction=False)
track_single_particle_nx1(SEED, apply_correction=True)
# %%