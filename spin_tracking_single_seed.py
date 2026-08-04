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
import my_functions as mf
import random


#%%

SEED = random.randint(0,int(1e6))
SEED = 1687758877

design = int(os.environ.get('DESIGN', 1))
config = int(os.environ.get('CONFIG', 1))
phase=int(os.environ.get('PHASE',90))

long_scan_turns = 20000  

# %%

pdr = xt.Environment.from_json(f'JSON Files/D{design}/C{config}/pdr_perfect_{phase}.json')
energy=2.86e9
pdr.lines['ring'].particle_ref.anomalous_magnetic_moment = 0.001159652181
pdr.lines['ring'].particle_ref.kinetic_energy0 = energy

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

scan = mf.spin_tune_resonance_scan(line, nu_min=5.5, nu_max=7.5, n_points=80,
                                 misalign_sigma=0.25e-3, seed=SEED)
mf.plot_spin_resonance_scan(scan, out_path=f'{results_dir}/spin_resonance_scan.png')


#%%


def deep_track_single(seed_val, apply_correction, transient_turns=8000):
    

    branch_label = 'corrected' if apply_correction else 'misaligned'
    print(f"Running deep track for Seed {seed_val} ({branch_label})...")

    line = base_line.copy()

    line.configure_radiation('mean')
    line.build_tracker(_context=xo.ContextCpu(omp_num_threads=0))
    line = mc.misalignments(line, 0.25e-3, seed=seed_val)

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

    particles = xp.generate_matched_gaussian_bunch(
        line=line,
        nemitt_x=tw.eq_nemitt_x,
        nemitt_y=tw.eq_nemitt_y,
        sigma_z=np.sqrt(tw.eq_gemitt_zeta * tw.bets0),
        num_particles=num_particles)

    particles.zeta += tw.zeta[0]
    particles.delta += tw.delta[0]
    particles.spin_x = tw.spin_x[0]
    particles.spin_y = tw.spin_y[0]
    particles.spin_z = tw.spin_z[0]

    '''rng = np.random.default_rng(seed_val)
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
    )'''
    

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



# %%


def n0_vs_spin_tune_scan(seed_val, nu_min, nu_max, n_points=60,apply_correction=True, at_element=None):

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
    ring0 = line
    a_gyro = ring0.particle_ref.anomalous_magnetic_moment[0]
    mass0 = ring0.particle_ref.mass0  # eV

    nu_targets = np.linspace(nu_min, nu_max, n_points)
    gammas = nu_targets / a_gyro
    energies = gammas * mass0

    nu_spin, n0x, n0y, n0z = [], [], [], []

    for nu_target, energy in zip(nu_targets, energies):
        line = ring0.copy()
        line.particle_ref.kinetic_energy0 = energy - mass0
        line.configure_spin('auto')
        try:
            tw = line.twiss(method='6d', radiation_integrals=True,
                            eneloss_and_damping=True, spin=True, polarization=True)
            idx = 0 if at_element is None else tw.rows.indices[at_element]
            nu_spin.append(tw.spin_tune_fractional + np.floor(nu_target))
            n0x.append(tw.spin_x[idx])
            n0y.append(tw.spin_y[idx])
            n0z.append(tw.spin_z[idx])
        except Exception as e:
            print(f"  nu_target={nu_target:.4f}: failed ({type(e).__name__}: {e})")
            nu_spin.append(np.nan); n0x.append(np.nan)
            n0y.append(np.nan);     n0z.append(np.nan)

    return {k: np.array(v) for k, v in
            dict(nu_spin=nu_spin, n0x=n0x, n0y=n0y, n0z=n0z).items()}


def plot_n0_vs_spin_tune(results, out_path=None):
    nu = results['nu_spin']
    order = np.argsort(nu)  # spin tune isn't monotonic in nu_target near a
                            # resonance crossing, so sort by the x-axis itself
    nu = nu[order]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(nu, results['n0x'][order], 'o-', ms=4, label='$n_{0,x}$', color='tab:blue')
    ax.plot(nu, results['n0y'][order], 'o-', ms=4, label='$n_{0,y}$', color='tab:green')
    ax.plot(nu, results['n0z'][order], 'o-', ms=4, label='$n_{0,z}$', color='tab:red')

    ax.set_xlabel(r'Spin tune $\nu_{spin}$')
    ax.set_ylabel(r'Invariant spin vector $n_0$ component')
    ax.set_title(r'Invariant spin vector vs spin tune')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300)
    return fig, ax



def check_qy_spin_coupling(line, dqy=1e-3, qy_knobs=('kQFarc', 'kQDarc'),
                           max_order=5, verbose=True):


    tw0 = line.twiss(method='6d', radiation_integrals=True,
                     eneloss_and_damping=True, spin=True, polarization=True)
    Qx0, Qy0 = tw0.qx, tw0.qy
    nu0 = tw0.spin_tune_fractional
    n0_vec0 = np.array([tw0.spin_x[0], tw0.spin_y[0], tw0.spin_z[0]])

    n_center = int(round(nu0))
    candidates = []
    for n in range(n_center - max_order, n_center + max_order + 1):
        for sign in (+1, -1):
            res = (n + sign * Qy0)
            candidates.append((abs(res - nu0), n, sign, res))
    candidates.sort(key=lambda c: c[0])
    dist, n_near, sign_near, res_near = candidates[0]

    if verbose:
        sign_str = '+' if sign_near > 0 else '-'
        print(f"[coupling] Qx={Qx0:.5f}  Qy={Qy0:.5f}  nu_spin={nu0:.5f}")
        print(f"[coupling] nearest vertical intrinsic resonance: "
              f"nu = {n_near} {sign_str} Qy = {res_near:.5f}  "
              f"(distance {dist:.5f})")

    line2 = line.copy()
    k0 = {kn: line2.vars[kn]._value for kn in qy_knobs}
    limits = {kn: (v - 0.05 * abs(v) - 1e-6, v + 0.05 * abs(v) + 1e-6)
             for kn, v in k0.items()}
    try:
        opt = line2.match(
            method='6d', solve=False, verbose=False,
            vary=[xt.Vary(kn, step=1e-6, limits=limits[kn]) for kn in qy_knobs],
            targets=[
                xt.Target('qx', Qx0, tol=1e-7),
                xt.Target('qy', Qy0 + dqy, tol=1e-7),
            ],
        )
        try:
            opt.solve(n_steps=20)
        except (RuntimeError, np.linalg.LinAlgError):
            for kn in qy_knobs:
                line2.vars[kn] = k0[kn]
            opt.step(20, broyden=True, rcond=1e-3)
        tw1 = line2.twiss(method='6d', radiation_integrals=True,
                          eneloss_and_damping=True, spin=True, polarization=True)
        Qx1, Qy1 = tw1.qx, tw1.qy
        nu1 = tw1.spin_tune_fractional
        n0_vec1 = np.array([tw1.spin_x[0], tw1.spin_y[0], tw1.spin_z[0]])

        dQy_actual = Qy1 - Qy0
        dQx_actual = Qx1 - Qx0
        dnu = nu1 - nu0
        dn0 = np.linalg.norm(n0_vec1 - n0_vec0)

        dnu_dQy = dnu / dQy_actual if abs(dQy_actual) > 1e-12 else np.nan
        dn0_dQy = dn0 / abs(dQy_actual) if abs(dQy_actual) > 1e-12 else np.nan

        if verbose:
            print(f"[coupling] Qy nudged by {dQy_actual:+.2e} "
                  f"(Qx held to within {dQx_actual:+.2e})")
            print(f"[coupling] d(nu_spin)/d(Qy) ~= {dnu_dQy:.4f}")
            print(f"[coupling] |d(n0)|/d(Qy)    ~= {dn0_dQy:.4f}  "
                  f"(tilt of invariant spin axis per unit Qy)")
    except Exception as e:
        print(f"[coupling] could not isolate Qy via {qy_knobs} "
              f"({type(e).__name__}: {e}); reporting resonance proximity only")
        dnu_dQy = dn0_dQy = np.nan

    return dict(Qx=Qx0, Qy=Qy0, nu_spin=nu0,
                nearest_resonance=res_near, nearest_order=n_near,
                nearest_sign=sign_near, distance_to_resonance=dist,
                dnu_dQy=dnu_dQy, dn0_dQy=dn0_dQy)


def compare_qy_spin_coupling_across_branches(base_line, seed_val,
                                             misalign_sigma=0.25e-3,
                                             dqy=1e-3, qy_knobs=('kQFarc', 'kQDarc'),
                                             max_order=5, results_dir=results_dir):
    rows = {}

    line_perfect = base_line.copy()
    print("=== Coupling check: PERFECT lattice ===")
    rows['perfect'] = check_qy_spin_coupling(
        line_perfect, dqy=dqy, qy_knobs=qy_knobs, max_order=max_order)

    line_mis = base_line.copy()
    line_mis.configure_radiation('mean')
    line_mis.build_tracker(_context=xo.ContextCpu(omp_num_threads=0))
    line_mis = mc.misalignments(line_mis, misalign_sigma, seed=seed_val)
    print(f"\n=== Coupling check: MISALIGNED lattice (seed {seed_val}) ===")
    rows['misaligned'] = check_qy_spin_coupling(
        line_mis, dqy=dqy, qy_knobs=qy_knobs, max_order=max_order)

    line_cor = base_line.copy()
    line_cor.configure_radiation('mean')
    line_cor.build_tracker(_context=xo.ContextCpu(omp_num_threads=0))
    line_cor = mc.misalignments(line_cor, misalign_sigma, seed=seed_val)
    tw_cor = line_cor.twiss(method='6d', radiation_integrals=True,
                            eneloss_and_damping=True, spin=True, polarization=True)
    try:
        mc.orbit_correction(line_cor, tw_cor, threading=False)
    except Exception as e:
        print(f"  [seed {seed_val}] orbit_correction(threading=False) raised: "
              f"{type(e).__name__}: {e} -- retrying with threading=True")
        mc.orbit_correction(line_cor, tw_cor, threading=True)
    print(f"\n=== Coupling check: CORRECTED lattice (seed {seed_val}) ===")
    rows['corrected'] = check_qy_spin_coupling(
        line_cor, dqy=dqy, qy_knobs=qy_knobs, max_order=max_order)

    print("\n=== Summary: Qy / nu_spin across branches ===")
    header = f"{'branch':<12}{'Qy':>12}{'nu_spin':>12}{'dist_to_res':>14}{'dnu_dQy':>12}{'dn0_dQy':>12}"
    print(header)
    for branch, r in rows.items():
        print(f"{branch:<12}{r['Qy']:>12.6f}{r['nu_spin']:>12.6f}"
              f"{r['distance_to_resonance']:>14.6f}{r['dnu_dQy']:>12.4f}{r['dn0_dQy']:>12.4f}")

    dQy_mis = rows['misaligned']['Qy'] - rows['perfect']['Qy']
    dnu_mis = rows['misaligned']['nu_spin'] - rows['perfect']['nu_spin']
    dQy_cor = rows['corrected']['Qy'] - rows['perfect']['Qy']
    dnu_cor = rows['corrected']['nu_spin'] - rows['perfect']['nu_spin']
    print(f"\nMisalignment shift:  dQy = {dQy_mis:+.2e}   d(nu_spin) = {dnu_mis:+.2e}")
    print(f"Residual after correction: dQy = {dQy_cor:+.2e}   d(nu_spin) = {dnu_cor:+.2e}")

    if results_dir is not None:
        import csv
        out_path = f'{results_dir}/qy_spin_coupling_seed{seed_val}.csv'
        with open(out_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['branch', 'Qx', 'Qy', 'nu_spin', 'nearest_resonance',
                       'distance_to_resonance', 'dnu_dQy', 'dn0_dQy'])
            for branch, r in rows.items():
                w.writerow([branch, r['Qx'], r['Qy'], r['nu_spin'],
                           r['nearest_resonance'], r['distance_to_resonance'],
                           r['dnu_dQy'], r['dn0_dQy']])
        print(f"\nSaved summary to {out_path}")

    return rows



def assess_seed_resonance_excitation(seed_val, apply_correction, deep_track_single_fn,
                                     check_qy_spin_coupling_fn, base_line,
                                     misalign_sigma=0.26e-3, p_eq_suppression_flag=0.9,
                                     results_dir=None):
    branch_label = 'corrected' if apply_correction else 'misaligned'

    line = base_line.copy()
    line.configure_radiation('mean')
    line.build_tracker(_context=xo.ContextCpu(omp_num_threads=0))
    line = mc.misalignments(line, misalign_sigma, seed=seed_val)

    if apply_correction:
        tw = line.twiss(method='6d', radiation_integrals=True,
                        eneloss_and_damping=True, spin=True, polarization=True)
        try:
            mc.orbit_correction(line, tw, threading=False)
        except Exception as e:
            print(f"orbit_correction(threading=False) raised: {type(e).__name__}: {e}"
                  f" -- retrying with threading=True")
            mc.orbit_correction(line, tw, threading=True)

    coupling = check_qy_spin_coupling_fn(line)

    track_result = deep_track_single_fn(seed_val, apply_correction)

    p_bks = track_result['p_bks']
    p_eq_long = track_result['p_eq_long']
    t_dep_turns = track_result['t_dep_turns_long']
    fit_reliable = track_result['fit_reliable']

    suppression_ratio = p_eq_long / p_bks if p_bks > 0 else np.nan
    excited = bool(np.isfinite(suppression_ratio) and
                  suppression_ratio < p_eq_suppression_flag)

    print(f"\n=== Resonance excitation assessment: seed {seed_val} ({branch_label}) ===")
    print(f"Qx={coupling['Qx']:.5f}  Qy={coupling['Qy']:.5f}  "
          f"nu_spin={coupling['nu_spin_full']:.5f}")
    print(f"nearest resonance: nu = {coupling['nearest_order']} "
          f"{'+' if coupling['nearest_sign']>0 else '-'} Qy = {coupling['nearest_resonance']:.5f}  "
          f"(distance {coupling['distance_to_resonance']:.5f})")
    print(f"tau_depol = {t_dep_turns:.1f} turns (fit_reliable={fit_reliable})")
    print(f"p_bks = {p_bks:.2f}%   p_eq_long = {p_eq_long:.2f}%   "
          f"suppression_ratio = {suppression_ratio:.3f}")
    print(f"excited (p_eq_long < {p_eq_suppression_flag} * p_bks): {excited}")

    row = dict(seed=seed_val, branch=branch_label, Qx=coupling['Qx'], Qy=coupling['Qy'],
              nu_spin_full=coupling['nu_spin_full'],
              nearest_order=coupling['nearest_order'],
              nearest_sign=coupling['nearest_sign'],
              nearest_resonance=coupling['nearest_resonance'],
              distance_to_resonance=coupling['distance_to_resonance'],
              t_dep_turns=t_dep_turns, fit_reliable=fit_reliable,
              p_bks=p_bks, p_eq_long=p_eq_long,
              suppression_ratio=suppression_ratio, excited=excited)

    if results_dir is not None:
        out_path = f'{results_dir}/resonance_excitation_seed{seed_val}_{branch_label}.csv'
        with open(out_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)
        print(f"Saved to {out_path}")

    return row


def assess_resonance_excitation_multi_seed(seed_list, apply_correction, deep_track_single_fn,
                                           check_qy_spin_coupling_fn, base_line,
                                           misalign_sigma=0.26e-3, p_eq_suppression_flag=0.9,
                                           results_dir=None, out_name='resonance_excitation_summary.csv'):
    rows = []
    for seed_val in seed_list:
        row = assess_seed_resonance_excitation(
            seed_val, apply_correction, deep_track_single_fn, check_qy_spin_coupling_fn,
            base_line, mc, xo, misalign_sigma=misalign_sigma,
            p_eq_suppression_flag=p_eq_suppression_flag, results_dir=None)
        rows.append(row)

    if results_dir is not None:
        out_path = f'{results_dir}/{out_name}'
        with open(out_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved summary of {len(rows)} seeds to {out_path}")

    n_excited = sum(r['excited'] for r in rows)
    print(f"\n{n_excited}/{len(rows)} seeds show resonance excitation "
          f"(suppression_ratio < {p_eq_suppression_flag})")

    return rows

#%%

row=assess_seed_resonance_excitation(SEED, True, deep_track_single,
                                     check_qy_spin_coupling, base_line)
print(row)
plot_invariant_spin_vector(SEED, apply_correction=False)
plot_invariant_spin_vector(SEED, apply_correction=True)

track_single_particle_nx1(SEED, apply_correction=False)
track_single_particle_nx1(SEED, apply_correction=True)

results = n0_vs_spin_tune_scan(SEED, nu_min=5.5, nu_max=7.5,
                                 n_points=80, apply_correction=False)
plot_n0_vs_spin_tune(results, out_path=f'{results_dir}/n0_vs_spin_tune_misaligned.png')

results = n0_vs_spin_tune_scan(SEED, nu_min=5.5, nu_max=7.5,
                                 n_points=80, apply_correction=True)
plot_n0_vs_spin_tune(results, out_path=f'{results_dir}/n0_vs_spin_tune_corrected.png')

coupling = compare_qy_spin_coupling_across_branches(line, SEED)
print(coupling)

data_misaligned = deep_track_single(SEED, apply_correction=False)
data_corrected = deep_track_single(SEED, apply_correction=True)



plot_seed_with_textbox(
    data_misaligned, 'Misaligned',
    f'{results_dir}/Polarization_Misaligned.png')
plot_seed_with_textbox(
    data_corrected, 'Corrected',
    f'{results_dir}/Polarization_Corrected.png')