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

SEED = 3578530

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


# ===========================================================================
# Shared exponential-decay fit for the depolarization time.
#
# BOUNDED, matching the supervisor's fitfu/curve_fit approach:
#   - amplitude 'a' constrained to [0.9, 1.1]  (pol starts at ~1 by
#     construction, so a well-behaved fit should stay near there)
#   - decay rate 'tauinv' constrained to [0, 1/TAU_MIN_TURNS]
#     i.e. tau_depol >= TAU_MIN_TURNS turns.
#
# Without these bounds, an unconstrained curve_fit on a polarization trace
# that is dominated by shot noise (no visible trend) is free to converge on
# an almost-zero decay rate, which inverts to an astronomically large and
# meaningless tau_depol. Bounding tauinv away from zero forces the fit to
# report a physically plausible decay time even when it can't resolve the
# true one, and the reliability check below then flags those cases
# explicitly instead of silently reporting nonsense.
# ===========================================================================

TAU_MIN_TURNS = 1000  # floor on fitted depolarization time, in turns (matches
                       # supervisor's tauinv upper bound of 1/1000 per turn)


def exp_decay(t, a, tauinv):
    return a * np.exp(-tauinv * t)


def two_step_exp_fit(t, p, amp_bounds=(0.9, 1.1), tau_min_turns=TAU_MIN_TURNS):
    """Two-step fit: lstsq gives a robust starting guess, curve_fit refines
    it within bounds. Returns (popt, pcov) with popt = [a, tauinv]."""
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)

    A = np.vstack([t, np.ones_like(t)]).T
    lin_slope, intercept = np.linalg.lstsq(A, p, rcond=None)[0]
    tauinv0 = -lin_slope / intercept if intercept != 0 else 0.0

    tauinv_max = 1.0 / tau_min_turns
    p0 = [np.clip(intercept, *amp_bounds), np.clip(tauinv0, 0.0, tauinv_max)]
    bounds = ([amp_bounds[0], 0.0], [amp_bounds[1], tauinv_max])

    popt, pcov = curve_fit(exp_decay, t, p, p0=p0, bounds=bounds, maxfev=10000)
    return popt, pcov


def fit_is_reliable(t, p, popt, min_snr=3.0):
    """A fitted decay is only trustworthy if the amplitude it predicts the
    signal actually moves by, over the tracked window, is well above the
    noise floor of the residuals. Otherwise the fit is just tracking noise."""
    a, tauinv = popt
    if not (np.isfinite(tauinv) and tauinv > 0):
        return False
    n = len(t)
    predicted_amplitude_change = a * (1 - np.exp(-tauinv * n))
    residual_rms = np.std(p - exp_decay(t, *popt))
    if residual_rms == 0:
        return True
    return predicted_amplitude_change > min_snr * residual_rms




def generate_bunch(tw, particle_ref, num_particles, fct=0.0,
                                     seed=None, match_supervisor_epsy_bug=True):
   
    rng = np.random.default_rng(seed)

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

    eps_y_row = epsx if match_supervisor_epsy_bug else epsy

    normat = np.array([
        [(epsx*betxin)**.5,          0,                       0, 0, 0, dxin*(epsl/betlin)**.5],
        [-alfxin*(epsx/betxin)**.5,  (epsx/betxin)**.5,        0, 0, 0, dpxin*(epsl/betlin)**.5],
        [0, 0,  (eps_y_row*betyin)**.5,          0,                       0, 0],
        [0, 0,  -alfyin*(eps_y_row/betyin)**.5,  (eps_y_row/betyin)**.5,  0, 0],
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
        p0c=particle_ref.p0c[0],
        mass0=particle_ref.mass0,
        q0=particle_ref.q0,
        anomalous_magnetic_moment=0.00,
        x=parts[0], px=parts[1],
        y=parts[2], py=parts[3],
        zeta=parts[4], delta=parts[5],
        spin_x=parts[6], spin_y=parts[7], spin_z=parts[8],
    )
    return particles


#%%


def deep_track_single(seed_val, apply_correction, transient_turns=8000):
    

    branch_label = 'corrected' if apply_correction else 'misaligned'
    print(f"Running deep track for Seed {seed_val} ({branch_label})...")

    line = base_line.copy()

    line.configure_radiation('mean')
    line.build_tracker(_context=xo.ContextCpu(omp_num_threads=0))
    #line = mc.misalignments(line, 0.25e-3, seed=seed_val)

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


    particles = generate_bunch(
        tw, line.particle_ref, num_particles=300, fct=0.0, seed=seed_val)

    line.configure_radiation('quantum')
    line.build_tracker(_context=xo.ContextCpu(omp_num_threads='auto'))

    line.track(particles, num_turns=long_scan_turns, turn_by_turn_monitor=True, with_progress=10)
    mon = line.record_last_track

    num_particles = 300
    pol_x = mon.spin_x.sum(axis=0) / num_particles
    pol_y = mon.spin_y.sum(axis=0) / num_particles
    pol_z = mon.spin_z.sum(axis=0) / num_particles
    pol = np.sqrt(pol_x**2 + pol_y**2 + pol_z**2)

    mask_alive = mon.state > 0
    n_alive = mask_alive.sum(axis=0)
    if n_alive.min() < num_particles:
        print(f"  [seed {seed_val}] WARNING: particle losses detected "
              f"(min alive = {n_alive.min()}/{num_particles}) -- the fixed-"
              f"denominator polarization is biased low after losses begin.")

    n_turns_rec = len(pol)
    turns = np.arange(n_turns_rec)

    icTrns = int(np.clip(transient_turns, 0, n_turns_rec - 2))

   
    try:
        popt_all, _ = two_step_exp_fit(turns, pol)
        popt_cut, _ = two_step_exp_fit(turns[icTrns:], pol[icTrns:])
        amp_all, tauinv_all = popt_all
        amp_cut, tauinv_cut = popt_cut
        t_dep_turns_all = 1.0 / tauinv_all if tauinv_all > 0 else np.nan
        t_dep_turns_long = 1.0 / tauinv_cut if tauinv_cut > 0 else np.nan
        fit_curve = exp_decay(turns, amp_cut, tauinv_cut)     
        fit_curve_all = exp_decay(turns, amp_all, tauinv_all)  
        fit_ok = (
            np.isfinite(t_dep_turns_long)
            and t_dep_turns_long < 100 * long_scan_turns
            and fit_is_reliable(turns[icTrns:], pol[icTrns:], popt_cut)
        )
    except (RuntimeError, ValueError) as e:
        print(f"  Exponential fit failed for seed {seed_val}: {e}")
        amp_cut = np.nan
        t_dep_turns_all = np.nan
        t_dep_turns_long = np.nan
        fit_curve = np.full_like(pol, np.nan)
        fit_curve_all = np.full_like(pol, np.nan)
        fit_ok = False

    # Fall back to the analytic depol time if the tracked fit isn't reliable
    # (e.g. no resolvable decay above the noise floor over the tracked window).
    if not fit_ok:
        t_depol = tw.spin_t_depol_component_s
        t_dep_turns_long = t_depol / tw.T_rev0
        print(f"  Seed {seed_val}: fit unreliable -- "
              f"using analytic t_depol = {t_dep_turns_long:.3e} turns")

    n_over_tau = n_turns_rec / t_dep_turns_long if t_dep_turns_long > 0 else np.nan
    print(f"  Seed {seed_val}: tau_fit(all) = {t_dep_turns_all:.1f} turns, "
          f"tau_fit(cut@{icTrns}) = {t_dep_turns_long:.1f} turns, "
          f"N/tau = {n_over_tau:.2f}, fit_reliable = {fit_ok}")

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
        'fit_reliable': fit_ok,
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

    reliability_tag = ('fit reliable' if data.get('fit_reliable', True)
                        else 'FIT UNRELIABLE -- using analytic tau_depol')
    info_text = (
        f"$P_{{BKS}}$ = {data['p_bks']:.2f}%   "
        f"$P_{{eq}}$ (long track) = {data['p_eq_long']:.2f}%\n"
        f"$\\tau_{{depol}}$ = {data['t_dep_turns_all']:.1f} turns   "
        f"$t_{{pol}}$ = {data['t_pol']:.4e} s   [{reliability_tag}]\n"
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

    #line=mc.misalignments(line,2.5e-3,seed_val)
    

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
    line = mc.misalignments(line, 0.25e-3, seed=seed_val)
 
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