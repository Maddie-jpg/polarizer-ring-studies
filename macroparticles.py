"""
Injection-efficiency study with the measured injected positron beam.

  1. Load the injected-beam distribution (config.INJECTED_BEAM_FILE), select the
     main bunch, optionally apply the energy-compressor model.
  2. Characterise it: emittances, moment-based and fitted Twiss, phase-space plots.
  3. Match it to the ring at the injection marker and track: survival vs turns
     at the nominal, optimal and bunch-average energies, and an energy scan.
  4. For misaligned/corrected modes, repeat over several error seeds and overlay.

Inputs   lattice chosen by DESIGN/CONFIG/MODE/PHASE/CHANGES (see paths.study_from_env)
         ENERGY_COMPRESSOR=1 to use the compressed beam
Outputs  Results/.../InjectionEfficiency/

Run:     python macroparticles.py   (or via run_full_sims.py)
"""
# %%
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import xtrack as xt
import xpart as xp
import xobjects as xo
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit
import my_functions as mf
import config as cfg
import paths
xo.context_cpu.allow_no_prebuilt_kernel = True

# %%
study = paths.study_from_env()
design, config, mode = study['design'], study['config'], study['mode']
phase, changes = study['phase'], study['changes']

ENERGY_COMPRESSOR_ON = os.environ.get('ENERGY_COMPRESSOR', 'false').strip().lower() not in ('0', 'false', 'off', 'no')

# %%
df = pd.read_csv(cfg.INJECTED_BEAM_FILE, sep=r'\s+')
print(list(df.columns))

# %%

def filter_beam_core(df_raw, mahalanobis_r=3):
    """
    6-D Mahalanobis-radius cut. NOT the same convention as filter_by_action's
    n_sigma, which is a 2-D action cut -- the fraction kept here follows the
    chi-squared CDF with 6 dof, so mahalanobis_r=3 keeps ~88%, not ~99%:

        r = 2 -> ~45%      r = 3 -> ~88%      r = 5 -> ~97%

    Renamed from n_sigma to stop it being read as an n-sigma beam cut.
    """
    n_sigma = mahalanobis_r
    cols = ['x[mm]', 'xp[mrad]', 'y[mm]', 'yp[mrad]', 't[mm/c]', 'p[MeV/c]']
    df_centered = df_raw[cols] - df_raw[cols].mean()
    
    cov_matrix = df_centered.cov()
    inv_cov_matrix = np.linalg.inv(cov_matrix.values)
    
    distances_sq = np.sum(np.dot(df_centered.values, inv_cov_matrix) * df_centered.values, axis=1)
    
    is_core = distances_sq <= (n_sigma ** 2)
    
    print(f"--> Filtering Beam Core (Mahalanobis r={n_sigma}, 6-D): "
          f"Retained {np.sum(is_core)} / {len(df_raw)} particles.")
    return df_raw[is_core].copy()

def select_main_bunch(df_in, t_col='t[mm/c]', mad_k=10):
    t_vals = np.asarray(df_in[t_col].values, dtype=float)
    med = np.median(t_vals)
    mad = np.median(np.abs(t_vals - med))
    if mad == 0:
        return df_in.copy()
    window = mad_k * mad * 1.4826
    mask = np.abs(t_vals - med) < window
    if mask.sum() < 2:
        return df_in.copy()
    return df_in[mask].copy()

def evaluate_ecs_performance(params, df_raw, ring, ring_tw, p0c_ref, marker_name, n_particles=500, num_turns=100, seed=74):
    R56, Vdeb, Phasdeb = params
    
    c = 299792458
    freq = 3e9
    k = 2 * np.pi * freq / c
    fixed_e_mev = p0c_ref / 1e6

    
    df_subset = df_raw.sample(n=n_particles, random_state=seed)
    
    z_i = df_subset['t[mm/c]'].values * 1e-3
    p_i = df_subset['p[MeV/c]'].values
    Eref = np.mean(p_i)
    delta_i = p_i / Eref - 1

    z_f = z_i + R56 * delta_i
    delta_f = delta_i + Vdeb * np.sin(k * z_f + Phasdeb)
    p_f = Eref * (1 + delta_f)

    rms_spread = (np.std(p_f) / np.mean(p_f)) * 100

    x_in = df_subset['x[mm]'].values / 1000.0
    px_in = df_subset['xp[mrad]'].values / 1000.0
    y_in = df_subset['y[mm]'].values / 1000.0
    py_in = df_subset['yp[mrad]'].values / 1000.0
    
    delta_in = (p_f - fixed_e_mev) / fixed_e_mev
    zeta_in = z_f

    marker_twiss = ring_tw.rows[marker_name]
    x_matched = x_in + marker_twiss['dx'][0] * delta_in
    px_matched = px_in + marker_twiss['dpx'][0] * delta_in
    y_matched = y_in + marker_twiss['dy'][0] * delta_in
    py_matched = py_in + marker_twiss['dpy'][0] * delta_in

    p_test = xp.Particles(
        p0c=p0c_ref,
        mass0=xp.ELECTRON_MASS_EV,
        x=x_matched, px=px_matched,
        y=y_matched, py=py_matched,
        delta=delta_in,
        zeta=zeta_in
    )

    ring.track(p_test, num_turns=num_turns)
    survived = np.sum(p_test.state > 0)
    efficiency = (survived / n_particles) * 100

    return rms_spread, efficiency

def run_multi_objective_scan(df_raw, ring, ring_tw, p0c_ref, marker_name, n_samples=150):
    np.random.seed(42)
    
    R56_samples = np.random.uniform(0.20, 0.45, n_samples)
    Vdeb_samples = np.random.uniform(-0.15, 0.0, n_samples)
    Phasdeb_samples = np.random.uniform(0.10, 0.45, n_samples)
    
    scan_results = []
    
    print(f"Starting joint scan over {n_samples} configurations...")
    for i in range(n_samples):
        R56 = R56_samples[i]
        Vdeb = Vdeb_samples[i]
        Phasdeb = Phasdeb_samples[i]
        
        spread, eff = evaluate_ecs_performance(
            [R56, Vdeb, Phasdeb], df_raw, ring, ring_tw, p0c_ref, marker_name
        )
        
        scan_results.append({
            'R56': R56,
            'Vdeb': Vdeb,
            'Phasdeb': Phasdeb,
            'RMS_Spread_%': spread,
            'Efficiency_%': eff
        })
        
        if (i + 1) % 15 == 0:
            print(f"Progress: {i + 1}/{n_samples} configurations evaluated.")
            
    df_res = pd.DataFrame(scan_results)
    
    df_res = df_res.sort_values(by=['Efficiency_%', 'RMS_Spread_%'], ascending=[False, True])
    return df_res

#df_raw=filter_beam_core(df, mahalanobis_r=3)   # 6-D cut, keeps ~88%

# %%
'''
scan_df = run_multi_objective_scan(df_raw, ring, ring_tw, p0c_reference, n_samples=150)

print("\nTop 10 configurations (Highest Efficiency + Lowest Spread):")
display(scan_df.head(10))'''

# %%
df_copy = df.copy()

Eref = np.mean(df['p[MeV/c]'])
R56= 0.352963
Vdeb= -0.048365
Phasdeb= 0.29229

c = 299792458
freq = 3e9
k = 2*np.pi*freq/c

main_bunch = select_main_bunch(df)

if ENERGY_COMPRESSOR_ON:
    V_int=abs(Vdeb)*Eref
    print(f'{V_int} MeV')

    delta_i = df_copy['p[MeV/c]']/Eref - 1

    z_i = df_copy['t[mm/c]']*1e-3

    z_f = z_i + R56*delta_i

    delta_f = delta_i + Vdeb*np.sin(k*z_f + Phasdeb)

    p_f = Eref*(1+delta_f)

    df_copy['t[mm/c]'] = z_f
    df_copy['delta_final'] = delta_f
    df_copy['p[MeV/c]'] = p_f

    main_bunch_copy = select_main_bunch(df_copy)

    plt.scatter(main_bunch['t[mm/c]'], main_bunch['p[MeV/c]'], s=0.2)
    plt.title('Before Energy Compressor')
    plt.show()

    plt.scatter(main_bunch_copy['t[mm/c]'], main_bunch_copy['p[MeV/c]'], s=0.2)
    plt.title('After Energy Compressor')
    plt.show()
else:
    print('Energy Compressor OFF -- using raw beam distribution unchanged.')
    plt.scatter(main_bunch['t[mm/c]'], main_bunch['p[MeV/c]'], s=0.2)
    plt.title('Beam Distribution (Energy Compressor OFF)')
    plt.show()

# %%
df=df_copy
def CalcEmittanceManual(df, position, angle):
    pos = df[position].values
    ang = df[angle].values

    x_c = pos - np.mean(pos)
    xp_c = ang - np.mean(ang)

    x2_mean = np.mean(x_c**2)
    xp2_mean = np.mean(xp_c**2)
    xxp_mean = np.mean(x_c * xp_c)
    
    emittance = np.sqrt(x2_mean * xp2_mean - xxp_mean**2)
    return emittance

def CalcEmittanceAuto(df, position, angle):
    pos = df[position].values
    ang = df[angle].values
    
    cov = np.cov(pos, ang, ddof=0) 
    emittance = np.sqrt(np.linalg.det(cov))
    return emittance

def NormalisedEmittance(emittance, E_total_mev):
    m0 = 0.510998 
    
    gamma_rel = E_total_mev / m0
    beta_rel = np.sqrt(1 - 1/gamma_rel**2)
    
    n_emittance = beta_rel * gamma_rel * emittance
    return n_emittance

def get_twiss(df, position, angle, p_col='p[MeV/c]', p0_mev=2860.0):
    pos = df[position].values
    ang = df[angle].values
    
    eps = CalcEmittanceAuto(df, position, angle)
    
    cov_matrix = np.cov(pos, ang, ddof=0)
    
    beta = cov_matrix[0, 0] / eps
    alpha = -cov_matrix[0, 1] / eps
    gamma = (1 + alpha**2) / beta
    
    delta = (df[p_col].values - p0_mev) / p0_mev
    dispersion_x = np.cov(pos, delta, ddof=0)[0, 1] / np.var(delta)
    disp_prime_x = np.cov(ang, delta, ddof=0)[0, 1] / np.var(delta)
    
    return alpha, beta, gamma, dispersion_x, disp_prime_x

def remove_dispersion(df, position, angle, p_col='p[MeV/c]', p0_mev=2860.0):
    df_out = df.copy()
    delta = (df_out[p_col].values - p0_mev) / p0_mev
    disp = np.cov(df_out[position].values, delta, ddof=0)[0, 1] / np.var(delta)
    ddisp = np.cov(df_out[angle].values, delta, ddof=0)[0, 1] / np.var(delta)
    df_out[position] = df_out[position].values - disp * delta
    df_out[angle] = df_out[angle].values - ddisp * delta
    return df_out

def filter_by_action(df, position, angle, n_sigma=3, max_iter=8, min_particles=20):
    df_iter = df.copy()
    for _ in range(max_iter):
        df_iter = remove_dispersion(df_iter, position, angle)
        alpha, beta, gamma, _, _ = get_twiss(df_iter, position, angle)
        eps = CalcEmittanceAuto(df_iter, position, angle)

        x = df_iter[position].values - df_iter[position].values.mean()
        xp = df_iter[angle].values - df_iter[angle].values.mean()
        J = gamma * x**2 + 2 * alpha * x * xp + beta * xp**2

        mask = J <= n_sigma**2 * eps
        if mask.sum() == len(df_iter):
            break
        if mask.sum() < min_particles:
            break
        df_iter = df_iter[mask].copy()
    return df_iter

def filter_by_action_xy(df, n_sigma=5, max_iter=5):
    df_x = filter_by_action(df, 'x[mm]', 'xp[mrad]', n_sigma, max_iter)
    df_xy = filter_by_action(df_x, 'y[mm]', 'yp[mrad]', n_sigma, max_iter)
    return df_xy

def betatron_mismatch(beta1, alpha1, beta2, alpha2):
    """
    Mismatch factor between two Twiss ellipses (beta1,alpha1) and
    (beta2,alpha2) -- e.g. an actual beam vs. a design/reference lattice,
    or one seed's optics vs. the baseline ring's. Symmetric: swapping
    which is "beam" and which is "reference" gives the same result.

    Standard formula (Sagan & Rubin; same one MAD-X/Bmad report as "Bmag"):
        gamma1 = (1+alpha1^2)/beta1,  gamma2 = (1+alpha2^2)/beta2
        H    = 0.5*(beta2*gamma1 - 2*alpha1*alpha2 + beta1*gamma2)
        Bmag = H + sqrt(H^2 - 1)

    Returns (H, Bmag):
      H    -- 1.0 when perfectly matched, >1 otherwise.
      Bmag -- the emittance growth factor after filamentation:
              eps_effective = Bmag * eps_true. Bmag=1 means no mismatch.

    beta1/beta2 must be in the SAME unit convention (both meters, or both
    the mm/mrad convention used elsewhere in this project -- confirmed
    numerically interchangeable earlier in this project).
    """
    gamma1 = (1 + alpha1**2) / beta1
    gamma2 = (1 + alpha2**2) / beta2
    H = 0.5 * (beta2*gamma1 - 2*alpha1*alpha2 + beta1*gamma2)
    Bmag = H + np.sqrt(max(H**2 - 1, 0.0))
    return H, Bmag

def optics_match_transform(pos, ang, beta1, alpha1, beta2, alpha2):
    pos_matched = np.sqrt(beta2 / beta1) * pos
    ang_matched = ((alpha1 - alpha2) / np.sqrt(beta1 * beta2)) * pos + np.sqrt(beta1 / beta2) * ang
    return pos_matched, ang_matched

def getrms_gaussfit(xis, maxnmax=50, max_doublings=20):
    """Robust average/variance via histogram + Gaussian fit, adapted from
    the supervisor's getrms(). Doubles nbins until the tallest bin drops
    below maxnmax counts, then fits a Gaussian to the histogram -- the fit
    itself is what makes this robust to a handful of far-out particles,
    since they barely move where a fitted peak sits, unlike a raw
    np.var() which they can dominate."""
    xis = np.asarray(xis, dtype=float)
    minxi = xis.min() - 0.001 * (xis.max() - xis.min())
    maxxi = xis.max() + 0.001 * (xis.max() - xis.min())
    nbins = max(2 ** int(np.floor(np.log2(len(xis) / maxnmax))), 1)
    nmax = 2 * maxnmax
    n_doublings = 0
    counts, edges = None, None
    while nmax > maxnmax and n_doublings < max_doublings:
        nbins *= 2
        counts, edges = np.histogram(xis, nbins, range=[minxi, maxxi])
        nmax = counts.max()
        n_doublings += 1
    centers = (edges[:-1] + edges[1:]) / 2

    def gauss(x, amp, xav, xvar):
        return amp * np.exp(-(x - xav) ** 2 / (2 * xvar))

    p0 = [counts.max(), xis.mean(), xis.var()]
    popt, _ = curve_fit(gauss, centers, counts, p0=p0)
    return popt[1], popt[2]

def moment_based_twiss(df, position, angle, p_col='p[MeV/c]', p0_mev=2860.0,
                        iters=5, maxnmax=50):
    """Iterative moment/projection-based Twiss reconstruction, ported from
    the supervisor's analysedataProj(). At each iteration: normalize the
    (dispersion-subtracted) coordinates using the current beta/alpha
    guess, take three projections at 0/120/240 degrees in that normalized
    frame, get each projection's variance via getrms_gaussfit(), and
    reconstruct an updated beta/alpha/emittance from the three variances.
    Returns (history, Dt, Dpt) where history is a list of
    (iteration, eps, beta, alfa) tuples, one per iteration, so the
    convergence itself can be plotted -- distinct from filter_by_action(),
    which this is meant to be compared against, not replace."""
    pos = df[position].values
    ang = df[angle].values
    delta = (df[p_col].values - p0_mev) / p0_mev

    pos_av, ang_av, delta_av = pos.mean(), ang.mean(), delta.mean()
    var_delta = np.var(delta)

    Dt = np.cov(pos, delta, ddof=0)[0, 1] / var_delta
    Dpt = np.cov(ang, delta, ddof=0)[0, 1] / var_delta

    pos_beta = pos - Dt * (delta - delta_av)
    ang_beta = ang - Dpt * (delta - delta_av)

    xxbet = np.mean((pos - pos_av - Dt * (delta - delta_av)) ** 2)
    xpxpbet = np.mean((ang - ang_av - Dpt * (delta - delta_av)) ** 2)
    xxpbet = np.mean((pos - pos_av - Dt * (delta - delta_av)) * (ang - ang_av - Dpt * (delta - delta_av)))
    eps_ref = np.sqrt(xxbet * xpxpbet - xxpbet ** 2)
    beta_ref, alfa_ref = xxbet / eps_ref, -xxpbet / eps_ref

    history = []
    for it in range(iters):
        xi = pos_beta / np.sqrt(beta_ref)
        xip = np.sqrt(beta_ref) * ang_beta + alfa_ref * pos_beta / np.sqrt(beta_ref)
        xi1 = (-xi + np.sqrt(3) * xip) / 2
        xi2 = (xi + np.sqrt(3) * xip) / 2

        _, xivar = getrms_gaussfit(xi, maxnmax)
        _, xi1var = getrms_gaussfit(xi1, maxnmax)
        _, xi2var = getrms_gaussfit(xi2, maxnmax)

        betnoreps = xivar
        alfnoreps = (xi1var - xi2var) / np.sqrt(3)
        gamnoreps = (2 * (xi1var + xi2var) - xivar) / 3
        eps = np.sqrt(betnoreps * gamnoreps - alfnoreps ** 2)
        betn, alfn = betnoreps / eps, alfnoreps / eps
        alfa_ref = betn * alfa_ref + alfn
        beta_ref = betn * beta_ref

        history.append((it + 1, eps, beta_ref, alfa_ref))
    return history, Dt, Dpt

def plot_twiss_ellipse(beta, alpha, beta2, alpha2, emittance,ax):
    gamma = (1 + alpha**2) / beta
    theta = np.linspace(0, 2*np.pi, 100)
    
    x1 = np.sqrt(emittance * beta) * np.cos(theta)
    xp1 = -np.sqrt(emittance / beta) * (alpha * np.cos(theta) - np.sin(theta))
    x2 = np.sqrt(emittance * beta2) * np.cos(theta)
    xp2 = -np.sqrt(emittance / beta2) * (alpha2 * np.cos(theta) - np.sin(theta))
    
    ax.plot(x1, xp1, label=f'Particle distribution',color='blue')
    ax.plot(x2, xp2, label=f'Initial ring parameters',color='red')
    ax.axhline(0, color='black', lw=0.5, ls='--')
    ax.axvline(0, color='black', lw=0.5, ls='--')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    ax.axis('equal')

def normalisation_matrix(beta_r,alpha,beta_l,phi,x,xp):
    mat_A=np.array([
        [np.sqrt(beta_r),0],
        [-alpha/np.sqrt(beta_l),1/np.sqrt(beta_r)]
        ])

    mat_B=np.array([
        [np.cos(phi),np.sin(phi)],
        [-np.sin(phi),np.cos(phi)]
    ])

    mat_C=np.array([
        [1/np.sqrt(beta_l),0],
        [alpha/np.sqrt(beta_l),np.sqrt(beta_l)]
    ])

    x_col=np.array([
        [x],[xp]
    ])

    result=mat_A @ mat_B @ mat_C @ x_col

    zeta=result[0,0]
    zeta_prime=result[1,0]
    
    return zeta, zeta_prime

def plot_twiss_ellipse_normalised(beta_l,alpha_l,beta_r, alpha_r,disp,ddisp,delta,emittance,ax):

    phi=np.linspace(0,2*np.pi,500)

    x = np.sqrt(emittance * beta_l) * np.cos(phi)
    xp = - (np.sqrt(emittance / beta_l)) * (alpha_l * np.cos(phi) + np.sin(phi))

    zeta=(1/(np.sqrt(beta_l)))*x
    zeta_prime=np.sqrt(beta_l)*xp+(alpha_l/np.sqrt(beta_l))*x

    zeta_r=(1/(np.sqrt(beta_r)))*x
    zeta_prime_r=np.sqrt(beta_r)*xp+(alpha_r/np.sqrt(beta_r))*x

    x_b=x+disp*delta
    xp_b=xp+ddisp*delta

    zeta_r_d=(1/(np.sqrt(beta_r)))*x_b
    zeta_prime_r_d=np.sqrt(beta_r)*xp_b+(alpha_r/np.sqrt(beta_r))*x_b

    ax.plot(zeta, zeta_prime, color='blue', label='Particle distribution')
    ax.plot(zeta_r, zeta_prime_r, color='red', label='Initial ring parameters')
    if delta != 0:
        ax.plot(zeta_r_d, zeta_prime_r_d, color='green', linestyle='--',
                label=f'Ring parameters (shifted by beam-ring $\\delta$={delta:.4f})')
    ax.axis('equal') 
    ax.grid(True, linestyle=':')
    ax.legend()

def plot_twiss_with_particles(df, pos_col, ang_col, alpha, beta, emittance):
    gamma = (1 + alpha**2) / beta
    
    plt.figure(figsize=(7, 7))
    ax = plt.gca()
    
    ax.scatter(df[pos_col], df[ang_col], s=1, alpha=0.3, label='Particles')
    
    theta = np.linspace(0, 2*np.pi, 200)
    x = np.sqrt(emittance * beta) * np.cos(theta)
    xp = -np.sqrt(emittance / beta) * (alpha * np.cos(theta) - np.sin(theta))
    
    ax.plot(x, xp, color='red', lw=2, label=f'RMS Ellipse (ε={emittance:.2f})')
    
    
    ax.axhline(0, color='black', lw=1, ls='--')
    ax.axvline(0, color='black', lw=1, ls='--')
    ax.set_xlabel(pos_col)
    ax.set_ylabel(ang_col)
    ax.set_title(f'Phase Space: {pos_col} vs {ang_col}')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.show()

def density_scatter(ax, x, y, s=2, cmap='viridis', **kwargs):
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        print(f"[density_scatter] Only {len(x)} point(s) after filtering -- "
              f"skipping KDE coloring, plotting plain scatter instead.")
        return ax.scatter(x, y, s=s, **kwargs)

    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    z = z / z.max()

    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    sc = ax.scatter(x, y, c=z, s=s, cmap=cmap, **kwargs)
    return sc


def draw_action_ellipse(ax_plot, df_plot, pos_col, ang_col,
                        alpha, beta, emittance, n_sigma,
                        color='hotpink', lw=2, ls='-', label=None):
    """
    Draw the n_sigma action ellipse -- the boundary of the particles kept by
    filter_by_action at the same n_sigma -- and label it with the emittance
    and how many of the PLOTTED particles lie inside it.

    Uses the same action as filter_by_action,
        J = gamma*x^2 + 2*alpha*x*xp + beta*xp^2 <= n_sigma^2 * eps,
    so an n_sigma here means exactly what it means there. The count is taken
    from the points actually on the axes, so the label always describes the
    curve that is drawn rather than the sample the Twiss was fitted to.
    """
    theta = np.linspace(0, 2 * np.pi, 400)
    amp = n_sigma * np.sqrt(emittance)
    e_pos = amp * np.sqrt(beta) * np.cos(theta)
    e_ang = -amp / np.sqrt(beta) * (alpha * np.cos(theta) - np.sin(theta))

    gamma = (1.0 + alpha ** 2) / beta
    x = df_plot[pos_col].values - df_plot[pos_col].values.mean()
    xp = df_plot[ang_col].values - df_plot[ang_col].values.mean()
    J = gamma * x ** 2 + 2 * alpha * x * xp + beta * xp ** 2
    n_in, n_tot = int(np.sum(J <= n_sigma ** 2 * emittance)), len(df_plot)

    head = label if label is not None else f'{n_sigma:g}$\\sigma$ beam core'
    ax_plot.plot(e_pos, e_ang, color=color, lw=lw, ls=ls,
        label=f'{head} ($\\epsilon$={emittance:.3g} $\\mu$m)\n'
              f'{n_in}/{n_tot} particles inside ({100*n_in/n_tot:.1f}%)')
    return n_in, n_tot


# %%

moment_iters = 5

history_x, Dt_x, Dpt_x = moment_based_twiss(df, 'x[mm]', 'xp[mrad]', iters=moment_iters)
history_y, Dt_y, Dpt_y = moment_based_twiss(df, 'y[mm]', 'yp[mrad]', iters=moment_iters)

def filter_by_moment_core(df_in, position, angle, iters=5, max_passes=20,
                          p_col='p[MeV/c]', p0_mev=2860.0):
    df_iter = df_in.copy()
    for _ in range(max_passes):
        history, Dt, Dpt = moment_based_twiss(df_iter, position, angle,
                                              p_col=p_col, p0_mev=p0_mev, iters=iters)
        _, eps, beta, alfa = history[-1]
        delta = (df_iter[p_col].values - p0_mev) / p0_mev
        pos = df_iter[position].values - Dt * (delta - delta.mean())
        ang = df_iter[angle].values - Dpt * (delta - delta.mean())
        pos = pos - pos.mean()
        ang = ang - ang.mean()
        gamma = (1 + alfa**2) / beta
        J = gamma * pos**2 + 2 * alfa * pos * ang + beta * ang**2
        J_max = 2 * eps * np.log(2 * len(df_iter))
        mask = J <= J_max
        if mask.all():
            break
        df_iter = df_iter[mask].copy()
    return df_iter, {'eps': eps, 'beta': beta, 'alfa': alfa, 'J_max': J_max, 'Dt': Dt, 'Dpt': Dpt}

df_core_x, core_x = filter_by_moment_core(df, 'x[mm]', 'xp[mrad]', iters=moment_iters)
df_clean, core_y = filter_by_moment_core(df_core_x, 'y[mm]', 'yp[mrad]', iters=moment_iters)
print(f"Moment-based core: kept {len(df_clean)}/{len(df)} particles "
      f"({100*len(df_clean)/len(df):.1f}%)")

core_amp_x = np.sqrt(core_x['J_max'] / core_x['eps'])
core_amp_y = np.sqrt(core_y['J_max'] / core_y['eps'])
CORE_AMPLITUDE = max(core_amp_x, core_amp_y)
print(f"Core boundary from the moment-based fit: x = {core_amp_x:.2f} rms, y = {core_amp_y:.2f} rms")

df_core_betatron = remove_dispersion(df_clean, 'x[mm]', 'xp[mrad]')
df_core_betatron = remove_dispersion(df_core_betatron, 'y[mm]', 'yp[mrad]')

emittance_x = CalcEmittanceAuto(df_core_betatron, 'x[mm]', 'xp[mrad]')
m_emittance_x = CalcEmittanceManual(df_core_betatron, 'x[mm]', 'xp[mrad]')
print(f'Horizontal geometric emittance (core): {emittance_x} um, {m_emittance_x} um')

emittance_y = CalcEmittanceAuto(df_core_betatron, 'y[mm]', 'yp[mrad]')
m_emittance_y = CalcEmittanceManual(df_core_betatron, 'y[mm]', 'yp[mrad]')
print(f'Vertical geometric emittance (core): {emittance_y} um, {m_emittance_y} um')

ax, bx, gx, _, _ = get_twiss(df_core_betatron, 'x[mm]', 'xp[mrad]')
ay, by, gy, _, _ = get_twiss(df_core_betatron, 'y[mm]', 'yp[mrad]')
_, _, _, dx, ddx = get_twiss(df_clean, 'x[mm]', 'xp[mrad]')
_, _, _, dy, ddy = get_twiss(df_clean, 'y[mm]', 'yp[mrad]')
print(f"Core beam Twiss: alpha_x={ax:.3f}, beta_x={bx:.3f} m, dx={dx:.3f} m, ddx={ddx:.3f} m")
print(f"Core beam Twiss: alpha_y={ay:.3f}, beta_y={by:.3f} m, dy={dy:.3f} m, ddy={ddy:.3f} m")

p_array=df['p[MeV/c]']

p_avg = np.mean(p_array)
rms_spread = np.std(p_array)  
relative_spread = (rms_spread / p_avg) *100

print(f"RMS Momentum Spread: {relative_spread} %")

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

def draw_emittance_ellipse(ax_plot, df_all, df_core, position, angle, alpha, beta, eps):
    pc, ac = df_core[position].values.mean(), df_core[angle].values.mean()
    theta = np.linspace(0, 2 * np.pi, 400)
    e_pos = pc + np.sqrt(eps * beta) * np.cos(theta)
    e_ang = ac - np.sqrt(eps / beta) * (alpha * np.cos(theta) - np.sin(theta))
    ax_plot.plot(e_pos, e_ang, color='hotpink', lw=2,
        label=f'Core emittance ellipse ($\\epsilon$={eps:.3g} $\\mu$m)\n'
              f'{len(df_core)}/{len(df_all)} particles in core ({100*len(df_core)/len(df_all):.1f}%)')

sc0 = density_scatter(axes[0, 0], df['x[mm]'], df['xp[mrad]'])
axes[0, 0].set_xlabel('x [mm]')
axes[0, 0].set_ylabel('xp [mrad]')
axes[0, 0].set_title('Horizontal Phase Space')
fig.colorbar(sc0, ax=axes[0, 0], label='Relative Density')

sc1 = density_scatter(axes[0, 1], df['y[mm]'], df['yp[mrad]'])
axes[0, 1].set_xlabel('y [mm]')
axes[0, 1].set_ylabel('yp [mrad]')
axes[0, 1].set_title('Vertical Phase Space')
fig.colorbar(sc1, ax=axes[0, 1], label='Relative Density')

draw_emittance_ellipse(axes[0, 0], df, df_clean, 'x[mm]', 'xp[mrad]', ax, bx, emittance_x)
axes[0, 0].legend(fontsize='small', loc='upper left', framealpha=0.9)

draw_emittance_ellipse(axes[0, 1], df, df_clean, 'y[mm]', 'yp[mrad]', ay, by, emittance_y)
axes[0, 1].legend(fontsize='small', loc='upper left', framealpha=0.9)

main_bunch = select_main_bunch(df)
sc2 = density_scatter(axes[1, 0], main_bunch['t[mm/c]'], main_bunch['p[MeV/c]'])
axes[1, 0].set_xlabel('t [mm/c]')
axes[1, 0].set_ylabel('p [MeV/c]')
axes[1, 0].set_title('Longitudinal Phase Space')
fig.colorbar(sc2, ax=axes[1, 0], label='Relative Density')

sc3 = density_scatter(axes[1, 1], df['x[mm]'], df['y[mm]'])
axes[1, 1].set_xlabel('x [mm]')
axes[1, 1].set_ylabel('y [mm]')
axes[1, 1].set_title('Transverse Real-Space Profile')
fig.colorbar(sc3, ax=axes[1, 1], label='Relative Density')

folder=mf.results_dir(design, config, phase, changes=changes, metric='InjectionEfficiency', sub='BeamSource')
plt.savefig(f'{folder}/phase_space_plots.png')

# %%
print("\nMoment-based (Gaussian-fit) Twiss, horizontal:")
for it, eps, beta, alfa in history_x:
    print(f"  iter {it}: eps={eps:.4f} um, beta={beta:.4f} m, alfa={alfa:.4f}")
print("Moment-based (Gaussian-fit) Twiss, vertical:")
for it, eps, beta, alfa in history_y:
    print(f"  iter {it}: eps={eps:.4f} um, beta={beta:.4f} m, alfa={alfa:.4f}")

theta_cmp = np.linspace(0, 2 * np.pi, 200)
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

for ax_i, history, beta_action, alfa_action, eps_action, plane_label in [
    (axes[0], history_x, bx, ax, emittance_x, 'Horizontal'),
    (axes[1], history_y, by, ay, emittance_y, 'Vertical'),
]:
    colors_it = plt.cm.viridis(np.linspace(0, 1, len(history)))
    for (it, eps, beta, alfa), c in zip(history, colors_it):
        x_ell = np.sqrt(eps * beta) * np.cos(theta_cmp)
        xp_ell = -np.sqrt(eps / beta) * (alfa * np.cos(theta_cmp) - np.sin(theta_cmp))
        ax_i.plot(x_ell, xp_ell, color=c, label=f'Moment-based iter {it} (\u03b5={eps:.3f})')

    x_action = np.sqrt(eps_action * beta_action) * np.cos(theta_cmp)
    xp_action = -np.sqrt(eps_action / beta_action) * (alfa_action * np.cos(theta_cmp) - np.sin(theta_cmp))
    ax_i.plot(x_action, xp_action, color='red', lw=2.5, linestyle='--',
              label=f'Moment-filtered core beam (\u03b5={eps_action:.3f})')

    ax_i.set_title(f'{plane_label}: Moment-based Iterations vs Core Beam Emittance')
    ax_i.axhline(0, color='black', lw=0.5, ls='--')
    ax_i.axvline(0, color='black', lw=0.5, ls='--')
    ax_i.axis('equal')
    ax_i.grid(True, linestyle=':', alpha=0.5)
    ax_i.legend(fontsize='small')

plt.tight_layout()
plt.savefig(f'{folder}/moment_vs_action_emittance.png')
plt.show()

# %%
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

main_bunch_hist = select_main_bunch(df)

t = main_bunch_hist['t[mm/c]']
p = main_bunch_hist['p[MeV/c]']

plt.style.use('ggplot')

fig = plt.figure(figsize=(8,9))
gs = GridSpec(
    4, 6,
    width_ratios=[1, 1, 1, 0.9, 0.3, 0.08],
    height_ratios=[0.8, 1, 1, 1], 
    hspace=0.08,
    wspace=0.15
)

ax_main  = fig.add_subplot(gs[1:4, 0:3])
ax_top   = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
cax      = fig.add_subplot(gs[1:4, 4])

sc = density_scatter(ax_main, t, p, s=4, alpha=0.8)

ax_main.set_xlabel(r'$t$ [mm/c]')
ax_main.set_ylabel(r'$p$ [MeV/c]')

fig.colorbar(sc, cax=cax, label='Relative Density')

ax_top.hist(
    t,
    bins=100,
    color='steelblue',
    edgecolor='black'
)
ax_top.set_ylabel('Counts')
ax_top.tick_params(axis='x', labelbottom=False)

ax_right.hist(
    p,
    bins=100,
    orientation='horizontal',
    color='darkorange',
    edgecolor='black'
)
ax_right.set_xlabel('Counts')
ax_right.tick_params(axis='y', labelleft=False)

plt.savefig(f'{folder}/longitudinal_phase_space.png')
plt.show()

plt.style.use('default')
# %%

plot_twiss_with_particles(df_clean, 'x[mm]', 'xp[mrad]', ax, bx, emittance_x)
plot_twiss_with_particles(df_clean, 'y[mm]', 'yp[mrad]', ay, by, emittance_y)

# %%

p_core_mean_mev = float(df_clean['p[MeV/c]'].mean())
bg_core = p_core_mean_mev / 0.51099895000
nemittance_x = emittance_x * 1e-6 * bg_core
nemittance_y = emittance_y * 1e-6 * bg_core
print(f"Normalised emittance (core): x = {nemittance_x:.4e} m rad, y = {nemittance_y:.4e} m rad")

beam_results = {
    "x": {
        "alpha": float(ax),
        "beta": float(bx),
        "gamma": float(gx),
        "dispersion": float(dx),
        "dispersion_prime": float(ddx),
        "emittance_geo": float(emittance_x),
        "emittance_norm": float(nemittance_x)
    },
    "y": {
        "alpha": float(ay),
        "beta": float(by),
        "gamma": float(gy),
        "dispersion": float(dy),
        "dispersion_prime": float(ddy),
        "emittance_geo": float(emittance_y),
        "emittance_norm": float(nemittance_y)
    },
    "longitudinal": {
        "delta_rms": float(relative_spread / 100)
    },
    "metadata": {
        "particle_type": "positron",
        "design_energy_mev": 2860.0,
        "p_mean_mev": p_core_mean_mev,
        "n_particles": len(df_clean['ID'])
    }
}

with open(f'{folder}/TwissResults.json', 'w') as f:
    json.dump(beam_results, f, indent=4)

# %%

def insert_marker_in_drift(pdr, ring, drift_name='DrTripl', occurrence=0,
                            marker_name='TripletInject'):
    """Insert a marker at the midpoint of the `occurrence`-th (0-indexed,
    in s-order) PLAIN instance of a drift named `drift_name`. Excludes
    already-split DriftSlice pieces (e.g. 'DrTripl..102').

    Matches EITHER a bare name ('DrTripl') OR that name with an xtrack
    auto-disambiguation suffix ('DrTripl::0', 'DrTripl::1', ...) -- which
    convention shows up depends on the xtrack version/environment that
    built the ring (confirmed directly: one ring used bare 'DrTripl'
    repeated as-is, another used 'DrTripl::N' for the same construction).
    An exact-only match would silently find zero instances on rings using
    the '::N' convention. The '..NNN' pattern used for sliced pieces is
    still excluded either way, since neither branch of this pattern
    matches it.

    Looks up the drift's actual position/length at RUNTIME rather than
    assuming a fixed index or s-value, so this works across different
    design/config/phase rings that may have different l_cell, l_tripl,
    N_cells_S, etc.

    Returns marker_name for convenience (so it can be passed straight into
    ring_tw.rows[...] and ring.track(ele_start=...)).
    """
    import re
    names = ring.element_names
    pattern = re.compile(rf'^{re.escape(drift_name)}(::\d+)?$')
    indices = [i for i, n in enumerate(names) if pattern.match(n)]
    if not indices:
        raise ValueError(f"No element matching '{drift_name}' or '{drift_name}::N' found "
                          f"in this ring. If this ring was built with a different lattice "
                          f"function than three_fold_periodicity_long, the triplet drift may "
                          f"have a different name -- check ring.element_names for "
                          f"anything containing 'Tripl'.")
    if occurrence >= len(indices):
        raise ValueError(f"Requested occurrence {occurrence}, but only "
                          f"{len(indices)} instance(s) matching '{drift_name}' exist "
                          f"in this ring.")

    idx = indices[occurrence]
    s_positions = ring.get_s_position()
    s_start = s_positions[idx]
    length = ring.element_dict[names[idx]].length
    s_center = s_start + length / 2

    ring.insert(pdr.new(marker_name, xt.Marker), at=s_center)
    return marker_name


pdr = paths.load_lattice(design, config, mode, phase, changes)
ring = pdr.lines['ring']
marker_name = insert_marker_in_drift(pdr, ring, occurrence=3)
# Start the ring at the injection marker, so every track() call below begins
# where the beam is injected (and matched). Without this, tracking starts at
# element 0 even though the particles are matched to the optics at the marker.
ring.cycle(name_first_element=marker_name, inplace=True)
ring.element_dict['RFCav'].voltage = 8e6
ring.element_dict['RFCav_1'].voltage = 8e6

ring_tw=ring.twiss6d()
print(ring_tw.cols)
initial_twiss = ring_tw.rows[marker_name]

betx0 = initial_twiss['betx'][0]
alfx0 = initial_twiss['alfx'][0]
dx0   = initial_twiss['dx'][0]
ddx0   = initial_twiss['dpx'][0]
x     = initial_twiss['x'][0]
xp_v    = initial_twiss['px'][0]

bety0 = initial_twiss['bety'][0]
alfy0 = initial_twiss['alfy'][0]
dy0   = initial_twiss['dy'][0]
ddy0   = initial_twiss['dpy'][0]
y     = initial_twiss['y'][0]
yp    = initial_twiss['py'][0]

delta_ring_ref = initial_twiss['delta'][0]
p0_ring_mev = ring.particle_ref.p0c[0] / 1e6
p0_beam_mev = df['p[MeV/c]'].mean()
delta_beam_vs_ring = (p0_beam_mev - p0_ring_mev) / p0_ring_mev
print(f"Beam mean momentum: {p0_beam_mev:.2f} MeV vs ring reference: "
      f"{p0_ring_mev:.2f} MeV  ->  delta_beam_vs_ring = {delta_beam_vs_ring:.5f}")

H_x, Bmag_x = betatron_mismatch(bx, ax, betx0, alfx0)
H_y, Bmag_y = betatron_mismatch(by, ay, bety0, alfy0)
print(f"Beam-vs-ring betatron mismatch: "
      f"H_x={H_x:.4f} (Bmag_x={Bmag_x:.4f}), H_y={H_y:.4f} (Bmag_y={Bmag_y:.4f})")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

plot_twiss_ellipse(bx,ax,betx0, alfx0, emittance_x,axes[0, 0])
plot_twiss_ellipse(by,ay,bety0, alfy0, emittance_y,axes[1, 0])

plot_twiss_ellipse_normalised(bx,ax,betx0, alfx0,dx0,ddx0,delta_beam_vs_ring, emittance_x,axes[0,1])
plot_twiss_ellipse_normalised(by,ay,bety0, alfy0,dy0,ddy0,delta_beam_vs_ring, emittance_y,axes[1, 1])

axes[0, 0].set_title("Horizontal: Physical Phase Space")
axes[0, 0].set_xlabel("x [mm]")
axes[0, 0].set_ylabel("xp [mrad]")

axes[0, 1].set_title("Horizontal: Normalized Phase Space")
axes[0, 1].set_xlabel(r"$\zeta_x$")
axes[0, 1].set_ylabel(r"$\zeta'_x$")

axes[1, 0].set_title("Vertical: Physical Phase Space")
axes[1, 0].set_xlabel("y [mm]")
axes[1, 0].set_ylabel("yp [mrad]")

axes[1, 1].set_title("Vertical: Normalized Phase Space")
axes[1, 1].set_xlabel(r"$\zeta_y$")
axes[1, 1].set_ylabel(r"$\zeta'_y$")
plt.xlabel
plt.tight_layout()
plt.savefig(f'{folder}/twiss_ellipses.png')
plt.show()

# %%
def normalize_coords(pos, ang, beta, alpha):
    zeta = pos / np.sqrt(beta)
    zeta_prime = np.sqrt(beta) * ang + (alpha / np.sqrt(beta)) * pos
    return zeta, zeta_prime

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
theta = np.linspace(0, 2 * np.pi, 200)

for ax_i, pos_col, ang_col, beta_l, alpha_l, beta_r, alpha_r, eps, plane_label in [
    (axes[0], 'x[mm]', 'xp[mrad]', bx, ax, betx0, alfx0, emittance_x, 'Horizontal'),
    (axes[1], 'y[mm]', 'yp[mrad]', by, ay, bety0, alfy0, emittance_y, 'Vertical'),
]:
    pos = df_clean[pos_col].values
    ang = df_clean[ang_col].values
    zeta_p, zeta_prime_p = normalize_coords(pos, ang, beta_l, alpha_l)

    sc = density_scatter(ax_i, zeta_p, zeta_prime_p, s=1)
    fig.colorbar(sc, ax=ax_i, label='Relative Density')

    x_beam_ell = np.sqrt(eps * beta_l) * np.cos(theta)
    xp_beam_ell = -np.sqrt(eps / beta_l) * (alpha_l * np.cos(theta) - np.sin(theta))

    zeta_beam, zeta_prime_beam = normalize_coords(x_beam_ell, xp_beam_ell, beta_l, alpha_l)
    ax_i.plot(zeta_beam, zeta_prime_beam, color='blue', lw=2, label='Beam Twiss (self-normalised)')

    zeta_ring, zeta_prime_ring = normalize_coords(x_beam_ell, xp_beam_ell, beta_r, alpha_r)
    ax_i.plot(zeta_ring, zeta_prime_ring, color='red', lw=2, linestyle='--',
              label='Ring Twiss (beam-normalised frame)')

    ax_i.set_title(f'{plane_label}: Normalised Phase Space Density')
    ax_i.set_xlabel(r'$\zeta$')
    ax_i.set_ylabel(r"$\zeta'$")
    ax_i.axhline(0, color='black', lw=0.5, ls='--')
    ax_i.axvline(0, color='black', lw=0.5, ls='--')
    ax_i.axis('equal')
    ax_i.grid(True, linestyle=':', alpha=0.5)
    ax_i.legend(fontsize='small')

plt.tight_layout()
plt.savefig(f'{folder}/normalised_phase_space_density.png')
plt.show()

# %% Physical aperture and pole-tip fields from the injected beam's rms emittances
import matplotlib.patches as mpatches


def physical_aperture_study(ring, eps_x, eps_y, sigma_delta, mode_tag,
                            n_sigma_beam=3, chamber_margin=5e-3, wall_thickness=0.0,
                            B_limits=None, ds=0.02, chamber_3d='cylinder',
                            interactive_3d=False):
    """Size the vacuum chamber and magnet bores from the injected beam, and
    count how many rms emittances fit.
 
      beam size      sigma_x(s) = sqrt(beta_x eps_x + (D_x sigma_delta)^2)
                     sigma_y(s) = sqrt(beta_y eps_y)
      envelope       n_sigma_beam * sigma
      chamber        envelope + chamber_margin
      fit count      N_sigma = chamber / sigma  (rms beam sizes that fit)
                     N_eps   = N_sigma^2        (rms emittances that fit)
 
    eps_x, eps_y are geometric rms emittances in m rad.

    Pole-tip fields use the MAD/xtrack strengths directly: quad B = Brho k1 R,
    sextupole B = Brho k2 R^2 / 2. xtrack's k2 is k' = B''/Brho (an offset
    dx in a sextupole gives a gradient k1 = k2 dx), checked against the
    thin-lens chromaticity formula dQ'_x = (1/4pi) sum(beta_x D k2 L). Everything is local to
    this function, so it doesn't touch the script's globals (ax, dx, ...).
    Returns (df_mag, df_pole).
    """
    if B_limits is None:
        B_limits = {'Bend': 1.5, 'Quadrupole': 1.1, 'Sextupole': 0.8}
    magnet_types = ('Bend', 'Quadrupole', 'Sextupole')
    mu0 = 4e-7 * np.pi
    brho = ring.particle_ref.p0c[0] / 299792458.0
    folder_ap = mf.results_dir(design, config, phase, changes=changes,
                               metric='PhysicalAperture', sub=mode_tag)
    print(f'Injected beam (rms): eps_x = {eps_x*1e6:.3f} um, eps_y = {eps_y*1e6:.3f} um, '
          f'sigma_delta = {sigma_delta*100:.3f} % | envelope = {n_sigma_beam} sigma, '
          f'chamber = envelope + {chamber_margin*1e3:.0f} mm')
 
    # --- optics every ds metres (deep copy, radiation off) ---
    ring_ap = ring.copy()
    ring_ap.configure_radiation(model=None)
    ring_ap.cut_at_s(np.arange(ds, ring_ap.get_length() - ds / 2, ds))
    tw_ap = ring_ap.twiss4d()
    s_ap = np.asarray(tw_ap.s)
    sig_x = np.sqrt(tw_ap.betx * eps_x + (tw_ap.dx * sigma_delta)**2)
    sig_y = np.sqrt(tw_ap.bety * eps_y + (tw_ap.dy * sigma_delta)**2)
    env_x, env_y = n_sigma_beam * sig_x, n_sigma_beam * sig_y
    C_x, C_y = env_x + chamber_margin, env_y + chamber_margin
 
    # --- per magnet ---
    import bend_tools as bt          # groups field-expansion dipoles into one 'Bend'
    rows = []
    for mag in bt.magnets(ring, magnet_types):
        name, etype, s0, s1 = mag['name'], mag['type'], mag['s_start'], mag['s_end']
        i0, i1 = np.searchsorted(s_ap, s0 - 1e-9), np.searchsorted(s_ap, s1 + 1e-9)
        sl = slice(i0, max(i1, i0 + 1))
        ch_x, ch_y = float(C_x[sl].max()), float(C_y[sl].max())
        sx_max, sy_max = float(sig_x[sl].max()), float(sig_y[sl].max())
        rows.append({'name': name, 'family': name.split('::')[0].split('_')[0],
                     'type': etype, 's_start': s0, 's_end': s1, 'length': mag['length'],
                     'betx_max': float(np.max(tw_ap.betx[sl])),
                     'bety_max': float(np.max(tw_ap.bety[sl])),
                     'dx_max': float(np.max(np.abs(tw_ap.dx[sl]))),
                     'sigma_x_max': sx_max, 'sigma_y_max': sy_max,
                     'chamber_x': ch_x, 'chamber_y': ch_y, 'chamber_r': max(ch_x, ch_y),
                     'N_sigma_x': ch_x / sx_max, 'N_sigma_y': ch_y / sy_max,
                     'h': mag['h'] if etype == 'Bend' else 0.0,
                     'k1': mag['k1'], 'k2': mag['k2']})
    df_mag = pd.DataFrame(rows)
    df_mag['N_eps_x'] = df_mag.N_sigma_x**2
    df_mag['N_eps_y'] = df_mag.N_sigma_y**2
 
    # distinct magnet types: name prefix + length + strength
    df_mag['strength'] = np.where(df_mag.type == 'Bend', df_mag.h,
                         np.where(df_mag.type == 'Quadrupole', df_mag.k1, df_mag.k2))
    df_mag['key'] = list(zip(df_mag.family, df_mag.length.round(4), df_mag.strength.round(4)))
 
    def _variant_tag(names):
        suffixes = [n.split('::')[0].split('_', 1)[1] if '_' in n else '' for n in names]
        tag = ''
        for ch in os.path.commonprefix(suffixes):
            if not ch.isalpha():
                break
            tag += ch
        return tag
 
    variant_of_key = {}
    for fam, g in df_mag.groupby('family', sort=False):
        keys = list(dict.fromkeys(g.key))
        if len(keys) == 1:
            variant_of_key[keys[0]] = fam
            continue
        labels = {k_: (f'{fam}_{_variant_tag(g[g.key == k_].name)}'
                       if _variant_tag(g[g.key == k_].name) else fam) for k_ in keys}
        if len(set(labels.values())) < len(keys):
            labels = {k_: f'{fam} ({k_[2]:+.4g})' for k_ in keys}
        variant_of_key.update(labels)
    df_mag['variant'] = df_mag.key.map(variant_of_key)
    df_mag.drop(columns='key').to_csv(f'{folder_ap}/chamber_per_magnet_{mode_tag}.csv', index=False)
 
    i_wx, i_wy = df_mag.N_sigma_x.idxmin(), df_mag.N_sigma_y.idxmin()
    Nsx, Nsy = df_mag.N_sigma_x[i_wx], df_mag.N_sigma_y[i_wy]
    print(f'Ring acceptance with chamber = {n_sigma_beam} sigma + {chamber_margin*1e3:.0f} mm:')
    print(f'  x: {Nsx:.2f} rms beam sizes = {Nsx**2:.1f} rms emittances '
          f'(limited by {df_mag.name[i_wx]}, sigma_x = {df_mag.sigma_x_max[i_wx]*1e3:.2f} mm)')
    print(f'  y: {Nsy:.2f} rms beam sizes = {Nsy**2:.1f} rms emittances '
          f'(limited by {df_mag.name[i_wy]}, sigma_y = {df_mag.sigma_y_max[i_wy]*1e3:.2f} mm)')
 
    # --- plot along the ring ---
    colors_ap = {'Bend': 'tab:blue', 'Quadrupole': 'tab:red', 'Sextupole': 'tab:green'}
    fig, axs = plt.subplots(3, 1, figsize=(12, 9.5), sharex=True,
                            gridspec_kw={'height_ratios': (1, 1, 0.8)})
    for a, sig, env, C, lab in ((axs[0], sig_x, env_x, C_x, 'x'), (axs[1], sig_y, env_y, C_y, 'y')):
        for _, r in df_mag.iterrows():
            a.axvspan(r.s_start, r.s_end, color=colors_ap[r.type], alpha=0.15, lw=0)
        a.plot(s_ap, sig * 1e3, color='grey', lw=0.8, label='1 $\\sigma$')
        a.plot(s_ap, env * 1e3, 'k-', lw=1.2, label=f'{n_sigma_beam} $\\sigma$ beam envelope')
        if chamber_margin > 0:
            a.plot(s_ap, C * 1e3, 'k:', lw=1.3, label=f'chamber (+{chamber_margin*1e3:.0f} mm)')
        handles, _ = a.get_legend_handles_labels()
        handles += [mpatches.Patch(color=c_, alpha=0.3, label=t_) for t_, c_ in colors_ap.items()]
        a.legend(handles=handles, fontsize='small', ncol=6, loc='lower left',
                 bbox_to_anchor=(0, 1.0), frameon=False)
        a.set_ylabel(f'half-aperture {lab} [mm]')
        a.set_ylim(0, None)
        a.grid(alpha=0.3)
    axs[2].plot(s_ap, C_x / sig_x, color='tab:red', lw=1, label='x')
    axs[2].plot(s_ap, C_y / sig_y, color='tab:blue', lw=1, label='y')
    axs[2].axhline(n_sigma_beam, color='k', ls='--', lw=0.8)
    axs[2].set_ylabel('chamber / $\\sigma$')
    axs[2].set_xlabel('s [m]')
    axs[2].grid(alpha=0.3)
    axs[2].legend(fontsize='small', loc='upper right')
    fig.suptitle(f'Injected beam (rms $\\varepsilon_x$={eps_x*1e6:.2f} $\\mu$m, '
                 f'$\\varepsilon_y$={eps_y*1e6:.2f} $\\mu$m, $\\sigma_\\delta$={sigma_delta*100:.2f} %): '
                 f'acceptance {Nsx:.1f} $\\sigma_x$ / {Nsy:.1f} $\\sigma_y$', y=1.0)
    fig.tight_layout()
    fig.savefig(f'{folder_ap}/beam_envelope_chamber_{mode_tag}.png', dpi=200, bbox_inches='tight')
    plt.show()
 
    # --- 3D beam envelope along the ring, inside the vacuum chamber ---
    # chamber_3d:
    #   'cylinder' one constant bore at the worst case over the whole ring --
    #              what actually gets built, and the number the aperture
    #              sketch quotes
    #   'follow'   the s-dependent envelope + margin, a corrugated tube: shows
    #              where the margin is tight, but is not a buildable chamber
    #   None       beam only
    from matplotlib import cm as mcm, colors as mcolors
    stride = max(1, len(s_ap) // 3000)          # keep the surface light
    s3, sx3, sy3 = s_ap[::stride], env_x[::stride] * 1e3, env_y[::stride] * 1e3   # [mm]
    th3 = np.linspace(0, 2 * np.pi, 72)
    S3, TH3 = np.meshgrid(s3, th3, indexing='ij')
    X3 = sx3[:, None] * np.cos(TH3)
    Y3 = sy3[:, None] * np.sin(TH3)
    R3 = np.hypot(X3, Y3)
    norm3 = mcolors.Normalize(R3.min(), R3.max())
    fig = plt.figure(figsize=(12, 7))
    a3 = fig.add_subplot(111, projection='3d')

    Cx0, Cy0 = float(C_x.max()) * 1e3, float(C_y.max()) * 1e3
    if chamber_3d == 'cylinder':
        # drawn FIRST and left transparent so the beam stays visible through it
        s_ends = np.array([s3[0], s3[-1]])
        S_C, TH_C = np.meshgrid(s_ends, th3, indexing='ij')
        a3.plot_surface(S_C, Cx0 * np.cos(TH_C), Cy0 * np.sin(TH_C),
                        color='#6b7a8f', alpha=0.13, rstride=1, cstride=1,
                        linewidth=0, antialiased=True, shade=False)
        for i in np.linspace(0, len(s3) - 1, 9).astype(int):
            a3.plot(np.full_like(th3, s3[i]), Cx0 * np.cos(th3), Cy0 * np.sin(th3),
                    color='#3f4b5b', lw=0.8, alpha=0.45)
        for th_ in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            a3.plot(s_ends, [Cx0 * np.cos(th_)] * 2, [Cy0 * np.sin(th_)] * 2,
                    color='#3f4b5b', lw=0.6, alpha=0.3)
        ch_lab = f'chamber {Cx0:.1f} $\\times$ {Cy0:.1f} mm'
    elif chamber_3d == 'follow':
        cx3, cy3 = C_x[::stride] * 1e3, C_y[::stride] * 1e3
        a3.plot_surface(S3, cx3[:, None] * np.cos(TH3), cy3[:, None] * np.sin(TH3),
                        color='#6b7a8f', alpha=0.15, rstride=2, cstride=1,
                        linewidth=0, antialiased=True, shade=False)
        ch_lab = f'chamber = envelope + {chamber_margin*1e3:.0f} mm'
    else:
        ch_lab = None

    a3.plot_surface(S3, X3, Y3, facecolors=mcm.viridis(norm3(R3)), rstride=2, cstride=1,
                    linewidth=0, antialiased=False, shade=False)
    a3.set_box_aspect((3, 1, 1))
    a3.view_init(elev=25, azim=-60)
    if chamber_3d:
        lim = max(Cx0, Cy0) * 1.08
        a3.set_ylim(-lim, lim)
        a3.set_zlim(-lim, lim)
    a3.set_xlabel('s [m]', labelpad=10)
    a3.set_ylabel('x [mm]', labelpad=10)
    a3.set_zlabel('y [mm]', labelpad=10)
    fig.colorbar(mcm.ScalarMappable(norm=norm3, cmap=mcm.viridis), ax=a3, shrink=0.6, pad=0.1,
                 label='Beam radius R(s,$\\theta$) [mm]')
    a3.set_title(f'{n_sigma_beam}$\\sigma$ beam envelope (injected beam rms emittances, '
                 f'{mode_tag})' + (f' inside {ch_lab}' if ch_lab else ''))
    fig.tight_layout()
    fig.savefig(f'{folder_ap}/beam_envelope_3D_{mode_tag}.png', dpi=200, bbox_inches='tight')
    plt.show()

    # beam's-eye cross-section with the aperture requirement on it
    aperture_cross_section(s_ap, env_x, env_y, sig_x, sig_y, n_sigma_beam,
                           chamber_margin=chamber_margin, mode_tag=mode_tag,
                           folder=folder_ap)
    plt.show()

    # rotatable version of the same figure (needs plotly)
    if interactive_3d:
        try:
            beam_chamber_3d_interactive(s_ap, env_x, env_y, C_x, C_y,
                                        n_sigma_beam, mode_tag=mode_tag,
                                        folder=folder_ap)
        except ImportError:
            print('    interactive 3D skipped: pip install plotly')
 
    # --- pole-tip fields per magnet type ---
    pole_rows, worst = [], {}
    for var, g in df_mag.groupby('variant', sort=False):
        mtype = g.type.iloc[0]
        R = (g.chamber_y.max() if mtype == 'Bend' else g.chamber_r.max()) + wall_thickness
        worst[var] = g.loc[g.chamber_r.idxmax()]
        if mtype == 'Bend':
            h, k1 = g.h.iloc[0], g.k1.iloc[0]
            B = brho * (abs(h) + abs(k1) * R)
            strength = f'h = {h:.4f} 1/m' + (f', k1 = {k1:+.3f} 1/m2' if k1 != 0 else '')
            NI = brho * abs(h) * R / mu0
        elif mtype == 'Quadrupole':
            k1 = g.k1.iloc[0]
            B = brho * abs(k1) * R
            strength = f'k1 = {k1:+.4f} 1/m2'
            NI = B * R / (2 * mu0)
        else:
            k2 = g.k2.iloc[0]
            B = brho * abs(k2) * R**2 / 2
            strength = f'k2 = {k2:+.2f} 1/m3'
            NI = B * R / (3 * mu0)
        pole_rows.append({'Magnet type': var, 'Kind': mtype, 'N': len(g),
                          'L [m]': f'{g.length.iloc[0]:.3f}', 'Strength': strength,
                          'sigma x/y max [mm]': f'{g.sigma_x_max.max()*1e3:.2f} / {g.sigma_y_max.max()*1e3:.2f}',
                          'Chamber x/y [mm]': f'{g.chamber_x.max()*1e3:.1f} / {g.chamber_y.max()*1e3:.1f}',
                          'Fits x/y [sigma]': f'{g.N_sigma_x.min():.2f} / {g.N_sigma_y.min():.2f}',
                          'Fits x/y [eps]': f'{g.N_eps_x.min():.1f} / {g.N_eps_y.min():.1f}',
                          'R_pole [mm]': f'{R*1e3:.1f}', 'B_pole [T]': f'{B:.3f}',
                          'NI/pole [kA-turns]': f'{NI*1e-3:.2f}',
                          'Flag': 'ABOVE LIMIT' if B > B_limits[mtype] else '',
                          'Examples': ', '.join(g.name.iloc[:2]) + (', ...' if len(g) > 2 else ''),
                          '_R': R, '_B': B, '_t': mtype})
    df_pole = pd.DataFrame(pole_rows)
    df_show = df_pole.drop(columns=['_R', '_B', '_t'])
    df_show.to_csv(f'{folder_ap}/pole_tip_fields_{mode_tag}.csv', index=False)
 
    # --- cross-sections per magnet type ---
    th = np.linspace(0, 2 * np.pi, 200)

    def _draw_cross_section(a, r, lim, fs=9, lim_y=None):
        """Beam (1 and n sigma), chamber and pole for one magnet type, at the
        magnet of that type with the largest beam (`worst`)."""
        w = worst[r['Magnet type']]
        col = colors_ap[r['_t']]
        ex, ey = n_sigma_beam * w.sigma_x_max, n_sigma_beam * w.sigma_y_max
        X, Y = ex * np.cos(th), ey * np.sin(th)
        rr = np.hypot(X, Y)
        Xc, Yc = X * (rr + chamber_margin) / rr, Y * (rr + chamber_margin) / rr
        a.plot(w.sigma_x_max * np.cos(th) * 1e3, w.sigma_y_max * np.sin(th) * 1e3,
               color=col, lw=0.8, alpha=0.6, label='1 $\\sigma$ beam')
        a.plot(3 * w.sigma_x_max * np.cos(th) * 1e3, 3 * w.sigma_y_max * np.sin(th) * 1e3,
               '--', color=col, lw=1.1, label='3 $\\sigma$ beam envelope')
        a.plot(X * 1e3, Y * 1e3, '-', color=col, lw=1.6,
               label=f'physical aperture ({n_sigma_beam} $\\sigma$)')
        if chamber_margin > 0:
            a.plot(Xc * 1e3, Yc * 1e3, 'k:', lw=1.3,
                   label=f'aperture + {chamber_margin*1e3:.0f} mm (chamber)')
        title = (f"{r['Magnet type']}  ({w['name']})\n"
                 f"aperture x/y = {n_sigma_beam*w.sigma_x_max*1e3:.1f} / {n_sigma_beam*w.sigma_y_max*1e3:.1f} mm"
                 + (f", +{chamber_margin*1e3:.0f} mm = {w.chamber_x*1e3:.1f} / {w.chamber_y*1e3:.1f} mm"
                    if chamber_margin > 0 else ''))
        if r['_t'] != 'Bend':   # pole drawn for quads and sextupoles only
            Rm = r['_R'] * 1e3
            a.add_patch(mpatches.Circle((0, 0), Rm, fill=False, color='grey', lw=1.2,
                                        label=f'pole (R = {Rm:.1f} mm)'))
            title += f", pole R = {Rm:.1f} mm, B$_{{pole}}$ = {r['_B']:.3f} T"
        a.set_title(title,
                    fontsize=fs, color='crimson' if r['Flag'] else 'black')
        a.set_xlim(-lim, lim)
        a.set_ylim(-(lim_y or lim), lim_y or lim)
        a.set_aspect('equal')
        a.grid(alpha=0.3)
        a.tick_params(labelsize=fs - 1)

    n_var, ncols = len(df_pole), 4
    nrows = int(np.ceil(n_var / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.4 * nrows), squeeze=False)
    lim = 1.1 * max(df_mag.chamber_x.max(), df_mag.chamber_y.max(),
                    df_pole.loc[df_pole['_t'] != 'Bend', '_R'].max()) * 1e3
    for a, (_, r) in zip(axs.flat, df_pole.iterrows()):
        _draw_cross_section(a, r, lim)
    for a in list(axs.flat)[n_var:]:
        a.axis('off')
    for a in axs[-1]:
        a.set_xlabel('x [mm]', fontsize=9)
    for a in axs[:, 0]:
        a.set_ylabel('y [mm]', fontsize=9)
    fig.suptitle(f'Injected beam: 1 $\\sigma$ (faint), 3 $\\sigma$ (dashed), physical aperture = '
                 f'{n_sigma_beam} $\\sigma$ (solid), pole for quads/sextupoles (grey)'
                 + (f', chamber = +{chamber_margin*1e3:.0f} mm (dotted)' if chamber_margin > 0 else '')
                 + ', at the largest-beam magnet of each type', fontsize=11)
    fig.tight_layout()
    fig.savefig(f'{folder_ap}/beam_cross_section_per_magnet_type_{mode_tag}.png', dpi=200,
                bbox_inches='tight')
    plt.show()

    # --- one figure per kind (Bend, Quadrupole, Sextupole): the magnet type
    #     with the largest physical aperture (pole radius / half gap) ---
    for mtype in magnet_types:
        sub = df_pole[df_pole['_t'] == mtype]
        if sub.empty:
            continue
        r = sub.loc[sub['_R'].idxmax()]
        w = worst[r['Magnet type']]
        if mtype == 'Bend':
            lim_x = 1.15 * max(w.chamber_x, w.chamber_y) * 1e3
            lim_y = max(1.3 * w.chamber_y * 1e3, 0.3 * lim_x)  # true scale, no empty band
        else:                                                # whole pole circle in view
            lim_x = lim_y = 1.1 * max(r['_R'], w.chamber_x, w.chamber_y) * 1e3
        fig, a = plt.subplots(figsize=(9, 7 * lim_y / lim_x + 1.2))
        _draw_cross_section(a, r, lim_x, fs=11, lim_y=lim_y)
        a.set_xlabel('x [mm]', fontsize=11)
        a.set_ylabel('y [mm]', fontsize=11)
        a.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
        fig.tight_layout()
        out = f'{folder_ap}/pole_tip_aperture_{mtype}_{mode_tag}.png'
        fig.savefig(out, dpi=200, bbox_inches='tight')
        print(f'Largest {mtype} aperture: {r["Magnet type"]} (R = {r["_R"]*1e3:.1f} mm) -> {out}')
        plt.show()

    # --- table ---
    print(f'\nBrho = {brho:.4f} T m, envelope = {n_sigma_beam} sigma, '
          f'chamber margin = {chamber_margin*1e3:.0f} mm, wall = {wall_thickness*1e3:.1f} mm')
    print(df_show.drop(columns='Examples').to_string(index=False))
    cols_png = [c_ for c_ in df_show.columns if c_ != 'Examples']
    fig, a = plt.subplots(figsize=(19, 0.42 * len(df_show) + 1.2))
    a.axis('off')
    tab = a.table(cellText=df_show[cols_png].values.tolist(), colLabels=cols_png,
                  cellLoc='center', loc='center')
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    tab.scale(1.05, 1.5)
    for (r_, c_), cell in tab.get_celld().items():
        if r_ == 0:
            cell.set_text_props(weight='bold')
        elif df_show.iloc[r_ - 1]['Flag']:
            cell.set_facecolor('#fde0dc')
    fig.savefig(f'{folder_ap}/pole_tip_fields_{mode_tag}.png', dpi=200, bbox_inches='tight')
    plt.show()
    print(f'Results in {folder_ap}')
    return df_mag, df_pole





def aperture_cross_section(s_ap, env_x, env_y, sig_x, sig_y, n_sigma_beam,
                           chamber_margin=5e-3, mode_tag='', n_theta=721,
                           folder=None, ax=None):
    """
    Beam's-eye view: every n-sigma cross-section along the ring, superimposed,
    with the chamber that contains them all.

    Looking down s, the beam sweeps out the UNION of its cross-sections. For
    an axis-aligned ellipse of half-axes (a, b) the polar radius is
        r(theta) = a b / sqrt(b^2 cos^2 theta + a^2 sin^2 theta)
    so the union boundary is  R(theta) = max over s of r_s(theta).
    That is the lobed outline, and it is what the chamber must clear.

    A concentric axis-aligned ellipse contains another iff it is at least as
    big in BOTH half-axes, so the smallest elliptical chamber has half-axes
    max(env_x) and max(env_y) -- the dashed curve. The gap between it and the
    lobed union is aperture the beam never uses.
    """
    a = np.asarray(env_x); b = np.asarray(env_y)
    th = np.linspace(0, 2*np.pi, n_theta)
    ct, st = np.cos(th), np.sin(th)
    # r_s(theta) for every s, then the max over s
    r = (a[:, None]*b[:, None]) / np.sqrt((b[:, None]*ct)**2 + (a[:, None]*st)**2)
    R = r.max(axis=0)

    Ax, Ay = float(a.max()), float(b.max())
    isx, isy = int(a.argmax()), int(b.argmax())
    Cx, Cy = Ax + chamber_margin, Ay + chamber_margin
    # Smallest circle containing the elliptical chamber. The farthest point
    # of an ellipse from its centre is the end of the major axis, so this is
    # max(a, b) -- NOT hypot(a, b), which is the bounding rectangle's diagonal
    # and ~32% too large here. Matches 'chamber_r' in the per-magnet table.
    bore = float(max(Cx, Cy))

    mm = 1e3
    if ax is None:
        fig, ax = plt.subplots(figsize=(8.0, 8.8))
    else:
        fig = ax.figure

    # every cross-section, faintly
    step = max(1, len(a)//400)
    for i in range(0, len(a), step):
        ax.plot(a[i]*mm*ct, b[i]*mm*st, color='#5b9bd5', lw=0.4, alpha=0.12,
                zorder=1)

    ax.fill(R*mm*ct, R*mm*st, color='#5b9bd5', alpha=0.18, zorder=2)
    ax.plot(R*mm*ct, R*mm*st, color='#1f4e79', lw=2.0, zorder=5,
            label=f'union of all {n_sigma_beam:g}$\\sigma$ cross-sections')
    ax.plot(Ax*mm*ct, Ay*mm*st, color='#8c1c2b', lw=1.6, ls='--', zorder=4,
            label=f'required beam aperture  {Ax*mm:.2f} $\\times$ {Ay*mm:.2f} mm')
    ax.plot(Cx*mm*ct, Cy*mm*st, color='0.15', lw=2.4, zorder=6,
            label=f'chamber  {Cx*mm:.2f} $\\times$ {Cy*mm:.2f} mm '
                  f'(+{chamber_margin*mm:.0f} mm)')
    ax.plot(bore*mm*ct, bore*mm*st, color='0.55', lw=1.3, ls=':', zorder=3,
            label=f'circular bore  r = {bore*mm:.2f} mm')

    ax.annotate('', xy=(Cx*mm, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='<->', color='0.15', lw=1.1))
    ax.text(Cx*mm/2, -bore*mm*0.045, f'{Cx*mm:.1f} mm', ha='center', va='top',
            fontsize=9.5, color='0.15')
    ax.annotate('', xy=(0, Cy*mm), xytext=(0, 0),
                arrowprops=dict(arrowstyle='<->', color='0.15', lw=1.1))
    ax.text(bore*mm*0.015, Cy*mm/2, f'{Cy*mm:.1f} mm', rotation=90,
            ha='left', va='center', fontsize=9.5, color='0.15')

    # The union touches the bounding ellipse only on the axes -- there the
    # clearance is the margin by construction, so a "minimum clearance" is not
    # a useful number. What is useful: how much of the elliptical aperture the
    # beam never occupies.
    _trapz = getattr(np, 'trapezoid', None) or np.trapz   # numpy <2 fallback
    area_union = 0.5*_trapz(R**2, th)
    area_req = np.pi*Ax*Ay
    unused = 1.0 - area_union/area_req

    for (xx, yy, lab, s_at) in ((Ax, 0.0, 'widest in x', s_ap[isx]),
                                (0.0, Ay, 'widest in y', s_ap[isy])):
        ax.plot([xx*mm], [yy*mm], marker='o', ms=7, mfc='none',
                mec='#8c1c2b', mew=1.8, zorder=8)
        ax.annotate(f'{lab}\ns = {s_at:.1f} m',
                    xy=(xx*mm, yy*mm),
                    xytext=(-16, -26) if xx else (16, 12),
                    textcoords='offset points', fontsize=8.2, color='#8c1c2b',
                    ha='right' if xx else 'left', va='top' if xx else 'bottom',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white',
                              ec='#8c1c2b', lw=0.7, alpha=0.9))

    lim = bore*mm*1.1
    ax.axhline(0, color='0.85', lw=0.6, zorder=0)
    ax.axvline(0, color='0.85', lw=0.6, zorder=0)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
    ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
    ax.set_title(f'Aperture requirement, beam\'s-eye view ({mode_tag})\n'
                 f'all {n_sigma_beam:g}$\\sigma$ cross-sections of the ring '
                 f'superimposed', fontsize=11)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.085), ncol=2,
              fontsize=8.6, framealpha=0.95, borderaxespad=0.)

    Nsx = Cx/float(sig_x.max()); Nsy = Cy/float(sig_y.max())
    txt = (f'required beam half-aperture   x {Ax*mm:6.2f} mm  (widest at s = {s_ap[isx]:.1f} m)\n'
           f'                              y {Ay*mm:6.2f} mm  (widest at s = {s_ap[isy]:.1f} m)\n'
           f'chamber half-aperture         x {Cx*mm:6.2f} mm      y {Cy*mm:6.2f} mm\n'
           f'circular bore radius            {bore*mm:6.2f} mm\n'
           f'acceptance   {Nsx:.2f} $\\sigma_x$ / {Nsy:.2f} $\\sigma_y$'
           f'   =   {Nsx**2:.0f} / {Nsy**2:.0f} rms emittances\n'
           f'beam fills {100*(1-unused):.0f}% of the elliptical aperture')
    ax.text(0.5, -0.165, txt, transform=ax.transAxes, ha='center', va='top',
            fontsize=8.4, family='monospace', linespacing=1.5,
            bbox=dict(boxstyle='round,pad=0.45', fc='white', ec='0.7'))

    out = dict(A_x=Ax, A_y=Ay, chamber_x=Cx, chamber_y=Cy, bore_r=bore,
               s_widest_x=float(s_ap[isx]), s_widest_y=float(s_ap[isy]),
               N_sigma_x=Nsx, N_sigma_y=Nsy,
               area_union=float(area_union), area_required=float(area_req),
               unused_fraction=float(unused))
    if folder:
        fig.tight_layout()
        fig.savefig(f'{folder}/aperture_cross_section_{mode_tag}.png',
                    dpi=200, bbox_inches='tight')
        print(f'--> saved {folder}/aperture_cross_section_{mode_tag}.png')
    print(f'--> Aperture requirement ({mode_tag}), {n_sigma_beam:g} sigma:')
    for k, v in out.items():
        print(f'    {k:24s} {v:10.4f}')
    return out


def beam_chamber_3d_interactive(s_ap, env_x, env_y, C_x, C_y, n_sigma_beam,
                                mode_tag='', n_theta=72, max_s=1200,
                                html=None, show=True, folder=None):
    """
    The same 3D beam-in-chamber figure, but rotatable.

    Why not just %matplotlib widget on the mplot3d version: that surface is
    ~109k quads and matplotlib re-renders the whole thing in Python on every
    mouse-move (~2.4 s per redraw here), so dragging it is unusable. Plotly
    hands the mesh to the browser's WebGL once and rotation costs nothing.

    Needs plotly (pip install plotly). html='x.html' writes a standalone file
    that opens in any browser and keeps working without a kernel.
    """
    import plotly.graph_objects as go

    st = max(1, len(s_ap) // max_s)
    s3 = s_ap[::st]
    sx3, sy3 = env_x[::st] * 1e3, env_y[::st] * 1e3
    th = np.linspace(0, 2 * np.pi, n_theta)
    S, TH = np.meshgrid(s3, th, indexing='ij')
    X, Y = sx3[:, None] * np.cos(TH), sy3[:, None] * np.sin(TH)
    R = np.hypot(X, Y)

    Cx0, Cy0 = float(C_x.max()) * 1e3, float(C_y.max()) * 1e3
    S_C, TH_C = np.meshgrid(np.array([s3[0], s3[-1]]), th, indexing='ij')

    fig = go.Figure()
    fig.add_surface(x=S_C, y=Cx0 * np.cos(TH_C), z=Cy0 * np.sin(TH_C),
                    opacity=0.18, showscale=False, hoverinfo='skip',
                    colorscale=[[0, '#6b7a8f'], [1, '#6b7a8f']],
                    surfacecolor=np.zeros_like(S_C),
                    name='chamber', showlegend=True)
    fig.add_surface(x=S, y=X, z=Y, surfacecolor=R, colorscale='Viridis',
                    colorbar=dict(title='R(s,theta) [mm]', len=0.6),
                    name='beam', showlegend=True,
                    hovertemplate='s=%{x:.1f} m<br>x=%{y:.2f} mm'
                                  '<br>y=%{z:.2f} mm<extra></extra>')
    lim = max(Cx0, Cy0) * 1.08
    fig.update_layout(
        title=f'{n_sigma_beam} sigma beam envelope inside chamber '
              f'{Cx0:.1f} x {Cy0:.1f} mm ({mode_tag})',
        scene=dict(xaxis_title='s [m]', yaxis_title='x [mm]',
                   zaxis_title='y [mm]',
                   yaxis=dict(range=[-lim, lim]), zaxis=dict(range=[-lim, lim]),
                   aspectmode='manual', aspectratio=dict(x=3, y=1, z=1)),
        width=1150, height=650, margin=dict(l=0, r=0, t=50, b=0))

    # Always write the standalone file: it is the one route that works
    # everywhere (VS Code notebook, Interactive Window, plain `python x.py`,
    # over SSH). Inline display is attempted on top of that, and a missing
    # renderer or missing nbformat must not kill a long run.
    if html is None and folder is not None:
        html = f'{folder}/beam_chamber_3d_{mode_tag}.html'
    if html:
        fig.write_html(html, include_plotlyjs='cdn')
        print(f'--> interactive 3D written to {html}  (open it to rotate)')
    if show:
        try:
            fig.show()
        except Exception as e:
            print(f'    inline display unavailable ({type(e).__name__}: {e}); '
                  f'open the html file instead')
    return fig


def beam_amplitude(df_particles, df_core, position, angle, p_col='p[MeV/c]',
                   p0_mev=2860.0):
    """
    Amplitude of every particle in df_particles, in units of the rms beam size
    that the CORE emittance implies:

        A = sqrt(J / eps_core),      J = gamma x^2 + 2 alpha x x' + beta x'^2

    so A = 3 means "this particle reaches 3 sigma_x somewhere in betatron
    phase", with sigma_x = sqrt(beta eps_core).

    df_core sets alpha, beta and eps (the core Twiss); df_particles supplies
    the particles being measured. Passing the same frame for both measures the
    core against itself; passing the raw frame measures the halo too.
    """
    core = remove_dispersion(df_core, position, angle, p_col, p0_mev)
    alpha, beta, gamma, _, _ = get_twiss(core, position, angle, p_col, p0_mev)
    eps = CalcEmittanceAuto(core, position, angle)

    raw = remove_dispersion(df_particles, position, angle, p_col, p0_mev)
    u = raw[position].values - raw[position].values.mean()
    up = raw[angle].values - raw[angle].values.mean()
    J = gamma * u ** 2 + 2 * alpha * u * up + beta * up ** 2
    return np.sqrt(J / eps), eps


def aperture_n_sigma_from_beam(df_beam, df_core, n_sigma_core, cover='max',
                               safety=1.15, df_reference=None, p0_mev=2860.0,
                               override=None, verbose=True):
    """
    Envelope factor for the chamber, sized to df_beam.

    n_sigma_core (the action cut that DEFINES the core, and so eps) and this
    envelope factor are different quantities and must not be the same number:

      * n_sigma_core answers "where does beam stop and halo start?" Raising it
        makes eps bigger.
      * the envelope factor answers "how many rms beam sizes of clearance does
        the chamber need?"

    WHICH DISTRIBUTION TO PASS AS df_beam
      df_clean (the filtered core) -- treats the core as the beam to be
        transported and the discarded tail as not-beam. Because the action
        filter guarantees J <= n_sigma_core^2 eps, max amplitude comes out at
        n_sigma_core by construction, so the result is just
        ceil(safety * n_sigma_core). That is a policy multiplier, not a
        measurement: the beam contributes nothing to it. Correct when the
        discarded tail is instrumental junk or particles accepted as lost.
      df (the raw distribution) -- sizes the chamber to clear every particle
        actually present, halo included. The result then depends on the real
        tail and can be much larger.

    Pass df_reference (normally the raw frame) to have the tail reported for
    comparison without letting it drive the number.

    cover
        'max'  -- clear every particle in df_beam
        float  -- a percentile, e.g. 99.9, accepting the rest as loss

    override
        Set a number to FORCE the envelope (e.g. 5 for a 5 sigma chamber).
        The value the beam would have asked for is still computed and printed,
        and the coverage is reported against the forced value, so choosing it
        is a decision rather than an accident.

    Returns (n_aperture, amp_x, amp_y) with amp measured on df_beam.
    """
    amp_x, _ = beam_amplitude(df_beam, df_core, 'x[mm]', 'xp[mrad]',
                              p0_mev=p0_mev)
    amp_y, _ = beam_amplitude(df_beam, df_core, 'y[mm]', 'yp[mrad]',
                              p0_mev=p0_mev)
    if cover == 'max':
        a_cov = max(amp_x.max(), amp_y.max())
        how = 'max amplitude'
    else:
        a_cov = max(np.percentile(amp_x, cover), np.percentile(amp_y, cover))
        how = f'{cover:g}th percentile'
    n_derived = max(float(n_sigma_core), float(np.ceil(safety * a_cov)))
    n_ap = float(override) if override is not None else n_derived

    if verbose:
        print('--> Aperture envelope factor')
        print(f'    core cut defining eps : {n_sigma_core} sigma '
              f'(sets eps -- NOT an aperture)')
        print(f'    sized to              : {len(df_beam)} particles')
        print(f'    max amplitude  x / y  : '
              f'{amp_x.max():.3f} / {amp_y.max():.3f} sigma')
        print(f'    cover = {cover}  ({how} = {a_cov:.3f} sigma), '
              f'safety = {safety}')
        if override is None:
            print(f'    --> envelope          : {n_ap:g} sigma')
        else:
            print(f'    derived from the beam : {n_derived:g} sigma')
            print(f'    --> envelope (FORCED) : {n_ap:g} sigma'
                  + ('   <-- smaller than derived: the tail beyond it is '
                     'accepted as loss' if n_ap < n_derived else ''))
        if abs(a_cov - n_sigma_core) < 0.15 * n_sigma_core:
            print(f'    note: max amplitude sits at the core cut, as the '
                  f'action filter guarantees. This envelope is '
                  f'ceil({safety} x {n_sigma_core}) and carries no '
                  f'information from the beam itself.')
        if df_reference is not None:
            rx, _ = beam_amplitude(df_reference, df_core, 'x[mm]', 'xp[mrad]',
                                   p0_mev=p0_mev)
            ry, _ = beam_amplitude(df_reference, df_core, 'y[mm]', 'yp[mrad]',
                                   p0_mev=p0_mev)
            out = int(np.sum(rx > n_ap) + np.sum(ry > n_ap))
            print(f'    cross-check against {len(df_reference)} unfiltered '
                  f'particles (reported only, does not change the envelope):')
            print(f'      max amplitude x / y : {rx.max():.2f} / {ry.max():.2f} sigma')
            print(f'      {"n":>5s} {"beyond (x)":>12s} {"beyond (y)":>12s}')
            for n in (3, 5, 6, 8, 10, 12, 15):
                print(f'      {n:5d} {int(np.sum(rx > n)):12d} '
                      f'{int(np.sum(ry > n)):12d}')
            print(f'      outside the {n_ap:g} sigma envelope: {out} '
                  f'({100*out/(2*len(df_reference)):.3f}% of plane-particles) '
                  f'-- scraped at injection')
    return n_ap, amp_x, amp_y


# Physical aperture = N_SIGMA_APERTURE sigma of the injected beam,
#     A_x = n * sqrt(beta_x * eps_x + (D_x * sigma_delta)^2)
# and the vacuum chamber = aperture + 5 mm. Pole radii / B_pole use the chamber.
#
# N_SIGMA_APERTURE IS NOT N_SIGMA. They are different quantities:
#   N_SIGMA          action cut that DEFINES the core, and so eps. Raising it
#                    makes eps bigger. It decides what counts as halo.
#   N_SIGMA_APERTURE envelope factor for the chamber. It has to clear the halo
#                    that N_SIGMA threw away, so it is normally much larger.
# Setting the second equal to the first builds a chamber exactly at the cut
# used to discard particles, i.e. one that scrapes every discarded particle.
# Sizing to df_clean makes N_SIGMA_APERTURE = ceil(safety * N_SIGMA); sizing to
# df instead would make it track the real halo. Which is right depends on
# whether the discarded tail is real beam or instrumental junk.
#
# eps_x, eps_y: rms emittances of the beam core computed above (mm mrad = um).
# sigma_delta: rms momentum spread of the MAIN BUNCH. The whole-distribution
# spread (relative_spread, ~5 % with the tails) would give ~60 mm instead.
_main = select_main_bunch(df)
sigma_delta_for_aperture = float(np.std(_main['p[MeV/c]']) / np.mean(_main['p[MeV/c]']))
print(f'Aperture uses sigma_delta = {sigma_delta_for_aperture*1e3:.2f} per mil (main bunch), '
      f'eps_x = {emittance_x:.3f} um, eps_y = {emittance_y:.3f} um')

# Sized to df_clean: the filtered core IS the beam to be transported, and the
# tail the action cut removed is not treated as beam. Note the consequence --
# the action filter guarantees J <= N_SIGMA^2 eps, so max amplitude comes out
# at N_SIGMA and this envelope is just ceil(safety * N_SIGMA). It is a policy
# multiplier, not a measurement. df is passed as df_reference so the tail that
# would be scraped is still printed, without changing the number.
# Set to a number to force the chamber envelope, or None to let the beam
# decide. These are independent of N_SIGMA, which only defines eps.
N_SIGMA_APERTURE_OVERRIDE = 5

N_SIGMA_APERTURE, amp_x, amp_y = aperture_n_sigma_from_beam(
    df_clean, df_clean, CORE_AMPLITUDE, cover='max', safety=1.15,
    df_reference=df, p0_mev=2860.0, override=N_SIGMA_APERTURE_OVERRIDE)

aperture_mag, aperture_pole = physical_aperture_study(
    ring, emittance_x * 1e-6, emittance_y * 1e-6, sigma_delta_for_aperture, mode,
    n_sigma_beam=N_SIGMA_APERTURE, chamber_margin=5e-3,
    chamber_3d='cylinder', interactive_3d=True)

# %% Ring-wide worst-case aperture sketch
# One transverse cross-section standing in for the whole ring, built from
# the maximum beta in each plane and the rms momentum spread. Same sigma
# convention as physical_aperture_study above, and fed the same numbers,
# so the two agree by construction.

def _fmt_eps(e):
    return f'{e*1e6:.3g} $\\mu$m' if e >= 1e-7 else f'{e*1e9:.3g} nm'


def ring_beam_params(ring, coupling=0.01):
    """
    Equilibrium emittances and momentum spread from the ring's own radiation.
    eps_y comes out numerically zero with no coupling and no vertical
    excitation, so it is set from the coupling ratio instead.
    Returns (eps_x, eps_y, sigma_delta).
    """
    r = ring.copy()
    r.configure_radiation(model='mean')
    try:
        tw = r.twiss(radiation_analysis=True)
    except TypeError:
        tw = r.twiss(eneloss_and_damping=True)
    eps_x = float(tw.eq_gemitt_x)
    eps_y = float(tw.eq_gemitt_y)
    if not np.isfinite(eps_y) or eps_y < 1e-20:
        eps_y = coupling * eps_x
    sigma_delta = float(np.sqrt(tw.eq_gemitt_zeta / tw.bets0))
    return eps_x, eps_y, sigma_delta


def aperture_sketch(ring, eps_x, eps_y, sigma_delta, n_sigma=3,
                    chamber_margin=5e-3, ds=0.05, sizing='widest',
                    tw=None, title=None, show_profile=True, source=None):
    """
    Draw the worst-case transverse aperture cross-section for the whole ring.

    eps_x, eps_y   geometric rms emittances [m rad]
    sigma_delta    rms relative momentum spread
    n_sigma        beam envelope in rms beam sizes
    chamber_margin clearance added to the envelope [m]
    sizing
        'widest' (default) -- the widest point of the beam in each plane:
            sigma_x at the s where sigma_x(s) peaks, sigma_y at the s where
            sigma_y(s) peaks. beta and D are read off the lattice at those
            points, so each half-axis is a size the beam really reaches. The
            two s are usually a few metres apart; both are reported.
        'single_s' -- ONE location: the s where sqrt(sigma_x^2 + sigma_y^2) is
            largest, i.e. the worst point for a ROUND chamber. The ellipse is
            then a true cross-section, but one plane is normally well short of
            its own maximum, so the chamber is grown to cover the rest of the
            ring and the shortfall is drawn dashed.
        'composite' -- each plane from its own maximum beta combined with the
            ring's maximum |D|:
                sigma_x = sqrt(beta_x^max eps_x + (|D_x|^max sigma_delta)^2)
            Those maxima sit at different s, so this exceeds 'widest' whenever
            |D| peaks away from beta^max. Most conservative.

    Returns a dict of every number used.
    """
    if tw is None:
        r = ring.copy()
        try:
            r.configure_radiation(model=None)
        except Exception:
            pass
        r.cut_at_s(np.arange(ds, r.get_length() - ds / 2, ds))
        tw = r.twiss4d()

    s = np.asarray(tw.s)
    betx, bety = np.asarray(tw.betx), np.asarray(tw.bety)
    dx, dy = np.asarray(tw.dx), np.asarray(tw.dy)

    sig_x_s = np.sqrt(betx * eps_x + (dx * sigma_delta) ** 2)
    sig_y_s = np.sqrt(bety * eps_y + (dy * sigma_delta) ** 2)

    ibx, iby = int(np.argmax(betx)), int(np.argmax(bety))
    idx, idy = int(np.argmax(np.abs(dx))), int(np.argmax(np.abs(dy)))
    isx, isy = int(np.argmax(sig_x_s)), int(np.argmax(sig_y_s))

    bx_max, by_max = float(betx[ibx]), float(bety[iby])
    dx_max, dy_max = float(abs(dx[idx])), float(abs(dy[idy]))

    iw = int(np.argmax(np.hypot(sig_x_s, sig_y_s)))

    if sizing == 'widest':
        sx_bet, sx_dis = np.sqrt(betx[isx] * eps_x), abs(dx[isx]) * sigma_delta
        sy_bet, sy_dis = np.sqrt(bety[isy] * eps_y), abs(dy[isy]) * sigma_delta
        where = (f'widest point in each plane\n'
                 f'x: s = {s[isx]:.2f} m, $\\beta_x$ = {betx[isx]:.2f} m, '
                 f'$D_x$ = {dx[isx]:.3f} m     '
                 f'y: s = {s[isy]:.2f} m, $\\beta_y$ = {bety[isy]:.2f} m')
    elif sizing == 'single_s':
        sx_bet, sx_dis = np.sqrt(betx[iw] * eps_x), abs(dx[iw]) * sigma_delta
        sy_bet, sy_dis = np.sqrt(bety[iw] * eps_y), abs(dy[iw]) * sigma_delta
        where = (f'single widest s = {s[iw]:.2f} m  '
                 f'($\\beta_x$ = {betx[iw]:.2f}, $\\beta_y$ = {bety[iw]:.2f}, '
                 f'$D_x$ = {dx[iw]:.3f} m)')
    elif sizing == 'composite':
        sx_bet, sx_dis = np.sqrt(bx_max * eps_x), dx_max * sigma_delta
        sy_bet, sy_dis = np.sqrt(by_max * eps_y), dy_max * sigma_delta
        where = '$\\beta^{max}$ composite (no single s)'
    else:
        raise ValueError(
            "sizing must be 'widest', 'single_s' or 'composite'")

    sx, sy = float(np.hypot(sx_bet, sx_dis)), float(np.hypot(sy_bet, sy_dis))
    env_x, env_y = n_sigma * sx, n_sigma * sy

    # the chamber must clear the beam EVERYWHERE, not only at the drawn point
    req_x, req_y = n_sigma * float(sig_x_s[isx]), n_sigma * float(sig_y_s[isy])
    grown_x, grown_y = req_x > env_x * 1.0001, req_y > env_y * 1.0001
    ch_x = max(env_x, req_x) + chamber_margin
    ch_y = max(env_y, req_y) + chamber_margin
    # smallest circle containing the elliptical chamber = max half-axis
    # (see the note in aperture_cross_section)
    bore_r = float(max(ch_x, ch_y))

    out = dict(
        sizing=sizing, n_sigma=n_sigma,
        eps_x=eps_x, eps_y=eps_y, sigma_delta=sigma_delta,
        betx_max=bx_max, bety_max=by_max,
        s_betx_max=float(s[ibx]), s_bety_max=float(s[iby]),
        dx_max=dx_max, dy_max=dy_max,
        s_dx_max=float(s[idx]), s_dy_max=float(s[idy]),
        sigma_x=sx, sigma_y=sy,
        sigma_x_betatron=float(sx_bet), sigma_x_dispersive=float(sx_dis),
        sigma_y_betatron=float(sy_bet), sigma_y_dispersive=float(sy_dis),
        sigma_x_worst_s=float(sig_x_s[isx]), s_sigma_x_max=float(s[isx]),
        sigma_y_worst_s=float(sig_y_s[isy]), s_sigma_y_max=float(s[isy]),
        env_x=env_x, env_y=env_y,
        chamber_x=ch_x, chamber_y=ch_y, chamber_margin=chamber_margin,
        bore_r=bore_r,
        s_widest=float(s[iw]),
        betx_at_widest=float(betx[iw]), bety_at_widest=float(bety[iw]),
        dx_at_widest=float(dx[iw]),
        chamber_grown_x=bool(grown_x), chamber_grown_y=bool(grown_y),
    )

    mm = 1e3
    if show_profile:
        fig, (ax, axp) = plt.subplots(
            1, 2, figsize=(14.0, 7.0),
            gridspec_kw=dict(width_ratios=[1.0, 1.25]))
    else:
        fig, ax = plt.subplots(figsize=(8.0, 7.0))
        axp = None

    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(bore_r * mm * np.cos(th), bore_r * mm * np.sin(th),
            color='0.55', lw=1.4, ls=':', zorder=3,
            label=f'circular bore  r = {bore_r*mm:.1f} mm')
    ax.plot(ch_x * mm * np.cos(th), ch_y * mm * np.sin(th),
            color='0.2', lw=2.4, zorder=4,
            label=f'chamber  {ch_x*mm:.1f} $\\times$ {ch_y*mm:.1f} mm '
                  f'(+{chamber_margin*mm:.0f} mm margin)')
    ax.fill(env_x * mm * np.cos(th), env_y * mm * np.sin(th),
            color='#5b9bd5', alpha=0.22, zorder=1)
    ax.plot(env_x * mm * np.cos(th), env_y * mm * np.sin(th),
            color='#1f4e79', lw=2, zorder=5,
            label=f'{n_sigma:g}$\\sigma$ envelope  '
                  f'{env_x*mm:.2f} $\\times$ {env_y*mm:.2f} mm')
    ax.fill(sx * mm * np.cos(th), sy * mm * np.sin(th),
            color='#d1495b', alpha=0.5, zorder=2)
    ax.plot(sx * mm * np.cos(th), sy * mm * np.sin(th),
            color='#8c1c2b', lw=1.6, zorder=6,
            label=f'1$\\sigma$ beam  {sx*mm:.2f} $\\times$ {sy*mm:.2f} mm')
    if grown_x or grown_y:
        ax.plot(req_x * mm * np.cos(th), req_y * mm * np.sin(th),
                color='#4a7c59', lw=1.5, ls='--', zorder=5,
                label=f'{n_sigma:g}$\\sigma$ elsewhere in ring  '
                      f'{req_x*mm:.2f} $\\times$ {req_y*mm:.2f} mm')

    lim = bore_r * mm * 1.15
    ax.annotate('', xy=(ch_x * mm, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='<->', color='0.2', lw=1.1))
    ax.text(ch_x * mm / 2, -lim * 0.055, f'{ch_x*mm:.1f} mm',
            ha='center', va='top', fontsize=9.5, color='0.2')
    ax.annotate('', xy=(0, ch_y * mm), xytext=(0, 0),
                arrowprops=dict(arrowstyle='<->', color='0.2', lw=1.1))
    ax.text(lim * 0.02, ch_y * mm / 2, f'{ch_y*mm:.1f} mm', rotation=90,
            ha='left', va='center', fontsize=9.5, color='0.2')
    ax.annotate('', xy=(env_x * mm, -env_y * mm * 0.33),
                xytext=(sx * mm, -env_y * mm * 0.33),
                arrowprops=dict(arrowstyle='<->', color='#1f4e79', lw=1))
    ax.text((sx + env_x) * mm / 2, -env_y * mm * 0.37,
            f'{n_sigma:g}$\\sigma$', ha='center', va='top',
            fontsize=9, color='#1f4e79')

    ax.axhline(0, color='0.85', lw=0.6, zorder=0)
    ax.axvline(0, color='0.85', lw=0.6, zorder=0)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
    ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
    ax.set_title(title or 'Worst-case aperture cross-section', fontsize=11)
    ax.legend(loc='upper right', fontsize=8.0, framealpha=0.94)

    txt = (
        f'$\\epsilon_x$ = {_fmt_eps(eps_x)}     '
        f'$\\epsilon_y$ = {_fmt_eps(eps_y)}     '
        f'$\\sigma_\\delta$ = {sigma_delta*100:.3f} %\n'
        + (f'{source}\n' if source else '') +
        f'{where}\n'
        f'$\\sigma_x$ = $\\sqrt{{({sx_bet*mm:.3f})^2 + ({sx_dis*mm:.3f})^2}}$'
        f' = {sx*mm:.3f} mm     '
        f'$\\sigma_y$ = $\\sqrt{{({sy_bet*mm:.3f})^2 + ({sy_dis*mm:.3f})^2}}$'
        f' = {sy*mm:.3f} mm\n'
        f'(betatron $\\oplus$ dispersive)'
    )
    ax.text(0.5, -0.145, txt, transform=ax.transAxes, ha='center', va='top',
            fontsize=8.4, linespacing=1.45,
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='0.75'))

    if axp is not None:
        axp.plot(s, n_sigma * sig_x_s * mm, color='#1f4e79', lw=1.2,
                 label=f'{n_sigma:g}$\\sigma_x$(s)')
        axp.plot(s, n_sigma * sig_y_s * mm, color='#c05640', lw=1.2,
                 label=f'{n_sigma:g}$\\sigma_y$(s)')
        axp.axhline(ch_x * mm, color='#1f4e79', ls='--', lw=1.2,
                    label=f'chamber x  {ch_x*mm:.1f} mm')
        axp.axhline(ch_y * mm, color='#c05640', ls='--', lw=1.2,
                    label=f'chamber y  {ch_y*mm:.1f} mm')
        lo, hi = axp.get_ylim()
        axp.set_ylim(lo, hi + 0.30 * (hi - lo))
        axp.plot([s[isx]], [n_sigma * sig_x_s[isx] * mm], marker='v',
                 color='#1f4e79', ms=9, zorder=6, clip_on=False)
        axp.plot([s[isy]], [n_sigma * sig_y_s[isy] * mm], marker='v',
                 color='#c05640', ms=9, zorder=6, clip_on=False)
        if sizing == 'single_s':
            axp.axvline(s[iw], color='black', lw=1.6, alpha=0.85,
                        label=f'chosen s = {s[iw]:.1f} m')
        marks = ((isx, '#1f4e79', 'widest x'),
                 (isy, '#c05640', 'widest y'),
                 (idx, '#4a7c59', '$|D_x|^{max}$'))
        for j, (i, c, lab) in enumerate(sorted(marks, key=lambda m: s[m[0]])):
            axp.axvline(s[i], color=c, lw=0.9, alpha=0.55, ls='-.')
            axp.annotate(f'{lab}\n{s[i]:.0f} m', xy=(s[i], hi),
                         xytext=(0, 6 + 22 * (j % 2)),
                         textcoords='offset points', color=c, fontsize=8.2,
                         ha='center', va='bottom',
                         bbox=dict(boxstyle='round,pad=0.18', fc='white',
                                   ec=c, lw=0.7, alpha=0.9))
        axp.set_xlim(s[0], s[-1])
        axp.set_xlabel('s [m]')
        axp.set_ylabel(f'{n_sigma:g}$\\sigma$ beam envelope [mm]')
        axp.set_title('beam envelope around the ring  '
                      '(the maxima do not coincide)', fontsize=11)
        axp.grid(alpha=0.25)
        axp.legend(loc='lower right', fontsize=8.0, ncol=2, framealpha=0.94)

    print(f'--> Aperture sketch  [{sizing}]  n_sigma = {n_sigma:g}')
    print(f'    INPUT eps_x = {eps_x*1e6:.4f} um   eps_y = {eps_y*1e6:.4f} um'
          f'   sigma_delta = {sigma_delta*1e3:.3f} per mil')
    print(f'    beta_x max  {bx_max:8.3f} m   at s = {s[ibx]:7.2f} m')
    print(f'    beta_y max  {by_max:8.3f} m   at s = {s[iby]:7.2f} m')
    print(f'    |D_x| max   {dx_max:8.4f} m   at s = {s[idx]:7.2f} m')
    print(f'    |D_y| max   {dy_max:8.4f} m   at s = {s[idy]:7.2f} m')
    print(f'    sigma_x     {sx*mm:8.3f} mm  '
          f'(bet {sx_bet*mm:.3f} + disp {sx_dis*mm:.3f})')
    print(f'    sigma_y     {sy*mm:8.3f} mm  '
          f'(bet {sy_bet*mm:.3f} + disp {sy_dis*mm:.3f})')
    print(f'    {n_sigma:g}-sigma env {env_x*mm:7.2f} x {env_y*mm:.2f} mm')
    print(f'    chamber     {ch_x*mm:7.2f} x {ch_y*mm:.2f} mm   '
          f'bore r = {bore_r*mm:.2f} mm')
    print(f'    widest in x   s = {s[isx]:7.2f} m   sigma_x = '
          f'{sig_x_s[isx]*mm:.3f} mm  (beta_x {betx[isx]:.2f}, '
          f'D_x {dx[isx]:.4f} m)')
    print(f'    widest in y   s = {s[isy]:7.2f} m   sigma_y = '
          f'{sig_y_s[isy]*mm:.3f} mm  (beta_y {bety[isy]:.2f})')
    if abs(s[isx] - s[isy]) > ds:
        print(f'    the two widest points are {abs(s[isx]-s[isy]):.2f} m '
              f'apart, so this cross-section is a per-plane worst case')
    if grown_x or grown_y:
        g = ' and '.join(p for p, f in (('x', grown_x), ('y', grown_y)) if f)
        print(f'    NOTE: chamber grown in {g} to clear the beam elsewhere '
              f'in the ring (dashed curve on the sketch)')
    return out

# %%
aperture_sketch_data = aperture_sketch(
    ring,
    eps_x=emittance_x * 1e-6,
    eps_y=emittance_y * 1e-6,
    sigma_delta=sigma_delta_for_aperture,
    n_sigma=N_SIGMA_APERTURE,
    chamber_margin=5e-3,
    sizing='widest',
    source=f'$\\epsilon$ and envelope from df_clean '
           f'({len(df_clean)}/{len(df)} particles, moment-based core);'
           f'  envelope {N_SIGMA_APERTURE:g}$\\sigma$',
    title=f'{mode}  --  aperture at the widest point')
plt.tight_layout()
plt.show()

# %%
rand_num = 74
n_subset = 500
df_subset = df_clean.sample(n=min(n_subset, len(df_clean)), random_state=rand_num)

p0c_avg_mev = df_clean['p[MeV/c]'].mean()
p0c_avg = p0c_avg_mev * 1e6
ref_particle_avg = xp.Particles(p0c=p0c_avg, mass0=xp.ELECTRON_MASS_EV)

def match_coordinates(df_in, p0c_ref, ref_particle, dx, ddx, dy, ddy,
                       beam_disp_x=(0.0, 0.0), beam_disp_y=(0.0, 0.0),
                       beam_optics_x=None, beam_optics_y=None,
                       ring_optics_x=None, ring_optics_y=None):
    delta = (df_in['p[MeV/c]'].values * 1e6 - p0c_ref) / p0c_ref

    x_raw  = df_in['x[mm]'].values  * 1e-3
    px_raw = df_in['xp[mrad]'].values * 1e-3
    y_raw  = df_in['y[mm]'].values  * 1e-3
    py_raw = df_in['yp[mrad]'].values * 1e-3

    dxb, ddxb = beam_disp_x
    dyb, ddyb = beam_disp_y
    x_beta  = x_raw  - dxb * delta
    px_beta = px_raw - ddxb * delta
    y_beta  = y_raw  - dyb * delta
    py_beta = py_raw - ddyb * delta

    if beam_optics_x is not None and ring_optics_x is not None:
        x_beta, px_beta = optics_match_transform(x_beta, px_beta, *beam_optics_x, *ring_optics_x)
    if beam_optics_y is not None and ring_optics_y is not None:
        y_beta, py_beta = optics_match_transform(y_beta, py_beta, *beam_optics_y, *ring_optics_y)

    x_matched  = x_beta  + dx  * delta
    px_matched = px_beta + ddx * delta
    y_matched  = y_beta  + dy  * delta
    py_matched = py_beta + ddy * delta

    t_mm = df_in['t[mm/c]'].values
    zeta = (np.mean(t_mm) - t_mm) * 1e-3 * ref_particle.beta0[0]
    return x_matched, px_matched, y_matched, py_matched, delta, zeta

# %%
x_m_full, px_m_full, y_m_full, py_m_full, delta_full, zeta_full = match_coordinates(
    df_clean, p0c_avg, ref_particle_avg,
    dx0, ddx0, dy0, ddy0,
    beam_disp_x=(dx*1e-3, ddx*1e-3), beam_disp_y=(dy*1e-3, ddy*1e-3),
    beam_optics_x=(bx, ax), beam_optics_y=(by, ay),
    ring_optics_x=(betx0, alfx0), ring_optics_y=(bety0, alfy0))

def twiss_from_arrays(pos, ang):
    cov = np.cov(pos, ang, ddof=0)
    eps = np.sqrt(np.linalg.det(cov))
    beta = cov[0, 0] / eps
    alpha = -cov[0, 1] / eps
    return alpha, beta, eps

x_m_full_mm, px_m_full_mrad = x_m_full*1e3, px_m_full*1e3
y_m_full_mm, py_m_full_mrad = y_m_full*1e3, py_m_full*1e3

alpha_mx, beta_mx, eps_mx = twiss_from_arrays(x_m_full_mm, px_m_full_mrad)
alpha_my, beta_my, eps_my = twiss_from_arrays(y_m_full_mm, py_m_full_mrad)

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

for ax_i, pos_full, ang_full, beta_m, alpha_m, beta_r, alpha_r, eps_m, pos_label, ang_label, plane_label in [
    (axes[0], x_m_full_mm, px_m_full_mrad, beta_mx, alpha_mx, betx0, alfx0, eps_mx, 'x [mm]', "xp [mrad]", 'Horizontal'),
    (axes[1], y_m_full_mm, py_m_full_mrad, beta_my, alpha_my, bety0, alfy0, eps_my, 'y [mm]', "yp [mrad]", 'Vertical'),
]:
    sc = density_scatter(ax_i, pos_full, ang_full, s=1)
    fig.colorbar(sc, ax=ax_i, label='Relative Density')

    x1 = np.sqrt(eps_m * beta_m) * np.cos(theta)
    xp1 = -np.sqrt(eps_m / beta_m) * (alpha_m * np.cos(theta) - np.sin(theta))
    ax_i.plot(x1, xp1, color='blue', lw=2, label='Matched distribution Twiss')

    x2 = np.sqrt(eps_m * beta_r) * np.cos(theta)
    xp2 = -np.sqrt(eps_m / beta_r) * (alpha_r * np.cos(theta) - np.sin(theta))
    ax_i.plot(x2, xp2, color='red', lw=2, linestyle='--', label='Ring Twiss')

    ax_i.set_title(f'{plane_label}: Matched (Injected) Beam vs Ring Twiss')
    ax_i.set_xlabel(pos_label)
    ax_i.set_ylabel(ang_label)
    ax_i.axhline(0, color='black', lw=0.5, ls='--')
    ax_i.axvline(0, color='black', lw=0.5, ls='--')
    #ax_i.axis('equal')
    ax_i.set_xlim(-20,20)
    ax_i.set_ylim(-20,20)
    ax_i.grid(True, linestyle=':', alpha=0.5)
    ax_i.legend(fontsize='small')

plt.tight_layout()
plt.savefig(f'{folder}/injected_beam_vs_ring_twiss.png')
plt.show()

# %%
delta_before = (df_clean['p[MeV/c]'].values * 1e6 - p0c_avg) / p0c_avg
x_before = df_clean['x[mm]'].values * 1e-3 - (dx * 1e-3) * delta_before
px_before = df_clean['xp[mrad]'].values * 1e-3 - (ddx * 1e-3) * delta_before
y_before = df_clean['y[mm]'].values * 1e-3 - (dy * 1e-3) * delta_before
py_before = df_clean['yp[mrad]'].values * 1e-3 - (ddy * 1e-3) * delta_before
x_before, px_before = optics_match_transform(x_before, px_before, bx, ax, betx0, alfx0)
y_before, py_before = optics_match_transform(y_before, py_before, by, ay, bety0, alfy0)

x_before_mm, px_before_mrad = x_before * 1e3, px_before * 1e3
y_before_mm, py_before_mrad = y_before * 1e3, py_before * 1e3

fig, axes = plt.subplots(2, 2, figsize=(13, 12))

panels = [
    (axes[0, 0], x_before_mm, px_before_mrad, betx0, alfx0, 'x [mm]', "xp [mrad]",
     'Horizontal: Before Dispersion (Pure Betatron)'),
    (axes[0, 1], x_m_full_mm, px_m_full_mrad, betx0, alfx0, 'x [mm]', "xp [mrad]",
     'Horizontal: After Dispersion (Injected)'),
    (axes[1, 0], y_before_mm, py_before_mrad, bety0, alfy0, 'y [mm]', "yp [mrad]",
     'Vertical: Before Dispersion (Pure Betatron)'),
    (axes[1, 1], y_m_full_mm, py_m_full_mrad, bety0, alfy0, 'y [mm]', "yp [mrad]",
     'Vertical: After Dispersion (Injected)'),
]

for ax_i, pos_full, ang_full, beta_r, alpha_r, pos_label, ang_label, title in panels:
    alpha_m, beta_m, eps_m = twiss_from_arrays(pos_full, ang_full)

    sc = density_scatter(ax_i, pos_full, ang_full, s=1)
    fig.colorbar(sc, ax=ax_i, label='Relative Density')

    x1 = np.sqrt(eps_m * beta_m) * np.cos(theta)
    xp1 = -np.sqrt(eps_m / beta_m) * (alpha_m * np.cos(theta) - np.sin(theta))
    ax_i.plot(x1, xp1, color='blue', lw=2, label='Distribution Twiss')

    x2 = np.sqrt(eps_m * beta_r) * np.cos(theta)
    xp2 = -np.sqrt(eps_m / beta_r) * (alpha_r * np.cos(theta) - np.sin(theta))
    ax_i.plot(x2, xp2, color='red', lw=2, linestyle='--', label='Ring Twiss')

    ax_i.set_title(title)
    ax_i.set_xlabel(pos_label)
    ax_i.set_ylabel(ang_label)
    ax_i.axhline(0, color='black', lw=0.5, ls='--')
    ax_i.axvline(0, color='black', lw=0.5, ls='--')
    ax_i.set_xlim(-20, 20)
    ax_i.set_ylim(-20, 20)
    ax_i.grid(True, linestyle=':', alpha=0.5)
    ax_i.legend(fontsize='small')

plt.tight_layout()
plt.savefig(f'{folder}/injected_beam_before_after_dispersion.png')
plt.show()

# %%
fig, ax_long = plt.subplots(figsize=(8, 6))
sc = density_scatter(ax_long, zeta_full*1e3, delta_full*1e3, s=2)
fig.colorbar(sc, ax=ax_long, label='Relative Density')
ax_long.set_xlabel(r'$\zeta$ (mm)')
ax_long.set_ylabel(r'$\delta$ ($10^{-3}$)')
ax_long.set_title('Longitudinal Phase Space (Matched/Injected Beam)')
ax_long.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig(f'{folder}/injected_beam_longitudinal.png')
plt.show()

def compute_energy_scan(track_line, track_tw, df_subset, p0c_avg_mev, energy_range_mev=None):
    """Core of the energy scan, no plotting: scans reference energy vs.
    short (100-turn) survival efficiency for `track_line`. Matched
    coordinates are computed ONCE at p0c_avg (using track_tw's dispersion)
    and reused across the whole scan -- only `track_line.particle_ref` and
    each particle's own `delta` change per energy point. Restores
    track_line.particle_ref to p0c_avg afterward (RF cavities key off the
    line's particle_ref, not each particle's own p0c, so leaving this at
    the scan's last test energy would silently mis-set the RF bucket for
    anything tracked afterward). Returns (energy_range_mev, efficiency_results).
    """
    if energy_range_mev is None:
        energy_range_mev = np.linspace(2.5e3, 3.2e3, 100)

    p0c_avg = p0c_avg_mev * 1e6
    ref_particle_avg = xp.Particles(p0c=p0c_avg, mass0=xp.ELECTRON_MASS_EV)

    tw_row = track_tw.rows[marker_name]
    x_m, px_m, y_m, py_m, _, zeta_m = match_coordinates(
        df_subset, p0c_avg, ref_particle_avg,
        tw_row['dx'][0], tw_row['dpx'][0], tw_row['dy'][0], tw_row['dpy'][0],
        beam_disp_x=(dx*1e-3, ddx*1e-3), beam_disp_y=(dy*1e-3, ddy*1e-3),
        beam_optics_x=(bx, ax), beam_optics_y=(by, ay),
        ring_optics_x=(tw_row['betx'][0], tw_row['alfx'][0]),
        ring_optics_y=(tw_row['bety'][0], tw_row['alfy'][0]))

    efficiency_results = []
    for e_mev in energy_range_mev:
        p0c_test = e_mev * 1e6
        track_line.particle_ref = xp.Particles(p0c=p0c_test, mass0=xp.ELECTRON_MASS_EV)
        delta_test = (df_subset['p[MeV/c]'].values - e_mev) / e_mev

        p_test = xp.Particles(
            p0c=p0c_test, mass0=xp.ELECTRON_MASS_EV,
            x=x_m, px=px_m, y=y_m, py=py_m,
            delta=delta_test, zeta=zeta_m
        )

        track_line.track(p_test, num_turns=1000, with_progress=True)
        survived = np.sum(p_test.state > 0)
        efficiency = (survived / len(p_test.x)) * 100
        efficiency_results.append(efficiency)

    track_line.particle_ref = xp.Particles(p0c=p0c_avg, mass0=xp.ELECTRON_MASS_EV)
    return energy_range_mev, efficiency_results


def run_energy_scan(track_line, track_tw, mode_tag, df_subset, p0c_avg_mev,
                     design, config, phase, changes,
                     energy_range_mev=None):
    """Single-lattice energy scan: computes the scan via
    compute_energy_scan(), then plots/saves it and returns best_energy_mev.
    Used for the baseline (perfect) lattice, where the result actually
    determines the 'optimal' energy tracked everywhere else."""
    energy_range_mev, efficiency_results = compute_energy_scan(
        track_line, track_tw, df_subset, p0c_avg_mev, energy_range_mev)

    for e_mev, eff in zip(energy_range_mev, efficiency_results):
        print(f"[{mode_tag}] Energy: {e_mev:.3f} MeV | Efficiency: {eff:.1f}%")

    best_idx = np.argmax(efficiency_results)
    best_energy_mev = energy_range_mev[best_idx]

    folder3 = mf.results_dir(design, config, phase, changes=changes,
                              metric='InjectionEfficiency', sub=mode_tag)

    plt.figure(figsize=(10, 6))
    plt.plot(energy_range_mev, efficiency_results, 'o-', color='teal', linewidth=2)
    plt.axvline(best_energy_mev, color='red', linestyle='--',
                label=f'Optimal: {best_energy_mev:.3f} MeV')
    plt.axvline(p0c_avg_mev, color='blue', linestyle='--',
                label=f'Average: {p0c_avg_mev:.3f} MeV')
    plt.title(f'Injection Efficiency vs. Beam Energy ({mode_tag})', fontsize=14)
    plt.xlabel('Energy [MeV]', fontsize=12)
    plt.ylabel('Survival Efficiency [%]', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f'{folder3}/MPD_energy_scan.png')
    plt.show()

    print(f"\n[{mode_tag}] The best injection efficiency is at {best_energy_mev:.3f} MeV.")
    return best_energy_mev


best_energy_mev = run_energy_scan(ring, ring_tw, mode, df_subset, p0c_avg_mev,
                                   design, config, phase, changes)

# %%
compressor_params = {
    "RF_voltage": ring.element_dict['RFCav'].voltage,
    "energy_compressor_enabled": ENERGY_COMPRESSOR_ON,
    "R_56": R56 if ENERGY_COMPRESSOR_ON else None,
    "V_deb": Vdeb if ENERGY_COMPRESSOR_ON else None,
    "Phase_deb": Phasdeb if ENERGY_COMPRESSOR_ON else None,
}
folder3 = mf.results_dir(design, config, phase, changes=changes,
                          metric='InjectionEfficiency', sub=mode)
with open(f'{folder3}/CompressorParams.json', 'w') as f:
    json.dump(compressor_params, f, indent=4)

# %%
nominal_energy_mev = 2860.0

energies_to_track = {
    'average': p0c_avg_mev,
    'nominal': nominal_energy_mev,
    'optimal': best_energy_mev,
}

turns_to_plot = [0, 1, 2, 3]
trnplt = [1, 2000, 4000, 6000]

col_labels = [
    ("$x$ (mm)", "$x'$ (mrad)"),
    ("$y$ (mm)", "$y'$ (mrad)"),
    (r"$\zeta$ (mm)", r"$\delta$ ($10^{-3}$)"),
    ("$x$ (mm)", "$y$ (mm)")
]

def run_energy_diagnostics(track_line, track_tw, mode_tag, energies_to_track,
                            df_subset, rand_num, design, config, phase, changes):
    """Runs the same full tracking + diagnostic suite (injection tracking
    evolution, initial-turns phase-space grid, survival curve + lifetime,
    initial x-distribution) on `track_line`, using `track_tw`'s dispersion
    both for matching and for the dispersion-corrected scatter plot.
    Originally only ran on whichever `mode` JSON was loaded; factored out
    so it can also run on an in-memory misaligned line, same as corrected."""
    for label, e_mev in energies_to_track.items():
        print(f"\n=== [{mode_tag}] Tracking at {label} energy: {e_mev:.3f} MeV ===")

        p0c_reference = e_mev * 1e6
        ref_particle = xp.Particles(p0c=p0c_reference, mass0=xp.ELECTRON_MASS_EV)

        tw_row = track_tw.rows[marker_name]
        x_matched, px_matched, y_matched, py_matched, delta, zeta = match_coordinates(
            df_subset, p0c_reference, ref_particle,
            tw_row['dx'][0], tw_row['dpx'][0], tw_row['dy'][0], tw_row['dpy'][0],
            beam_disp_x=(dx*1e-3, ddx*1e-3), beam_disp_y=(dy*1e-3, ddy*1e-3),
            beam_optics_x=(bx, ax), beam_optics_y=(by, ay),
            ring_optics_x=(tw_row['betx'][0], tw_row['alfx'][0]),
            ring_optics_y=(tw_row['bety'][0], tw_row['alfy'][0]))

        particles = xp.Particles(
            p0c=p0c_reference, mass0=xp.ELECTRON_MASS_EV,
            x=x_matched, px=px_matched, y=y_matched, py=py_matched,
            zeta=zeta, delta=delta
        )

        n_track = len(particles.x)
        track_line.configure_radiation(model='quantum')
        track_line.track(particles, num_turns=7000, turn_by_turn_monitor=True, with_progress=True)
        data = track_line.record_last_track

        folder2 = mf.results_dir(design, config, phase, changes=changes,
                                  metric='InjectionEfficiency', sub=mode_tag,
                                  sub2=f'{label}_{int(e_mev)}MeV')

        fig, ax3 = plt.subplots(1, 3, figsize=(14, 4))
        fig.subplots_adjust(wspace=0.4)

        survival_count = np.sum(particles.state > 0)
        fig.suptitle(f'Particle Survival ({mode_tag}, {label}, {e_mev:.1f} MeV): {survival_count} / {n_track}')

        for i, (xl, yl) in enumerate(col_labels[:3]):
            ax3[i].set_xlabel(xl)
            ax3[i].set_ylabel(yl)

        for ind, turn in enumerate(trnplt):
            x_beta = 1000 * (data.x[:, turn] - tw_row['dx'][0] * data.delta[:, turn])
            px_beta = 1000 * (data.px[:, turn] - tw_row['dpx'][0] * data.delta[:, turn])
            ax3[0].scatter(x_beta, px_beta, s=2, color=f'C{ind}', label=f'Turn {turn}', alpha=0.6)
            ax3[1].scatter(1000 * data.y[:, turn], 1000 * data.py[:, turn], s=2, color=f'C{ind}', alpha=0.6)
            ax3[2].scatter(1000 * data.zeta[:, turn], 1000 * data.delta[:, turn], s=2, color=f'C{ind}', alpha=0.6)

        ax3[0].legend(fontsize='small')
        plt.savefig(f'{folder2}/injection_tracking_evolution_{rand_num}_{e_mev:.0f}MeV.png')
        plt.show()

        fig, axes = plt.subplots(4, 3, figsize=(15, 18))
        fig.subplots_adjust(hspace=0.4, wspace=0.35)

        for row, turn in enumerate(turns_to_plot):
            survived = np.sum(data.state[:, turn] > 0)

            axes[row, 0].scatter(data.x[:, turn]*1000, data.px[:, turn]*1000, s=2, color='C0', alpha=0.6)
            axes[row, 0].set_xlabel(col_labels[0][0])
            axes[row, 0].set_ylabel(f"Turn {turn}\n\n{col_labels[0][1]}")

            axes[row, 1].scatter(data.y[:, turn]*1000, data.py[:, turn]*1000, s=2, color='C1', alpha=0.6)
            axes[row, 1].set_xlabel(col_labels[1][0])
            axes[row, 1].set_ylabel(col_labels[1][1])
            axes[row, 1].set_title(f"Survivors: {survived}")

            axes[row, 2].scatter(data.zeta[:, turn]*1000, data.delta[:, turn]*1000, s=2, color='C2', alpha=0.6)
            axes[row, 2].set_xlabel(col_labels[2][0])
            axes[row, 2].set_ylabel(col_labels[2][1])

        fig.suptitle(f'Phase Space Evolution ({mode_tag}) at {label} energy ({e_mev:.2f} MeV)', fontsize=16, y=0.92)
        plt.savefig(f'{folder2}/initial_turns_{rand_num}_{e_mev:.0f}MeV.png')
        plt.show()

        survival_counts = np.sum(data.state > 0, axis=0)
        turns = np.arange(len(survival_counts))

        tau=mf.calculate_lifetime(survival_counts, track_line, ref_particle, fit_start_turn=500)
        print(f"Beam lifetime:{tau} Seconds")

        plt.figure(figsize=(10, 6))
        plt.plot(turns, survival_counts, color='firebrick', linewidth=2)
        plt.title(f'Particle Survival over {len(turns)} Turns ({mode_tag}, {label}, {e_mev:.1f} MeV)', fontsize=14)
        plt.xlabel('Turn Number', fontsize=12)
        plt.ylabel('Number of Surviving Particles', fontsize=12)
        plt.grid(True, which='both', linestyle=':', alpha=0.6)
        plt.ylim(min(survival_counts) - 5, survival_counts[0] + 5)
        plt.savefig(f'{folder2}/survival_vs_turns_{e_mev:.0f}MeV.png', bbox_inches='tight')
        plt.show()

        final_efficiency = (survival_counts[-1] / survival_counts[0]) * 100
        print(f"[{mode_tag}/{label}] Final Survival: {survival_counts[-1]} / {survival_counts[0]} "
              f"({final_efficiency:.2f}%)")

        c_light = 299792458
        T_rev0 = track_line.get_length() / (ref_particle.beta0[0] * c_light)
        with open(f'{folder2}/survival_curve.json', 'w') as f:
            json.dump({
                "label": label,
                "energy_mev": float(e_mev),
                "survival_counts": survival_counts.tolist(),
                "T_rev0": float(T_rev0),
            }, f, indent=4)

        plt.figure(figsize=(8, 6))
        plt.hist(data.x[:, 0] * 1000, bins=50, color='C0', edgecolor='black', alpha=0.7)
        plt.xlabel('$x$ (mm)')
        plt.ylabel('Number of Particles')
        plt.title(f'Initial x Distribution ({mode_tag}, {label}, {e_mev:.1f} MeV)')
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(f'{folder2}/initial_x_distribution_{e_mev:.0f}MeV.png')
        plt.show()


run_energy_diagnostics(ring, ring_tw, mode, energies_to_track,
                        df_subset, rand_num, design, config, phase, changes)

print("\nDone: tracked average, nominal, and optimal reference energies.")

# %%
# %% DA and MA from the injected beam
def beam_acceptance_scan(track_line, df_in, e_mev, mode_tag,
                         scales=(1.0, 1.5, 2.0, 3.0, 4.0), n_turns=2000,
                         radiation='mean', survival_target=0.99):
    """Track the matched injected beam, blown up by each factor in `scales`,
    from the injection marker, and record which particles survive.

    Each particle is labelled by its initial betatron amplitude in units of the
    injected beam's rms size (A_x, A_y) and its momentum deviation. Lost
    particles in the (A_x, A_y) plane show the DA as seen by the beam; lost
    particles in the (A_x, delta) plane show the MA. Scaling the beam up probes
    where the boundary is; the largest scale that still keeps `survival_target`
    of the beam is the injection margin.

    Scaling multiplies the betatron coordinates and the momentum deviation from
    the beam's mean by the same factor (i.e. a beam with s^2 the emittance and
    s times the energy spread).
    """
    p0c_ref = e_mev * 1e6
    ref_particle = xp.Particles(p0c=p0c_ref, mass0=xp.ELECTRON_MASS_EV)

    # Optics and closed orbit at injection, with the radiation model used for tracking
    track_line.configure_radiation(model='mean')
    tw_inj = track_line.twiss6d()
    track_line.configure_radiation(model=radiation)
    row = tw_inj.rows[marker_name]
    bx_r, ax_r, by_r, ay_r = row['betx'][0], row['alfx'][0], row['bety'][0], row['alfy'][0]
    Dx_r, Dpx_r, Dy_r, Dpy_r = row['dx'][0], row['dpx'][0], row['dy'][0], row['dpy'][0]
    gx_r, gy_r = (1 + ax_r**2) / bx_r, (1 + ay_r**2) / by_r

    # Matched betatron coordinates (ideal transfer line), ring dispersion added below
    xb, pxb, yb, pyb, delta, zeta = match_coordinates(
        df_in, p0c_ref, ref_particle, 0.0, 0.0, 0.0, 0.0,
        beam_disp_x=(dx*1e-3, ddx*1e-3), beam_disp_y=(dy*1e-3, ddy*1e-3),
        beam_optics_x=(bx, ax), beam_optics_y=(by, ay),
        ring_optics_x=(bx_r, ax_r), ring_optics_y=(by_r, ay_r))

    eps_x_m, eps_y_m = emittance_x * 1e-6, emittance_y * 1e-6   # [mm mrad] -> [m rad]
    d0 = np.mean(delta)

    folder_acc = mf.results_dir(design, config, phase, changes=changes,
                                metric='InjectionEfficiency', sub=mode_tag,
                                sub2='BeamAcceptance')
    summary, per_scale = [], []

    for s in scales:
        xs, pxs, ys, pys = s * xb, s * pxb, s * yb, s * pyb
        ds = d0 + s * (delta - d0)

        # initial amplitudes in units of the injected beam's rms size
        A_x = np.sqrt((gx_r * xs**2 + 2 * ax_r * xs * pxs + bx_r * pxs**2) / eps_x_m)
        A_y = np.sqrt((gy_r * ys**2 + 2 * ay_r * ys * pys + by_r * pys**2) / eps_y_m)

        p = xp.Particles(
            p0c=p0c_ref, mass0=xp.ELECTRON_MASS_EV,
            x=xs + Dx_r * ds + row['x'][0], px=pxs + Dpx_r * ds + row['px'][0],
            y=ys + Dy_r * ds + row['y'][0], py=pys + Dpy_r * ds + row['py'][0],
            zeta=zeta + row['zeta'][0], delta=ds)
        track_line.track(p, num_turns=n_turns, ele_start=marker_name)
        p.sort(by='particle_id', interleave_lost_particles=True)
        alive = np.asarray(p.state) > 0
        lost = ~alive

        surv = alive.mean()
        first_Ax = A_x[lost].min() if lost.any() else np.nan
        first_Ay = A_y[lost].min() if lost.any() else np.nan
        first_dd = np.abs(ds[lost] - d0).min() if lost.any() else np.nan
        print(f'[{mode_tag}] scale {s:.2f}: survival {surv*100:.1f}% '
              f'({lost.sum()} lost); smallest lost A_x={first_Ax:.2f}, '
              f'A_y={first_Ay:.2f} sigma, |delta-<delta>|={first_dd*100:.2f}%')
        summary.append({'scale': s, 'survival_%': 100 * surv, 'n_lost': int(lost.sum()),
                        'max_A_x': A_x.max(), 'max_A_y': A_y.max(),
                        'max_|ddelta|_%': 100 * np.abs(ds - d0).max(),
                        'first_lost_A_x': first_Ax, 'first_lost_A_y': first_Ay,
                        'first_lost_|ddelta|_%': 100 * first_dd})
        per_scale.append((s, A_x, A_y, ds, alive))

    df_acc = pd.DataFrame(summary)
    df_acc.to_csv(f'{folder_acc}/beam_acceptance_{e_mev:.0f}MeV.csv', index=False)

    # --- DA (A_x vs A_y) and MA (delta vs A_x) maps, one column per scale ---
    n = len(per_scale)
    fig, axs = plt.subplots(2, n, figsize=(3.6 * n, 7), squeeze=False)
    for j, (s, A_x, A_y, ds, alive) in enumerate(per_scale):
        a0, a1 = axs[0, j], axs[1, j]
        a0.scatter(A_x[alive], A_y[alive], s=2, color='grey', alpha=0.5, label='survived')
        a0.scatter(A_x[~alive], A_y[~alive], s=8, color='red', label='lost')
        a0.set_title(f'scale {s:g}: {alive.mean()*100:.1f}% survive', fontsize=10)
        a0.set_xlabel(r'$A_x$ [$\sigma_{x,inj}$]'); a0.grid(alpha=0.3)
        a1.scatter(ds[alive] * 100, A_x[alive], s=2, color='grey', alpha=0.5)
        a1.scatter(ds[~alive] * 100, A_x[~alive], s=8, color='red')
        a1.set_xlabel(r'$\delta$ [%]'); a1.grid(alpha=0.3)
    axs[0, 0].set_ylabel(r'$A_y$ [$\sigma_{y,inj}$]')
    axs[1, 0].set_ylabel(r'$A_x$ [$\sigma_{x,inj}$]')
    axs[0, 0].legend(fontsize='small', markerscale=2)
    fig.suptitle(f'DA (top) and MA (bottom) from the injected beam -- {mode_tag}, '
                 f'{e_mev:.1f} MeV, {n_turns} turns, radiation={radiation}')
    fig.tight_layout()
    fig.savefig(f'{folder_acc}/beam_DA_MA_maps_{e_mev:.0f}MeV.png', dpi=200)
    plt.show()

    # --- survival vs scale: the injection margin ---
    ok = df_acc[df_acc['survival_%'] >= 100 * survival_target]
    margin = ok.scale.max() if len(ok) else np.nan
    plt.figure(figsize=(7, 4.5))
    plt.plot(df_acc.scale, df_acc['survival_%'], 'o-', color='teal')
    plt.axhline(100 * survival_target, color='red', ls='--', lw=1,
                label=f'{survival_target*100:.0f}% target')
    plt.xlabel('beam blow-up factor'); plt.ylabel('survival [%]')
    plt.title(f'Injection margin = {margin:g}x ({mode_tag}, {e_mev:.1f} MeV)')
    plt.grid(alpha=0.3); plt.legend()
    plt.savefig(f'{folder_acc}/beam_margin_{e_mev:.0f}MeV.png', dpi=200, bbox_inches='tight')
    plt.show()

    print(df_acc.to_string(index=False))
    print(f'[{mode_tag}] injection margin: {margin:g}x the injected beam')
    return df_acc


df_acceptance = df_clean.sample(n=min(3000, len(df_clean)), random_state=rand_num)
acc_results = beam_acceptance_scan(ring, df_acceptance, best_energy_mev, mode)

if mode == 'perfect':
    import LatticeBuild.misalignments_corrections as mc
    from TuneDiagram.lib.TuneDiagram.tune_diagram import resonance_lines

    seeds = [100, 200, 300, 400, 500]
    misalignment_val = cfg.MISALIGN_SIGMA
    seed_energy_mev = best_energy_mev

    context_tracking = xo.ContextCpu(omp_num_threads=0)

    mc.insert_bpms_and_correctors(pdr, design, config)

    def prep_seed_line(base_line, seed, apply_correction):
        return mc.prepare_seed_line(base_line, seed, correct=apply_correction,
                                    sigma=misalignment_val, context=context_tracking)

    def track_seed_line(seed_line, e_mev):
        p0c_ref = e_mev * 1e6
        ref_particle = xp.Particles(p0c=p0c_ref, mass0=xp.ELECTRON_MASS_EV)

        seed_tw = seed_line.twiss6d()
        tw_row = seed_tw.rows[marker_name]

        x_m, px_m, y_m, py_m, delta_m, zeta_m = match_coordinates(
            df_subset, p0c_ref, ref_particle,
            tw_row['dx'][0], tw_row['dpx'][0], tw_row['dy'][0], tw_row['dpy'][0],
            beam_disp_x=(dx*1e-3, ddx*1e-3), beam_disp_y=(dy*1e-3, ddy*1e-3),
            beam_optics_x=(bx, ax), beam_optics_y=(by, ay),
            ring_optics_x=(tw_row['betx'][0], tw_row['alfx'][0]),
            ring_optics_y=(tw_row['bety'][0], tw_row['alfy'][0]))

        particles = xp.Particles(
            p0c=p0c_ref, mass0=xp.ELECTRON_MASS_EV,
            x=x_m, px=px_m, y=y_m, py=py_m, zeta=zeta_m, delta=delta_m
        )

        seed_line.configure_radiation(model='quantum')
        seed_line.track(particles, num_turns=7000,
                         turn_by_turn_monitor=True, with_progress=False)
        data_seed = seed_line.record_last_track
        survival_counts_seed = np.sum(data_seed.state > 0, axis=0)

        
        t0 = seed_tw.rows[marker_name]
        seed_line.configure_radiation(model='mean')
        return survival_counts_seed, t0['betx'][0], t0['alfx'][0], t0['bety'][0], t0['alfy'][0]

    '''misaligned_line = prep_seed_line(ring, seed=seeds[0], apply_correction=False)
    misaligned_tw = misaligned_line.twiss6d()
    run_energy_diagnostics(misaligned_line, misaligned_tw, 'misaligned_inmemory',
                            energies_to_track, df_subset, rand_num,
                            design, config, phase, changes)

    corrected_line = prep_seed_line(ring, seed=seeds[0], apply_correction=True)
    corrected_tw = corrected_line.twiss6d()
    run_energy_diagnostics(corrected_line, corrected_tw, 'corrected_inmemory',
                            energies_to_track, df_subset, rand_num,
                            design, config, phase, changes)'''

    folder_seeds = mf.results_dir(design, config, phase, changes=changes,
                                   metric='InjectionEfficiency', sub='SeedStudy')

    fig_surv_mis, ax_surv_mis = plt.subplots(figsize=(10, 6))
    fig_surv_cor, ax_surv_cor = plt.subplots(figsize=(10, 6))
    fig_ell_mis, axs_ell_mis = plt.subplots(1, 2, figsize=(13, 6))
    fig_ell_cor, axs_ell_cor = plt.subplots(1, 2, figsize=(13, 6))
    fig_escan_mis, ax_escan_mis = plt.subplots(figsize=(10, 6))
    fig_escan_cor, ax_escan_cor = plt.subplots(figsize=(10, 6))
    fig_tune_mis, ax_tune_mis = plt.subplots(figsize=(8, 8))
    fig_tune_cor, ax_tune_cor = plt.subplots(figsize=(8, 8))

    colors = plt.cm.viridis(np.linspace(0, 1, len(seeds)))
    theta = np.linspace(0, 2 * np.pi, 200)

    for apply_correction, ax_surv, axs_ell, ax_escan, ax_tune, tag in [
        (False, ax_surv_mis, axs_ell_mis, ax_escan_mis, ax_tune_mis, 'misaligned'),
        (True, ax_surv_cor, axs_ell_cor, ax_escan_cor, ax_tune_cor, 'corrected'),
    ]:
        for seed, c in zip(seeds, colors):
            print(f"\n=== Seed {seed} ({tag}) ===")
            seed_line = prep_seed_line(ring, seed, apply_correction)
            seed_tw_escan = seed_line.twiss6d()
            energy_range_seed, efficiency_seed = compute_energy_scan(
                seed_line, seed_tw_escan, df_subset, p0c_avg_mev)
            ax_escan.plot(energy_range_seed, efficiency_seed, color=c,
                          label=f'Seed {seed}')

            ax_tune.scatter(seed_tw_escan.qx, seed_tw_escan.qy, color=c,
                             s=90, edgecolors='black', linewidths=0.8,
                             label=f'Seed {seed} (Qx={seed_tw_escan.qx:.4f}, Qy={seed_tw_escan.qy:.4f})')

            survival_counts_seed, betx0_s, alfx0_s, bety0_s, alfy0_s = \
                track_seed_line(seed_line, seed_energy_mev)

            H_x_seed, Bmag_x_seed = betatron_mismatch(betx0_s, alfx0_s, betx0, alfx0)
            H_y_seed, Bmag_y_seed = betatron_mismatch(bety0_s, alfy0_s, bety0, alfy0)
            print(f"  Seed {seed} ({tag}) vs baseline ring: "
                  f"Bmag_x={Bmag_x_seed:.4f}, Bmag_y={Bmag_y_seed:.4f}")

            turns_seed = np.arange(len(survival_counts_seed))
            final_eff = 100 * survival_counts_seed[-1] / survival_counts_seed[0]
            ax_surv.plot(turns_seed, survival_counts_seed, color=c,
                         label=f'Seed {seed} ({final_eff:.1f}%)')

            for ax_e, beta0_s, alfa0_s, eps, Bmag_s in [
                (axs_ell[0], betx0_s, alfx0_s, emittance_x, Bmag_x_seed),
                (axs_ell[1], bety0_s, alfy0_s, emittance_y, Bmag_y_seed),
            ]:
                xs = np.sqrt(eps * beta0_s) * np.cos(theta)
                xps = -np.sqrt(eps / beta0_s) * (alfa0_s * np.cos(theta) - np.sin(theta))
                ax_e.plot(xs, xps, color=c, label=f'Seed {seed} (Bmag={Bmag_s:.2f})')

        ax_surv.set_title(f'Particle Survival over Turns -- {tag} seeds, '
                           f'{seed_energy_mev:.1f} MeV')
        ax_surv.set_xlabel('Turn Number')
        ax_surv.set_ylabel('Number of Surviving Particles')
        ax_surv.grid(True, which='both', linestyle=':', alpha=0.6)
        ax_surv.legend(fontsize='small')

        ax_escan.axvline(seed_energy_mev, color='red', linestyle='--',
                          label=f'Optimal (perfect machine): {seed_energy_mev:.3f} MeV')
        ax_escan.set_title(f'Injection Efficiency vs. Beam Energy -- {tag} seeds')
        ax_escan.set_xlabel('Energy [MeV]')
        ax_escan.set_ylabel('Survival Efficiency [%]')
        ax_escan.grid(True, alpha=0.3)
        ax_escan.legend(fontsize='small')

        qx_lim, qy_lim = ax_tune.get_xlim(), ax_tune.get_ylim()
        qx_center, qy_center = np.mean(qx_lim), np.mean(qy_lim)
        half_window = max(0.05, (qx_lim[1]-qx_lim[0])/2, (qy_lim[1]-qy_lim[0])/2)
        qx_range = (qx_center - half_window, qx_center + half_window)
        qy_range = (qy_center - half_window, qy_center + half_window)
        # periodicity=3 assumes three_fold_periodicity_long (per LatticeBuild) --
        # adjust if this ring uses a different lattice function
        res_lines = resonance_lines(qx_range, qy_range, orders=[1, 2, 3, 4], periodicity=3)
        res_lines.plot_resonance(figure_object=ax_tune.figure)
        ax_tune.set_xlim(qx_range)
        ax_tune.set_ylim(qy_range)

        ax_tune.set_title(f'Tune Diagram -- {tag} seeds')
        ax_tune.set_xlabel('Qx')
        ax_tune.set_ylabel('Qy')
        ax_tune.grid(True, linestyle=':', alpha=0.6)
        ax_tune.legend(fontsize='small')

        for ax_e, plane_name in zip(axs_ell, ['Horizontal', 'Vertical']):
            ax_e.axhline(0, color='black', lw=0.5, ls='--')
            ax_e.axvline(0, color='black', lw=0.5, ls='--')
            ax_e.set_title(f'{plane_name} ring ellipse -- {tag} seeds')
            ax_e.axis('equal')
            ax_e.grid(True, linestyle=':', alpha=0.6)
            ax_e.legend(fontsize='small')

    fig_surv_mis.tight_layout()
    fig_surv_mis.savefig(f'{folder_seeds}/survival_vs_turns_overlay_misaligned.png')
    fig_surv_cor.tight_layout()
    fig_surv_cor.savefig(f'{folder_seeds}/survival_vs_turns_overlay_corrected.png')
    fig_escan_mis.tight_layout()
    fig_escan_mis.savefig(f'{folder_seeds}/energy_scan_overlay_misaligned.png')
    fig_escan_cor.tight_layout()
    fig_escan_cor.savefig(f'{folder_seeds}/energy_scan_overlay_corrected.png')
    fig_tune_mis.tight_layout()
    fig_tune_mis.savefig(f'{folder_seeds}/tune_diagram_overlay_misaligned.png')
    fig_tune_cor.tight_layout()
    fig_tune_cor.savefig(f'{folder_seeds}/tune_diagram_overlay_corrected.png')
    fig_ell_mis.tight_layout()
    fig_ell_mis.savefig(f'{folder_seeds}/twiss_ellipse_overlay_misaligned.png')
    fig_ell_cor.tight_layout()
    fig_ell_cor.savefig(f'{folder_seeds}/twiss_ellipse_overlay_corrected.png')
    plt.show()

    print("\nSeed study done: overlaid survival-vs-turns and ring ellipses "
          "across misaligned and corrected realizations.")