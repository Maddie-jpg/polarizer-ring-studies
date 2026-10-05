# %%
import sys
import os

parent_dir = os.path.abspath('..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
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
xo.context_cpu.allow_no_prebuilt_kernel = True

# %%
design=int(os.environ.get('DESIGN',1))
config=int(os.environ.get('CONFIG',9))
mode=os.environ.get('MODE','perfect')
phase=int(os.environ.get('PHASE',90))
changes=os.environ.get('CHANGES',None)

ENERGY_COMPRESSOR_ON = os.environ.get('ENERGY_COMPRESSOR', 'false').strip().lower() not in ('0', 'false', 'off', 'no')

# %%
df = pd.read_csv('/home/mwatson/Documents/laughing-octo-bassoon/PositronBeam_2p86GeV/Beam_3GHzOption_2.86GeV_20260421.dat', sep=r'\s+')
print(list(df.columns))

# %%

def filter_beam_core(df_raw, n_sigma=3):
    cols = ['x[mm]', 'xp[mrad]', 'y[mm]', 'yp[mrad]', 't[mm/c]', 'p[MeV/c]']
    df_centered = df_raw[cols] - df_raw[cols].mean()
    
    cov_matrix = df_centered.cov()
    inv_cov_matrix = np.linalg.inv(cov_matrix.values)
    
    distances_sq = np.sum(np.dot(df_centered.values, inv_cov_matrix) * df_centered.values, axis=1)
    
    is_core = distances_sq <= (n_sigma ** 2)
    
    print(f"--> Filtering Beam Core ({n_sigma} sigma): Retained {np.sum(is_core)} / {len(df_raw)} particles.")
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

def evaluate_ecs_performance(params, df_raw, ring, ring_tw, p0c_ref, n_particles=500, num_turns=100, seed=74):
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

    x_matched = x_in + ring_tw.dx[0] * delta_in
    px_matched = px_in + ring_tw.dpx[0] * delta_in
    y_matched = y_in + ring_tw.dy[0] * delta_in
    py_matched = py_in + ring_tw.dpy[0] * delta_in

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

def run_multi_objective_scan(df_raw, ring, ring_tw, p0c_ref, n_samples=150):
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
            [R56, Vdeb, Phasdeb], df_raw, ring, ring_tw, p0c_ref
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

df_raw=filter_beam_core(df, n_sigma=8)

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

def filter_by_action(df, position, angle, n_sigma=3, max_iter=8):
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
        df_iter = df_iter[mask].copy()
    return df_iter

def filter_by_action_xy(df, n_sigma=3, max_iter=5):
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
# %%

df_clean = filter_by_action_xy(df, n_sigma=8)
print(f"Action filter: kept {len(df_clean)}/{len(df)} particles")

emittance_x=CalcEmittanceAuto(df_clean, 'x[mm]', 'xp[mrad]')
m_emittance_x=CalcEmittanceManual(df_clean, 'x[mm]', 'xp[mrad]')
print(f'Horizontal geometric emittance:{emittance_x} um, {m_emittance_x}um' )

emittance_y=CalcEmittanceAuto(df_clean, 'y[mm]', 'yp[mrad]')
m_emittance_y=CalcEmittanceManual(df_clean, 'y[mm]', 'yp[mrad]')
print(f'Vertical geometric emittance:{emittance_y} um, {m_emittance_y}um' )

ax, bx, gx, dx, ddx = get_twiss(df_clean, 'x[mm]', 'xp[mrad]')
ay, by, gy, dy, ddy = get_twiss(df_clean, 'y[mm]', 'yp[mrad]')
print(f"Beam Twiss: alpha_x={ax:.3f}, beta_x={bx:.3f} m, dx={dx:.3f} m,ddx={ddx:.3f} m")
print(f"Beam Twiss: alpha_y={ay:.3f}, beta_y={by:.3f} m, dy={dy:.3f} m,ddy={ddy:.3f} m")

p_array=df['p[MeV/c]']

p_avg = np.mean(p_array)
rms_spread = np.std(p_array)  
relative_spread = (rms_spread / p_avg) *100

print(f"RMS Momentum Spread: {relative_spread} %")

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

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

theta = np.linspace(0, 2 * np.pi, 200)
ellipse_x = np.sqrt(emittance_x * bx) * np.cos(theta)
ellipse_xp = -np.sqrt(emittance_x / bx) * (ax * np.cos(theta) - np.sin(theta))
axes[0, 0].plot(ellipse_x, ellipse_xp, color='hotpink', lw=2, label='RMS Ellipse')
axes[0, 0].legend(fontsize='small')

ellipse_y = np.sqrt(emittance_y * by) * np.cos(theta)
ellipse_yp = -np.sqrt(emittance_y / by) * (ay * np.cos(theta) - np.sin(theta))
axes[0, 1].plot(ellipse_y, ellipse_yp, color='hotpink', lw=2, label='RMS Ellipse')
axes[0, 1].legend(fontsize='small')

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
moment_iters = 5
history_x, Dt_x, Dpt_x = moment_based_twiss(df, 'x[mm]', 'xp[mrad]', iters=moment_iters)
history_y, Dt_y, Dpt_y = moment_based_twiss(df, 'y[mm]', 'yp[mrad]', iters=moment_iters)

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
              label=f'Action-based, n_sigma=8 (\u03b5={eps_action:.3f})')

    ax_i.set_title(f'{plane_label}: Moment-based vs Action-based Emittance')
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

# %%

plot_twiss_with_particles(df_clean, 'x[mm]', 'xp[mrad]', ax, bx, emittance_x)
plot_twiss_with_particles(df_clean, 'y[mm]', 'yp[mrad]', ay, by, emittance_y)

# %%

beam_results = {
    "x": {
        "alpha": float(ax),
        "beta": float(bx),
        "gamma": float(gx),
        "dispersion": float(dx),
        "dispersion_prime": float(ddx),
        "emittance_geo": float(emittance_x)
    },
    "y": {
        "alpha": float(ay),
        "beta": float(by),
        "gamma": float(gy),
        "dispersion": float(dy),
        "dispersion_prime": float(ddy),
        "emittance_geo": float(emittance_y)
    },
    "metadata": {
        "particle_type": "positron",
        "design_energy_mev": 2860.0,
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


if changes is not None:
    pdr= xt.Environment.from_json(f"JSON_Files/D{design}/C{config}/pdr_{mode}_{phase}_{changes}.json")
else:
    pdr= xt.Environment.from_json(f"JSON_Files/D{design}/C{config}/pdr_{mode}_{phase}.json")

    
ring=pdr.lines['ring']
marker_name = insert_marker_in_drift(pdr, ring, occurrence=3)
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

# %%
rand_num = 74
n_subset = 500
df_subset = df.sample(n=n_subset, random_state=rand_num)

p0c_avg_mev = df['p[MeV/c]'].mean()
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
    df, p0c_avg, ref_particle_avg,
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
delta_before = (df['p[MeV/c]'].values * 1e6 - p0c_avg) / p0c_avg
x_before = df['x[mm]'].values * 1e-3 - (dx * 1e-3) * delta_before
px_before = df['xp[mrad]'].values * 1e-3 - (ddx * 1e-3) * delta_before
y_before = df['y[mm]'].values * 1e-3 - (dy * 1e-3) * delta_before
py_before = df['yp[mrad]'].values * 1e-3 - (ddy * 1e-3) * delta_before
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

        track_line.track(p_test, num_turns=100)
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
        track_line.track(particles, num_turns=6100, turn_by_turn_monitor=True, with_progress=True)
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


df_acceptance = df.sample(n=min(3000, len(df)), random_state=rand_num)
acc_results = beam_acceptance_scan(ring, df_acceptance, best_energy_mev, mode)

if mode == 'perfect':
    import LatticeBuild.misalignments_corrections as mc
    from TuneDiagram.lib.TuneDiagram.tune_diagram import resonance_lines

    seeds = [100, 200, 300, 400, 500]
    misalignment_val = 0.25e-3
    seed_energy_mev = best_energy_mev

    context_tracking = xo.ContextCpu(omp_num_threads=0)

    if design == 1 and config == 1:
        mc.insert_BPMs_all_as_markers(pdr)
        mc.insert_correctors_var2(pdr)
    else:
        mc.insert_BPMs_all_as_markers(pdr)
        mc.insert_correctors(pdr)

    def prep_seed_line(base_line, seed, apply_correction):
        seed_line = base_line.copy()
        seed_line.configure_radiation(model='mean')
        seed_line.build_tracker(_context=context_tracking)
        seed_line = mc.misalignments(seed_line, misalignment_val, seed=seed)

        if apply_correction:
            tw = seed_line.twiss(method='6d', radiation_integrals=True,
                                  eneloss_and_damping=True)
            mc.misalignments_correctors(seed_line, misalignment_val, seed + 1)
            try:
                mc.orbit_correction(seed_line, tw, threading=False, seed=seed)
            except Exception as e:
                print(f"  [seed {seed}] orbit_correction(threading=False) raised: {e}")
                mc.orbit_correction(seed_line, tw, threading=True, seed=seed)

        return seed_line

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
        seed_line.track(particles, num_turns=6100,
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