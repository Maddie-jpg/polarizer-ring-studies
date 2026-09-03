# %%
#imports
import sys
import os

# Adds the parent directory to the search path
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
import my_functions as mf


# %%
design=int(os.environ.get('DESIGN',3))
config=int(os.environ.get('CONFIG',2))
mode=os.environ.get('MODE','perfect')
phase=int(os.environ.get('PHASE',90))
changes=os.environ.get('CHANGES',None)

# %%
#read in file
df = pd.read_csv('PositronBeam_2p86GeV/Beam_3GHzOption_2.86GeV_20260421.dat', sep=r'\s+')
print(list(df.columns))


# %%

def filter_beam_core(df_raw, n_sigma=4):
    """
    Filters the dataframe to retain only the core particles within a specified n_sigma ellipsoid.
    Uses coordinate columns: 'x[mm]', 'xp[mrad]', 'y[mm]', 'yp[mrad]', 't[mm/c]', 'p[MeV/c]'
    """
    cols = ['x[mm]', 'xp[mrad]', 'y[mm]', 'yp[mrad]', 't[mm/c]', 'p[MeV/c]']
    df_centered = df_raw[cols] - df_raw[cols].mean()
    
    # Calculate the covariance matrix and its inverse for the 6D ellipsoid
    cov_matrix = df_centered.cov()
    inv_cov_matrix = np.linalg.inv(cov_matrix.values)
    
    # Calculate Mahalanobis distance squared for each particle
    # This acts as a multi-dimensional sigma measure
    distances_sq = np.sum(np.dot(df_centered.values, inv_cov_matrix) * df_centered.values, axis=1)
    
    # Keep particles where distance is less than n_sigma^2
    is_core = distances_sq <= (n_sigma ** 2)
    
    print(f"--> Filtering Beam Core ({n_sigma} sigma): Retained {np.sum(is_core)} / {len(df_raw)} particles.")
    return df_raw[is_core].copy()

def evaluate_ecs_performance(params, df_raw, ring, ring_tw, p0c_ref, n_particles=500, num_turns=100, seed=74):
    """
    Evaluates a single combination of ECS parameters for both 
    RMS momentum spread and tracking survival efficiency.
    """
    R56, Vdeb, Phasdeb = params
    
    c = 299792458
    freq = 3e9
    k = 2 * np.pi * freq / c
    fixed_e_mev = p0c_ref / 1e6

    # Sample a subset for fast tracking (consistent with Cell 16)
    
    df_subset = df_raw.sample(n=n_particles, random_state=seed)
    
    # 1. Apply the ECS equations to the initial uncompressed coordinates
    z_i = df_subset['t[mm/c]'].values * 1e-3  # metres
    p_i = df_subset['p[MeV/c]'].values        # MeV/c
    Eref = np.mean(p_i)
    delta_i = p_i / Eref - 1

    z_f = z_i + R56 * delta_i
    delta_f = delta_i + Vdeb * np.sin(k * z_f + Phasdeb)
    p_f = Eref * (1 + delta_f)

    # Calculate relative RMS momentum spread (%)
    rms_spread = (np.std(p_f) / np.mean(p_f)) * 100

    # 2. Map coordinates to Xtrack Particles with dispersion matching
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

    # 3. Track through the ring to measure injection efficiency
    ring.track(p_test, num_turns=num_turns)
    survived = np.sum(p_test.state > 0)
    efficiency = (survived / n_particles) * 100

    return rms_spread, efficiency


def run_multi_objective_scan(df_raw, ring, ring_tw, p0c_ref, n_samples=150):
    """
    Performs a randomized parameter search to find the optimal trade-off space.
    """
    np.random.seed(42)
    
    # Define search bounds centered around your current best region
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
    
    # Sort first by highest injection efficiency, then by lowest momentum spread
    df_res = df_res.sort_values(by=['Efficiency_%', 'RMS_Spread_%'], ascending=[False, True])
    return df_res

# Run the scan
df_raw=filter_beam_core(df, n_sigma=4)


# %%
'''
scan_df = run_multi_objective_scan(df_raw, ring, ring_tw, p0c_reference, n_samples=150)

# Display the top 10 best configurations
print("\nTop 10 configurations (Highest Efficiency + Lowest Spread):")
display(scan_df.head(10))'''

# %%
df_copy = df_raw.copy()


Eref = np.mean(df['p[MeV/c]'])
R56= 0.352963
Vdeb= -0.048365
Phasdeb= 0.29229

V_int=abs(Vdeb)*Eref
print(f'{V_int} MeV')

# Relative momentum deviation
delta_i = df_copy['p[MeV/c]']/Eref - 1

# Convert z to meters
z_i = df_copy['t[mm/c]']*1e-3

# Compressor
z_f = z_i + R56*delta_i

# RF wave number
c = 299792458
freq = 3e9
k = 2*np.pi*freq/c

# RF kick
delta_f = delta_i + Vdeb*np.sin(k*z_f + Phasdeb)

# Final momentum
p_f = Eref*(1+delta_f)

# Store
df_copy['t[mm/c]'] = z_f
df_copy['delta_final'] = delta_f
df_copy['p[MeV/c]'] = p_f

main_bunch = df[df['t[mm/c]'] < 278540]
main_bunch_copy = df_copy[df_copy['t[mm/c]'] < 278.540]

plt.scatter(main_bunch['t[mm/c]'], main_bunch['p[MeV/c]'], s=0.2)
plt.show()

plt.scatter(main_bunch_copy['t[mm/c]'], main_bunch_copy['p[MeV/c]'], s=0.2)

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
    # Electron/Positron rest mass in MeV
    m0 = 0.510998 
    
    gamma_rel = E_total_mev / m0
    beta_rel = np.sqrt(1 - 1/gamma_rel**2)
    
    n_emittance = beta_rel * gamma_rel * emittance
    return n_emittance

def get_twiss(df, position, angle, p_col='p[MeV/c]', p0_mev=2860.0):
    pos = df[position].values
    ang = df[angle].values
    
    # Calculate emittance 
    eps = CalcEmittanceAuto(df, position, angle)
    
    cov_matrix = np.cov(pos, ang, ddof=0)
    
    beta = cov_matrix[0, 0] / eps
    alpha = -cov_matrix[0, 1] / eps
    gamma = (1 + alpha**2) / beta
    
    # Dispersion calculation
    delta = (df[p_col].values - p0_mev) / p0_mev
    dispersion_x = np.cov(pos, delta, ddof=0)[0, 1] / np.var(delta)
    disp_prime_x = np.cov(ang, delta, ddof=0)[0, 1] / np.var(delta)
    
    return alpha, beta, gamma, dispersion_x, disp_prime_x



def plot_twiss_ellipse(beta, alpha, beta2, alpha2, emittance,ax):
    gamma = (1 + alpha**2) / beta
    theta = np.linspace(0, 2*np.pi, 100)
    
    x1 = np.sqrt(emittance * beta) * np.cos(theta)
    xp1 = -np.sqrt(emittance / beta) * (alpha * np.cos(theta) - np.sin(theta))
    x2 = np.sqrt(emittance * beta2) * np.cos(theta)
    xp2 = -np.sqrt(emittance / beta2) * (alpha2 * np.cos(theta) - np.sin(theta))
    
    #ax.figure(figsize=(6,6))
    ax.plot(x1, xp1, label=f'Particle distribution',color='blue')#: α={alpha:.2f}, β={beta:.2f} m',color='blue')
    ax.plot(x2, xp2, label=f'Initial ring parameters',color='red')#: α={alpha2:.2f}, β={beta2:.2f} m',color='red')
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

    print(disp*delta)


    zeta_r_d=(1/(np.sqrt(beta_r)))*x_b
    zeta_prime_r_d=np.sqrt(beta_r)*xp_b+(alpha_r/np.sqrt(beta_r))*x_b

    ax.plot(zeta, zeta_prime, color='blue', label='Particle distribution')
    ax.plot(zeta_r, zeta_prime_r, color='red', label='Initial ring parameters')
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

    # Estimate density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    z = z / z.max()

    # Sort so densest points are plotted last (on top)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    sc = ax.scatter(x, y, c=z, s=s, cmap=cmap, **kwargs)
    return sc
# %%

emittance_x=CalcEmittanceAuto(df, 'x[mm]', 'xp[mrad]')
m_emittance_x=CalcEmittanceManual(df, 'x[mm]', 'xp[mrad]')
#nemitt_x=NormalisedEmittance(df['x[mm]'],df['xp[mrad]'],emittance_x)
#print(nemitt_x)
print(f'Horizontal geometric emittance:{emittance_x} um, {m_emittance_x}um' )

emittance_y=CalcEmittanceAuto(df, 'y[mm]', 'yp[mrad]')
m_emittance_y=CalcEmittanceManual(df, 'y[mm]', 'yp[mrad]')
print(f'Vertical geometric emittance:{emittance_y} um, {m_emittance_y}um' )
#nemitt_y=NormalisedEmittance(df['x[mm]'],df['xp[mrad]'],emittance_y)
#print(nemitt_y)

# Correct way to call it:
ax, bx, gx, dx, ddx = get_twiss(df, 'x[mm]', 'xp[mrad]')
# Correct way to call it:
ay, by, gy, dy, ddy = get_twiss(df, 'y[mm]', 'yp[mrad]')
print(f"Beam Twiss: alpha_x={ax:.3f}, beta_x={bx:.3f} m, dx={dx:.3f} m,ddx={ddx:.3f} m")
print(f"Beam Twiss: alpha_y={ay:.3f}, beta_y={by:.3f} m, dy={dx:.3f} m,ddy={ddx:.3f} m")

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

main_bunch = df[df['t[mm/c]'] < 278.540]
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
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

# Apply mask consistently
mask = df['t[mm/c]'] < 278.540

t = df.loc[mask, 't[mm/c]']
p = df.loc[mask, 'p[MeV/c]']

# Style
plt.style.use('ggplot')

# Figure layout — extra slim column (4) reserved for the colorbar
fig = plt.figure(figsize=(8,9))
gs = GridSpec(
    4, 6,
    width_ratios=[1, 1, 1, 0.9, 0.3, 0.08],  # last real col + gap + colorbar
    height_ratios=[0.8, 1, 1, 1], 
    hspace=0.08,
    wspace=0.15
)

# Axes
ax_main  = fig.add_subplot(gs[1:4, 0:3])
ax_top   = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
cax      = fig.add_subplot(gs[1:4, 4])

# =========================
# Main longitudinal phase space (density coloured)
# =========================
sc = density_scatter(ax_main, t, p, s=4, alpha=0.8)

ax_main.set_xlabel(r'$t$ [mm/c]')
ax_main.set_ylabel(r'$p$ [MeV/c]')
#ax_main.set_title('Longitudinal Phase Space')

fig.colorbar(sc, cax=cax, label='Relative Density')

# =========================
# Top histogram
# =========================
ax_top.hist(
    t,
    bins=100,
    color='steelblue',
    edgecolor='black'
)
ax_top.set_ylabel('Counts')
ax_top.tick_params(axis='x', labelbottom=False)

# =========================
# Right histogram
# =========================
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

plot_twiss_with_particles(df, 'x[mm]', 'xp[mrad]', ax, bx, emittance_x)
plot_twiss_with_particles(df, 'y[mm]', 'yp[mrad]', ay, by, emittance_y)

# %%

# Save twiss results
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
        "n_particles": len(df['ID'])
    }
}

with open(f'{folder}/TwissResults.json', 'w') as f:
    json.dump(beam_results, f, indent=4)


# %%


if changes is not None:
    pdr= xt.Environment.from_json(f"JSON_Files/D{design}/C{config}/pdr_{mode}_{phase}_{changes}.json")
else:
    pdr= xt.Environment.from_json(f"JSON_Files/D{design}/C{config}/pdr_{mode}_{phase}.json")
    
ring=pdr.lines['ring']
ring.element_dict['RFCav'].voltage = 20e6
ring.element_dict['RFCav_1'].voltage = 20e6

#ring twiss values
ring_tw=ring.twiss6d()
print(ring_tw.cols)
# Initial values
initial_twiss = ring_tw.rows[0]

#Horizontal
betx0 = initial_twiss['betx'][0]
alfx0 = initial_twiss['alfx'][0]
dx0   = initial_twiss['dx'][0]
ddx0   = initial_twiss['ddx'][0]
x     = initial_twiss['x'][0]
xp_v    = initial_twiss['px'][0]


#Vertical
bety0 = initial_twiss['bety'][0]
alfy0 = initial_twiss['alfy'][0]
dy0   = initial_twiss['dy'][0]
ddy0   = initial_twiss['ddy'][0]
y     = initial_twiss['y'][0]
yp    = initial_twiss['py'][0]

delta= initial_twiss['delta'][0]

plot_twiss_ellipse(bx,ax,betx0, alfx0, emittance_x,axes[0, 0])
plot_twiss_ellipse(by,ay,bety0, alfy0, emittance_y,axes[1, 0])

plot_twiss_ellipse_normalised(bx*1e-3,ax,betx0, alfx0,dx0,ddx0,delta, emittance_x,axes[0,1])
plot_twiss_ellipse_normalised(by*1e-3,ay,bety0, alfy0,dy0,ddy0,delta, emittance_y,axes[1, 1])

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

plot_twiss_ellipse(bx,ax,betx0, alfx0, emittance_x,axes[0, 0])
plot_twiss_ellipse(by,ay,bety0, alfy0, emittance_y,axes[1, 0])

plot_twiss_ellipse_normalised(bx,ax,betx0, alfx0,dx0,ddx0,delta, emittance_x,axes[0,1])
plot_twiss_ellipse_normalised(by,ay,bety0, alfy0,dy0,ddy0,delta, emittance_y,axes[1, 1])

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
# ============================================================
# ENERGY SCAN — run first, to find the optimal reference energy
# ============================================================
rand_num = 74
n_subset = 500
df_subset = df.sample(n=n_subset, random_state=rand_num)

p0c_avg_mev = df['p[MeV/c]'].mean()          # average measured beam momentum
p0c_avg = p0c_avg_mev * 1e6
ref_particle_avg = xp.Particles(p0c=p0c_avg, mass0=xp.ELECTRON_MASS_EV)


def match_coordinates(df_in, p0c_ref, ref_particle, dx, ddx, dy, ddy):
    """Apply dispersion-based matching to raw beam coordinates for a given
    reference momentum p0c_ref, returning arrays ready for xp.Particles().
    Centralised here so the energy scan and the full tracking loop below
    use the exact same matching logic instead of two separate copies."""
    delta = (df_in['p[MeV/c]'].values * 1e6 - p0c_ref) / p0c_ref
    x_matched  = df_in['x[mm]'].values  * 1e-3 + dx  * delta
    px_matched = df_in['xp[mrad]'].values * 1e-3 + ddx * delta
    y_matched  = df_in['y[mm]'].values  * 1e-3 + dy  * delta
    py_matched = df_in['yp[mrad]'].values * 1e-3 + ddy * delta
    t_mm = df_in['t[mm/c]'].values
    zeta = (np.mean(t_mm) - t_mm) * 1e-3 * ref_particle.beta0[0]
    return x_matched, px_matched, y_matched, py_matched, delta, zeta


x_m, px_m, y_m, py_m, _, zeta_m = match_coordinates(
    df_subset, p0c_avg, ref_particle_avg, dx, ddx, dy, ddy)

energy_range_mev = np.linspace(2.5e3, 3.2e3, 100)
efficiency_results = []

for e_mev in energy_range_mev:
    p0c_test = e_mev * 1e6
    ring.particle_ref = xp.Particles(p0c=p0c_test, mass0=xp.ELECTRON_MASS_EV)
    delta_test = (df_subset['p[MeV/c]'].values - e_mev) / e_mev

    p_test = xp.Particles(
        p0c=p0c_test, mass0=xp.ELECTRON_MASS_EV,
        x=x_m, px=px_m, y=y_m, py=py_m,
        delta=delta_test, zeta=zeta_m
    )

    ring.track(p_test, num_turns=100)
    survived = np.sum(p_test.state > 0)
    efficiency = (survived / len(p_test.x)) * 100
    efficiency_results.append(efficiency)

    print(f"Energy: {e_mev:.3f} MeV | Efficiency: {efficiency:.1f}%")

best_idx = np.argmax(efficiency_results)
best_energy_mev = energy_range_mev[best_idx]

folder3 = mf.results_dir(design, config, phase, changes=changes,
                          metric='InjectionEfficiency', sub=mode)

plt.figure(figsize=(10, 6))
plt.plot(energy_range_mev, efficiency_results, 'o-', color='teal', linewidth=2)
plt.axvline(best_energy_mev, color='red', linestyle='--',
            label=f'Optimal: {best_energy_mev:.3f} MeV')
plt.axvline(p0c_avg_mev, color='blue', linestyle='--',
            label=f'Average: {p0c_avg_mev:.3f} MeV')
plt.title('Injection Efficiency vs. Beam Energy', fontsize=14)
plt.xlabel('Energy [MeV]', fontsize=12)
plt.ylabel('Survival Efficiency [%]', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig(f'{folder3}/MPD_energy_scan.png')
plt.show()

print(f"\nThe best injection efficiency is at {best_energy_mev:.3f} MeV.")

# %%
# ============================================================
# COMPRESSOR PARAMETERS — computed once, independent of which
# reference energy is tracked below
# ============================================================
compressor_params = {
    "RF_voltage": ring.element_dict['RFCav'].voltage,
    "R_56": R56,
    "V_deb": Vdeb,
    "Phase_deb": Phasdeb,
}
with open(f'{folder3}/CompressorParams.json', 'w') as f:
    json.dump(compressor_params, f, indent=4)

# %%
# ============================================================
# FULL MULTI-TURN TRACKING — loop over average / nominal / optimal
# reference energies (nominal = design energy; optimal = result of
# the energy scan above)
# ============================================================
nominal_energy_mev = 2860.0   # design reference energy

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

for label, e_mev in energies_to_track.items():
    print(f"\n=== Tracking at {label} energy: {e_mev:.3f} MeV ===")

    p0c_reference = e_mev * 1e6
    ref_particle = xp.Particles(p0c=p0c_reference, mass0=xp.ELECTRON_MASS_EV)

    x_matched, px_matched, y_matched, py_matched, delta, zeta = match_coordinates(
        df_subset, p0c_reference, ref_particle, dx, ddx, dy, ddy)

    particles = xp.Particles(
        p0c=p0c_reference, mass0=xp.ELECTRON_MASS_EV,
        x=x_matched, px=px_matched, y=y_matched, py=py_matched,
        zeta=zeta, delta=delta
    )

    n_track = len(particles.x)
    ring.configure_radiation(model='quantum')
    ring.track(particles, num_turns=6100, turn_by_turn_monitor=True, with_progress=True)
    data = ring.record_last_track

    folder2 = mf.results_dir(design, config, phase, changes=changes,
                              metric='InjectionEfficiency', sub=mode,
                              sub2=f'{label}_{int(e_mev)}MeV')

    # --- phase-space snapshot at selected turns ---
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    fig.subplots_adjust(wspace=0.4)

    survival_count = np.sum(particles.state > 0)
    fig.suptitle(f'Particle Survival ({label}, {e_mev:.1f} MeV): {survival_count} / {n_track}')

    for i, (xl, yl) in enumerate(col_labels[:3]):
        ax[i].set_xlabel(xl)
        ax[i].set_ylabel(yl)

    for ind, turn in enumerate(trnplt):
        x_beta = 1000 * (data.x[:, turn] - ring_tw.dx[0] * data.delta[:, turn])
        px_beta = 1000 * (data.px[:, turn] - ring_tw.dpx[0] * data.delta[:, turn])
        ax[0].scatter(x_beta, px_beta, s=2, color=f'C{ind}', label=f'Turn {turn}', alpha=0.6)
        ax[1].scatter(1000 * data.y[:, turn], 1000 * data.py[:, turn], s=2, color=f'C{ind}', alpha=0.6)
        ax[2].scatter(1000 * data.zeta[:, turn], 1000 * data.delta[:, turn], s=2, color=f'C{ind}', alpha=0.6)

    ax[0].legend(fontsize='small')
    plt.savefig(f'{folder2}/injection_tracking_evolution_{rand_num}_{e_mev:.0f}MeV.png')
    plt.show()

    # --- early-turn phase space grid ---
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

    fig.suptitle(f'Phase Space Evolution at {label} energy ({e_mev:.2f} MeV)', fontsize=16, y=0.92)
    plt.savefig(f'{folder2}/initial_turns_{rand_num}_{e_mev:.0f}MeV.png')
    plt.show()

    # --- survival vs turn number ---
    survival_counts = np.sum(data.state > 0, axis=0)
    turns = np.arange(len(survival_counts))

    plt.figure(figsize=(10, 6))
    plt.plot(turns, survival_counts, color='firebrick', linewidth=2)
    plt.title(f'Particle Survival over {len(turns)} Turns ({label}, {e_mev:.1f} MeV)', fontsize=14)
    plt.xlabel('Turn Number', fontsize=12)
    plt.ylabel('Number of Surviving Particles', fontsize=12)
    plt.grid(True, which='both', linestyle=':', alpha=0.6)
    plt.ylim(min(survival_counts) - 5, survival_counts[0] + 5)
    plt.savefig(f'{folder2}/survival_vs_turns_{e_mev:.0f}MeV.png', bbox_inches='tight')
    plt.show()

    final_efficiency = (survival_counts[-1] / survival_counts[0]) * 100
    print(f"[{label}] Final Survival: {survival_counts[-1]} / {survival_counts[0]} "
          f"({final_efficiency:.2f}%)")

    # --- initial x distribution ---
    plt.figure(figsize=(8, 6))
    plt.hist(data.x[:, 0] * 1000, bins=50, color='C0', edgecolor='black', alpha=0.7)
    plt.xlabel('$x$ (mm)')
    plt.ylabel('Number of Particles')
    plt.title(f'Initial x Distribution ({label}, {e_mev:.1f} MeV)')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(f'{folder2}/initial_x_distribution_{e_mev:.0f}MeV.png')
    plt.show()

print("\nDone: tracked average, nominal, and optimal reference energies.")