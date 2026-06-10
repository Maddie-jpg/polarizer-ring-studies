# %%
import xtrack as xt
import xpart as xp
import xfields as xf 
import xobjects as xo
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import json
import os

# %%
design=int(os.environ.get('DESIGN',1))
config=int(os.environ.get('CONFIG',1))
mode=os.environ.get('MODE','misaligned')

# %%
pdr = xt.Environment.from_json(f'JSON Files/D{design}/C{config}/pdr_{mode}.json')
pdr.particle_ref.anomalous_magnetic_moment=0.001159652181

line=pdr.lines['ring']


line.configure_radiation('mean')


line.discard_tracker()
line.build_tracker()

tw = line.twiss(method='6d', radiation_integrals=True, eneloss_and_damping=True,
                   spin=True, polarization=True)

print(f"Global Coupling Strength (|C-|): {tw.c_minus:.5f}")
print(f"Horizontal Emittance (ex): {tw.eq_gemitt_x:.5e} m")
print(f"Vertical Emittance (ey): {tw.eq_gemitt_y:.5e} m")
print(f"Emittance Ratio (ey/ex): {tw.eq_gemitt_y / tw.eq_gemitt_x:.4f}")

print(tw.cols)
np.random.seed(0)
particles = xp.generate_matched_gaussian_bunch(
    line=line,
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

# Simulate bunch evolution with stochastic photon emission
line.configure_spin('auto')
line.configure_radiation(model='quantum')

# Enable parallelization
line.discard_tracker()
line.build_tracker(_context=xo.ContextCpu(omp_num_threads='auto'))

# Track
num_turns=10000
line.track(particles, num_turns=num_turns, turn_by_turn_monitor=True,
           with_progress=10)
mon = line.record_last_track


# Fit depolarization time
mask_alive = mon.state > 0
pol_x = mon.spin_x.sum(axis=0)/mask_alive.sum(axis=0)
pol_y = mon.spin_y.sum(axis=0)/mask_alive.sum(axis=0)
pol_z = mon.spin_z.sum(axis=0)/mask_alive.sum(axis=0)
pol = np.sqrt(pol_x**2 + pol_y**2 + pol_z**2)

i_start = 3 # Skip a few turns (small initial mismatch)
pol_to_fit = pol[i_start:]/pol[i_start]

turns = np.arange(len(pol_to_fit))
slope, intercept, r_value, p_value, std_err = linregress(turns, pol_to_fit)
# Calculate depolarization time
t_dep_turns = -1 / slope


# %%

# Plot polarization decay and fit
plt.figure()
plt.plot(pol_to_fit-1, label='Tracking')
plt.plot(turns, intercept*np.exp(-turns/t_dep_turns) - 1, label='Fit')
plt.ylabel(r'$P/P_0 - 1$')
plt.xlabel('Turn')
plt.subplots_adjust(left=.2)
plt.legend()

# Compute equilibrium polarization
p_inf = tw['spin_polarization_inf_no_depol']
print(p_inf)
t_pol_turns = tw['spin_t_pol_component_s']/tw.T_rev0

p_eq = p_inf * 1 / (1 + t_pol_turns/t_dep_turns)
print(p_eq)
plt.subplots_adjust(bottom=0.2)
plt.figtext(0.5, 0.04, f"Calculated Equilibrium Polarization ($P_{{eq}}$): {p_eq:.3f}", 
            ha="center", fontsize=11, 
            bbox=dict(facecolor='ghostwhite', alpha=0.8, boxstyle='round,pad=0.5'))

# Time constant in seconds
t_pol_seconds = tw['spin_t_pol_component_s']

# Time constant in turns
t_pol_turns = tw['spin_t_pol_component_s'] / tw.T_rev0

print(f"Sokolov-Ternov Build-up Time: {t_pol_seconds:.2f} seconds ({t_pol_turns:.1f} turns)")

plt.savefig(f'Results/D{design}/C{config}/{mode}/equilibrium_pol_{mode}.png')

# %%
print(tw.qx)
print(tw.qy)
print(tw.qs)

#%%
