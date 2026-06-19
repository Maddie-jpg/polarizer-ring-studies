#%%
import xtrack as xt
import xpart as xp
import xobjects as xo
import matplotlib.pyplot as plt
import numpy as np

#%%

#import lattice
pdr=xt.load('../External Files/circringwgl.seq', format='madx')
line=pdr.lines['circringwgl']

#%%
line.particle_ref = xt.Particles(kinetic_energy0=2.86e9, mass0 = xt.ELECTRON_MASS_EV)

tw=line.twiss()

# %%
#Beta functions
plt.figure(figsize=(12,8))
plt.plot(tw.s,tw.betx,color='blue',label=r'$\beta_x$')
plt.plot(tw.s,tw.bety,color='red',label=r'$\beta_y$')
plt.xlabel('S [m]')
plt.ylabel(r'$\beta$ [m]')
plt.legend()

# %%
#Position
plt.figure(figsize=(12,8))
plt.plot(tw.s,tw.x,color='blue',label=r'$x$')
plt.plot(tw.s,tw.y,color='red',label=r'$y$')
plt.xlabel('S [m]')
plt.ylabel('Position [m]')
plt.legend()
# %%
p0c_eV = 2859.999954 * 1e6  # Convert MeV/c to eV/c

line.insert_element(
    element=xt.LimitRect(min_x=-0.03, max_x=0.03, min_y=-0.015, max_y=0.015),
    name='global_aperture',
    at_s=0.0
)

nemitt_x = 7600 * 1e-9
nemitt_y = 7200 * 1e-9



twiss_initial = xt.TwissInit(
    element_name=line.element_names[0],  
    betx=9.989319917,
    bety=2.418837296,
    alfx=-0.0002540930971,
    alfy=7.379939621e-05
)

#%%
line.build_tracker(_context=xo.ContextCpu(omp_num_threads=200))
# Generate particle bunch
bunch = xp.build_particles(
    line=line,
    num_particles=500,
    nemitt_x=nemitt_x,
    nemitt_y=nemitt_y,
    sigma_z=1e-4,
    init=twiss_initial,
    particle_ref=line.particle_ref
)

# %%
# Initial phase space plot
initial_x  = bunch.x.copy() * 1e3            # m to mm
initial_px = bunch.px.copy() * p0c_eV/1e6    # dimensionless to MeV/c
initial_y  = bunch.y.copy() * 1e3            # m to mm
initial_py = bunch.py.copy() * p0c_eV/1e6
initial_z  = bunch.zeta.copy() * 1e3       # m to mm
initial_pz = bunch.delta.copy() * p0c_eV/1e6
#%%

num_turns = 10000

# When tracking with a monitor:
line.track(bunch, num_turns=num_turns, turn_by_turn_monitor=True, with_progress=5)
data=line.record_last_track

#%%
# Initial phase space plot
final_x  = bunch.x.copy() * 1e3            # m to mm
final_px = bunch.px.copy() * p0c_eV/1e6    # dimensionless to MeV/c
final_y  = bunch.y.copy() * 1e3            # m to mm
final_py = bunch.py.copy() * p0c_eV/1e6
final_z  = bunch.zeta.copy() * 1e3       # m to mm
final_pz = bunch.delta.copy() * p0c_eV/1e6

#%%
survival_counts = np.sum(data.state > 0, axis=0)
turns = np.arange(len(survival_counts))

# 2. Plotting
plt.figure(figsize=(10, 6))
plt.plot(turns, survival_counts, color='firebrick', linewidth=2)

plt.title(f'Particle Survival over {len(turns)} Turns', fontsize=14)
plt.xlabel('Turn Number', fontsize=12)
plt.ylabel('Number of Surviving Particles', fontsize=12)
plt.grid(True, which='both', linestyle=':', alpha=0.6)
plt.ylim(min(survival_counts)-5, survival_counts[0]+5)  # Set y-axis to start at 0
plt.legend()


# 3. Save the plot
plt.show()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))

ax1.scatter(initial_x, initial_px, color='blue', alpha=0.4, s=1.5, label='Initial')
ax1.scatter(final_x, final_px, color='red', alpha=0.5, s=1.5, label='Final (Turn 10k)')
ax1.set_xlabel('x [mm]')
ax1.set_ylabel('$P_x$ [MeV/c]')
ax1.set_title('Horizontal Phase Space ($x - P_x$)')
ax1.grid(True, linestyle=':', alpha=0.5)
ax1.legend(loc='upper right')

# Plot 2: Vertical Phase Space
ax2.scatter(initial_y, initial_py, color='blue', alpha=0.4, s=1.5, label='Initial')
ax2.scatter(final_y, final_py, color='red', alpha=0.5, s=1.5, label='Final (Turn 10k)')
ax2.set_xlabel('y [mm]')
ax2.set_ylabel('$P_y$ [MeV/c]')
ax2.set_title('Vertical Phase Space ($y - P_y$)')
ax2.grid(True, linestyle=':', alpha=0.5)
ax2.legend(loc='upper right')

# Plot 3: Longitudinal Phase Space
ax3.scatter(initial_z, initial_pz, color='blue', alpha=0.4, s=1.5, label='Initial')
ax3.scatter(final_z, final_pz, color='red', alpha=0.5, s=1.5, label='Final (Turn 10k)')
ax3.set_xlabel('$z$ [mm]')
ax3.set_ylabel('$\Delta P_z$ [MeV/c]')
ax3.set_title('Longitudinal Phase Space ($z - \Delta P_z$)')
ax3.grid(True, linestyle=':', alpha=0.5)
ax3.legend(loc='upper right')

#%%
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))

ax1.scatter(final_x, final_px, color='red', alpha=0.5, s=1.5, label='Final (Turn 10k)')
ax1.set_xlabel('x [mm]')
ax1.set_ylabel('$P_x$ [MeV/c]')
ax1.set_title('Horizontal Phase Space ($x - P_x$)')
ax1.grid(True, linestyle=':', alpha=0.5)
ax1.legend(loc='upper right')

# Plot 2: Vertical Phase Space

ax2.scatter(final_y, final_py, color='red', alpha=0.5, s=1.5, label='Final (Turn 10k)')
ax2.set_xlabel('y [mm]')
ax2.set_ylabel('$P_y$ [MeV/c]')
ax2.set_title('Vertical Phase Space ($y - P_y$)')
ax2.grid(True, linestyle=':', alpha=0.5)
ax2.legend(loc='upper right')

# Plot 3: Longitudinal Phase Space

ax3.scatter(final_z, final_pz, color='red', alpha=0.5, s=1.5, label='Final (Turn 10k)')
ax3.set_xlabel('$z$ [mm]')
ax3.set_ylabel('$\Delta P_z$ [MeV/c]')
ax3.set_title('Longitudinal Phase Space ($z - \Delta P_z$)')
ax3.grid(True, linestyle=':', alpha=0.5)
ax3.legend(loc='upper right')