#%%
# %%
import sys
import os

# Adds the parent directory to the search path
parent_dir = os.path.abspath('..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import xtrack as xt
import numpy as np
import matplotlib.pyplot  as plt
import matplotlib.patches as patches
import xpart as xp
import xobjects as xo
import math
from TuneDiagram.lib.TuneDiagram.tune_diagram import resonance_lines
from prettytable import PrettyTable
import xutil_DA_CC.xsuite_plot_functions as my_xpf
import xutil_DA_CC.xsuite_utilities as xutil
import constants
import my_functions as mf
from matplotlib.backends.backend_pdf import PdfPages
import tune_wp_scan as twps

# %%
design=int(os.environ.get('DESIGN',1))
config=int(os.environ.get('CONFIG',2))
mode=os.environ.get('MODE','perfect')


# %%


# %%
pdr= xt.Environment.from_json(f"JSON Files/D{design}/C{config}/pdr_{mode}.json")

ring=pdr.lines['ring']
print(ring.element_names)
period=pdr.lines['period']

variable_name = f"WP_D{design}"

current_wp = getattr(constants, variable_name)

# %%
E0 = constants.E0; VRF = constants.VRF

U0 = (0.88463e-31)*E0**4*(2.*np.pi)/(6*(2*pdr['N_cells_S']*pdr['l_bend'] + pdr['l_bendDS']) )

period_sliced = period.select()
period_sliced.cut_at_s( np.linspace(.05, period.get_length()-.05, int(period.get_length()/.05-.5)) )


ring.configure_radiation(model=None)
fRev = 1./(ring.twiss(method='4d').T_rev0)
fRF  = fRev*round(4.e8/fRev)  # at integer harmonics and close to 400 MHz

# %%
ring.configure_radiation(model='mean')
ring_tw=ring.twiss(method='6d', radiation_integrals=True, eneloss_and_damping=True,
                   spin=True, polarization=True )

# %%
print(ring.element_names)

new_results_folder=f'Results/D{design}/C{config}/{mode}'

if not os.path.exists(new_results_folder):
    os.makedirs(new_results_folder)
# %%
#%matplotlib widget
if mode in ['perfect','misaligned']:
    ring.survey().plot()
    #plt.savefig(f'Results/D{design}/C{config}/{mode}/ring_survey_{mode}.png')
else:
    mf.survey_plot(ring)
    #plt.tight_layout()
    #plt.savefig(f'Results/D{design}/C{config}/{mode}/ring_survey_{mode}.png')

# %%
print(ring.element_names)

# %%

ring_tw.plot(f'delta (zeta+0.00504) x y-0.001')
ring_tw.plot(f'x y-0.001')
plt.tight_layout()
#plt.savefig(f'Results/D{design}/C{config}/{mode}/ring_closed_orbit_{mode}.png')


# %%

ring_tw.plot()
plt.tight_layout()
plt.savefig(f'Results/D{design}/C{config}/{mode}/ring_twiss_{mode}.png')


def calc_damping_time_constant(m):
    num=2*E0*ring_tw.T_rev0
    den=ring_tw.partition_numbers[m]*U0
    tau=num/den
    return tau

# Initialize the table
brho = ring.particle_ref.p0c[0] / 299792458.0 
max_k0 = np.max(np.abs(ring_tw.k0l))
max_field_tesla = max_k0 * brho
# 1. Prepare your data in a list of lists
data = [
    ["Horizontal Chromaticity (dqx)", f"{ring_tw.dqx:.4f}", ""],
    ["Vertical Chromaticity (dqy)", f"{ring_tw.dqy:.4f}", ""],
    ["Max Betx", f"{max(ring_tw.betx):.2f}", "m"],
    ["Max Bety", f"{max(ring_tw.bety):.2f}", "m"],
    ["Harmonic Number", f"{fRF/fRev:10.5f}", ""],
    ["RF Frequency", f"{fRF*1e-6:8.5f}", "MHz"],
    ["Length of short drifts", f"{pdr['l_drift']:7.3f}", "m"],
    ["Circumference", f"{ring_tw.circumference:8.4f}", "m"],
    ["Revolution Time", f"{1e6*ring_tw.T_rev0:8.5f}", "us"],
    ["Energy Loss (Manual)", f"{U0*1e-3:10.2f}", "keV"],
    ["Energy Loss (Twiss)", f"{ring_tw.eneloss_turn:10.2f}", "keV"],
    ["Vertical Damping Time", f"{1/ring_tw.damping_constants_turns[1]:10.3f}", "turns"],
    ["Horizontal Tune (qx)", f"{ring_tw.qx:10.5f}", ""],
    ["Vertical Tune (qy)", f"{ring_tw.qy:10.5f}", ""],
    ["Equilibrium Polarization", f"{ring_tw.spin_polarization_eq:8.5f}", ""],
    ["Equilibrium Emittance x", f"{ring_tw.eq_gemitt_x * 1e9:.4f}", "nm"],
    ["Equilibrium Emittance y", f"{ring_tw.eq_gemitt_y * 1e9:.4f}", "nm"],
    ["Polarization Build-up Time", f"{(ring_tw.spin_t_pol_buildup_s)/60:10.2f}", "minutes"],
    ["Horiz. Damping Constant", f"{(1/ring_tw.damping_constants_s[0])*1e3:.5f}", "ms"],
    ["Vert. Damping Constant", f"{(1/ring_tw.damping_constants_s[1])*1e3:.5f}", "ms"],
    ["Long. Damping Constant", f"{(1/ring_tw.damping_constants_s[2])*1e3:.5f}", "ms"],
    ["Maximum bending field", f"{max_field_tesla:.4f}", "T"]
]

column_headers = ["Parameter", "Value", "Unit"]

betx0 = ring_tw.betx[0]
bety0 = ring_tw.bety[0]
alfx0 = ring_tw.alfx[0]
alfy0 = ring_tw.alfy[0]


print(f"Initial Beta X: {betx0:.4f} m")
print(f"Initial Alpha X: {alfx0:.4f}")
print(f"Initial Beta Y: {bety0:.4f} m")
print(f"Initial Alpha Y: {alfy0:.4f}")
dx_start = ring_tw.dx[0]
dpx_start = ring_tw.dpx[0]

print(f"Initial Periodic Dispersion x:  {dx_start:.6f} m")
print(f"Initial Periodic Dispersion px: {dpx_start:.6f}")

# To verify it is periodic, check the last element as well
dx_end = ring_tw.dx[-1]
print(f"End Periodic Dispersion x:      {dx_end:.6f} m")

# 2. Create the figure
fig, ax = plt.subplots(figsize=(8, 10)) # Adjust size to fit your rows
ax.axis('tight')
ax.axis('off')

# 3. Create the table
table = ax.table(
    cellText=data, 
    colLabels=column_headers, 
    cellLoc='center', 
    loc='center'
)

# 4. Optional: Styling
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5) # Scale width and height of cells

# Bold the headers
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold')

#plt.savefig(f'Results/D{design}/C{config}/{mode}/table_{mode}.png')


#Working point plot
resonance_orders = (1,2,3,4)
Qx_range = (15,16)
Qy_range = (15,16)
resonances = resonance_lines(Qx_range,Qy_range,resonance_orders,3)
fig, ax = plt.subplots(1, figsize=(8,8), alpha=0.3)
ax.plot(ring_tw.qx, ring_tw.qy,'o', 
                color='green', label='Current working point', markersize=5)
ax.legend()
resonances.plot_resonance(fig)
plt.tight_layout()
#plt.savefig(f'Results/D{design}/C{config}/{mode}/WP_{mode}.png')

print('2nd order chrom x', ring_tw.ddqx)
print('2nd order chrom y', ring_tw.ddqy)

# %%

#mf.SpuckParsAus( period_sliced.twiss(method='4d'), period,(0., 38.), (0., 12.), (0., 1.), .08,pdr, f"Results/D{design}/C{config}/{mode}/Standard.png" )

# %%
max_dp = 4e-2
deltas = np.arange(-max_dp,max_dp+2.5e-3,2.5e-3)

delta=[]
qx=[]
qy=[]
qx_shift=[]
qy_shift=[]
for d in deltas:
    ring.configure_radiation(model=None)
    try:
        tw=ring.twiss4d(delta0=d)
        qx.append(tw.qx)
        qy.append(tw.qy)
        qx_s=ring_tw.qx+ring_tw.dqx*d
        qx_shift.append(qx_s)
        qy_s=ring_tw.qy+ring_tw.dqy*d
        qy_shift.append(qy_s)
        delta.append(d)
    except:
        print('deltap of %1.2e not stable'%d)


nominal_tw = ring.twiss4d(delta0=0)

plt.figure(figsize=(8,4))
WP = (nominal_tw.qx,nominal_tw.qy)
Qx_range = (np.floor(WP[0])-0.05,np.floor(WP[0])+0.55)
Qy_range = (np.floor(WP[1])-0.05,np.floor(WP[1])+0.55)

 
plt.plot(delta, qx-np.floor(WP[0]), label='$Q_x$ (twiss)', color='tab:blue')
plt.plot(delta, qy-np.floor(WP[1]), label='$Q_y$ (twiss)', color='tab:red')

# Dotted/Dashed lines for the linear model
plt.plot(delta, qx_shift-np.floor(WP[0]), '--', label=f'$Q_x^{{linear}} = Q_{{x0}} + \\xi_x \\delta$', color='tab:green')
plt.plot(delta, qy_shift-np.floor(WP[1]), '--', label=f'$Q_y^{{linear}} = Q_{{y0}} + \\xi_y \\delta$', color='tab:pink')
plt.xlabel('Relative momentum deviation ')
plt.ylabel('Fractional tune')
plt.legend()
plt.grid(True)
plt.xlim(-max_dp*1.05,max_dp*1.05)
plt.savefig(f'Results/D{design}/C{config}/{mode}/momentum_deviation{current_wp}_{mode}.png')

nominal_tw = ring.twiss4d(delta0=0)
'''Qx_range = (nominal_tw.qx-0.005,tw.qx+0.005)
Qy_range = (nominal_tw.qy-0.1,tw.qy+0.1)'''
resonances = resonance_lines(Qx_range,Qy_range,resonance_orders,3)
fig, ax = plt.subplots(1, figsize=(8,8), alpha=0.3)

ax.plot(qx, qy, label='Chromatic Footprint', color='tab:blue')

# Plot the actual center (delta=0)
ax.plot(nominal_tw.qx,nominal_tw.qy, 'ro', label='Nominal ($\delta=0$)')
ax.legend()
resonances.plot_resonance(fig)
plt.tight_layout()
plt.savefig(f'Results/D{design}/C{config}/{mode}/momentum_dev_working_point{current_wp}_{mode}.png')

