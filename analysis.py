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
xo.context_cpu.allow_no_prebuilt_kernel = True

# %%
design=int(os.environ.get('DESIGN',3))
config=int(os.environ.get('CONFIG',2))
mode=os.environ.get('MODE','perfect')
phase=int(os.environ.get('PHASE',90))
changes=os.environ.get('CHANGES',None)


# %%
pdf_run=False

if pdf_run is True:
    pdf = PdfPages(f"Results/D{design}/C{config}/{mode}/AnalysisResults.pdf")

    _old_savefig = plt.savefig

    def _new_savefig(*args, **kwargs):
        pdf.savefig(plt.gcf())
        _old_savefig(*args, **kwargs)

    plt.savefig = _new_savefig

# %%
if changes is not None:
    pdr= xt.Environment.from_json(f"JSON Files/D{design}/C{config}/pdr_{mode}_{phase}_{changes}.json")
else:
    pdr= xt.Environment.from_json(f"JSON Files/D{design}/C{config}/pdr_{mode}_{phase}.json")

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

new_results_folder=f'Results/D{design}/C{config}/{phase}deg_PhaseAdvance/{mode}'

if not os.path.exists(new_results_folder):
    os.makedirs(new_results_folder)
# %%
#%matplotlib widget
if mode in ['perfect','misaligned']:
    ring.survey().plot()
    plt.savefig(f'{new_results_folder}/ring_survey_{mode}.png')
else:
    mf.survey_plot(ring)
    #plt.tight_layout()
    plt.savefig(f'{new_results_folder}/ring_survey_{mode}.png')

# %%
print(ring.element_names)

# %%

ring_tw.plot(f'delta (zeta+0.00504) x y-0.001')
ring_tw.plot(f'x y-0.001')
plt.tight_layout()
plt.savefig(f'{new_results_folder}/ring_closed_orbit_{mode}.png')


# %%

ring_tw.plot()
plt.tight_layout()
plt.savefig(f'{new_results_folder}/ring_twiss_{mode}.png')


# %% Straight-section zoom: optics and phase advances

def straight_section_window(tw, pad=2.0, cluster_gap=20.0):
    """s-range covering the first straight section, found from the
    doublet/triplet quads that bound it, padded by `pad` metres.
    cluster_gap: gap (m) between quad clusters above which a new
    straight section is assumed to start."""
    names = list(tw.name)
    s_marks = [tw['s', nn] for nn in names
               if nn.startswith(('QFDoub_', 'QDDoub_', 'QDTrip_', 'QFTripC_'))]
    if len(s_marks) == 0:
        raise ValueError("No doublet/triplet quads found -- check name prefixes")
    s_marks = np.sort(np.asarray(s_marks))
    breaks = np.where(np.diff(s_marks) > cluster_gap)[0]
    end_first = s_marks[breaks[0]] if len(breaks) else s_marks[-1]
    start_first = s_marks[0]
    return start_first - pad, end_first + pad


s_lo, s_hi = straight_section_window(ring_tw)
mask = (ring_tw.s >= s_lo) & (ring_tw.s <= s_hi)

# --- Plot 2: optics zoom in the straight section ---
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(ring_tw.s[mask], ring_tw.betx[mask], color='red', label=r'$\beta_x$')
ax1.plot(ring_tw.s[mask], ring_tw.bety[mask], color='blue', label=r'$\beta_y$')
ax1.set_xlabel('s (m)')
ax1.set_ylabel(r'$\beta_x$, $\beta_y$ [m]')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1b = ax1.twinx()
ax1b.plot(ring_tw.s[mask], ring_tw.dx[mask], 'k--', label='$D_x$')
ax1b.set_ylabel('$D_x$ [m]')
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax1b.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc='best')
ax1.set_xlim(s_lo, s_hi)
ax1.set_title(f'Straight section optics ({mode})')
plt.tight_layout()
plt.savefig(f'{new_results_folder}/straight_optics_{mode}.png')

# --- Plot 3: same zoom, optics + phase advances on a shared s-axis ---
mux0 = ring_tw.mux[mask][0]
muy0 = ring_tw.muy[mask][0]

fig, (axo, axm) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                height_ratios=(3, 2))
axo.plot(ring_tw.s[mask], ring_tw.betx[mask], color='red', label=r'$\beta_x$')
axo.plot(ring_tw.s[mask], ring_tw.bety[mask], color='blue', label=r'$\beta_y$')
axo.set_ylabel(r'$\beta_x$, $\beta_y$ [m]')
axo.grid(True, linestyle=':', alpha=0.6)
axob = axo.twinx()
axob.plot(ring_tw.s[mask], ring_tw.dx[mask], 'k--', label='$D_x$')
axob.set_ylabel('$D_x$ [m]')
h1, l1 = axo.get_legend_handles_labels()
h2, l2 = axob.get_legend_handles_labels()
axo.legend(h1 + h2, l1 + l2, loc='best')
axo.set_title(f'Straight section optics and phase advance ({mode})')

axm.plot(ring_tw.s[mask], ring_tw.mux[mask] - mux0, color='red', label=r'$\mu_x$')
axm.plot(ring_tw.s[mask], ring_tw.muy[mask] - muy0, color='blue', label=r'$\mu_y$')
axm.set_xlabel('s (m)')
axm.set_ylabel(r'$\Delta\mu$ from straight entrance [$2\pi$]')
axm.grid(True, linestyle=':', alpha=0.6)
axm.legend(loc='best')
axo.set_xlim(s_lo, s_hi)
plt.tight_layout()
plt.savefig(f'{new_results_folder}/straight_phase_advance_{mode}.png')


def calc_damping_time_constant(m):
    num=2*E0*ring_tw.T_rev0
    den=ring_tw.partition_numbers[m]*U0
    tau=num/den
    return tau

# Initialize the table
brho = ring.particle_ref.p0c[0] / 299792458.0 
_mask = ring_tw.length > 0
max_k0 = np.max(np.abs(ring_tw.k0l[_mask]) / ring_tw.length[_mask])
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

plt.savefig(f'{new_results_folder}/table_{mode}.png')


def integer_tune_ranges(qx, qy, half_width=1):
    """
    Range of half_width on each side of the nearest integer to each tune.
    e.g. qx=15.38 -> round to 15 -> (13.5, 16.5) with half_width=1.5,
    or (14, 16) with half_width=1.0.
    """
    nx = round(float(qx) * 2) / 2
    ny = round(float(qy) * 2) / 2
    return (nx - half_width, nx + half_width), (ny - half_width, ny + half_width)


#Working point plot
resonance_orders = (1,2,3,4)
Qx_range, Qy_range = integer_tune_ranges(ring_tw.qx, ring_tw.qy)
resonances = resonance_lines(Qx_range,Qy_range,resonance_orders,3)
fig, ax = plt.subplots(1, figsize=(8,8), alpha=0.3)
ax.plot(ring_tw.qx, ring_tw.qy,'o', 
                color='green', label='Current working point', markersize=5)
ax.legend()
resonances.plot_resonance(fig)
ax.set_xlim(Qx_range)
ax.set_ylim(Qy_range)
plt.tight_layout()
plt.savefig(f'{new_results_folder}/WP_{mode}.png')

print('2nd order chrom x', ring_tw.ddqx)
print('2nd order chrom y', ring_tw.ddqy)

# %%

mf.SpuckParsAus( ring, period_sliced.twiss(method='4d'), period,(0., 38.), (0., 12.), (0., 1.), .08,pdr, f"{new_results_folder}/Standard.png" )

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
    except Exception:
        print('deltap of %1.2e not stable'%d)


nominal_tw = ring.twiss4d(delta0=0)

plt.figure(figsize=(8,4))
WP = (nominal_tw.qx,nominal_tw.qy)

 
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
plt.savefig(f'{new_results_folder}/momentum_deviation{current_wp}_{mode}.png')



nominal_tw = ring.twiss4d(delta0=0)
'''Qx_range = (nominal_tw.qx-0.005,tw.qx+0.005)
Qy_range = (nominal_tw.qy-0.1,tw.qy+0.1)'''
Qx_range, Qy_range = integer_tune_ranges(ring_tw.qx, ring_tw.qy)
resonances = resonance_lines(Qx_range,Qy_range,resonance_orders,3)
fig, ax = plt.subplots(1, figsize=(8,8), alpha=0.3)

ax.plot(qx, qy, label='Chromatic Footprint', color='tab:blue')

# Plot the actual center (delta=0)
ax.plot(nominal_tw.qx,nominal_tw.qy, 'ro', label='Nominal ($\delta=0$)')
ax.legend()
resonances.plot_resonance(fig)
ax.set_xlim(Qx_range)
ax.set_ylim(Qy_range)
plt.tight_layout()
plt.savefig(f'{new_results_folder}/momentum_dev_working_point{current_wp}_{mode}.png')



# %%
# r_max is capped well below the retry loop's old 16 sigma: at that large an
# amplitude, particles are plausibly crossing real resonance lines (visible
# on this same plot) and undergoing genuine chaotic tune jumps, or the FFT
# peak-picking in get_footprint is just noisy once the motion is chaotic.
# Either way that's not representative of the actual beam and it dominates
# the plot's scale, squeezing out the near-WP detail that's actually
# actionable. 6 sigma is a more standard footprint range.
r_max = 8.0
fp0 = None
nemitt_x = 1.3e-3
nemitt_y = 1.2e-3 

while r_max >= 1:
    try:
        n_r = int(r_max * 2)  # denser radial sampling than r_max alone gave
        fp0 = ring.get_footprint(
            nemitt_x=nemitt_x, 
            nemitt_y=nemitt_y, 
            r_range=(0.1, r_max), 
            n_r=n_r,
            n_theta=20,  # default is 10 -- too coarse, produces a jagged,
                        # self-crossing connect-the-dots look rather than a
                        # smooth fan when Footprint.plot() draws the grid
            freeze_longitudinal=True,   # avoid synchrotron-motion smearing the
                                        # FFT tune peaks -- RF+radiation are on
        )
        print(f"Successfully generated footprint for r_max = {r_max}")
        break 
    except AssertionError:
        print(f"Particles lost at r_max = {r_max}. Reducing amplitude...")
        r_max -= 1.0

if fp0 is not None:
    q_offset_x = np.floor(WP[0])
    q_offset_y = np.floor(WP[1])

    # get_footprint's internal FFT uses rfftfreq, which only resolves
    # frequencies in [0, 0.5] (real-signal Nyquist limit) -- if the true
    # fractional tune is above 0.5 it aliases and this offset reconstruction
    # would silently be wrong. Guard against that rather than fail silently.
    frac_x_max, frac_y_max = np.nanmax(fp0.qx), np.nanmax(fp0.qy)
    if frac_x_max > 0.5 or frac_y_max > 0.5:
        print(f"WARNING: footprint fractional tune near/above 0.5 "
              f"(max frac Qx={frac_x_max:.3f}, Qy={frac_y_max:.3f}) -- "
              f"FFT peak detection may be aliased; treat this footprint "
              f"with caution.")

    # get_footprint returns fractional tunes -> shift to absolute.
    fp0.qx += q_offset_x
    fp0.qy += q_offset_y

    # NOTE: qx/qy (from twiss4d) and nominal_tw.qx/qy are ALREADY absolute
    # tunes -- do not add the integer offset to them again.
    # min_span widened from the default 0.02 -- with a tight 6-sigma
    # footprint and a narrow chromatic sweep, the data-driven window alone
    # was too small to show any resonance context (previous plot had only
    # one or two lines barely clipping the edge).
    Qx_range, Qy_range = integer_tune_ranges(ring_tw.qx, ring_tw.qy,half_width=0.25)

    fig, ax = plt.subplots(figsize=(9, 9))

    # Layer 1 (back): the classic systematic (red) / non-systematic (blue)
    # resonance grid -- same scheme used elsewhere in this file via
    # resonance_lines().plot_resonance(), but reimplemented with tunable
    # alpha/linewidth so it's actually visible here rather than the
    # library's fixed alpha=0.3.
    mf.plot_resonance_grid_red_blue(
        ax, Qx_range, Qy_range, resonance_orders, 3,
        alpha=0.6, lw_systematic=2.2, lw_nonsystematic=1.1)

    # Layer 1b: proximity-to-WP danger tiers, back on top of the classic
    # grid -- recolored (black/gray) so "close to WP" doesn't collide with
    # "systematic" (red) from the layer above; those are two different
    # pieces of information (structural type vs. distance to WP) and
    # deserve visually distinct colors.
    '''mf.plot_dangerous_resonances(
        ring, ring_tw.qx, ring_tw.qy, max_order=(1, 2, 3, 4, 5),
        ax=ax, qx_range=Qx_range, qy_range=Qy_range,
        legend_tiers=(1, 2), tier_colors={1: 'black', 2: 'dimgray'},
        draw_background_grid=False)'''

    # Layer 2 (middle): amplitude footprint -- the widest-reaching data on
    # the plot, so give it a muted, translucent color that reads as a
    # "region" rather than competing with the resonance lines for attention.
    fp0.plot(ax=ax, color='tab:green', alpha=0.55,
             label=f'Amplitude Footprint ({r_max}$\\sigma$)')

    # Layer 3 (front): chromatic footprint -- a single clean curve, the
    # most important "how far do we move" indicator, drawn last so it
    # stays visually on top.
    ax.plot(qx, qy, '.-', lw=2, color='tab:purple', ms=4,
            label=f'Chromatic shift ($\\delta$: {min(delta):.1e} to {max(delta):.1e})')

    # Nominal WP marker -- distinct from the Tier-1 red resonance lines,
    # so use a black star rather than red.
    ax.plot(nominal_tw.qx, nominal_tw.qy, 'ro',
            label='Nominal ($\\delta=0$)')

    ax.set_xlim(Qx_range)
    ax.set_ylim(Qy_range)
    ax.set_xlabel('$Q_x$')
    ax.set_ylabel('$Q_y$')
    ax.set_title(f'Working point overview for {mode} lattice')
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(f'{new_results_folder}/momentum_dev_working_point_{mode}.png', dpi=200)
    plt.show()
else:
    print("Could not generate a stable footprint even at 1 sigma.")

# %%
tt=ring.get_table()
print(tt)

# %%


try:
   
    res_df = mf.analyse_verdier_resonances_from_line(
        ring, max_order=5, tune_tolerance=0.02
    )

    if res_df is not None and not res_df.empty:
        print("\n=== VERDIER RESONANCE FILTER RESULTS ===")
        print(res_df.to_string(index=False))
    else:
        print("\nNo dangerous resonance lines found within the specified tune tolerance.")
except NameError:
    print(
        "Please check the name of your Xtrack Line variable inside your workspace cell."
    )


mf.plot_dangerous_resonances(ring, ring_tw.qx, ring_tw.qy, max_order=(1,2,3,4,5), tune_range=0.1)
plt.savefig(f'{new_results_folder}/dangerous_resonances_{mode}.png')

# %%

line=ring
line.config.XTRACK_USE_EXACT_DRIFTS = True
xutil.set_integrator (line)
particle = 'positron'
operation_mode = 'z'
modes = {'z': 45.6e9, # in eV 
        'w': 80e9, # in eV
        'h': 120e9, # in eV 
        't': 182.5e9 # in eV
        }
energy = modes[operation_mode]
parameters = xutil.log_parameters (None, operation_mode, particle_type=particle, modes=modes)
# ## Choose a context
context = xo.ContextCpu()         # For CPU
context_tracking = xo.ContextCpu(omp_num_threads=0) # For CPU with activate multi-core CPU parallelization


# ## Transfer lattice on context and compile tracking code
# line.build_tracker(_context=context)

# line.configure_radiation(model='mean')
# line.compensate_radiation_energy_loss()

tw = line.twiss(eneloss_and_damping=True)

#xutil.update_reference_parameters_from_line (line, parameters, BS_scale_factor=0, update_type='all', max_bb_param=0)
# xutil.correct_parameters_conflicts(parameters, update_study_parameters_from_reference=False)

gamma0 = ring.particle_ref.gamma0[0]
beta0 = ring.particle_ref.beta0[0]

n_emittancex = 1.354177116369456e-6 * gamma0 * beta0
n_emittancey = 1.420755089827341e-6 * gamma0 * beta0

parameters['study_parameters'] = {
    'ini_cond_type' : 'grid_DA', # grid_DA, grid_MA, distribution_matched, distribution_injected
    'output_dir' : 'out',
    'number_of_turns' : 2500,
    'number_of_particles' : 1000, 
    'inv1': 0, # np.arange(2)+1,
    'inv2': 0, # np.arange(2,2+3)+1,
    'start_element' : 'QD1_R1', # 'ca1.1','ip' #'rf400'
    'ini_cond_nemittance_x':n_emittancex,
    'ini_cond_nemittance_y': n_emittancey,
    'ini_cond_bunch_length': 4.8e-3,
    'ini_cond_energy_spread': 2e-3,
    'ini_cond_energy_offset': None,
    'new_closed_orbit': None, # {'x': -x_co_inj_marker, 'px': None, 'y': None, 'py': None, 'zeta': None, 'delta': None}
    'covariance_dispertion_free': False
}




## Initial conditions
#particles = xutil.generate_particle_distribution (line, parameters['study_parameters'], beambeam_strength_used=1, radiation_off=True)
particles, grid_details = xutil.generate_particle_grid (line, parameters['study_parameters'])

line.discard_tracker()
line.build_tracker(_context=context_tracking)


# %%
line.discard_tracker()
## Tracking studies
line.configure_radiation(model='mean')


## Change context for multy CPU for tracking

line.build_tracker(_context=context_tracking)


# Use tracking
line.track(particles, num_turns=parameters['study_parameters']['number_of_turns'], turn_by_turn_monitor=True, time=True, with_progress=10) #, freeze_longitudinal=True
particles.sort(interleave_lost_particles=True)

# dic_particles_all = xutil.tracking_data_process (tracking_data=line.record_last_track, 
#                             monitor_twiss=ring_tw.rows[0], 
#                             norm_emit_x=parameters['reference_parameters']['normalized_emittance_x'],
#                             norm_emit_y=parameters['reference_parameters']['normalized_emittance_y'],
#                             sigma_z=parameters['reference_parameters']['bunch_length'], 
#                             sigma_delta=parameters['reference_parameters']['energy_spread'],
#                             particle_id_to_use='all')
# xutil.save_dict_to_h5(f'{output_dir}/dic_particles_all.h5', dic_particles_all)


# liftime = my_xpf.particle_population_vs_turns (dic_particles_all['state'], T_rev0=tw.T_rev0)
# my_xpf.phase_space_evolution_difference (dic_particles_all['zeta'][:,0], dic_particles_all['delta'][:,0], dic_particles_all['zeta'][:,-1], dic_particles_all['delta'][:,-1], x_axis_label=r'$z~[m]$', y_axis_label=r'$\delta$')
# my_xpf.phase_space_evolution_difference (dic_particles_all['x'][:,0], dic_particles_all['px'][:,0], dic_particles_all['x'][:,-1], dic_particles_all['px'][:,-1], x_axis_label=r'$x~[m]$', y_axis_label=r'$px$')
# my_xpf.emittances_vs_turns (dic_particles_all['emit_x'], dic_particles_all['emit_y'], dic_particles_all['emit_zeta'], reference_emit_x=ref_param['emittance_x'], reference_emit_y=ref_param['emittance_y'], reference_emit_z=ref_param['energy_spread']*ref_param['bunch_length'], log_y_axis=True)
# my_xpf.coordinates_vs_turns (dic_particles_all['x_sigma'], dic_particles_all['px_sigma'], dic_particles_all['at_turn'], r'$x~[\sigma_x]$', r'$p_x~[\sigma_x]$', particle_id_list=dic_particles_all['particle_id_list'], full_init_cond=study_param['number_of_particles'])
# my_xpf.MA_vs_turns(particles, grid_details['num_r_y_points'], grid_details['num_delta'], grid_details['x_normalized'], grid_details['y_normalized'], grid_details['delta_init'])
x_DA,y_DA,_,_=my_xpf.DA_vs_turns(particles, grid_details['num_r_y_points'], grid_details['num_theta_x_points'], grid_details['x_normalized'], grid_details['y_normalized'], grid_details['delta_init'],delta_plots=True)

ax = plt.gca()


plt.savefig(f"{new_results_folder}/DA_plot_{mode}_WP{current_wp}_full.png", dpi=300, bbox_inches='tight')

ax.relim(); ax.autoscale_view()          
(x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
ax.set_xlim(-15, 15)
ax.set_ylim(None, 12)
plt.savefig(f"{new_results_folder}/DA_plot_{mode}_WP{current_wp}_zoom.png", dpi=300, bbox_inches='tight')

# %%
#%matplotlib widget
'''
line=ring
line.config.XTRACK_USE_EXACT_DRIFTS = True
xutil.set_integrator (line)
particle = 'positron'
operation_mode = 'z'
modes = {'z': 45.6e9, # in eV 
        'w': 80e9, # in eV
        'h': 120e9, # in eV 
        't': 182.5e9 # in eV
        }
energy = modes[operation_mode]



context = xo.ContextCpu()         # For CPU
context_tracking = xo.ContextCpu(omp_num_threads=0) # For CPU with activate multi-core CPU 
# parameters = xutil.log_parameters (None, operation_mode, particle_type=particle, modes=modes)
parameters = {}
line.configure_radiation(model=None)
parameters['study_parameters'] = {
    'ini_cond_type' : 'grid_DA', # grid_DA, grid_MA, distribution_matched, distribution_injected
    'output_dir' : 'out',
    'number_of_turns' : 2500,
    'number_of_particles' : 1000, 
    'inv1': 0, # np.arange(2)+1,
    'inv2': 0, # np.arange(2,2+3)+1,
    'start_element' : 'QD1_R1', # 'ca1.1','ip' #'rf400'
    'ini_cond_nemittance_x':n_emittancex,
    'ini_cond_nemittance_y': n_emittancey,
    'ini_cond_bunch_length': 4.8e-3,
    'ini_cond_energy_spread': 2e-3,
    'ini_cond_energy_offset': None,
    'new_closed_orbit': None, # {'x': -x_co_inj_marker, 'px': None, 'y': None, 'py': None, 'zeta': None, 'delta': None}
    'covariance_dispertion_free': False
}


particles, grid_details = xutil.generate_particle_grid (line, parameters['study_parameters'], min_r_y=0, max_r_y=4, num_r_y_points=50, min_theta_x=0, max_theta_x=np.pi/2, 
                            num_theta_x_points=50, cartesian_polar='polar')
print(grid_details)

## Tracking studies
line.configure_radiation(model='mean')


## Change context for multy CPU for tracking
line.discard_tracker()
line.build_tracker(_context=context_tracking)


# Use tracking
line.track(particles, num_turns=parameters['study_parameters']['number_of_turns'], turn_by_turn_monitor=True, time=True, with_progress=10) #, freeze_longitudinal=True
particles.sort(interleave_lost_particles=True)

Qx_start = xutil.nafflib_tune_calculation(line.record_last_track.x[:, :1000], pq_coordinates=line.record_last_track.px[:, :1000], number_harmonics=1) 
Qy_start = xutil.nafflib_tune_calculation(line.record_last_track.y[:, :1000], pq_coordinates=line.record_last_track.py[:, :1000], number_harmonics=1) 

# Window 2: Last 1000 turns (e.g., from turn 1000 to 2000)
Qx_end = xutil.nafflib_tune_calculation(line.record_last_track.x[:, 1500:2500], pq_coordinates=line.record_last_track.px[:, 1500:2500], number_harmonics=1) 
Qy_end = xutil.nafflib_tune_calculation(line.record_last_track.y[:, 1500:2500], pq_coordinates=line.record_last_track.py[:, 1500:2500], number_harmonics=1) 


my_xpf.tune_diffusion (Qx_start['Q1']+15, Qx_end['Q1']+15, Qy_start['Q1']+15, Qy_end['Q1']+15, initial_conditions_x_axis=grid_details['x_normalized'], initial_conditions_y_axis=grid_details['y_normalized'], xlabel='x [$\sigma$]', ylabel='y [$\sigma$]', resonance_orders=(1,2,3,4), annotate=True, delta_value= None)

'''

# %%
'''fp = line.get_footprint(
    nemitt_x=13000e-6, 
    nemitt_y=12000e-6,
    mode='uniform_action_grid',
    x_norm_range=(0.1, 4),  # Lowered from 6 to 4 sigmas
    y_norm_range=(0.1, 4),  # Lowered from 6 to 4 sigmas
    #n_x_norm=10, 
    #n_y_norm=10,
    freeze_longitudinal=True # Good for 4D/transverse-only footprints
)
fp.qx += 15
fp.qy += 15

# 3. Plotting
fig, ax = plt.subplots(figsize=(8,8))
fp.plot(ax=ax, color='blue', label='Amplitude Footprint')

# Set the limits to the specific integer cell
ax.set_xlim(15.0, 16.0)
ax.set_ylim(15.0, 16.0)

# Optional: Add the Twiss working point for comparison
ax.plot(ring_tw.qx, ring_tw.qy, 'ro', label='Nominal WP')
ax.legend()
'''

# %%
line.configure_radiation(model=None)

parameters['study_parameters'] = {
    'ini_cond_type' : 'grid_MA', # grid_DA, grid_MA, distribution_matched, distribution_injected
    'output_dir' : 'out',
    'number_of_turns' : 2500,
    'number_of_particles' : 1000, 
    'inv1': 0, # np.arange(2)+1,
    'inv2': 0, # np.arange(2,2+3)+1,
    'start_element' : 'QD1_R1', # 'ca1.1','ip' #'rf400'
    'ini_cond_nemittance_x':n_emittancex,
    'ini_cond_nemittance_y': n_emittancey,
    'ini_cond_bunch_length': 4.8e-3,
    'ini_cond_energy_spread': 6e-3,
    'ini_cond_energy_offset': None,
    'new_closed_orbit': None, # {'x': -x_co_inj_marker, 'px': None, 'y': None, 'py': None, 'zeta': None, 'delta': None}
    'covariance_dispertion_free': False
}


## Initial conditions
#particles = xutil.generate_particle_distribution (line, parameters['study_parameters'], beambeam_strength_used=1, radiation_off=True)
particles, grid_details = xutil.generate_particle_grid (line, parameters['study_parameters'])
print(grid_details)

## Tracking studies
line.configure_radiation(model='mean')


## Change context for multy CPU for tracking
line.discard_tracker()
line.build_tracker(_context=context_tracking)


# Use tracking
line.track(particles, num_turns=parameters['study_parameters']['number_of_turns'], turn_by_turn_monitor=True, time=True, with_progress=10) #, freeze_longitudinal=True
particles.sort(interleave_lost_particles=True)


# %%
my_xpf.MA_vs_turns(particles, grid_details['num_r_y_points'], 51, grid_details['x_normalized'], grid_details['y_normalized'], grid_details['delta_init'])

ax = plt.gca()

plt.savefig(f"{new_results_folder}/MA_plot_{mode}_WP{current_wp}_full.png", dpi=300, bbox_inches='tight')

ax.relim(); ax.autoscale_view()          
(x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
ax.set_xlim(-7, 7)
ax.set_ylim(None, 12)
plt.savefig(f"{new_results_folder}/MA_plot_{mode}_WP{current_wp}_zoom.png", dpi=300, bbox_inches='tight')

# %%

if pdf_run is True:
    pdf.close()

# %%
'''min_DAs=[]

line.discard_tracker()
line.build_tracker(_context=context_tracking)

qx_vals=np.linspace(15.3,15.4,2)
qy_vals=np.linspace(15.3,15.4,2)

for qx_val in qx_vals:
    for qy_val in qy_vals:

        try:
            lo2.matchingWP(qx_val,qy_val)
            parameters['study_parameters'] = {
            'ini_cond_type' : 'grid_DA', # grid_DA, grid_MA, distribution_matched, distribution_injected
            'output_dir' : 'out',
            'number_of_turns' : 100,
            'number_of_particles' : 100, 
            'inv1': 0, # np.arange(2)+1,
            'inv2': 0, # np.arange(2,2+3)+1,
            'start_element' : 'QD1_R1', # 'ca1.1','ip' #'rf400'
            'ini_cond_nemittance_x':13000e-6,
            'ini_cond_nemittance_y': 12000e-6,
            'ini_cond_bunch_length': 4.8e-3,
            'ini_cond_energy_spread': 2e-3,
            'ini_cond_energy_offset': None,
            'new_closed_orbit': None, # {'x': -x_co_inj_marker, 'px': None, 'y': None, 'py': None, 'zeta': None, 'delta': None}
            'covariance_dispertion_free': False
                }
                ## Initial conditions
                #particles = xutil.generate_particle_distribution (line, parameters['study_parameters'], beambeam_strength_used=1, radiation_off=True)

            line.configure_radiation(model=None)
            particles, grid_details = xutil.generate_particle_grid (line, parameters['study_parameters'])

            ## Tracking studies
            line.configure_radiation(model='mean')


            ## Change context for multy CPU for tracking



            # Use tracking
            max_turns = parameters['study_parameters']['number_of_turns']
            line.track(particles, num_turns=max_turns, turn_by_turn_monitor=True, time=True, with_progress=10) 
            particles.sort(interleave_lost_particles=True)

            if isinstance(particles, dict):
                max_turns = np.shape(particles['x'])[1]-1 
                part_at_turn = np.nanmax(particles['at_turn'],axis=1)
            else:
                max_turns = np.max(particles.filter(particles.at_element==0).at_turn) 
                part_at_turn = particles.at_turn
                
            delta_plots=False
            delta_initial=grid_details['delta_init']
            x_norm=grid_details['x_normalized']
            y_norm=grid_details['y_normalized']
            num_r_steps=grid_details['num_r_y_points']
            num_theta_steps=grid_details['num_theta_x_points']
            
            if not delta_plots and np.size(delta_initial) > 1:
                closest_to_zero_delta = delta_initial[(np.abs(delta_initial - 0)).argmin()]
                delta_index = np.where(delta_initial==closest_to_zero_delta)[0]
                x_norm_1d = x_norm[delta_index]
                y_norm_1d = y_norm[delta_index]
                part_at_turn_1d = part_at_turn[delta_index]
            else:
                x_norm_1d = x_norm
                y_norm_1d = y_norm      
                part_at_turn_1d = part_at_turn

            x_norm_2d = x_norm_1d.reshape(num_r_steps, num_theta_steps)
            y_norm_2d = y_norm_1d.reshape(num_r_steps, num_theta_steps)
            part_at_turn_2d = part_at_turn_1d.reshape(num_r_steps, num_theta_steps)
            
            x_DA = np.full(num_theta_steps, np.nan)
            y_DA = np.full(num_theta_steps, np.nan)
            
            for jj in range(num_theta_steps):
                for ii in range(num_r_steps):
                    if part_at_turn_2d[ii,jj] != max_turns:
                        x_DA[jj] = x_norm_2d[ii,jj]
                        y_DA[jj] = y_norm_2d[ii,jj]
                        break

            min_DA = np.nanmin(np.round(np.sqrt(x_DA**2+y_DA**2),1)) 
            where_min_DA = np.where(np.round(np.sqrt(x_DA**2+y_DA**2),1) == min_DA)[0]

            min_DAs.append({
                'qx': qx_val,
                'qy': qy_val,
                'min_da': min_DA
            })
            print(f"WP ({qx_val:.3f}, {qy_val:.3f}) computed successfully. DA: {min_DA}")
            
        except RuntimeError:
            continue'''

        


# %%
'''import pandas as pd
%matplotlib widget
df = pd.DataFrame(min_DAs)

# 2. Pivot the data to create a 2D matrix (Qy rows, Qx columns)
# If a specific tune pair crashed or has multiples, 'min' handles it gracefully
heatmap_data = df.pivot_table(index='qy', columns='qx', values='min_da', aggfunc='min')

# Sort index descendingly so higher vertical tunes are at the top of the plot
heatmap_data = heatmap_data.sort_index(ascending=False)

# 3. Define the spatial extent for boundary alignment
qx_centers = heatmap_data.columns.values
qy_centers = heatmap_data.index.values

dx = (qx_centers[1] - qx_centers[0]) / 2.0 if len(qx_centers) > 1 else 0.05
dy = (qy_centers[1] - qy_centers[0]) / 2.0 if len(qy_centers) > 1 else 0.05

extent = [
    qx_centers.min() - dx, qx_centers.max() + dx,
    qy_centers.min() - dy, qy_centers.max() + dy
]

# 4. Generate the plot
fig, ax = plt.subplots(figsize=(9, 8))

# Using 'viridis' or 'jet'. 'inferno' works great for structural DA scans.
cax = ax.imshow(heatmap_data.values, extent=extent, cmap='viridis', aspect='auto')

# Labels and aesthetics
ax.set_xlabel(r'Horizontal Tune ($q_x$)', fontsize=12)
ax.set_ylabel(r'Vertical Tune ($q_y$)', fontsize=12)
ax.set_title('Working Point Scan: Minimum Dynamic Aperture', fontsize=14, pad=15, fontweight='bold')

# Set tick markers directly on your scanned tune steps
ax.set_xticks(qx_vals)
ax.set_yticks(qy_vals)
plt.xticks(rotation=45)

# Colorbar setup
cbar = fig.colorbar(cax, ax=ax)
cbar.set_label(r'Minimum DA [$\sigma$]', fontsize=12)

ax.grid(True, which='both', linestyle='--', alpha=0.5, color='grey')

plt.tight_layout()
plt.show()'''

# %%
'''context = xo.ContextCpu(omp_num_threads=None)

#  Simple test of tracking with synchrotron radiation - takes some time
ring.configure_radiation(model='mean')
ring_tw=ring.twiss(method='6d', radiation_integrals=True, eneloss_and_damping=True)
#ring_tw.plot()
#print( ring_tw.keys() )
print( f' Energy loss per turn from twiss {ring_tw.eneloss_turn:10.2f} and me {U0:10.2f}')
print( 1/ring_tw.damping_constants_turns )
print( 2*E0/ring_tw.eneloss_turn )

npts = 200
epsx, epsy, epsl = 1.0e-7, 1.0e-7, 1.0e-4  # starting with somewhat large emittances
betxin, alfxin  = ring_tw.betx[0], ring_tw.alfx[0] 
xin, pxin       = ring_tw.x[0], ring_tw.px[0]
dxin, dpxin     = ring_tw.dx[0], ring_tw.dpx[0]
betyin, alfyin  = ring_tw.bety[0], ring_tw.alfy[0] 
yin, pyin       = ring_tw.y[0], ring_tw.py[0]   # should be zero for perfect case
#epsx, epsl      = 2.0*ring_tw.eq_gemitt_x, 2.0*ring_tw.eq_gemitt_zeta
betlin          = ring_tw.bets0
zetain, deltain = ring_tw.zeta[0], ring_tw.delta[0]
print( f'At reference location: D ={dxin:7.4f} m, Dp ={dpxin:7.4f},' +
       f' betx ={betxin:7.4f} m, alfx ={alfxin:7.4f},' + 
       f' betl ={betlin:7.4f} m ')

normat = np.array([
    [(epsx*betxin)**.5,         0,                 0,  0, 0, dxin*(epsl/betlin)**.5],
    [-alfxin*(epsx/betxin)**.5, (epsx/betxin)**.5, 0,  0, 0, dpxin*(epsl/betlin)**.5],
    [0,  0,  (epsx*betyin)**.5,         0,                 0, 0],
    [0,  0,  -alfyin*(epsx/betyin)**.5, (epsx/betyin)**.5, 0, 0],
    [0, 0, 0, 0, (epsl*betlin)**.5, 0                ],
    [0, 0, 0, 0, 0,                 (epsl/betlin)**.5]               
    ])

parts = np.array([
     [xin, pxin, yin, pyin, zetain, deltain] + normat@np.random.randn(6) 
         for ind in range(npts) ]).T
# quit()
p2 = xt.Particles(kinetic_energy0 = 2860.e6, mass0 = xt.ELECTRON_MASS_EV,
                  x    = parts[0], px   = parts[1],
                  y    = parts[2], py   = parts[3],
                  zeta = parts[4], delta = parts[5] )
ring.configure_radiation(model='quantum')
ring.track( p2, num_turns = 6100, turn_by_turn_monitor = True,
            with_progress = True )

data = ring.record_last_track
trnplt = [1, 2000, 4000, 6000]
#trnplt = range( len(data.x.T) ) # superimpose all turns, rendering is slowish
fig = plt.figure( figsize=(14., 4.) )
fig.suptitle( f'Phase Space Plots - particle survival {int(((p2.state + 1)/2).sum()):5d} out of {npts:5d}' )
ax = fig.subplots(1, 3)
fig.subplots_adjust( wspace=0.4 )
ax[0].set_xlabel('x (mm)')
ax[0].set_ylabel("x' (mrad)")
ax[1].set_xlabel('y (mm)')
ax[1].set_ylabel("y' (mrad)")
ax[2].set_xlabel('z (mm)')
ax[2].set_ylabel(r'rel. mom. offset ($10^{-3}$)')
for ind in range( len(trnplt) ):  # data.x.T[ind] is nparray (behaviour of multiplication)!
   print( f' ==> Plotting turn"{trnplt[ind]:5d}')
   ax[0].scatter( 1000*(data.x.T[trnplt[ind]] - 1.*ring_tw.dx[0]*data.delta.T[trnplt[ind]]), 
            1000*(data.px.T[trnplt[ind]] - 1.*ring_tw.dpx[0]*data.delta.T[trnplt[ind]]),
            s=.4, c = f'C{ind:1d}')
   ax[1].scatter( 1000*(data.y.T[trnplt[ind]]), 1000*(data.py.T[trnplt[ind]]), 
            s=.2, c = f'C{ind:1d}' )
   ax[2].scatter(1000*data.zeta.T[trnplt[ind]], 1000*data.delta.T[trnplt[ind]], 
            s=.2, color=f'C{ind:1d}' )

print( [pdr['RFCav_1'].lag, pdr['RFCav_1'].voltage] )

print( f' Particle survival: {int(((p2.state + 1)/2).sum()):5d} out of {npts:5d}' )
plt.savefig(f'config_D{n}/Particle_survival_{n}.png')'''