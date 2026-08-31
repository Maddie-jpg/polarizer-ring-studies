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
config=int(os.environ.get('CONFIG',4))
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

U0 = (0.88463e-31) * E0**4 * pdr['hBarc']

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


# %%
#%matplotlib widget
folder=mf.results_dir(design, config, phase, changes=changes, metric='LatticeOptics', sub=None)
if mode in ['perfect','misaligned']:
    ring.survey().plot()
    plt.savefig(f'{folder}/ring_survey_{mode}.png')
else:
    mf.survey_plot(ring)
    #plt.tight_layout()
    plt.savefig(f'{folder}/ring_survey_{mode}.png')

# %%
print(ring.element_names)

# %%
folder1=mf.results_dir(design, config, phase, changes=changes, metric='LatticeOptics', sub=mode)
ring_tw.plot(f'delta (zeta+0.00504) x y-0.001')
ring_tw.plot(f'x y-0.001')
plt.tight_layout()
plt.savefig(f'{folder1}/ring_closed_orbit_{mode}.png')


# %%

ring_tw.plot()
plt.tight_layout()
plt.savefig(f'{folder1}/ring_twiss_{mode}.png')


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
plt.savefig(f'{folder1}/straight_optics_{mode}.png')

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
plt.savefig(f'{folder1}/straight_phase_advance_{mode}.png')


def calc_damping_time_constant(m):
    num=2*E0*ring_tw.T_rev0
    den=ring_tw.partition_numbers[m]*U0
    tau=num/den
    return tau

# Initialize the table
brho = ring.particle_ref.p0c[0] / 299792458.0 
length_arr = np.asarray(ring_tw.length)
k0l_arr = np.asarray(ring_tw.k0l)

mask = length_arr > 0         
max_k0 = np.max(np.abs(k0l_arr[mask] / length_arr[mask]))
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
    ["Energy Loss (Twiss)", f"{ring_tw.eneloss_turn*1e-3:10.2f}", "keV"],
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

plt.savefig(f'{folder1}/table_{mode}.png')


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
plt.savefig(f'{folder1}/WP_{mode}.png')

print('2nd order chrom x', ring_tw.ddqx)
print('2nd order chrom y', ring_tw.ddqy)

# %%

mf.SpuckParsAus(
    period, period_sliced.twiss(method='4d'), period,
    (0., 38.), (0., 12.), (0., 1.), .08, pdr,
    f"{folder}/PeriodOptics.png"
)

mf.SpuckParsAus(
    ring, ring_tw, ring,
    (0., ring.get_length()), (0., 12.), (0., 1.), .08, pdr,
    f"{folder}/RingOptics.png",
    n_periods=1
)
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
plt.savefig(f'{folder1}/momentum_deviation{current_wp}_{mode}.png')



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
plt.savefig(f'{folder1}/momentum_dev_working_point{current_wp}_{mode}.png')



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
    # get_footprint's internal FFT uses rfftfreq, which only ever resolves
    # frequencies in [0, 0.5] (real-signal Nyquist limit). If the TRUE
    # fractional tune on an axis is above 0.5, the FFT peak for it is
    # indistinguishable from one at (1 - true_frac) and gets reported
    # there instead -- i.e. it silently folds back into [0, 0.5]. Because
    # of that fold, the raw fp0.qx/fp0.qy values can never actually reveal
    # whether aliasing happened (they're always <= 0.5 by construction), so
    # checking np.nanmax(fp0.qx) > 0.5 -- as this block used to -- can never
    # fire and is not a real guard. The only reliable way to know is to look
    # at the true fractional part of the known nominal WP itself, decided
    # BEFORE trusting the FFT output, and unfold accordingly.
    def _reconstruct_absolute_tune(frac_from_fft, wp_value):
        n = np.floor(wp_value)
        true_frac = wp_value - n
        if true_frac > 0.5:
            # rfftfreq folded the true peak back to (1 - true_frac); undo it.
            return (n + 1) - frac_from_fft
        else:
            return n + frac_from_fft

    if (WP[0] % 1) > 0.5 or (WP[1] % 1) > 0.5:
        print(f"NOTE: fractional tune above 0.5 detected from the nominal "
              f"WP directly (Qx frac={WP[0] % 1:.3f}, Qy frac={WP[1] % 1:.3f}) "
              f"-- applying fold-back correction to reconstruct the absolute "
              f"footprint tunes on the affected axis.")

    # get_footprint returns fractional tunes -> reconstruct to absolute,
    # correctly unfolding any axis where the true fractional tune is > 0.5.
    fp0.qx = _reconstruct_absolute_tune(fp0.qx, WP[0])
    fp0.qy = _reconstruct_absolute_tune(fp0.qy, WP[1])

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
    plt.savefig(f'{folder1}/momentum_dev_working_point_{mode}.png', dpi=200)
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
plt.savefig(f'{folder1}/dangerous_resonances_{mode}.png')

# %%
if mode=='perfect':
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

    tw = line.twiss(eneloss_and_damping=True)

    gamma0 = ring.particle_ref.gamma0[0]
    beta0 = ring.particle_ref.beta0[0]

    n_emittancex = 1.354177116369456e-6 * gamma0 * beta0
    n_emittancey = 1.420755089827341e-6 * gamma0 * beta0

    parameters['study_parameters'] = {
        'ini_cond_type' : 'grid_DA', # grid_DA, grid_MA, distribution_matched, distribution_injected
        'output_dir' : 'out',
        'number_of_turns' : 5000,
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

    particles, grid_details = xutil.generate_particle_grid (line, parameters['study_parameters'])

    line.discard_tracker()
    line.build_tracker(_context=context_tracking)


    
    line.discard_tracker()
    ## Tracking studies
    line.configure_radiation(model='mean')


    ## Change context for multy CPU for tracking

    line.build_tracker(_context=context_tracking)


    # Use tracking
    line.track(particles, num_turns=parameters['study_parameters']['number_of_turns'], turn_by_turn_monitor=True, time=True, with_progress=10) #, freeze_longitudinal=True
    particles.sort(interleave_lost_particles=True)

    x_DA,y_DA,_,_=my_xpf.DA_vs_turns(particles, grid_details['num_r_y_points'], grid_details['num_theta_x_points'], grid_details['x_normalized'], grid_details['y_normalized'], grid_details['delta_init'],delta_plots=True)

    ax = plt.gca()

    folder2=mf.results_dir(design, config, phase, changes=changes, metric='DA_MA', sub=None)

    plt.savefig(f"{folder2}/DA_plot_{mode}_WP{current_wp}_full.png", dpi=300, bbox_inches='tight')

    ax.relim(); ax.autoscale_view()          
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    ax.set_xlim(-15, 15)
    ax.set_ylim(None, 12)
    plt.savefig(f"{folder2}/DA_plot_{mode}_WP{current_wp}_zoom.png", dpi=300, bbox_inches='tight')

    
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


    line.configure_radiation(model=None)

    parameters['study_parameters'] = {
        'ini_cond_type' : 'grid_MA', # grid_DA, grid_MA, distribution_matched, distribution_injected
        'output_dir' : 'out',
        'number_of_turns' : 5000,
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


    my_xpf.MA_vs_turns(particles, grid_details['num_r_y_points'], 51, grid_details['x_normalized'], grid_details['y_normalized'], grid_details['delta_init'])

    ax = plt.gca()

    plt.savefig(f"{folder2}/MA_plot_{mode}_WP{current_wp}_full.png", dpi=300, bbox_inches='tight')

    ax.relim(); ax.autoscale_view()          
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    ax.set_xlim(-7, 7)
    ax.set_ylim(None, 12)
    plt.savefig(f"{folder2}/MA_plot_{mode}_WP{current_wp}_zoom.png", dpi=300, bbox_inches='tight')

   
    import numpy as np
    import matplotlib.pyplot as plt
    import xobjects as xo
    import LatticeBuild.misalignments_corrections as mc
    import xutil_DA_CC.xsuite_utilities as xutil

    context_tracking = xo.ContextCpu(omp_num_threads=0)
    misalignment_val = 0.2e-3


    if design == 1 and config == 1:
        mc.insert_BPMs_all_as_markers(pdr)
        mc.insert_correctors_var2(pdr)
    else:
        mc.insert_BPMs_all_as_markers(pdr)
        mc.insert_correctors(pdr)

    # ---- lightweight boundary extractors (logic copied from xsuite_plot_functions,
    #      but with plotting stripped out so nothing pops open/gets thrown away) ----

    def get_DA_boundary(particles, num_r_steps, num_theta_steps, x_norm, y_norm):
        if isinstance(particles, dict):
            max_turns = np.shape(particles['x'])[1] - 1
            part_at_turn = np.nanmax(particles['at_turn'], axis=1)
        else:
            max_turns = np.max(particles.filter(particles.at_element == 0).at_turn)
            part_at_turn = particles.at_turn

        x_2d = x_norm.reshape(num_r_steps, num_theta_steps)
        y_2d = y_norm.reshape(num_r_steps, num_theta_steps)
        p_2d = part_at_turn.reshape(num_r_steps, num_theta_steps)

        x_DA = np.full(num_theta_steps, np.nan)
        y_DA = np.full(num_theta_steps, np.nan)
        for jj in range(num_theta_steps):
            for ii in range(num_r_steps):
                if p_2d[ii, jj] != max_turns:
                    x_DA[jj], y_DA[jj] = x_2d[ii, jj], y_2d[ii, jj]
                    break

        min_DA = np.nanmin(np.round(np.sqrt(x_DA**2 + y_DA**2), 1))
        return x_DA, y_DA, min_DA


    def get_MA_boundary(particles, num_r_steps, num_delta_steps, x_norm, y_norm, delta_initial):
        if isinstance(particles, dict):
            max_turns = np.shape(particles['x'])[1] - 1
            part_at_turn = np.nanmax(particles['at_turn'], axis=1)
        else:
            max_turns = np.max(particles.filter(particles.at_element == 0).at_turn)
            part_at_turn = particles.at_turn

        x_2d = x_norm.reshape(num_delta_steps, num_r_steps)
        delta_2d = delta_initial.reshape(num_delta_steps, num_r_steps)
        p_2d = part_at_turn.reshape(num_delta_steps, num_r_steps)

        x_MA = np.full(num_delta_steps, np.nan)
        delta_MA = np.full(num_delta_steps, np.nan)
        for jj in range(num_delta_steps):
            for ii in range(num_r_steps):
                if p_2d[jj, ii] != max_turns:
                    x_MA[jj], delta_MA[jj] = x_2d[jj, ii], delta_2d[jj, ii]
                    break

        min_MA = np.nanmin(np.round(np.sqrt(x_MA**2 + delta_MA**2), 1))
        return x_MA, delta_MA, min_MA


    # ---- per-seed branch prep, mirrors prep_branch() in spin_tracking.py ----

    def prep_seed_line(base_line, seed, apply_correction):
        seed_line = base_line.copy()
        seed_line.configure_radiation(model='mean')
        seed_line.build_tracker(_context=xo.ContextCpu(omp_num_threads=0))
        seed_line = mc.misalignments(seed_line, misalignment_val, seed=seed)

        if apply_correction:
            tw = seed_line.twiss(method='6d', radiation_integrals=True, eneloss_and_damping=True)
            mc.misalignments_correctors(seed_line, 0.2e-3, seed + 1)
            try:
                mc.orbit_correction(seed_line, tw, threading=False, seed=seed)
            except Exception as e:
                print(f"  [seed {seed}] orbit_correction(threading=False) raised: {e}")
                mc.orbit_correction(seed_line, tw, threading=True, seed=seed)

        return seed_line


    def get_DA_for_seed(base_line, seed, apply_correction, study_params_DA):
        seed_line = prep_seed_line(base_line, seed, apply_correction)
        seed_line.discard_tracker()
        seed_line.build_tracker(_context=context_tracking)
        seed_line.configure_radiation(model='mean')

        particles, grid_details = xutil.generate_particle_grid(seed_line, study_params_DA)
        seed_line.track(particles, num_turns=study_params_DA['number_of_turns'],
                        turn_by_turn_monitor=True, time=True, with_progress=10)
        particles.sort(interleave_lost_particles=True)

        return get_DA_boundary(particles, grid_details['num_r_y_points'],
                                grid_details['num_theta_x_points'],
                                grid_details['x_normalized'], grid_details['y_normalized'])


    def get_MA_for_seed(base_line, seed, apply_correction, study_params_MA):
        seed_line = prep_seed_line(base_line, seed, apply_correction)
        seed_line.discard_tracker()
        seed_line.build_tracker(_context=context_tracking)
        seed_line.configure_radiation(model='mean')

        particles, grid_details = xutil.generate_particle_grid(seed_line, study_params_MA)
        seed_line.track(particles, num_turns=study_params_MA['number_of_turns'],
                        turn_by_turn_monitor=True, time=True, with_progress=10)
        particles.sort(interleave_lost_particles=True)

        return get_MA_boundary(particles, grid_details['num_r_y_points'], 51,
                                grid_details['x_normalized'], grid_details['y_normalized'],
                                grid_details['delta_init'])


    # ---- run: misaligned seeds first, then corrected seeds ----

    # NOTE: don't reuse parameters['study_parameters'] here -- by this point in the
    # script it's been overwritten (first by the polar-DA cell, then by the MA cell
    # at ini_cond_type='grid_MA'), so grabbing it directly silently hands the DA scan
    # an MA-shaped grid (fixed theta=45deg, x_normalized==y_normalized everywhere).
    # Build both dicts explicitly and self-contained instead.

    seeds = [100, 200, 300, 400, 500]
    colors = plt.cm.gist_rainbow(np.linspace(0, 0.95, len(seeds)))  # more spread than tab10

    study_params_DA = {
        'ini_cond_type': 'grid_DA',
        'output_dir': 'out',
        'number_of_turns': 5000,
        'number_of_particles': 1000,
        'inv1': 0,
        'inv2': 0,
        'start_element': 'QD1_R1',
        'ini_cond_nemittance_x': n_emittancex,
        'ini_cond_nemittance_y': n_emittancey,
        'ini_cond_bunch_length': 4.8e-3,
        'ini_cond_energy_spread': 2e-3,
        'ini_cond_energy_offset': None,
        'new_closed_orbit': None,
        'covariance_dispertion_free': False,
    }

    study_params_MA = dict(study_params_DA)
    study_params_MA['ini_cond_type'] = 'grid_MA'
    study_params_MA['ini_cond_energy_spread'] = 6e-3  # matches the MA cell's value, not DA's

    fig_da_mis, ax_da_mis = plt.subplots(figsize=(8, 8))
    fig_da_cor, ax_da_cor = plt.subplots(figsize=(8, 8))
    fig_ma_mis, ax_ma_mis = plt.subplots(figsize=(8, 6))
    fig_ma_cor, ax_ma_cor = plt.subplots(figsize=(8, 6))

    # --- pass 1: misaligned ---
    for seed, c in zip(seeds, colors):
        x_DA, y_DA, min_DA = get_DA_for_seed(ring, seed, apply_correction=False,
                                            study_params_DA=study_params_DA)
        ax_da_mis.plot(x_DA, y_DA, '-', color=c,
                    label=f'seed {seed} (DA$_{{min}}$={min_DA:.1f}$\\sigma$)')

        x_MA, delta_MA, min_MA = get_MA_for_seed(ring, seed, apply_correction=False,
                                                study_params_MA=study_params_MA)
        ax_ma_mis.plot(delta_MA * 100, x_MA, '-', color=c, label=f'seed {seed}')

    # --- pass 2: corrected ---
    for seed, c in zip(seeds, colors):
        x_DA, y_DA, min_DA = get_DA_for_seed(ring, seed, apply_correction=True,
                                            study_params_DA=study_params_DA)
        ax_da_cor.plot(x_DA, y_DA, '-', color=c,
                    label=f'seed {seed} (DA$_{{min}}$={min_DA:.1f}$\\sigma$)')

        x_MA, delta_MA, min_MA = get_MA_for_seed(ring, seed, apply_correction=True,
                                                study_params_MA=study_params_MA)
        ax_ma_cor.plot(delta_MA * 100, x_MA, '-', color=c, label=f'seed {seed}')

    # ---- formatting + saving, one block per plot ----

    for ax, title in [(ax_da_mis, 'DA — misaligned seeds'), (ax_da_cor, 'DA — corrected seeds')]:
        ax.set_xlabel(r'$\hat{x}$ [$\sqrt{\varepsilon_x}$]')
        ax.set_ylabel(r'$\hat{y}$ [$\sqrt{\varepsilon_y}$]')
        ax.set_xlim(-16, 16); ax.set_ylim(0, 20)
        ax.set_title(title)
        ax.legend(fontsize='x-small', loc='best', ncol=2)

    for ax, title in [(ax_ma_mis, 'MA — misaligned seeds'), (ax_ma_cor, 'MA — corrected seeds')]:
        ax.set_xlabel(r'$\delta$ [%]')
        ax.set_ylabel(r'$\hat{x}$ [$\sqrt{\varepsilon_x}$]')
        ax.set_xlim(-5, 5)
        ax.set_title(title)
        ax.legend(fontsize='x-small', loc='best', ncol=2)

    fig_da_mis.tight_layout()
    fig_da_mis.savefig(f'{folder2}/DA_overlay_misaligned.png')

    fig_da_cor.tight_layout()
    fig_da_cor.savefig(f'{folder2}/DA_overlay_corrected.png')

    fig_ma_mis.tight_layout()
    fig_ma_mis.savefig(f'{folder2}/MA_overlay_misaligned.png')

    fig_ma_cor.tight_layout()
    fig_ma_cor.savefig(f'{folder2}/MA_overlay_corrected.png')
# %%
if pdf_run is True:
    pdf.close()