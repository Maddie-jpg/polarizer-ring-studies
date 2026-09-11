#%%

import sys
import os
import xtrack as xt

# Adds the parent directory to the search path
parent_dir = os.path.abspath('..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import linear_optics as lo
import sextupole_configs as sc
import misalignments_corrections as mc
import constants
from pathlib import Path
import xobjects as xo
xo.context_cpu.allow_no_prebuilt_kernel = True


SCRIPT_DIR = Path(__file__).resolve().parent
parent_dir = SCRIPT_DIR.parent   # one level up from this script

if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

#%%

design=1
config=9
mode='perfect' # 'perfect', 'misaligned', or 'corrected'
phase=90
changes=None

#%%
SEED=123456789

if mode == 'perfect':
    #Linear optics - uncomment desired optics
    pdr=lo.three_fold_periodicity(fringe_fields=True,matched=True,WP=(15.42,15.38),phase_advance=0.25)
    sc.config_D1_C9(pdr)
    '''pdr=lo.two_fold_racetrack_3straight(fringe_fields=True,matched=True,WP=(13.65,13.23),phase_advance=0.25,betay_DS_target=False)
    sc.config_D2_C1(pdr)'''


#Misalignments
elif mode == 'misaligned':
    if changes is not None:
        pdr=xt.Environment.from_json(f'../JSON Files/D{design}/C{config}/pdr_perfect_{phase}_{changes}.json')
    else:
        pdr=xt.Environment.from_json(f'../JSON Files/D{design}/C{config}/pdr_perfect_{phase}.json')
    ring=pdr.lines['ring']
    mc.misalignments(ring, 0.25e-3,SEED)

elif mode=='corrected':
    if changes is not None:
        pdr=xt.Environment.from_json(f'../JSON Files/D{design}/C{config}/pdr_misaligned_{phase}_{changes}.json')
    else:
        pdr=xt.Environment.from_json(f'../JSON Files/D{design}/C{config}/pdr_misaligned_{phase}.json')
    ring=pdr.lines['ring']

    mc.insert_BPMs_all_as_markers(pdr)
    mc.insert_correctors_var2(pdr)
    mc.misalignments_correctors(ring,0.25e-3,SEED)

    twiss=ring.twiss(method='6d', radiation_integrals=True, eneloss_and_damping=True,
                   spin=True, polarization=True )



    try:
        ring.discard_tracker()
        mc.orbit_correction(ring, twiss, threading=False, rcond_x=1e-4, rcond_y=1e-2,seed=SEED)
        
    except:
        mc.orbit_correction(ring, twiss, threading=True, rcond_x=1e-4, rcond_y=1e-2,seed=SEED)
        


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
design=int(os.environ.get('DESIGN',1))
config=int(os.environ.get('CONFIG',9))
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

ring=pdr.lines['ring']
print(ring.element_names)
period=pdr.lines['period']

variable_name = f"WP_D{design}"

current_wp = getattr(constants, variable_name)

# %%
E0 = constants.E0; VRF = constants.VRF

U0 = (0.88463e-31) * E0**4 * pdr['hBarc']

period_sliced = period.select()
#period_sliced.cut_at_s( np.linspace(.05, period.get_length()-.05, int(period.get_length()/.05-.5)) )


ring.configure_radiation(model=None)
fRev = 1./(ring.twiss(method='4d').T_rev0)
fRF  = fRev*round(4.e8/fRev)  # at integer harmonics and close to 400 MHz

# %%
ring.configure_radiation(model='mean')
ring_tw=ring.twiss(method='6d', radiation_integrals=True, eneloss_and_damping=True,
                   spin=True, polarization=True )

# %%
print(ring.element_names)

p_test = xt.Particles(x=0.01, px=0.0, y=0.005, py=0.0,
                       mass0=xt.ELECTRON_MASS_EV, kinetic_energy0=2.86e9)
p_before = p_test.copy()
pdr.lines['Bend'].track(p_test)
print("x:  ", p_before.x[0], "->", p_test.x[0])
print("px: ", p_before.px[0], "->", p_test.px[0])
print("y:  ", p_before.y[0], "->", p_test.y[0])
print("py: ", p_before.py[0], "->", p_test.py[0])

for scale in [0.5, 1.0, 2.0, 4.0]:
    p = xt.Particles(x=0.01*scale, y=0.005*scale, px=0.0, py=0.0,
                      mass0=xt.ELECTRON_MASS_EV, kinetic_energy0=2.86e9)
    pdr.lines['Bend'].track(p)
    print(f"scale={scale:4.1f}  px/x={p.px[0]/(0.01*scale):.6e}   py/y={p.py[0]/(0.005*scale):.6e}")
# %%
def set_integrator(line):
    tt = line.get_table()
    tt_bend = tt.rows[
        (tt.element_type=='Bend') |
        (tt.element_type=='RBend') |
        (tt.element_type=='ThickSliceBend') |
        (tt.element_type=='ThickSliceRBend')]
    tt_wigg = tt.rows['mw.*']
    tt_quad = tt.rows[(tt.element_type=='Quadrupole')|
        (tt.element_type=='ThickSliceQuadrupole')]
    tt_sext = tt.rows[(tt.element_type=='Sextupole')]
    tt_fieldexp = tt.rows[
        (tt.element_type=='StraightFieldExpansion') |
        (tt.element_type=='BentFieldExpansion')]

    line.set(tt_bend, integrator='uniform', num_multipole_kicks=3, model='mat-kick-mat')
    line.set(tt_wigg, integrator='teapot', num_multipole_kicks=11, model='mat-kick-mat')
    line.set(tt_quad, integrator='uniform', num_multipole_kicks=7, model='mat-kick-mat')
    line.set(tt_sext, integrator='yoshida4', num_multipole_kicks=1)

    # FieldExpansion elements don't have integrator/model/num_multipole_kicks --
    # their resolution knob is nstep (number of integration steps through the
    # field profile). Bump it up here for the same reason the others get
    # explicit accuracy settings, rather than silently keeping whatever was
    # passed at construction time (nstep=10 in make_fringe_bend).
    if len(tt_fieldexp.name) > 0:
        line.set(tt_fieldexp, integrator='uniform', num_multipole_kicks=3, model='mat-kick-mat')

    return

folder2=mf.results_dir(design, config, phase, changes=changes, metric='DA_MA', sub=None)
print("Saving to:", os.path.abspath(folder2))

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


    # ---- combined DA+MA for one seed: prep the line ONCE (misalign + optional
    #      correction), then run both studies on it, instead of prepping twice ----

    def get_DA_and_MA_for_seed(base_line, seed, apply_correction, study_params_DA, study_params_MA):
        seed_line = prep_seed_line(base_line, seed, apply_correction)
        seed_line.discard_tracker()
        seed_line.build_tracker(_context=context_tracking)
        seed_line.configure_radiation(model='mean')

        # --- DA ---
        particles_DA, grid_DA = xutil.generate_particle_grid(seed_line, study_params_DA)
        seed_line.track(particles_DA, num_turns=study_params_DA['number_of_turns'],
                         turn_by_turn_monitor=False, time=True, with_progress=10)
        particles_DA.sort(interleave_lost_particles=True)
        da_result = get_DA_boundary(particles_DA, grid_DA['num_r_y_points'],
                                     grid_DA['num_theta_x_points'],
                                     grid_DA['x_normalized'], grid_DA['y_normalized'])

        # --- MA ---
        particles_MA, grid_MA = xutil.generate_particle_grid(seed_line, study_params_MA)
        seed_line.track(particles_MA, num_turns=study_params_MA['number_of_turns'],
                         turn_by_turn_monitor=False, time=True, with_progress=10)
        particles_MA.sort(interleave_lost_particles=True)
        ma_result = get_MA_boundary(particles_MA, grid_MA['num_r_y_points'], 51,
                                     grid_MA['x_normalized'], grid_MA['y_normalized'],
                                     grid_MA['delta_init'])

        return da_result, ma_result


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
    '''for seed, c in zip(seeds, colors):
        (x_DA, y_DA, min_DA), (x_MA, delta_MA, min_MA) = get_DA_and_MA_for_seed(
            ring, seed, apply_correction=False,
            study_params_DA=study_params_DA, study_params_MA=study_params_MA)

        ax_da_mis.plot(x_DA, y_DA, '-', color=c,
                    label=f'seed {seed} (DA$_{{min}}$={min_DA:.1f}$\\sigma$)')
        ax_ma_mis.plot(delta_MA * 100, x_MA, '-', color=c, label=f'seed {seed}')'''

    # --- pass 2: corrected ---
    for seed, c in zip(seeds, colors):
        (x_DA, y_DA, min_DA), (x_MA, delta_MA, min_MA) = get_DA_and_MA_for_seed(
            ring, seed, apply_correction=True,
            study_params_DA=study_params_DA, study_params_MA=study_params_MA)

        ax_da_cor.plot(x_DA, y_DA, '-', color=c,
                    label=f'seed {seed} (DA$_{{min}}$={min_DA:.1f}$\\sigma$)')
        ax_ma_cor.plot(delta_MA * 100, x_MA, '-', color=c, label=f'seed {seed}')

    # ---- formatting + saving, one block per plot ----

    for ax, title in [(ax_da_mis, 'DA — misaligned seeds'), (ax_da_cor, 'DA — corrected seeds')]:
        ax.set_xlabel(r'$\hat{x}$ [$\sqrt{\varepsilon_x}$]')
        ax.set_ylabel(r'$\hat{y}$ [$\sqrt{\varepsilon_y}$]')
        ax.set_xlim(-16, 16); ax.set_ylim(0, 12)
        ax.set_title(title)
        ax.legend(fontsize='x-small', loc='best', ncol=2)

    for ax, title in [(ax_ma_mis, 'MA — misaligned seeds'), (ax_ma_cor, 'MA — corrected seeds')]:
        ax.set_xlabel(r'$\delta$ [%]')
        ax.set_ylabel(r'$\hat{x}$ [$\sqrt{\varepsilon_x}$]')
        ax.set_xlim(-7, 7)
        ax.set_ylim(None,12)
        ax.set_title(title)
        ax.legend(fontsize='x-small', loc='best', ncol=2)

    #fig_da_mis.tight_layout()
    
    #fig_da_mis.savefig(f'{folder2}/DA_overlay_misaligned.png')

    fig_da_cor.tight_layout()
    fig_da_cor.savefig(f'{folder2}/DA_overlay_corrected.png')

    #fig_ma_mis.tight_layout()
    #fig_ma_mis.savefig(f'{folder2}/MA_overlay_misaligned.png')

    #fig_ma_cor.tight_layout()
    fig_ma_cor.savefig(f'{folder2}/MA_overlay_corrected.png')
# %%
if pdf_run is True:
    pdf.close()