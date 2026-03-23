#%%
import os
import time
import argparse
import numpy as np
import xpart as xp
import xtrack as xt
import xcoll as xc
import xobjects as xo
import xutil_DA_CC.xsuite_utilities as xutil
import scipy.constants as const


output_dir = 'out'
line = xt.Line.from_json('Lattice_corrected_0_1_1.json')
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
context_tracking = xo.ContextCpu(omp_num_threads='auto') # For CPU with activate multi-core CPU parallelization


# ## Transfer lattice on context and compile tracking code
# line.build_tracker(_context=context)

# line.configure_radiation(model='mean')
# line.compensate_radiation_energy_loss()

# tw = line.twiss(eneloss_and_damping=True)

# xutil.update_reference_parameters_from_line (line, parameters, BS_scale_factor=0, update_type='all', max_bb_param=0)
# xutil.correct_parameters_conflicts(parameters, update_study_parameters_from_reference=False)

parameters['study_parameters'] = {
    'ini_cond_type' : 'grid_DA', # grid_DA, grid_MA, distribution_matched, distribution_injected
    'output_dir' : 'out',
    'number_of_turns' : 2500,
    'number_of_particles' : 1000, 
    'inv1': 0, # np.arange(2)+1,
    'inv2': 0, # np.arange(2,2+3)+1,
    'start_element' : 'ca1.1', # 'ca1.1','ip' #'rf400'
    'ini_cond_nemittance_x': inj_emitn_x,
    'ini_cond_nemittance_y': inj_emitn_y,
    'ini_cond_bunch_length': inj_bunch_length,
    'ini_cond_energy_spread': inj_energy_spread,
    'ini_cond_energy_offset': None,
    'new_closed_orbit': None, # {'x': -x_co_inj_marker, 'px': None, 'y': None, 'py': None, 'zeta': None, 'delta': None}
    'covariance_dispertion_free': False
}


## Initial conditions
#particles = xutil.generate_particle_distribution (line, parameters['study_parameters'], beambeam_strength_used=1, radiation_off=True)
particles, grid_details = xutil.generate_particle_grid (line, parameters['study_parameters'])

## Tracking studies
line.configure_radiation(model='quantum')


## Change context for multy CPU for tracking
line.discard_tracker()
line.build_tracker(_context=context_tracking)


# Use tracking
line.track(particles, num_turns=parameters['study_parameters']['number_of_turns'], turn_by_turn_monitor=True, time=True, with_progress=10) #, freeze_longitudinal=True
particles.sort(interleave_lost_particles=True)

dic_particles_all = xutil.tracking_data_process (tracking_data=line.record_last_track, 
                            monitor_twiss=tw.rows[0], 
                            norm_emit_x=parameters['reference_parameters']['normalized_emittance_x'],
                            norm_emit_y=parameters['reference_parameters']['normalized_emittance_y'],
                            sigma_z=parameters['reference_parameters']['bunch_length'], 
                            sigma_delta=parameters['reference_parameters']['energy_spread'],
                            particle_id_to_use='all')
# xutil.save_dict_to_h5(f'{output_dir}/dic_particles_all.h5', dic_particles_all)



import xutil_DA_CC.xsuite_plot_functions as my_xpf
# liftime = my_xpf.particle_population_vs_turns (dic_particles_all['state'], T_rev0=tw.T_rev0)
# my_xpf.phase_space_evolution_difference (dic_particles_all['zeta'][:,0], dic_particles_all['delta'][:,0], dic_particles_all['zeta'][:,-1], dic_particles_all['delta'][:,-1], x_axis_label=r'$z~[m]$', y_axis_label=r'$\delta$')
# my_xpf.phase_space_evolution_difference (dic_particles_all['x'][:,0], dic_particles_all['px'][:,0], dic_particles_all['x'][:,-1], dic_particles_all['px'][:,-1], x_axis_label=r'$x~[m]$', y_axis_label=r'$px$')
# my_xpf.emittances_vs_turns (dic_particles_all['emit_x'], dic_particles_all['emit_y'], dic_particles_all['emit_zeta'], reference_emit_x=ref_param['emittance_x'], reference_emit_y=ref_param['emittance_y'], reference_emit_z=ref_param['energy_spread']*ref_param['bunch_length'], log_y_axis=True)
# my_xpf.coordinates_vs_turns (dic_particles_all['x_sigma'], dic_particles_all['px_sigma'], dic_particles_all['at_turn'], r'$x~[\sigma_x]$', r'$p_x~[\sigma_x]$', particle_id_list=dic_particles_all['particle_id_list'], full_init_cond=study_param['number_of_particles'])
# my_xpf.MA_vs_turns(particles, grid_details['num_r_y_points'], grid_details['num_delta'], grid_details['x_normalized'], grid_details['y_normalized'], grid_details['delta_init'])
my_xpf.DA_vs_turns(particles, grid_details['num_r_y_points'], grid_details['num_theta_x_points'], grid_details['x_normalized'], grid_details['y_normalized'], grid_details['delta_init'])
    
       

# %%
