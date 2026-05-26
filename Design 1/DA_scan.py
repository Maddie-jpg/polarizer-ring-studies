import sys
import os

# Adds the parent directory to the search path
parent_dir = os.path.abspath('..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import numpy as np
import xtrack as xt
import linear_optics_variation_2 as lo2
import xutil_DA_CC.xsuite_plot_functions as my_xpf
import xutil_DA_CC.xsuite_utilities as xutil
import xobjects as xo
import pandas as pd
import matplotlib.pyplot as plt

line=xt.Line.from_json('/home/mwatson/Documents/laughing-octo-bassoon/Junk Draw/polarizer_ring_current_config.json')
line_tw=line.twiss()

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

min_DAs=[]

line.discard_tracker()
line.build_tracker(_context=context_tracking)

qx_up = np.arange(15.38, 15.95, 0.05)
qy_up = np.arange(15.42, 15.95, 0.05)

# Stream 2 goes DOWN towards 15.0 (inverted using [::-1] or negative steps)
qx_down = np.arange(15.05, 15.38, 0.1)[::-1]
qy_down = np.arange(15.05, 15.42, 0.1)[::-1]

paths = [
    ("Stream UP", qx_up, qy_up),
    ("Stream DOWN", qx_down, qy_down)
]

qx=line_tw.qx
qy=line_tw.qy
radii=np.linspace(0,0.5,25)
angles=np.linspace(0,2*np.pi,80)

for r in radii:
    for theta in angles:
            
            qx_val= qx + r*np.cos(theta)
            qy_val= qy + r*np.sin(theta)

            try:
                new_strengths = lo2.matchingWP(qx_val,qy_val)

                
                for knob, value in new_strengths.items():
                    line.vars[knob] = value

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
                
            except (RuntimeError, ValueError):
                min_DAs.append({
                    'qx': qx_val,
                    'qy': qy_val,
                    'min_da': 0
                })

        


df = pd.DataFrame(min_DAs)

df['qx_round'] = df['qx'].round(2)
df['qy_round'] = df['qy'].round(2)

# 2. Pivot using the rounded, structured coordinate system
heatmap_data = df.pivot_table(index='qy_round', columns='qx_round', values='min_da', aggfunc='min')

# Sort index descendingly so higher vertical tunes are at the top of the plot
heatmap_data = heatmap_data.sort_index(ascending=False)

# Check if we have a valid, dense 2D grid matrix
if heatmap_data.notna().sum().sum() < (len(df) * 0.1):
    # FALLBACK: If your points do not lie on a perfect grid, a Scatter Plot 
    # will display the data accurately without needing a dense matrix grid.
    print("Data is irregularly spaced. Generating a continuous scatter map instead...")
    fig, ax = plt.subplots(figsize=(9, 8))
    cax = ax.scatter(df['qx'], df['qy'], c=df['min_da'], cmap='viridis', s=40, marker='s')
    
else:
    # GRID RENDER: Proceed with standard grid image array mapping
    qx_centers = heatmap_data.columns.values
    qy_centers = heatmap_data.index.values

    dx = (qx_centers[1] - qx_centers[0]) / 2.0 if len(qx_centers) > 1 else 0.05
    dy = (qy_centers[1] - qy_centers[0]) / 2.0 if len(qy_centers) > 1 else 0.05

    extent = [
        qx_centers.min() - dx, qx_centers.max() + dx,
        qy_centers.min() - dy, qy_centers.max() + dy
    ]

    fig, ax = plt.subplots(figsize=(9, 8))
    cax = ax.tripcolor(df['qx'], df['qy'], df['min_da'], cmap='viridis', shading='flat')

    # Update axis markers directly on your clean binned values
    ax.set_xticks(np.arange(15.0, 16.0, 0.1))
    ax.set_yticks(np.arange(15.0, 16.0, 0.1))

# Labels and aesthetics
ax.set_xlabel(r'Horizontal Tune ($q_x$)', fontsize=12)
ax.set_ylabel(r'Vertical Tune ($q_y$)', fontsize=12)
ax.set_title('Working Point Scan: Minimum Dynamic Aperture', fontsize=14, pad=15, fontweight='bold')
plt.xticks(rotation=45)

# Colorbar setup
cbar = fig.colorbar(cax, ax=ax)
cbar.set_label(r'Minimum DA [$\sigma$]', fontsize=12)

plt.tight_layout()
plt.savefig('DA_scan.png')
plt.show()