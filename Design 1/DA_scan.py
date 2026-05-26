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

line=xt.Line.from_json('/Users/maddiewatson/Library/CloudStorage/OneDrive-Personal/University/Research year/polarizer-ring-studies/Junk Draw/polarizer_ring_current_config.json')


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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xtrack as xt

# ---------------------------------------------------
# Nominal working point
# ---------------------------------------------------

qx0 = 15.38
qy0 = 15.42

# ---------------------------------------------------
# Radial scan parameters
# ---------------------------------------------------

radii = np.linspace(0, 0.30, 25)
angles = np.linspace(0, 2*np.pi, 80)

# ---------------------------------------------------
# Storage
# ---------------------------------------------------

results = []

# ---------------------------------------------------
# Build tracker once
# ---------------------------------------------------

line.discard_tracker()
line.build_tracker(_context=context_tracking)

# ---------------------------------------------------
# Main scan
# ---------------------------------------------------

for r in radii:

    print(f'\nRadius = {r:.3f}')

    for theta in angles:

        qx_target = qx0 + r*np.cos(theta)
        qy_target = qy0 + r*np.sin(theta)

        try:

            # ----------------------------------------
            # Match tunes
            # ----------------------------------------

            new_strengths = lo2.matchingWP(
                qx_target,
                qy_target
            )

            for knob, value in new_strengths.items():
                line.vars[knob] = value

            # ----------------------------------------
            # Generate particles
            # ----------------------------------------

            line.configure_radiation(model=None)

            particles, grid_details = xutil.generate_particle_grid(
                line,
                parameters['study_parameters']
            )

            # ----------------------------------------
            # Tracking
            # ----------------------------------------

            line.configure_radiation(model='mean')

            max_turns = parameters['study_parameters']['number_of_turns']

            line.track(
                particles,
                num_turns=max_turns,
                turn_by_turn_monitor=False
            )

            particles.sort(interleave_lost_particles=True)

            # ----------------------------------------
            # DA extraction
            # ----------------------------------------

            part_at_turn = particles.at_turn

            x_norm = grid_details['x_normalized']
            y_norm = grid_details['y_normalized']

            nr = grid_details['num_r_y_points']
            nth = grid_details['num_theta_x_points']

            x2d = x_norm.reshape(nr, nth)
            y2d = y_norm.reshape(nr, nth)
            t2d = part_at_turn.reshape(nr, nth)

            x_DA = np.full(nth, np.nan)
            y_DA = np.full(nth, np.nan)

            for jj in range(nth):
                for ii in range(nr):

                    if t2d[ii, jj] != max_turns:

                        x_DA[jj] = x2d[ii, jj]
                        y_DA[jj] = y2d[ii, jj]

                        break

            r_DA = np.sqrt(x_DA**2 + y_DA**2)

            min_DA = np.nanmin(r_DA)

            # ----------------------------------------
            # Store
            # ----------------------------------------

            results.append({
                'qx': qx_target,
                'qy': qy_target,
                'radius': r,
                'theta': theta,
                'min_DA': min_DA
            })

            print(
                f'Qx={qx_target:.3f} '
                f'Qy={qy_target:.3f} '
                f'DA={min_DA:.2f}'
            )

        except Exception:

            results.append({
                'qx': qx_target,
                'qy': qy_target,
                'radius': r,
                'theta': theta,
                'min_DA': np.nan
            })

            print(
                f'FAILED: '
                f'Qx={qx_target:.3f}, '
                f'Qy={qy_target:.3f}'
            )

# ---------------------------------------------------
# Convert to dataframe
# ---------------------------------------------------

df = pd.DataFrame(results)

# ---------------------------------------------------
# Plot
# ---------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 8))

sc = ax.scatter(
    df['qx'],
    df['qy'],
    c=df['min_DA'],
    s=40,
    cmap='viridis'
)

cb = plt.colorbar(sc)
cb.set_label('Minimum DA [$\\sigma$]')

# nominal point
ax.plot(
    qx0,
    qy0,
    'r*',
    markersize=14,
    label='Nominal WP'
)

ax.set_xlabel(r'$Q_x$')
ax.set_ylabel(r'$Q_y$')

ax.set_title('Radial Working Point Scan')

ax.legend()

plt.tight_layout()
plt.show()