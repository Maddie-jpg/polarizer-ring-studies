#%%
import sys
import os

# Adds the parent directory to the search path
parent_dir = os.path.abspath('..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import xtrack as xt
import xpart as xp
import xfields as xf
import xobjects as xo
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import LatticeBuild.misalignments_corrections as mc
#%%

design=int(os.environ.get('DESIGN',1))
config=int(os.environ.get('CONFIG',1))
mode=os.environ.get('MODE','corrected')

# Which seed to plot: 'top' (highest P_eq), 'bottom' (lowest P_eq), or a specific
# integer seed value to look up directly.
SEED_CHOICE = os.environ.get('SEED_CHOICE', 'bottom')

#%%

if mode == 'corrected':
    pdr = xt.Environment.from_json(f'JSON Files/D{design}/C{config}/pdr_perfect.json')
    if design ==1 and config==1:
        mc.insert_BPMs_all_as_markers(pdr)
        mc.insert_correctors_var2(pdr)
    else:
        mc.insert_BPMs_all_as_markers(pdr)
        mc.insert_correctors(pdr)
else:
    pdr = xt.Environment.from_json(f'JSON Files/D{design}/C{config}/pdr_{mode}.json')
pdr.particle_ref.anomalous_magnetic_moment=0.001159652181

base_line = pdr.lines['ring'].copy()

#%%

# Pick the seed from the existing scan results, same convention as the deep-track script.
df = pd.read_csv(f'/Users/maddiewatson/Library/CloudStorage/OneDrive-Personal/University/Research year/polarizer-ring-studies/Results/D1/C1/Comparison/SpinTrackingResults_MisalignedVsCorrected.dat')

if SEED_CHOICE == 'top':
    row = df.nlargest(1, 'P_eq').iloc[0]
    seed_val = int(row['Seed'])
    seed_label = 'Best Seed'
elif SEED_CHOICE == 'bottom':
    row = df.nsmallest(1, 'P_eq').iloc[0]
    seed_val = int(row['Seed'])
    seed_label = 'Worst Seed'
else:
    seed_val = int(SEED_CHOICE)
    seed_label = f'Seed {seed_val}'

print(f"Plotting invariant spin vector for {seed_label} (Seed {seed_val})...")

#%%

line = base_line.copy()
line.discard_tracker()
line.build_tracker(_context=xo.ContextCpu(omp_num_threads=0))

line = mc.misalignments(line, 0.2e-3, seed=seed_val)

line.configure_radiation('mean')
tw = line.twiss(method='6d', radiation_integrals=True, eneloss_and_damping=True,
                spin=True, polarization=True)

if mode == 'corrected':
    try:
        line.discard_tracker()
        mc.orbit_correction(pdr, tw, threading=False, rcond_x=1e-4, rcond_y=1e-2)
    except:
        mc.orbit_correction(pdr, tw, threading=False, rcond_x=1e-4, rcond_y=1e-2)
    # Orbit correction changes the closed orbit/optics, so re-twiss to get the
    # n0 vector consistent with the corrected lattice.
    tw = line.twiss(method='6d', radiation_integrals=True, eneloss_and_damping=True,
                    spin=True, polarization=True)

#%%

s = tw.s
sx = tw.spin_x
sy = tw.spin_y
sz = tw.spin_z

# Sanity check: |n0| should be 1 everywhere (it's a unit vector by construction).
# Large deviations from 1 indicate a numerical issue with the twiss/spin solve.
n0_mag = np.sqrt(sx**2 + sy**2 + sz**2)
print(f"  |n0| range: [{n0_mag.min():.6f}, {n0_mag.max():.6f}] (should be ~1.0)")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(s, sx, label='$n_{0,x}$', color='tab:blue')
ax.plot(s, sy, label='$n_{0,y}$', color='tab:green')
ax.plot(s, sz, label='$n_{0,z}$', color='tab:red')

ax.set_xlabel('s (m)')
ax.set_ylabel('Invariant spin vector $n_0$ component')
ax.set_title(f'Invariant Spin Vector Along the Ring — {seed_label} (Seed {seed_val})')
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend()

plt.tight_layout()
out_path = f'Results/D{design}/C{config}/{mode}/InvariantSpinVector_{seed_label.replace(" ", "")}.png'
plt.savefig(out_path, dpi=300)
plt.close()

print(f"  Saved to {out_path}")