# %%
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
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
import LatticeBuild.misalignments_corrections as mc
import my_functions as mf
import random
import csv


#%%

SEED = random.randint(0,int(1e6))
SEED = 283792

design = int(os.environ.get('DESIGN', 1))
config = int(os.environ.get('CONFIG', 9))
phase=int(os.environ.get('PHASE',90))
changes=os.environ.get('CHANGES',None)

long_scan_turns = 20000  

# %%
if changes is not None:
    pdr= xt.Environment.from_json(f"JSON Files/D{design}/C{config}/pdr_perfect_{phase}_{changes}.json")
else:
    pdr= xt.Environment.from_json(f"JSON Files/D{design}/C{config}/pdr_perfect_{phase}.json")

energy=2.86e9
pdr.lines['ring'].particle_ref.anomalous_magnetic_moment = 0.001159652181
pdr.lines['ring'].particle_ref.kinetic_energy0 = energy

if design == 1 and config == 1:
    mc.insert_BPMs_all_as_markers(pdr)
    mc.insert_correctors_var2(pdr)
else:
    mc.insert_BPMs_all_as_markers(pdr)
    mc.insert_correctors(pdr)


line = pdr.lines['ring']
mc.misalignments_correctors(line,0.2e-3,SEED+1)

line.configure_spin('auto')

base_line = line.copy()

results_dir = f'Results/D{design}/C{config}/SingleSeed_{SEED}'
os.makedirs(results_dir, exist_ok=True)

scan = mf.spin_tune_resonance_scan(line, nu_min=5.5, nu_max=7.5, n_points=80,
                                 misalign_sigma=0.25e-3, seed=SEED)
mf.plot_spin_resonance_scan(scan, out_path=f'{results_dir}/spin_resonance_scan.png')



#%%

#row=mf.assess_seed_resonance_excitation(SEED, True, base_line,long_scan_turns)
#print(row)
mf.plot_invariant_spin_vector(base_line,SEED, apply_correction=False,out_path=f'{results_dir}/InvariantSpinVector_Misaligned.png')
mf.plot_invariant_spin_vector(base_line,SEED, apply_correction=True,out_path=f'{results_dir}/InvariantSpinVector_Corrected.png')

mf.track_single_particle_nx1(base_line,SEED, apply_correction=False,out_path=f'{results_dir}/SpinVector_nx1_Misaligned.png')
mf.track_single_particle_nx1(base_line,SEED, apply_correction=True,out_path=f'{results_dir}/SpinVector_nx1_Corrected.png')

results = mf.n0_vs_spin_tune_scan(base_line,SEED, nu_min=5.5, nu_max=7.5,
                                 n_points=80, apply_correction=False)
mf.plot_n0_vs_spin_tune(results, out_path=f'{results_dir}/n0_vs_spin_tune_misaligned.png')

results = mf.n0_vs_spin_tune_scan(base_line,SEED, nu_min=5.5, nu_max=7.5,
                                 n_points=80, apply_correction=True)
mf.plot_n0_vs_spin_tune(results, out_path=f'{results_dir}/n0_vs_spin_tune_corrected.png')

coupling = mf.compare_qy_spin_coupling_across_branches(line, SEED)
print(coupling)

data_misaligned = mf.deep_track_single(base_line,SEED,long_scan_turns, apply_correction=False)

data_corrected = mf.deep_track_single(base_line,SEED,long_scan_turns, apply_correction=True)



mf.plot_seed_with_textbox(
    data_misaligned, 'Misaligned',
    f'{results_dir}/Polarization_Misaligned.png')
mf.plot_seed_with_textbox(
    data_corrected, 'Corrected',
    f'{results_dir}/Polarization_Corrected.png')