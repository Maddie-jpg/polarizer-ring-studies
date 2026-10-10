"""
Detailed spin study of a single error seed.

For one seed (SEED below) this runs: a spin-tune resonance scan on the perfect
ring, n0 vs spin tune for the misaligned and corrected machine, the Qy-spin
coupling comparison, and long tracking (mf.deep_track_single) of the
misaligned and corrected machines.

Inputs   lattice chosen by DESIGN/CONFIG/PHASE/CHANGES (see paths.study_from_env)
Outputs  Results/.../Spin/Seed_<SEED>/

Run:     python spin_tracking_single_seed.py   (or via run_full_sims.py)
"""
# %%
import os

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
import paths
import random
import csv


#%%

# Fixed seed so results are reproducible; use random.randint(0, int(1e6)) for a new one.
SEED = 283792

study = paths.study_from_env()
design, config, phase, changes = study['design'], study['config'], study['phase'], study['changes']

long_scan_turns = 20000  

# %%
pdr = paths.load_lattice(design, config, 'perfect', phase, changes)
mc.setup_spin_reference(pdr.lines['ring'])
mc.insert_bpms_and_correctors(pdr, design, config)

line = pdr.lines['ring']
# Corrector errors are applied per seed inside the mf.* helpers, not here.

line.configure_spin('auto')

base_line = line.copy()

results_dir = mf.results_dir(design, config, phase, changes=changes, metric='Spin', sub=f'Seed_{SEED}')
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