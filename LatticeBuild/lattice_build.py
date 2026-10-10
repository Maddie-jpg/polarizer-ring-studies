"""
Build a polarizer-ring lattice and save it as an xtrack Environment JSON.

Three modes, run in this order for a new design/config:
  perfect     build the linear optics (linear_optics.py) and add sextupoles
              (sextupole_configs.py)                    -> pdr_perfect_*.json
  misaligned  load pdr_perfect_*, apply magnet errors   -> pdr_misaligned_*.json
  corrected   load pdr_misaligned_*, add BPMs/correctors, misalign correctors,
              correct the orbit                         -> pdr_corrected_*.json

Files are saved to JSON_Files/D{design}/C{config}/pdr_{mode}_{phase}[_{changes}].json
(see paths.py). Analysis scripts then load them by the same naming scheme.

Edit the "Study settings" cell and run (works from any directory):
    python LatticeBuild/lattice_build.py
"""
#%%
import sys
from pathlib import Path

# Make the repository root importable (config.py, paths.py).
REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import xobjects as xo

import config as cfg
import paths
import LatticeBuild.linear_optics as lo
import LatticeBuild.sextupole_configs as sc
import LatticeBuild.misalignments_corrections as mc

xo.context_cpu.allow_no_prebuilt_kernel = True


#%% Study settings
design = 1
config = 9
mode = 'perfect'        # 'perfect', 'misaligned' or 'corrected'
phase = 90
changes = 'FDF'         # variant tag in the file name, or None

SEED = cfg.DEFAULT_SEED

#%% Build
if mode == 'perfect':
    # Linear optics - choose the builder for this design
    pdr = lo.three_fold_periodicity(fringe_fields=True, matched=True, WP=(15.80, 13.87),
                                    phase_advance=0.25, betay_DS_target=7.0,
                                    triplet_seed=(1.8, -3.2))
    sc.config_D1_C9(pdr)

    # Example for design 2:
    # pdr = lo.two_fold_racetrack_3straight(fringe_fields=True, matched=True, WP=(13.65, 13.23),
    #                                       phase_advance=0.25, betay_DS_target=False)
    # sc.config_D2_C1(pdr)

elif mode == 'misaligned':
    pdr = paths.load_lattice(design, config, 'perfect', phase, changes)
    ring = pdr.lines['ring']
    mc.misalignments(ring, cfg.MISALIGN_SIGMA, SEED)

elif mode == 'corrected':
    pdr = paths.load_lattice(design, config, 'misaligned', phase, changes)
    ring = pdr.lines['ring']

    # NB: the saved corrected lattices use the var2 corrector layout for every
    # design, while the analysis scripts use mc.insert_bpms_and_correctors
    # (var2 only for D1 C1). Kept as before - check which one you intend.
    mc.insert_BPMs_all_as_markers(pdr)
    mc.insert_correctors_var2(pdr)
    mc.misalignments_correctors(ring, cfg.CORRECTOR_SIGMA, SEED)

    twiss = ring.twiss(method='6d', radiation_integrals=True, eneloss_and_damping=True,
                       spin=True, polarization=True)
    ring.discard_tracker()
    mc.correct_orbit_with_fallback(ring, twiss, SEED,
                                   rcond_x=cfg.RCOND_X, rcond_y=cfg.RCOND_Y)

else:
    raise ValueError(f"Unknown mode {mode!r}; use one of {paths.MODES}")

#%% Save
output_file = paths.lattice_json_path(design, config, mode, phase, changes)
output_file.parent.mkdir(parents=True, exist_ok=True)
pdr.to_json(str(output_file))
print(f'Saved {output_file}')
