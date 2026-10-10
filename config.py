"""
Shared physical constants and study defaults for the polarizer-ring studies.

This is the ONE place to change a beam energy, working point, error sigma or
seed. Every script (lattice build, analysis, tracking) should import from
here rather than defining its own copy.

Units: energies in eV, lengths in m, angles in rad, frequencies in Hz.
"""

# -----------------------------------------------------------------------------
# Beam and RF
# -----------------------------------------------------------------------------
E0 = 2.86e9                         # Reference beam energy [eV]
VRF = 8.000e6                       # Total RF voltage [V]
F_RF_TARGET = 4.0e8                 # RF frequency is the harmonic of f_rev closest to this [Hz]
ANOMALOUS_MAGNETIC_MOMENT = 0.001159652181   # Electron/positron a = (g-2)/2, for spin tracking

# -----------------------------------------------------------------------------
# Default working points (Qx, Qy) used by the lattice builders in
# LatticeBuild/linear_optics.py when no WP is passed explicitly.
# The working point a lattice actually has is always read from its twiss,
# not from here.
# -----------------------------------------------------------------------------
WP_D1 = (15.42, 15.38)              # D1, 90 deg arcs
WP_D1_120 = (18.43, 18.36)          # D1, 120 deg arcs
WP_D2 = (11.72, 11.375)
WP_D3 = (13.375, 12.775)            # NB: root constants.py had (16.53, 16.35); see README

# -----------------------------------------------------------------------------
# Alignment / field errors and orbit correction
# -----------------------------------------------------------------------------
MISALIGN_SIGMA = 0.25e-3            # RMS magnet shift [m] / roll [rad], truncated at MISALIGN_CUT sigma
CORRECTOR_SIGMA = 0.25e-3           # RMS corrector misalignment
BPM_SIGMA = 0.25e-3                 # RMS BPM reading error
MISALIGN_CUT = 2.5                  # Gaussian truncation [sigma]
RCOND_X = 1e-4                      # SVD cut-off for horizontal orbit correction
RCOND_Y = 1e-2                      # SVD cut-off for vertical orbit correction

DEFAULT_SEED = 123456789            # Seed used to build pdr_misaligned / pdr_corrected JSONs

# -----------------------------------------------------------------------------
# Default study: which lattice a script uses when run on its own (not via
# run_full_sims.py). Override with the DESIGN/CONFIG/MODE/PHASE/CHANGES
# environment variables.
# -----------------------------------------------------------------------------
DEFAULT_STUDY = dict(design=1, config=9, mode='perfect', phase=90, changes='FDF')

# -----------------------------------------------------------------------------
# External input files (override with environment variables if needed)
# -----------------------------------------------------------------------------
import os as _os
from pathlib import Path as _Path
INJECTED_BEAM_FILE = _os.environ.get(
    'BEAM_FILE',
    str(_Path(__file__).resolve().parent / 'PositronBeam_2p86GeV_PolarizedEbeam' / 'beam_ECS_04092026.dat'))
