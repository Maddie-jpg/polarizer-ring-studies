"""
File locations for lattices and results.

All paths are built from the repository root (the folder this file lives in),
so scripts work no matter which directory they are launched from.

Naming scheme
-------------
Lattices: JSON_Files/D{design}/C{config}/pdr_{mode}_{phase}[_{changes}].json
Results:  Results/D{design}/C{config}/{phase}deg[_{changes}]/{metric}/{sub}/{sub2}

    design   lattice layout family, e.g. 1 = three-fold periodicity
    config   sextupole configuration within that design (see sextupole_configs.py)
    mode     'perfect' | 'misaligned' | 'corrected'
    phase    arc-cell phase advance in degrees (90 or 120)
    changes  optional variant tag, e.g. 'FDF', 'DSchange'

This module only imports the standard library (xtrack is imported lazily
inside load_lattice), so lightweight scripts can use it cheaply.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
JSON_DIR = REPO_ROOT / 'JSON_Files'
RESULTS_DIR = REPO_ROOT / 'Results'

MODES = ('perfect', 'misaligned', 'corrected')


def _tag(changes):
    """'' for no variant, otherwise '_<changes>'."""
    return f'_{changes}' if changes else ''


def lattice_json_path(design, config, mode='perfect', phase=90, changes=None):
    """Path of the saved xtrack Environment for one lattice variant."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    return JSON_DIR / f'D{design}' / f'C{config}' / f'pdr_{mode}_{phase}{_tag(changes)}.json'


def results_dir(design, config, phase, changes=None, metric=None, sub=None, sub2=None):
    """Create (if needed) and return the output folder for one study, as a str.

    Example: results_dir(1, 9, 90, 'FDF', metric='DA_MA')
             -> <repo>/Results/D1/C9/90deg_FDF/DA_MA
    """
    path = RESULTS_DIR / f'D{design}' / f'C{config}' / f'{phase}deg{_tag(changes)}'
    for part in (metric, sub, sub2):
        if part:
            path = path / str(part)
    os.makedirs(path, exist_ok=True)
    return str(path)


def apertures_json_path(design, config, phase, changes=None, mode='perfect'):
    """Chamber sizes written by macroparticles.py (physical_aperture_study)."""
    return (RESULTS_DIR / f'D{design}' / f'C{config}' / f'{phase}deg{_tag(changes)}'
            / 'PhysicalAperture' / mode / 'apertures.json')


def load_lattice(design, config, mode='perfect', phase=90, changes=None):
    """Load a saved lattice and return the xtrack Environment.

    The ring is pdr.lines['ring']; the single period is pdr.lines['period'].
    """
    import xtrack as xt
    path = lattice_json_path(design, config, mode, phase, changes)
    if not path.exists():
        raise FileNotFoundError(
            f'{path} not found. Build it first with LatticeBuild/lattice_build.py '
            f'(mode={mode!r}).')
    return xt.Environment.from_json(str(path))


def study_from_env():
    """Read DESIGN, CONFIG, MODE, PHASE, CHANGES set by run_full_sims.py.

    Anything not set falls back to config.DEFAULT_STUDY, so every script
    picks the same lattice when run on its own. CHANGES='' (or 'None') means
    explicitly no variant tag.

    Returns a dict with keys design, config, mode, phase, changes.
    """
    from config import DEFAULT_STUDY
    env = os.environ
    changes = env.get('CHANGES', DEFAULT_STUDY['changes'])
    if changes in ('', 'None', 'none'):
        changes = None
    return dict(
        design=int(env.get('DESIGN', DEFAULT_STUDY['design'])),
        config=int(env.get('CONFIG', DEFAULT_STUDY['config'])),
        mode=env.get('MODE', DEFAULT_STUDY['mode']),
        phase=int(env.get('PHASE', DEFAULT_STUDY['phase'])),
        changes=changes,
    )
