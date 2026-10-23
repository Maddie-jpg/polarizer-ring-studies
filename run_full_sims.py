"""
Run the full analysis chain for one or more lattices.

For each lattice the scripts below are run in turn as separate processes, with
the lattice passed through the DESIGN, CONFIG, MODE, PHASE and CHANGES
environment variables (read by paths.study_from_env in each script):

    perfect              macroparticles.py, analysis.py,
                         spin_tracking_single_seed.py, spin_tracking.py
    misaligned/corrected macroparticles.py, analysis.py

macroparticles.py runs first: analysis.py takes the normalised emittances for
its DA/MA studies from the TwissResults.json it writes.

The lattice JSONs must already exist (build them with LatticeBuild/lattice_build.py).

Run:  python run_full_sims.py
"""
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

SCRIPTS = {
    'perfect':    ['macroparticles.py', 'analysis.py',
                   'spin_tracking_single_seed.py', 'spin_tracking.py'],
    'misaligned': ['macroparticles.py', 'analysis.py'],
    'corrected':  ['macroparticles.py', 'analysis.py'],
}


def run_configuration(design, config, mode, phase, changes):
    """Run every script for `mode` on lattice D{design} C{config}.

    changes : variant tag in the JSON file name (e.g. 'FDF'), or None for none.
    Stops the whole run if a script fails.
    """
    env = os.environ.copy()
    env.update(DESIGN=str(design), CONFIG=str(config), MODE=mode, PHASE=str(phase),
               CHANGES=changes or '')   # '' = explicitly no tag

    for script in SCRIPTS[mode]:
        print(f'\n[Master] Starting {script} (D{design} C{config}, {phase}deg, '
              f'mode={mode}, changes={changes})')
        result = subprocess.run([sys.executable, script], env=env, cwd=REPO_ROOT)
        if result.returncode != 0:
            print(f'[Master Error] {script} failed for D{design} C{config} mode={mode}.')
            sys.exit(1)


if __name__ == '__main__':
    for mode in ('perfect', 'misaligned', 'corrected'):
        run_configuration(1, 9, mode, 90, 'FDF')
