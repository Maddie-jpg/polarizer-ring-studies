import os
import subprocess
import sys

def run_configuration(design, config, mode):
    """
    Runs the appropriate scripts for a given design, config, and mode.
    """
    env = os.environ.copy()
    env['DESIGN'] = str(design)
    env['CONFIG'] = str(config)
    env['MODE'] = mode

    # Dynamically determine which scripts to run based on the mode
    if mode == 'perfect':
        scripts = ['analysis.py', 'macroparticles.py']
    else:
        scripts = ['spin_tracking.py','spin_tracking_deep.py']

    for script in scripts:
        print(f'\n[Master] Starting run of: {script} (Design: {design}, Config: {config}, Mode: {mode})')

        result = subprocess.run([sys.executable, script], env=env)

        if result.returncode != 0:
            print(f'[Master Error] {script} failed for Mode: {mode}, Config: {config}.')
            sys.exit(1)

if __name__ == '__main__':

    run_configuration(1, 2, 'misaligned')
    run_configuration(1, 2, 'corrected')
