import os
import subprocess
import sys

def run_configuration(design, config, mode,phase,changes):
    """
    Runs the appropriate scripts for a given design, config, mode, phase, and changes.
    """
    env = os.environ.copy()
    env['DESIGN'] = str(design)
    env['CONFIG'] = str(config)
    env['MODE'] = mode
    env['PHASE'] = str(phase)
    
    if changes is not None:
        env['CHANGES'] = str(changes)
    else:
        env.pop('CHANGES', None) 

    # Dynamically determine which scripts to run based on the mode
    if mode == 'perfect':
        scripts = ['analysis.py', 'macroparticles.py','spin_tracking_single_seed.py','spin_tracking.py']
    else:
        scripts = ['analysis.py', 'macroparticles.py']

    for script in scripts:
        print(f'\n[Master] Starting run of: {script} (Design: {design}, Config: {config}, Mode: {mode})')

        result = subprocess.run([sys.executable, script], env=env)

        if result.returncode != 0:
            print(f'[Master Error] {script} failed for Mode: {mode}, Config: {config}.')
            sys.exit(1)

if __name__ == '__main__':
    run_configuration(1, 9, 'perfect', 90, None)
    run_configuration(1, 9, 'misaligned', 90, None)
    run_configuration(1, 9, 'corrected', 90, None)