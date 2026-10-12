"""
Parallel "realistic" tune scan of the dynamic aperture.

For every (Qx, Qy) on a grid the ring is re-matched with the production
matchingWP from LatticeBuild/linear_optics.py, then a DA study is tracked
with the same particle grid (xutil.generate_particle_grid) and the same DA
boundary definition as analysis.py. Returns the min-DA map.

Why it is faster than the serial version in wp_optimisation.py:
  * grid points run in separate worker processes (one per CPU core);
    each worker loads its own copy of the lattice, so the caller's
    ring / arc1R are never modified and no knob restoring is needed
  * each worker walks its rows in a snake order, so every match starts
    from a neighbouring solution (shorter matchingWP ramps)
  * the tracker is built once per worker, not once per point
  * turns / particles / radiation are arguments, so a coarse scan can
    be run first and refined around the best region

Use from wp_optimisation.py (or anywhere):
    import wp_da_scan
    QX, QY, DA = wp_da_scan.run_parallel(lattice_json, qx_vals, qy_vals,
                                         n_turns=512, n_particles=200)

Workers are separate Python processes started with this file
(`python wp_da_scan.py --worker ...`), which works the same from a script,
VS Code interactive cells or Jupyter.
"""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# =============================================================================
# DA helpers (same definition as analysis.py / wp_optimisation.py)
# =============================================================================
def get_DA_boundary(particles, num_r_steps, num_theta_steps, x_norm, y_norm):
    """First lost radius per angle; returns (x_DA, y_DA, min_DA) in sigma."""
    max_turns = np.max(particles.filter(particles.at_element == 0).at_turn)
    part_at_turn = particles.at_turn
    x_2d = x_norm.reshape(num_r_steps, num_theta_steps)
    y_2d = y_norm.reshape(num_r_steps, num_theta_steps)
    p_2d = part_at_turn.reshape(num_r_steps, num_theta_steps)
    x_DA = np.full(num_theta_steps, np.nan)
    y_DA = np.full(num_theta_steps, np.nan)
    for jj in range(num_theta_steps):
        for ii in range(num_r_steps):
            if p_2d[ii, jj] != max_turns:
                x_DA[jj], y_DA[jj] = x_2d[ii, jj], y_2d[ii, jj]
                break
    min_DA = np.nanmin(np.round(np.sqrt(x_DA ** 2 + y_DA ** 2), 1))
    return x_DA, y_DA, min_DA


def _study_params(s):
    return {
        'ini_cond_type': 'grid_DA', 'output_dir': 'out',
        'number_of_turns': s['n_turns'], 'number_of_particles': s['n_particles'],
        'inv1': 0, 'inv2': 0, 'start_element': s.get('start_element'),
        'ini_cond_nemittance_x': s['nemitt_x'], 'ini_cond_nemittance_y': s['nemitt_y'],
        'ini_cond_bunch_length': 4.8e-3, 'ini_cond_energy_spread': 2e-3,
        'ini_cond_energy_offset': None, 'new_closed_orbit': None,
        'covariance_dispertion_free': False,
    }


# =============================================================================
# Worker: runs a list of points in order, appends one JSON line per point
# =============================================================================
def _worker(cfg_file, out_file):
    import xobjects as xo
    import xtrack as xt
    xo.context_cpu.allow_no_prebuilt_kernel = True
    import LatticeBuild.linear_optics as lo
    import LatticeBuild.sextupole_configs as sc
    import xutil_DA_CC.xsuite_utilities as xutil

    s = json.loads(Path(cfg_file).read_text())
    quiet = lambda: contextlib.redirect_stdout(io.StringIO())

    pdr = xt.Environment.from_json(s['lattice_json'])
    ring, arc1R = pdr.lines['ring'], pdr.lines['arc1R']
    cell_arc, cell_tr = pdr.lines['cell_arc'], pdr.lines['cell_tr']

    # The straight-cell match needs the triplet near its seed; on a fully
    # built lattice it sits elsewhere and the periodic cell is unstable.
    if s.get('triplet_seed'):
        pdr['kQFtr'], pdr['kQDtr'] = s['triplet_seed']
    with quiet():
        cell_arc_opt, cell_tr_opt = lo._match_cells_3fold(
            pdr, cell_arc, cell_tr, mu_cell=s['mu_cell'])

    ring.build_tracker(_context=xo.ContextCpu())         # serial: one core per worker
    study = _study_params(s)

    for qx, qy in s['points']:
        row = dict(qx_t=qx, qy_t=qy, DA=None)
        t0 = time.time()
        try:
            with quiet():
                lo.matchingWP(qx, qy, cell_arc_opt, cell_arc, arc1R,
                              n_periods=s['n_periods'],
                              betay_DS_target=s.get('betay_DS_target'))
            ring.configure_radiation(model=None)
            tw = ring.twiss(method='4d')
            row.update(qx=float(tw.qx), qy=float(tw.qy),
                       betx_max=float(tw.betx.max()), bety_max=float(tw.bety.max()))
            if abs(tw.qx - qx) > s['tune_tol'] or abs(tw.qy - qy) > s['tune_tol']:
                raise RuntimeError(f'tune match missed ({tw.qx:.4f}, {tw.qy:.4f})')
            if s.get('sext_knobs'):                       # re-zero chromaticity here
                with quiet():
                    sc.ChromCorrect(ring, pdr, s['sext_knobs'])
            ring.configure_radiation(model=s['radiation'])
            with quiet():
                particles, grid = xutil.generate_particle_grid(ring, study)
                ring.track(particles, num_turns=s['n_turns'],
                           turn_by_turn_monitor=False, with_progress=False)
            particles.sort(interleave_lost_particles=True)
            _, _, da = get_DA_boundary(particles, grid['num_r_y_points'],
                                       grid['num_theta_x_points'],
                                       grid['x_normalized'], grid['y_normalized'])
            row['DA'] = float(da)
        except Exception as e:
            row['error'] = f'{type(e).__name__}: {str(e)[:150]}'
        row['time'] = time.time() - t0
        with open(out_file, 'a') as f:
            f.write(json.dumps(row) + '\n')


# =============================================================================
# Driver
# =============================================================================
def _snake(qx_vals, qy_vals, rows):
    pts = []
    for k, i in enumerate(rows):
        cols = range(len(qx_vals)) if k % 2 == 0 else reversed(range(len(qx_vals)))
        pts += [(float(qx_vals[j]), float(qy_vals[i])) for j in cols]
    return pts


def run_parallel(lattice_json, qx_vals, qy_vals, *, nemitt_x, nemitt_y,
                 n_workers=None, n_turns=1000, n_particles=200, radiation='mean',
                 sext_knobs=None, n_periods=6, mu_cell=0.25, betay_DS_target=None,
                 triplet_seed=(1.8, -3.2), tune_tol=2e-3, start_element=None,
                 progress_every=10):
    """Scan min DA over qx_vals x qy_vals using n_workers processes.

    lattice_json   saved Environment to scan (e.g. paths.lattice_json_path(...))
    n_workers      default: number of CPU cores - 1
    n_turns, n_particles, radiation
                   tracking settings; 'mean' as before, None is ~15 % faster
    sext_knobs     e.g. ['k2XFarc', 'k2XDarc'] to re-correct chromaticity to
                   zero at every point (slower, but compares like with like);
                   None keeps the lattice's sextupole strengths
    Returns QX, QY, DA_map (np.nan where the match or tracking failed) and a
    list of per-point records (tunes reached, beta max, errors, time).
    """
    qx_vals, qy_vals = np.asarray(qx_vals, float), np.asarray(qy_vals, float)
    n_workers = n_workers or max(1, (os.cpu_count() or 2) - 1)
    n_workers = min(n_workers, len(qy_vals) * len(qx_vals))
    # contiguous row blocks per worker -> neighbours stay in the same worker
    row_blocks = np.array_split(np.arange(len(qy_vals)), n_workers)
    tmp = Path(tempfile.mkdtemp(prefix='wp_da_scan_'))
    base = dict(lattice_json=str(lattice_json), n_turns=int(n_turns),
                n_particles=int(n_particles), radiation=radiation,
                nemitt_x=float(nemitt_x), nemitt_y=float(nemitt_y),
                sext_knobs=sext_knobs, n_periods=n_periods, mu_cell=mu_cell,
                betay_DS_target=betay_DS_target, triplet_seed=triplet_seed,
                tune_tol=tune_tol, start_element=start_element)
    procs, outs = [], []
    total = 0
    for w, rows in enumerate(row_blocks):
        if len(rows) == 0:
            continue
        pts = _snake(qx_vals, qy_vals, rows)
        total += len(pts)
        cfg, out = tmp / f'cfg{w}.json', tmp / f'out{w}.jsonl'
        cfg.write_text(json.dumps({**base, 'points': pts}))
        outs.append(out)
        procs.append(subprocess.Popen(
            [sys.executable, '-W', 'ignore', str(Path(__file__).resolve()),
             '--worker', str(cfg), str(out)],
            cwd=str(REPO_ROOT), stdout=subprocess.DEVNULL,
            stderr=open(tmp / f'log{w}.txt', 'w')))
    print(f'[wp_da_scan] {total} points on {len(procs)} workers '
          f'({n_turns} turns, {n_particles} particles, radiation={radiation}); '
          f'temp files in {tmp}')

    t0, last = time.time(), 0
    while any(p.poll() is None for p in procs):
        time.sleep(5)
        done = sum(sum(1 for _ in open(o)) for o in outs if o.exists())
        if done - last >= progress_every:
            last = done
            el = time.time() - t0
            print(f'[wp_da_scan] {done}/{total} done, {el/60:.1f} min elapsed, '
                  f'~{el/done*(total-done)/60:.1f} min left')
    for w, p in enumerate(procs):
        if p.returncode != 0:
            print(f'[wp_da_scan] worker {w} exited with code {p.returncode}; '
                  f'see {tmp / f"log{w}.txt"}')

    records = [json.loads(l) for o in outs if o.exists() for l in open(o)]
    QX, QY = np.meshgrid(qx_vals, qy_vals)
    DA = np.full(QX.shape, np.nan)
    ix = {round(v, 10): j for j, v in enumerate(qx_vals)}
    iy = {round(v, 10): i for i, v in enumerate(qy_vals)}
    for r in records:
        if r.get('DA') is not None:
            DA[iy[round(r['qy_t'], 10)], ix[round(r['qx_t'], 10)]] = r['DA']
    n_ok = int(np.isfinite(DA).sum())
    print(f'[wp_da_scan] finished in {(time.time()-t0)/60:.1f} min: '
          f'{n_ok}/{total} points with a DA, {total - n_ok} failed/unreachable')
    return QX, QY, DA, records


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--worker':
        _worker(sys.argv[2], sys.argv[3])
    else:
        sys.exit('usage: python wp_da_scan.py --worker <cfg.json> <out.jsonl> '
                 '(normally started by run_parallel)')
