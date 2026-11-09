#%%
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

At every point it can also record, from that point's re-matched optics
(cheap compared with the tracking):
  * first-order RDTs (xt.rdt_first_order_perturbation), max and rms of |f|
    around the ring -> keys '<rdt>_max', '<rdt>_rms'
  * amplitude-detuning coefficients dQx/dJx, dQy/dJy, dQx/dJy, dQy/dJx
    [1/m] (amplitude_detuning) -> 'det_xx', 'det_yy', 'det_xy', 'det_yx'
  * second-order chromaticity Q'' (ring.get_non_linear_chromaticity)
    -> 'ddqx', 'ddqy'
metric_maps() turns those records into maps on the same grid as the DA.

Use from wp_optimisation.py (or anywhere):
    import wp_da_scan
    QX, QY, DA = wp_da_scan.run_parallel(lattice_json, qx_vals, qy_vals,
                                         n_turns=512, n_particles=200)

Workers are separate Python processes started with this file
(`python wp_da_scan.py --worker ...`), which works the same from a script,
VS Code interactive cells or Jupyter.
"""
import collections
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
    """First lost radius per angle; returns (x_DA, y_DA, min_DA) in sigma.

    An angle where nothing is lost gets the largest radius tracked at that
    angle (a lower bound), so a point whose DA is beyond the whole grid
    returns the grid edge instead of NaN -- before, those points were
    counted as failed. analysis.py's definition is otherwise unchanged.
    """
    max_turns = np.max(particles.filter(particles.at_element == 0).at_turn)
    part_at_turn = particles.at_turn
    x_2d = x_norm.reshape(num_r_steps, num_theta_steps)
    y_2d = y_norm.reshape(num_r_steps, num_theta_steps)
    p_2d = part_at_turn.reshape(num_r_steps, num_theta_steps)
    x_DA = x_2d[-1].astype(float).copy()          # default: grid edge
    y_DA = y_2d[-1].astype(float).copy()
    for jj in range(num_theta_steps):
        for ii in range(num_r_steps):
            if p_2d[ii, jj] != max_turns:
                x_DA[jj], y_DA[jj] = x_2d[ii, jj], y_2d[ii, jj]
                break
    min_DA = np.nanmin(np.round(np.sqrt(x_DA ** 2 + y_DA ** 2), 1))
    return x_DA, y_DA, min_DA


# Sextupole-driven (3rd-order) RDTs: resonances 3Qx, Qx, Qx+2Qy, Qx, Qx-2Qy
DEFAULT_RDTS = ('f3000', 'f2100', 'f1020', 'f1011', 'f1002')


def amplitude_detuning(ring, gemitt_x, gemitt_y, amps=(1, 2, 3, 4),
                       other=0.5, turns=1024):
    """dQ/dJ [1/m] from tracking: particles at amps x sigma in one plane
    (other x sigma in the other), tunes by NAFF on the complex normalised
    signal, linear fit of Q against J = a^2 eps / 2.

    The RF is switched off while tracking (and restored), so delta stays at
    0. With it on, particles launched on the 4D orbit do a ~0.7 mm
    synchrotron oscillation through the dispersion and NAFF returns Qs
    instead of the betatron tune -- this is also why
    ring.get_amplitude_detuning_coefficients gives nonsense on this ring.
    Returns det_xx, det_yy, det_xy (dQx/dJy) and det_yx (dQy/dJx); the last
    two should agree, which is a useful check.
    """
    import nafflib as nl
    amps = np.asarray(amps, float)
    k = len(amps)
    p0 = ring.particle_ref
    bg = float(p0.beta0[0] * p0.gamma0[0])
    tw = ring.twiss(method='4d')
    part = ring.build_particles(
        method='4d', zeta=0, delta=0,
        x_norm=np.r_[amps, np.full(k, other)], y_norm=np.r_[np.full(k, other), amps],
        nemitt_x=gemitt_x * bg, nemitt_y=gemitt_y * bg)
    cav = [n for n, e in zip(ring.element_names, ring.elements)
           if e.__class__.__name__ == 'Cavity']
    volts = [ring[n].voltage for n in cav]
    for n in cav:
        ring[n].voltage = 0
    try:
        ring.track(part, num_turns=turns, turn_by_turn_monitor=True)
    finally:
        for n, v in zip(cav, volts):
            ring[n].voltage = v
    if np.any(part.state <= 0):
        raise RuntimeError('detuning particles lost')
    m = ring.record_last_track

    def tune(u, pu, b, a):
        u, pu = u - u.mean(), pu - pu.mean()
        return nl.get_tune(u / np.sqrt(b) - 1j * (a * u + b * pu) / np.sqrt(b)) % 1.0

    qx = np.array([tune(m.x[i], m.px[i], tw.betx[0], tw.alfx[0]) for i in range(2 * k)])
    qy = np.array([tune(m.y[i], m.py[i], tw.bety[0], tw.alfy[0]) for i in range(2 * k)])
    if abs(qx[0] - tw.qx % 1) > 0.01 or abs(qy[k] - tw.qy % 1) > 0.01:
        raise RuntimeError(f'NAFF tunes ({qx[0]:.4f}, {qy[k]:.4f}) do not match '
                           f'twiss ({tw.qx % 1:.4f}, {tw.qy % 1:.4f})')
    Jx, Jy = amps ** 2 * gemitt_x / 2, amps ** 2 * gemitt_y / 2
    slope = lambda J, q: float(np.polyfit(J, q, 1)[0])
    return dict(det_xx=slope(Jx, qx[:k]), det_yx=slope(Jx, qy[:k]),
                det_xy=slope(Jy, qx[k:]), det_yy=slope(Jy, qy[k:]))


def optics_metrics(ring, rdts=DEFAULT_RDTS, detuning=True, nl_chroma=True,
                   nemitt_x=1e-6, nemitt_y=1e-6, detuning_turns=1024,
                   eq_emittance=True):
    """RDTs, amplitude detuning, Q'' and equilibrium emittance of `ring` as
    it is now.

    Radiation must be off on entry, and is off again on return. Returns a
    flat dict of floats; anything that fails is reported under
    'metrics_error' instead of raising, so a failed metric never costs the DA
    of that point.

    eq_emittance : also record the synchrotron-radiation equilibrium
        emittances 'eq_gemitt_x', 'eq_gemitt_y', 'eq_gemitt_zeta' [m rad] and
        the energy loss per turn 'eneloss_turn' [eV]. This is the one metric
        here that needs radiation ON, so it is switched on for that twiss
        only and switched back off afterwards. eq_gemitt_y is ~0 for a flat
        lattice with no coupling, so the horizontal one is the useful map.
    """
    import xtrack as xt
    out = {}
    try:
        if rdts:
            tw = ring.twiss(method='4d')
            st = ring.get_table(attr=True)
            res = xt.rdt_first_order_perturbation(list(rdts), twiss=tw,
                                                  strengths=st, feed_down=False)
            for r in rdts:
                a = np.abs(np.asarray(res[r]))
                out[f'{r}_max'] = float(a.max())
                out[f'{r}_rms'] = float(np.sqrt(np.mean(a ** 2)))
        if detuning:
            p0 = ring.particle_ref
            bg = float(p0.beta0[0] * p0.gamma0[0])
            out.update(amplitude_detuning(ring, nemitt_x / bg, nemitt_y / bg,
                                          turns=detuning_turns))
        if nl_chroma:
            nlc = ring.get_non_linear_chromaticity()
            out['ddqx'] = float(nlc.dnqx[2])
            out['ddqy'] = float(nlc.dnqy[2])
        if eq_emittance:
            # The only metric here that needs radiation on. Put it back off in
            # the finally, so the contract of this function is unchanged and a
            # failure here cannot leave radiation on for the next caller.
            try:
                ring.configure_radiation(model='mean')
                try:
                    tw6 = ring.twiss(method='6d', radiation_analysis=True)
                except TypeError:   # xtrack < 0.118 spells it differently
                    tw6 = ring.twiss(method='6d', eneloss_and_damping=True)
                for k in ('eq_gemitt_x', 'eq_gemitt_y', 'eq_gemitt_zeta'):
                    if hasattr(tw6, k):
                        out[k] = float(np.asarray(getattr(tw6, k)).ravel()[0])
                # 'eneloss_turn' was renamed 'energy_loss'; prefer the new
                # name so newer xtrack does not warn, keep the old key.
                for k in ('energy_loss', 'eneloss_turn'):
                    if hasattr(tw6, k):
                        out['eneloss_turn'] = float(
                            np.asarray(getattr(tw6, k)).ravel()[0])
                        break
            finally:
                ring.configure_radiation(model=None)
    except Exception as e:
        out['metrics_error'] = f'{type(e).__name__}: {str(e)[:150]}'
    return out


def metric_maps(records, qx_vals, qy_vals, keys=None):
    """Turn per-point records into {key: 2-D map} on the scan grid (same
    shape and orientation as the DA map). keys=None takes every metric found."""
    qx_vals, qy_vals = np.asarray(qx_vals, float), np.asarray(qy_vals, float)
    ix = {round(v, 10): j for j, v in enumerate(qx_vals)}
    iy = {round(v, 10): i for i, v in enumerate(qy_vals)}
    skip = {'qx_t', 'qy_t', 'qx', 'qy', 'DA', 'DA_at_grid_edge', 'time', 'error', 'metrics_error',
            'betx_max', 'bety_max'}
    if keys is None:
        keys = sorted({k for r in records for k in r if k not in skip})
    maps = {k: np.full((len(qy_vals), len(qx_vals)), np.nan) for k in keys}
    for r in records:
        i, j = iy[round(r['qy_t'], 10)], ix[round(r['qx_t'], 10)]
        for k in keys:
            if r.get(k) is not None:
                maps[k][i, j] = r[k]
    return maps


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

            # Second half of the production match (see _run_standard_matching).
            # The sextant match alone only lands on the tune AT the lattice's
            # own working point; 0.9 away it misses by up to ~0.4, which is
            # 200x tune_tol -- so without this every off-centre point in a scan
            # dies on 'tune match missed'. The trim closes it exactly in ~2 s
            # for a ~1 % gradient change. row['trim_pct'] records how hard it
            # had to pull: above a few per cent the sextant match did not
            # converge and the trim is buying tune by spending beta.
            trim = s.get('trim_knobs') or ()
            if trim:
                before = {k: ring.varval[k] for k in trim}
                with quiet():
                    ring.match(method='4d',
                               vary=[xt.Vary(k, step=1e-5) for k in trim],
                               targets=[xt.Target('qx', qx, tol=1e-6),
                                        xt.Target('qy', qy, tol=1e-6)])
                row['trim_pct'] = 100 * max(
                    abs(ring.varval[k] - before[k]) / max(abs(before[k]), 1e-12)
                    for k in trim)

            tw = ring.twiss(method='4d')
            row.update(qx=float(tw.qx), qy=float(tw.qy),
                       betx_max=float(tw.betx.max()), bety_max=float(tw.bety.max()))
            if abs(tw.qx - qx) > s['tune_tol'] or abs(tw.qy - qy) > s['tune_tol']:
                raise RuntimeError(f'tune match missed ({tw.qx:.4f}, {tw.qy:.4f})')
            if s.get('sext_knobs'):                       # re-zero chromaticity here
                with quiet():
                    sc.ChromCorrect(ring, pdr, s['sext_knobs'])
            if s.get('metrics'):
                ring.configure_radiation(model=None)
                with quiet():
                    row.update(optics_metrics(
                        ring, rdts=s['metrics'].get('rdts', DEFAULT_RDTS),
                        detuning=s['metrics'].get('detuning', True),
                        nl_chroma=s['metrics'].get('nl_chroma', True),
                        eq_emittance=s['metrics'].get('eq_emittance', True),
                        nemitt_x=s['nemitt_x'], nemitt_y=s['nemitt_y']))
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
            r_edge = np.round(np.hypot(grid['x_normalized'], grid['y_normalized']).max(), 1)
            row['DA_at_grid_edge'] = bool(da >= r_edge - 1e-9)   # True: DA is a lower bound
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
                 sext_knobs=None, n_periods=6, mu_cell=None, betay_DS_target=None,
                 triplet_seed=(1.8, -3.2), tune_tol=2e-3, start_element=None,
                 trim_knobs=('kQFarc', 'kQDarc', 'kQFarcM', 'kQDarcM'),
                 progress_every=10, metrics=True, stall_timeout=1200):
    """Scan min DA over qx_vals x qy_vals using n_workers processes.

    lattice_json   saved Environment to scan (e.g. paths.lattice_json_path(...))
    n_workers      default: number of CPU cores - 1. Each worker holds its own
                   lattice + tracker at ~0.17 GB, so cap this on a many-core
                   box rather than letting it default
    stall_timeout  seconds with no point finishing before the remaining
                   workers are reported and killed (default 1200). None waits
                   forever, which is what used to hang whole scans
    trim_knobs     ring gradients trimmed to the exact tune after the sextant
                   match, as _run_standard_matching does. None skips the trim,
                   which makes every point off the lattice's own working point
                   fail on 'tune match missed'
    mu_cell        arc cell phase advance the cells are re-matched to before
                   every point. None (default) reads it off the lattice, which
                   is what you want -- a value that disagrees with the lattice
                   retunes the ring before matching and fails every point
    n_turns, n_particles, radiation
                   tracking settings; 'mean' as before, None is ~15 % faster
    sext_knobs     e.g. ['k2XFarc', 'k2XDarc'] to re-correct chromaticity to
                   zero at every point (slower, but compares like with like);
                   None keeps the lattice's sextupole strengths
    metrics        True (default): also record RDTs, amplitude detuning and
                   Q'' at every point (see optics_metrics; adds a few s per
                   point). A dict like {'rdts': ['f3000'], 'detuning': False,
                   'nl_chroma': True} picks which; False/None skips them.
                   Get the maps with metric_maps(records, qx_vals, qy_vals).
    Returns QX, QY, DA_map (np.nan where the match or tracking failed) and a
    list of per-point records (tunes reached, beta max, errors, time).
    """
    qx_vals, qy_vals = np.asarray(qx_vals, float), np.asarray(qy_vals, float)

    # The arc cells set most of the ring tune, and the per-point prep calls
    # _match_cells_3fold, which rewrites kQFarc/kQDarc/kQFarcM to whatever
    # mu_cell says. A hardcoded mu_cell therefore silently RETUNES the lattice
    # being scanned before any matching: forcing 0.25 on a 120 deg/cell ring
    # drops Qx by ~6 units, and matchingWP only varies the DS/straight/triplet
    # knobs, so it can never win that back -- every point then dies on
    # 'tune match missed' having reached the same wrong tune. Read the cell
    # phase advance off the lattice instead, and complain if a caller's value
    # disagrees with it.
    if mu_cell is None:
        import xtrack as _xt
        _env = _xt.Environment.from_json(str(lattice_json))
        _ca = _env.lines['cell_arc']
        try:
            _ca.configure_radiation(model=None)
        except Exception:
            pass
        mu_cell = float(_ca.twiss(method='4d').qx)
        del _env, _ca
        print(f'[wp_da_scan] arc cell phase advance read from the lattice: '
              f'mu_cell={mu_cell:.6f} ({mu_cell * 360:.1f} deg/cell)')

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
                tune_tol=tune_tol, start_element=start_element,
                trim_knobs=list(trim_knobs) if trim_knobs else None,
                metrics=(dict(rdts=list(DEFAULT_RDTS), detuning=True,
                              nl_chroma=True, eq_emittance=True)
                         if metrics is True else (metrics or None)))
    procs, outs, assigned = [], [], []
    total = 0
    for w, rows in enumerate(row_blocks):
        if len(rows) == 0:
            continue
        pts = _snake(qx_vals, qy_vals, rows)
        total += len(pts)
        assigned.extend(pts)
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

    # A point that never returns used to hang the whole scan: the wait loop had
    # no timeout, so one stuck worker stalled a multi-hour run with no output
    # and no way to tell which point did it. Watch for no progress at all and
    # name the outstanding points before killing the stragglers, so the
    # remaining chunks still run.
    t0, last, seen, last_change = time.time(), 0, -1, time.time()
    while any(p.poll() is None for p in procs):
        time.sleep(5)
        done = sum(sum(1 for _ in open(o)) for o in outs if o.exists())
        if done != seen:
            seen, last_change = done, time.time()
        if done - last >= progress_every:
            last = done
            el = time.time() - t0
            print(f'[wp_da_scan] {done}/{total} done, {el/60:.1f} min elapsed, '
                  f'~{el / done * (total - done) / 60:.1f} min left')
        if stall_timeout and time.time() - last_change > stall_timeout:
            finished = set()
            for o in outs:
                if o.exists():
                    for l in open(o):
                        r = json.loads(l)
                        finished.add((round(r['qx_t'], 10), round(r['qy_t'], 10)))
            stuck = [q for q in assigned
                     if (round(q[0], 10), round(q[1], 10)) not in finished]
            alive = [w for w, p in enumerate(procs) if p.poll() is None]
            print(f'[wp_da_scan] STALLED: nothing finished for '
                  f'{stall_timeout / 60:.0f} min at {done}/{total}. '
                  f'Workers still running: {alive}. {len(stuck)} points '
                  f'outstanding; each live worker is stuck on its first one:')
            for q in stuck[:8]:
                print(f'    outstanding ({q[0]:.3f}, {q[1]:.3f})')
            print(f'[wp_da_scan] killing the stragglers so the scan continues; '
                  f'their stderr is in {tmp}/log*.txt')
            for p in procs:
                if p.poll() is None:
                    p.kill()
                    p.wait()          # reap, so returncode is -9 not None
            break
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
    n_edge = sum(bool(r.get('DA_at_grid_edge')) for r in records)
    print(f'[wp_da_scan] finished in {(time.time()-t0)/60:.1f} min: '
          f'{n_ok}/{total} points with a DA, {total - n_ok} failed/unreachable')
    if n_edge:
        print(f'[wp_da_scan] {n_edge} points lost nothing on the whole grid: their DA '
              f'is the grid edge (a lower bound). Enlarge the grid to resolve them.')

    # Why the failures failed. Without this the scan reports only a count, so a
    # run where every point raises the same exception looks exactly like a run
    # where the whole window is genuinely unstable -- and costs hours either way.
    if total - n_ok:
        missing = total - len(records)
        if missing:
            print(f'[wp_da_scan] {missing} points produced no record at all '
                  f'(worker died before reaching them; see {tmp}/log*.txt)')
        errs = collections.Counter(r['error'] for r in records if r.get('error'))
        if errs:
            print(f'[wp_da_scan] {sum(errs.values())} points raised:')
            for msg, n in errs.most_common(8):
                eg = next(r for r in records if r.get('error') == msg)
                print(f'    {n:4d} x  {msg}')
                print(f'           e.g. at target ({eg["qx_t"]:.3f}, {eg["qy_t"]:.3f})'
                      + (f' -> reached ({eg["qx"]:.4f}, {eg["qy"]:.4f}), '
                         f'betx_max={eg.get("betx_max", float("nan")):.1f} m, '
                         f'bety_max={eg.get("bety_max", float("nan")):.1f} m'
                         if 'qx' in eg else ' -> no twiss'))
    return QX, QY, DA, records


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--worker':
        _worker(sys.argv[2], sys.argv[3])
    else:
        sys.exit('usage: python wp_da_scan.py --worker <cfg.json> <out.jsonl> '
                 '(normally started by run_parallel)')
