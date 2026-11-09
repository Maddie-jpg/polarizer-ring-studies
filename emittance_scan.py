"""Equilibrium emittance over a Qx-Qy grid.

Unlike wp_da_scan this never tracks. The equilibrium emittance comes from a
radiation twiss of the re-matched optics, so a point costs the match plus a
few seconds instead of the ~13 min a tracked DA point costs once the DA is
high enough that nothing is lost. It also means the map is smooth, so a
coarse grid is usually enough -- there is no loss boundary to resolve.

Each point is re-matched exactly as the DA scan does it: matchingWP for the
sextant, then the global tune trim that _run_standard_matching applies.
Without the trim every point away from the lattice's own working point
misses the tune by far more than tune_tol and is thrown away.

Run it directly:

    python emittance_scan.py path/to/pdr_perfect_90_FDF.json

or from a script / notebook:

    import emittance_scan as es
    QX, QY, EX, records = es.run_parallel(lattice_json, qx_vals, qy_vals)
    fig = es.plot_emittance_scan(QX, QY, EX)
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

from wp_da_scan import _snake

TRIM_KNOBS = ('kQFarc', 'kQDarc', 'kQFarcM', 'kQDarcM')
# Written by the worker for every point that matches.
EQ_KEYS = ('eq_gemitt_x', 'eq_gemitt_y', 'eq_gemitt_zeta', 'eneloss_turn',
           'alpha_p', 'betx_max', 'bety_max', 'dx_max')


def _radiation_twiss(ring):
    """6d twiss with radiation on, across xtrack versions.

    The keyword is radiation_analysis from 0.118 and eneloss_and_damping
    before it; eneloss_turn was likewise renamed energy_loss.
    """
    try:
        return ring.twiss(method='6d', radiation_analysis=True)
    except TypeError:
        return ring.twiss(method='6d', eneloss_and_damping=True)


def _worker(cfg_file, out_file):
    import xobjects as xo
    import xtrack as xt
    xo.context_cpu.allow_no_prebuilt_kernel = True
    import LatticeBuild.linear_optics as lo
    import LatticeBuild.sextupole_configs as sc

    s = json.loads(Path(cfg_file).read_text())
    quiet = lambda: contextlib.redirect_stdout(io.StringIO())

    pdr = xt.Environment.from_json(s['lattice_json'])
    ring, arc1R = pdr.lines['ring'], pdr.lines['arc1R']
    cell_arc, cell_tr = pdr.lines['cell_arc'], pdr.lines['cell_tr']

    if s.get('triplet_seed'):
        pdr['kQFtr'], pdr['kQDtr'] = s['triplet_seed']
    with quiet():
        cell_arc_opt, cell_tr_opt = lo._match_cells_3fold(
            pdr, cell_arc, cell_tr, mu_cell=s['mu_cell'])

    ring.build_tracker(_context=xo.ContextCpu())       # serial: one core each

    for qx, qy in s['points']:
        row = dict(qx_t=qx, qy_t=qy)
        t0 = time.time()
        try:
            with quiet():
                lo.matchingWP(qx, qy, cell_arc_opt, cell_arc, arc1R,
                              n_periods=s['n_periods'],
                              betay_DS_target=s.get('betay_DS_target'))
            ring.configure_radiation(model=None)

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
                       betx_max=float(tw.betx.max()),
                       bety_max=float(tw.bety.max()),
                       dx_max=float(np.abs(tw.dx).max()))
            if abs(tw.qx - qx) > s['tune_tol'] or abs(tw.qy - qy) > s['tune_tol']:
                raise RuntimeError(f'tune match missed ({tw.qx:.4f}, {tw.qy:.4f})')

            # Sextupoles do not change the equilibrium emittance to the order
            # that matters here (it is set by the H-function in the dipoles),
            # so chromaticity correction is skipped unless asked for.
            if s.get('sext_knobs'):
                with quiet():
                    sc.ChromCorrect(ring, pdr, list(s['sext_knobs']))

            ring.configure_radiation(model='mean')
            try:
                tw6 = _radiation_twiss(ring)
                for k in ('eq_gemitt_x', 'eq_gemitt_y', 'eq_gemitt_zeta'):
                    if hasattr(tw6, k):
                        row[k] = float(np.asarray(getattr(tw6, k)).ravel()[0])
                for k in ('energy_loss', 'eneloss_turn'):
                    if hasattr(tw6, k):
                        row['eneloss_turn'] = float(
                            np.asarray(getattr(tw6, k)).ravel()[0])
                        break
                for k in ('momentum_compaction_factor', 'slip_factor'):
                    if hasattr(tw6, k):
                        row['alpha_p'] = float(
                            np.asarray(getattr(tw6, k)).ravel()[0])
                        break
            finally:
                ring.configure_radiation(model=None)
        except Exception as e:
            row['error'] = f'{type(e).__name__}: {str(e)[:150]}'
        row['time'] = time.time() - t0
        with open(out_file, 'a') as f:
            f.write(json.dumps(row) + '\n')


def emittance_map(records, qx_vals, qy_vals, key='eq_gemitt_x'):
    """records -> a (len(qy), len(qx)) array of `key`, NaN where missing."""
    ix = {round(float(v), 10): j for j, v in enumerate(qx_vals)}
    iy = {round(float(v), 10): i for i, v in enumerate(qy_vals)}
    M = np.full((len(qy_vals), len(qx_vals)), np.nan)
    for r in records:
        if r.get(key) is None:
            continue
        i = iy.get(round(float(r['qy_t']), 10))
        j = ix.get(round(float(r['qx_t']), 10))
        if i is not None and j is not None:
            M[i, j] = float(r[key])
    return M


def run_parallel(lattice_json, qx_vals, qy_vals, *, n_workers=None,
                 n_periods=6, mu_cell=None, betay_DS_target=None,
                 triplet_seed=(1.8, -3.2), tune_tol=2e-3, sext_knobs=None,
                 trim_knobs=TRIM_KNOBS, key='eq_gemitt_x',
                 progress_every=10, stall_timeout=1200):
    """Equilibrium emittance over qx_vals x qy_vals on n_workers processes.

    Returns QX, QY, EX, records -- EX in m rad, NaN where the point failed.

    n_workers   default: CPU count - 1. Each worker holds its own lattice at
                roughly 0.3 GB, so cap it on a many-core box.
    mu_cell     arc cell phase advance the cells are re-matched to. None
                (default) reads it off the lattice, which is what you want:
                a value that disagrees with the lattice retunes the ring
                before matching and fails every point.
    sext_knobs  normally None -- the equilibrium emittance does not depend on
                the sextupoles, so correcting chromaticity only costs time.
    """
    qx_vals = np.asarray(qx_vals, float)
    qy_vals = np.asarray(qy_vals, float)

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
        print(f'[emittance_scan] arc cell phase advance read from the lattice: '
              f'mu_cell={mu_cell:.6f} ({mu_cell * 360:.1f} deg/cell)')

    n_workers = n_workers or max(1, (os.cpu_count() or 2) - 1)
    row_blocks = np.array_split(np.arange(len(qy_vals)), n_workers)
    tmp = Path(tempfile.mkdtemp(prefix='emittance_scan_'))

    base = dict(lattice_json=str(lattice_json), n_periods=n_periods,
                mu_cell=mu_cell, betay_DS_target=betay_DS_target,
                triplet_seed=list(triplet_seed) if triplet_seed else None,
                tune_tol=tune_tol, trim_knobs=list(trim_knobs) if trim_knobs
                else None,
                sext_knobs=list(sext_knobs) if sext_knobs else None)

    procs, outs, assigned = [], [], []
    total = 0
    for w, rows in enumerate(row_blocks):
        if len(rows) == 0:
            continue
        pts = _snake(qx_vals, qy_vals, rows)
        total += len(pts)
        assigned.extend(pts)
        cfg = tmp / f'cfg{w}.json'
        out = tmp / f'out{w}.jsonl'
        cfg.write_text(json.dumps(dict(base, points=pts)))
        outs.append(out)
        procs.append(subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), '--worker',
             str(cfg), str(out)],
            stdout=subprocess.DEVNULL,
            stderr=open(tmp / f'log{w}.txt', 'w')))

    print(f'[emittance_scan] {total} points on {len(procs)} workers '
          f'(no tracking); temp files in {tmp}')

    t0, last, seen, last_change = time.time(), 0, -1, time.time()
    while any(p.poll() is None for p in procs):
        time.sleep(5)
        done = sum(sum(1 for _ in open(o)) for o in outs if o.exists())
        if done != seen:
            seen, last_change = done, time.time()
        if done - last >= progress_every:
            last = done
            el = time.time() - t0
            print(f'[emittance_scan] {done}/{total} done, {el/60:.1f} min '
                  f'elapsed, ~{el / done * (total - done) / 60:.1f} min left')
        if stall_timeout and time.time() - last_change > stall_timeout:
            print(f'[emittance_scan] STALLED: nothing finished for '
                  f'{stall_timeout/60:.0f} min at {done}/{total}; killing the '
                  f'stragglers (stderr in {tmp}/log*.txt)')
            for p in procs:
                if p.poll() is None:
                    p.kill()
                    p.wait()
            break

    records = []
    for o in outs:
        if o.exists():
            records += [json.loads(l) for l in open(o)]

    EX = emittance_map(records, qx_vals, qy_vals, key=key)
    QX, QY = np.meshgrid(qx_vals, qy_vals)
    n_ok = int(np.isfinite(EX).sum())
    print(f'[emittance_scan] finished in {(time.time()-t0)/60:.1f} min: '
          f'{n_ok}/{total} points with an emittance')

    if total - n_ok:
        missing = total - len(records)
        if missing:
            print(f'[emittance_scan] {missing} points produced no record at '
                  f'all (worker died; see {tmp}/log*.txt)')
        errs = collections.Counter(r['error'] for r in records if r.get('error'))
        if errs:
            print(f'[emittance_scan] {sum(errs.values())} points raised:')
            for msg, n in errs.most_common(8):
                eg = next(r for r in records if r.get('error') == msg)
                print(f'    {n:4d} x  {msg}')
                print(f'           e.g. at target ({eg["qx_t"]:.3f}, '
                      f'{eg["qy_t"]:.3f})')
    return QX, QY, EX, records


def plot_emittance_scan(QX, QY, EX, current_qx=None, current_qy=None,
                        plane='x', title=None):
    """Qx-Qy heatmap of the equilibrium emittance. EX in m rad, drawn in nm.

    Failed points are masked grey rather than set to 0: zero emittance would
    read as the best point on the map instead of a missing one.
    """
    import matplotlib.pyplot as plt
    ex_nm = np.asarray(EX, dtype=float) * 1e9
    masked = np.ma.masked_invalid(ex_nm)
    cmap = plt.get_cmap('magma').copy()
    cmap.set_bad('lightgrey')

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.pcolormesh(QX, QY, masked, shading='auto', cmap=cmap)
    fig.colorbar(im, ax=ax, label=rf'Equilibrium $\varepsilon_{plane}$ '
                                  rf'[nm rad]  (grey = failed/unreachable)')
    if current_qx is not None:
        ax.plot(current_qx, current_qy, 'o', color='red', ms=10, mec='black',
                label='Current working point')
        ax.legend()
    ax.set_xlabel('$Q_x$')
    ax.set_ylabel('$Q_y$')
    ax.set_title(title or rf'Tune scan: equilibrium $\varepsilon_{plane}$')
    plt.tight_layout()
    return fig


def grid(centre, half_width, per_unit):
    """per_unit points per unit tune, snapped to a 1/per_unit lattice."""
    lo_, hi = centre - half_width, centre + half_width
    first = np.ceil(lo_ * per_unit) / per_unit
    n = int(np.floor((hi - first) * per_unit)) + 1
    return first + np.arange(n) / per_unit


def main(lattice_json, half_width=0.9, per_unit=10, n_workers=None,
         out_dir='.', **kw):
    """Scan a window centred on the lattice's own tune and save the map."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import xobjects as xo
    import xtrack as xt
    xo.context_cpu.allow_no_prebuilt_kernel = True

    env = xt.Environment.from_json(str(lattice_json))
    ring = env.lines['ring']
    ring.build_tracker(_context=xo.ContextCpu())
    ring.configure_radiation(model=None)
    tw = ring.twiss(method='4d')
    cqx, cqy = float(tw.qx), float(tw.qy)
    print(f'[emittance_scan] lattice tune Qx={cqx:.4f} Qy={cqy:.4f}, '
          f'C={ring.get_length():.2f} m')
    del ring, env

    qx_vals = grid(cqx, half_width, per_unit)
    qy_vals = grid(cqy, half_width, per_unit)
    print(f'[emittance_scan] grid {len(qx_vals)} x {len(qy_vals)} = '
          f'{len(qx_vals)*len(qy_vals)} points')

    QX, QY, EX, records = run_parallel(lattice_json, qx_vals, qy_vals,
                                       n_workers=n_workers, **kw)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = plot_emittance_scan(QX, QY, EX, current_qx=cqx, current_qy=cqy)
    png = out_dir / 'tune_scan_eq_emittance.png'
    fig.savefig(png, dpi=300, bbox_inches='tight')
    npz = out_dir / 'tune_scan_eq_emittance.npz'
    np.savez(npz, QX=QX, QY=QY, EX=EX, qx_vals=qx_vals, qy_vals=qy_vals,
             **{k: emittance_map(records, qx_vals, qy_vals, key=k)
                for k in EQ_KEYS})
    print(f'[emittance_scan] wrote {png} and {npz}')

    if np.isfinite(EX).any():
        nm = EX * 1e9
        i, j = np.unravel_index(np.nanargmin(nm), nm.shape)
        print(f'[emittance_scan] eq emittance over the window: '
              f'{np.nanmin(nm):.2f} - {np.nanmax(nm):.2f} nm rad')
        print(f'[emittance_scan] smallest at Qx={QX[i, j]:.3f}, '
              f'Qy={QY[i, j]:.3f}')
    return QX, QY, EX, records


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--worker':
        _worker(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 2:
        main(sys.argv[1],
             half_width=float(sys.argv[2]) if len(sys.argv) > 2 else 0.9,
             per_unit=int(sys.argv[3]) if len(sys.argv) > 3 else 10,
             n_workers=int(sys.argv[4]) if len(sys.argv) > 4 else None)
    else:
        sys.exit('usage: python emittance_scan.py <lattice.json> '
                 '[half_width] [points_per_unit_tune] [n_workers]')
