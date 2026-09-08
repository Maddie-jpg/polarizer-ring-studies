#%%
#---------------------
# phase advance scan
#---------------------

import numpy as np
import matplotlib.pyplot as plt
import xtrack as xt


def _twiss_4d_radiation_safe(line, **twiss_kwargs):
    """
    Run a method='4d' twiss even if `line` currently has radiation
    configured -- a plain 4d twiss can't run with radiation on (4d assumes
    no average energy loss per turn, which radiation breaks).

    Temporarily disables radiation, runs the 4d twiss, then restores
    whatever radiation/beamstrahlung model was active before.
    """
    had_radiation = getattr(line, '_radiation_model', None) is not None
    if had_radiation:
        SR_model = line._radiation_model
        BS_model = line._beamstrahlung_model
        line.configure_radiation(model=None, model_beamstrahlung=None)

    try:
        tw = line.twiss(method='4d', **twiss_kwargs)
    finally:
        if had_radiation:
            line.configure_radiation(model=SR_model, model_beamstrahlung=BS_model)

    return tw


def scan_mux_muy_emittance(cell_arc, mux_range, muy_range, n_mux=15, n_muy=15,
                           vary_knobs=('kQFarc', 'kQDarc'),
                           knob_limits=None, n_steps=40, verbose=False):
    """
    Scan the arc cell's phase advance over an independent (mu_x, mu_y) grid,
    re-matching the cell at each grid point, and record the resulting
    equilibrium horizontal emittance, momentum compaction factor, and
    horizontal/vertical chromaticity.

    cell_arc: a single periodic arc cell line, already carrying the
        design's magnet knobs. Matched IN PLACE at each grid point --
        pass a copy if you need the original preserved afterwards.
    mux_range, muy_range: (min, max) phase advance per cell, in units of
        2*pi (e.g. (0.15, 0.35) to scan roughly 55-125 degrees/cell).
    vary_knobs: two knobs used to independently hit qx and qy. Defaults to
        the standard arc F/D pair.
    knob_limits: optional dict {knob_name: (lo, hi)} bounding each vary
        knob so a bad grid point's Newton step can't run away into an
        unstable cell. Defaults to +-3x each knob's starting value.

    Returns dict with 'mux_grid', 'muy_grid' (1D arrays) and three 2D
    arrays -- 'emit_x', 'alpha_c', 'dqx', 'dqy' -- NaN wherever the match
    failed to converge or the cell went unstable at that grid point.
    """
    k0 = {kn: cell_arc.vars[kn]._value for kn in vary_knobs}
    if knob_limits is None:
        knob_limits = {kn: (-3 * abs(v) - 1e-3, 3 * abs(v) + 1e-3)
                       for kn, v in k0.items()}

    mux_grid = np.linspace(mux_range[0], mux_range[1], n_mux)
    muy_grid = np.linspace(muy_range[0], muy_range[1], n_muy)
    shape = (n_muy, n_mux)  # rows=muy, cols=mux, for pcolormesh
    emit_x = np.full(shape, np.nan)
    alpha_c = np.full(shape, np.nan)
    dqx = np.full(shape, np.nan)
    dqy = np.full(shape, np.nan)

    # radiation_integrals=True is purely analytic (integrals of curvature
    # over the optics) and doesn't need real radiation kicks enabled, but
    # a plain method='4d' twiss still can't run AT ALL if radiation is
    # configured on the line -- disable it once for the whole scan rather
    # than per grid point, and restore it afterward regardless of whether
    # the scan finishes cleanly or raises.
    had_radiation = getattr(cell_arc, '_radiation_model', None) is not None
    if had_radiation:
        SR_model = cell_arc._radiation_model
        BS_model = cell_arc._beamstrahlung_model
        cell_arc.configure_radiation(model=None, model_beamstrahlung=None)

    try:
        for iy, muy in enumerate(muy_grid):
            # reset knobs to the starting point each row, so failures/drift
            # in one row don't bias the next row's starting guess
            for kn, v in k0.items():
                cell_arc.vars[kn] = v

            for ix, mux in enumerate(mux_grid):
                try:
                    opt = cell_arc.match(
                        method='4d', solve=False, verbose=False,
                        vary=[xt.Vary(kn, step=1e-5, limits=knob_limits[kn])
                             for kn in vary_knobs],
                        targets=[
                            xt.Target('qx', mux, tol=1e-6),
                            xt.Target('qy', muy, tol=1e-6),
                        ],
                    )
                    try:
                        opt.solve(n_steps=n_steps)
                    except Exception:
                        opt.step(n_steps, broyden=True, rcond=1e-3)

                    tw = cell_arc.twiss(method='4d', radiation_integrals=True)
                    emit_x[iy, ix] = tw.rad_int_eq_gemitt_x
                    alpha_c[iy, ix] = tw.momentum_compaction_factor
                    dqx[iy, ix] = tw.dqx
                    dqy[iy, ix] = tw.dqy

                    if verbose:
                        print(f"mux={mux:.4f} muy={muy:.4f} -> "
                              f"emit_x={emit_x[iy, ix]:.4e}  "
                              f"alpha_c={alpha_c[iy, ix]:.4e}  "
                              f"dqx={dqx[iy, ix]:.3f}  dqy={dqy[iy, ix]:.3f}")
                except Exception as e:
                    if verbose:
                        print(f"mux={mux:.4f} muy={muy:.4f} -> failed "
                              f"({type(e).__name__})")
    finally:
        # Restore the vary knobs to their ORIGINAL starting values, not
        # just whatever the very last grid point happened to leave them
        # at. The loop resets to k0 at the start of every ROW, but never
        # after the last row's last point -- and if these knobs are
        # global (shared with `ring` via the same pdr environment, not
        # local to cell_arc), that leaves the ACTUAL RING's magnet
        # strengths corrupted for the rest of the script, not just
        # cell_arc. This is what was silently breaking every twiss()
        # call on `ring` after this scan ran, even with no copy/match
        # involved -- confirmed via debug_tune_match's isolation tests.
        for kn, v in k0.items():
            cell_arc.vars[kn] = v
        if had_radiation:
            cell_arc.configure_radiation(model=SR_model, model_beamstrahlung=BS_model)

    return dict(mux_grid=mux_grid, muy_grid=muy_grid, emit_x=emit_x,
               alpha_c=alpha_c, dqx=dqx, dqy=dqy)


def _find_best_point(data, mux_grid, muy_grid, objective='min'):
    """
    Find the grid point optimizing `data` under `objective`:
      'min'     -> smallest value
      'max'     -> largest value
      'abs_min' -> value closest to zero

    NaNs (failed/unstable grid points) are ignored. Returns
    (best_mux, best_muy, best_value), or None if every point is NaN.
    """
    if objective == 'min':
        score = data
    elif objective == 'max':
        score = -data
    elif objective == 'abs_min':
        score = np.abs(data)
    else:
        raise ValueError(f"Unknown objective: {objective!r}")

    if np.all(np.isnan(score)):
        return None

    iy, ix = np.unravel_index(np.nanargmin(score), score.shape)
    return mux_grid[ix], muy_grid[iy], data[iy, ix]


def _annotate_best_point(ax, best, label='Best', value_fmt='{:.4e}',
                         color='white', marker='D'):
    """Mark `best` (from _find_best_point) with a diamond and a textbox
    giving its (mu_x, mu_y) coordinates and the metric's value there."""
    if best is None:
        return
    bx, by, bval = best
    ax.plot(bx, by, marker=marker, color=color, markeredgecolor='black',
           markersize=10, linestyle='', zorder=6)
    ax.annotate(
        f'{label}\n' + r'$\mu_x$=' + f'{bx:.4f}, ' + r'$\mu_y$=' + f'{by:.4f}'
        f'\nvalue={value_fmt.format(bval)}',
        xy=(bx, by), xytext=(0.03, 0.97), textcoords='axes fraction',
        va='top', ha='left', fontsize=8.5,
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.85),
        arrowprops=dict(arrowstyle='->', color='black', lw=1))


def plot_mux_muy_4panel(scan_result, cmap='viridis', log_emittance=False,
                        show_best=True, objectives=None):
    """
    Four-panel (mu_x, mu_y) scan plot:

        top-left     = horizontal equilibrium emittance
        top-right    = momentum compaction factor (alpha_c)
        bottom-left  = horizontal chromaticity (dqx)
        bottom-right = vertical chromaticity (dqy)
    """
    mux_grid = scan_result['mux_grid']
    muy_grid = scan_result['muy_grid']

    default_objectives = dict(emit_x='min', alpha_c='abs_min',
                              dqx='abs_min', dqy='abs_min')
    if objectives:
        default_objectives.update(objectives)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    panels = [
        (axes[0, 0], 'emit_x', r'$\epsilon_x$',
         (lambda d: np.log10(d)) if log_emittance else (lambda d: d),
         r'$\log_{10}(\epsilon_x)$ [m]' if log_emittance else r'$\epsilon_x$ [m]',
         'Min emittance', '{:.3e} m'),
        (axes[0, 1], 'alpha_c', r'$\alpha_c$',
         lambda d: d, r'$\alpha_c$ (momentum compaction)',
         'Min $|\\alpha_c|$', '{:.4e}'),
        (axes[1, 0], 'dqx', r'$\xi_x$',
         lambda d: d, r'$\xi_x$',
         'Min $|\\xi_x|$', '{:.3f}'),
        (axes[1, 1], 'dqy', r'$\xi_y$',
         lambda d: d, r'$\xi_y$',
         'Min $|\\xi_y|$', '{:.3f}'),
    ]

    for ax, key, title, transform, cbar_label, best_label, value_fmt in panels:
        data = scan_result[key]
        im = ax.pcolormesh(mux_grid, muy_grid, transform(data),
                           shading='auto', cmap=cmap)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)

        ax.set_xlim(mux_grid.min(), mux_grid.max())
        ax.set_ylim(muy_grid.min(), muy_grid.max())

        ax.set_xlabel(r'Cell $\mu_x$ (fraction of $2\pi$)')
        ax.set_ylabel(r'Cell $\mu_y$ (fraction of $2\pi$)')
        ax.set_title(title)
        if show_best:
            best = _find_best_point(data, mux_grid, muy_grid,
                                    objective=default_objectives[key])
            _annotate_best_point(ax, best, label=best_label, value_fmt=value_fmt)

    fig.suptitle('Phase advance scan: emittance, momentum compaction, '
                 'and chromaticity', fontsize=13)
    plt.tight_layout()
    return fig, axes




pdr = xt.Environment.from_json('/home/mwatson/Documents/laughing-octo-bassoon/JSON_Files/D1/C9/pdr_perfect_90.json')
ring = pdr.lines['ring']
cell_arc = pdr.lines['cell_arc']
ref_twiss = ring.twiss6d()
line_table = ring.get_table()

scan = scan_mux_muy_emittance(
    cell_arc, mux_range=(0.30, 0.45), muy_range=(0.15, 0.35),
    n_mux=20, n_muy=20,
)
fig, axes = plot_mux_muy_4panel(scan)
fig.savefig('wp_plots.png')

# Guard: kQFarc/kQDarc are shared globally between cell_arc and ring (both
# come from `pdr`), so if the scan above ever again leaves them at a bad
# grid point instead of restoring k0, this catches it here with a clear
# message instead of a confusing "Invalid n1" much later in the script.
try:
    _sanity_tw = ring.twiss6d()
    print(f"[sanity check] ring.twiss6d() OK after scan_mux_muy_emittance "
          f"(qx={_sanity_tw.qx:.4f}, qy={_sanity_tw.qy:.4f})")
except Exception as e:
    raise RuntimeError(
        "ring.twiss6d() FAILED right after scan_mux_muy_emittance() -- the "
        "arc-cell scan above shares its vary knobs with the actual ring "
        "and left them at a bad value. Fix scan_mux_muy_emittance's knob "
        "restoration before trusting anything downstream in this script."
    ) from e

#%%
#-----------
# RDTs scan
#-----------

import numpy as np
from math import factorial
import matplotlib.pyplot as plt


def _parse_rdt_key(rdt):
    pqrt = rdt[1:]
    if len(pqrt) > 4:
        if '_' in pqrt:
            p, q, r, t = (int(v) for v in pqrt.split('_'))
        else:
            raise ValueError(
                "RDT key must look like 'f1020' (one letter f + four digits) "
                "or 'f11_0_2_0' (one letter f + more than four digits separated by '_')."
            )
    elif len(pqrt) == 4:
        p, q, r, t = (int(c) for c in pqrt)
    else:
        raise ValueError(f"RDT key '{rdt}' is too short -- expected 4+ digits after 'f'.")
    return p, q, r, t


def debug_rdt_element(ref_twiss, line, line_table, elem_name):
    """Tests each step precompute_rdt_elements() performs on ONE element,
    with no broad except hiding failures -- run this on a known sextupole
    name to see exactly which step breaks, instead of it being silently
    swallowed inside the main scan loop."""
    print(f"--- debug_rdt_element('{elem_name}') ---")

    try:
        et = line_table.rows[elem_name].element_type[0]
        print(f"1. line_table.rows[elem].element_type[0] = {et!r}  OK")
    except Exception as e:
        print(f"1. line_table.rows[elem].element_type[0] FAILED: {type(e).__name__}: {e}")
        return

    try:
        betx = ref_twiss.rows[elem_name].betx[0]
        bety = ref_twiss.rows[elem_name].bety[0]
        mux = ref_twiss.rows[elem_name].mux[0]
        muy = ref_twiss.rows[elem_name].muy[0]
        print(f"2. ref_twiss.rows[elem] betx={betx:.4g} bety={bety:.4g} "
              f"mux={mux:.4g} muy={muy:.4g}  OK")
    except Exception as e:
        print(f"2. ref_twiss.rows[elem] FAILED: {type(e).__name__}: {e}")
        return

    try:
        elem_obj = line[elem_name]
        print(f"3. line[elem] = {elem_obj!r} (type {type(elem_obj).__name__})  OK")
    except Exception as e:
        print(f"3. line[elem] FAILED: {type(e).__name__}: {e}")
        return

    try:
        knl, ksl, method = _get_multipole_strength(elem_obj, order=2)
        print(f"4. _get_multipole_strength(order=2) -> knl={knl}, ksl={ksl}, "
              f"method={method}  OK")
    except Exception as e:
        print(f"4. _get_multipole_strength FAILED: {type(e).__name__}: {e}")
        return

    print("--- all steps completed without exception ---")


def _get_multipole_strength(elem_obj, order):
    """Tries the two xtrack strength-parametrization conventions and
    returns (knl, ksl, method) for the given multipole order (2 for
    sextupole, 3 for octupole, ...). 'method' says which convention
    actually matched, for diagnostics -- (0.0, 0.0, None) if neither did.

    Tries the scalar convention FIRST: confirmed via debug_rdt_element()
    that dedicated element classes here (Sextupole etc.) carry BOTH a
    real .k2/.length AND a stale, always-zero .knl array simultaneously
    -- checking .knl first (as this used to) silently "succeeds" with 0.0
    and never reaches the real value. The scalar attribute, when present,
    is authoritative for these dedicated classes; .knl is the fallback
    for generic Multipole-class elements that only have that array.
    """
    k_scalar = getattr(elem_obj, f'k{order}', None)
    length = getattr(elem_obj, 'length', None)
    if k_scalar is not None and length is not None:
        ks_scalar = getattr(elem_obj, f'k{order}s', None)
        knl = k_scalar * length
        ksl = (ks_scalar * length) if ks_scalar is not None else 0.0
        return knl, ksl, 'scalar_times_length'

    knl_arr = getattr(elem_obj, 'knl', None)
    if knl_arr is not None and len(knl_arr) > order:
        ksl_arr = getattr(elem_obj, 'ksl', None)
        ksl = ksl_arr[order] if ksl_arr is not None and len(ksl_arr) > order else 0.0
        return knl_arr[order], ksl, 'knl_array'

    return 0.0, 0.0, None


def precompute_rdt_elements(rdt, ref_twiss, line, line_table, observation_point, source_elements):
    """One-time pass over the lattice for a single RDT key. Extracts
    per-element beta functions, multipole strength, and base phase advance
    to the observation point -- everything the RDT sum needs EXCEPT the
    actual (Qx, Qy) choice. Returns a dict of numpy arrays ready for a fast
    scan over many candidate working points via evaluate_rdt_at().

    Strength is read via _get_multipole_strength(), which tries both the
    generic Multipole knl-array convention and the dedicated-element
    scalar-gradient-times-length convention -- table column naming for
    strengths varies across xtrack versions/element types, so going
    straight to the element and trying both is more robust than betting on
    one specific attribute name. line_table is still used for element_type
    (Marker/Drift filtering).
    """
    tt = line_table
    p, q, r, t = _parse_rdt_key(rdt)
    n = p + q + r + t
    order = n - 1

    mu0_x = ref_twiss.rows[observation_point].mux[0] * 2 * np.pi
    mu0_y = ref_twiss.rows[observation_point].muy[0] * 2 * np.pi

    h_list, dmux0_list, dmuy0_list, wrap_x_list, wrap_y_list = [], [], [], [], []
    diagnostic_dumped = False

    for elem in source_elements:
        try:
            if tt.rows[elem].element_type[0] in ['Marker', 'Drift']:
                continue

            beta_x = ref_twiss.rows[elem].betx[0]
            beta_y = ref_twiss.rows[elem].bety[0]
            mu_x = ref_twiss.rows[elem].mux[0] * 2 * np.pi
            mu_y = ref_twiss.rows[elem].muy[0] * 2 * np.pi

            dmux0 = mu0_x - mu_x
            wrap_x = dmux0 < 0
            dmuy0 = mu0_y - mu_y
            wrap_y = dmuy0 < 0

            elem_obj = line[elem]
            knl, ksl, method = _get_multipole_strength(elem_obj, order)

            expected_type = {0: 'Bend', 1: 'Quadrupole', 2: 'Sextupole',
                              3: 'Octupole'}.get(order)
            is_expected_type = (expected_type is not None
                                 and tt.rows[elem].element_type[0] == expected_type)
            if method is None and is_expected_type and not diagnostic_dumped:
                # Only worth dumping diagnostics for an element that's
                # actually SUPPOSED to carry this order's strength (a
                # Quadrupole legitimately has no k2 -- that's normal, not
                # a problem). If a genuine Sextupole/Octupole/etc still
                # matches neither convention, something's really off --
                # print exactly what IS on it instead of guessing again.
                diagnostic_dumped = True
                print(f"[precompute_rdt_elements] DIAGNOSTIC for element '{elem}' "
                      f"(type {type(elem_obj).__name__}, element_type='{expected_type}'), "
                      f"order={order}: neither '.knl' array nor '.k{order}'+'.length' "
                      f"scalar convention matched. Available non-private attributes: "
                      f"{[a for a in dir(elem_obj) if not a.startswith('_')]}")

            k = np.real(1j ** (r + t) * (knl + 1j * ksl))
            if k == 0.0:
                continue  # no relevant multipole here -- skip, pure speed-up

            factorial_prod = factorial(p) * factorial(q) * factorial(r) * factorial(t)
            h = -k * (beta_x ** ((p + q) / 2) * beta_y ** ((r + t) / 2)) / (factorial_prod * 2 ** n)

            h_list.append(h)
            dmux0_list.append(dmux0)
            dmuy0_list.append(dmuy0)
            wrap_x_list.append(wrap_x)
            wrap_y_list.append(wrap_y)
        except (IndexError, ValueError, TypeError, KeyError):
            # KeyError specifically covers synthetic table-only rows like
            # '_end_point' that get_table() reports but that aren't real,
            # indexable elements in the Line object -- skip them like any
            # other non-physical entry rather than crashing the scan.
            continue

    h_arr = np.array(h_list, dtype=complex)
    if h_arr.size == 0:
        # No guessing this time -- show exactly what element_type strings
        # (and how many of each) actually exist in source_elements, so the
        # real name for your sextupoles (whatever it's called) is visible
        # instead of assuming it's literally 'Sextupole'.
        type_counts = {}
        for elem in source_elements:
            try:
                et = tt.rows[elem].element_type[0]
            except (IndexError, ValueError, TypeError, KeyError):
                continue
            type_counts[et] = type_counts.get(et, 0) + 1
        print(f"[precompute_rdt_elements] WARNING: found ZERO elements with nonzero "
              f"order-{order} multipole strength for RDT '{rdt}'. No element in "
              f"source_elements had element_type=='{expected_type}' (that's why no "
              f"DIAGNOSTIC line fired above) -- here's every element_type actually "
              f"present and how many of each: {type_counts}")

    return {
        'p': p, 'q': q, 'r': r, 't': t,
        'h': h_arr,
        'dmux0': np.array(dmux0_list), 'dmuy0': np.array(dmuy0_list),
        'wrap_x': np.array(wrap_x_list), 'wrap_y': np.array(wrap_y_list),
    }


def evaluate_rdt_at(precomp, Qx, Qy):
    """Fast |f_pqrt| evaluation at one candidate working point, reusing the
    per-element arrays from precompute_rdt_elements()."""
    p, q, r, t = precomp['p'], precomp['q'], precomp['r'], precomp['t']
    h = precomp['h']
    if h.size == 0:
        return 0.0

    dmux = precomp['dmux0'] + np.where(precomp['wrap_x'], 2 * np.pi * Qx, 0.0)
    dmuy = precomp['dmuy0'] + np.where(precomp['wrap_y'], 2 * np.pi * Qy, 0.0)

    phase = np.exp(1j * ((p - q) * dmux + (r - t) * dmuy))
    denom = 1 - np.exp(2j * np.pi * ((p - q) * Qx + (r - t) * Qy))
    if denom == 0:
        return np.inf

    return np.abs(np.sum(h * phase) / denom)


def scan_working_point(rdts, ref_twiss, line, line_table, observation_point, source_elements,
                        qx_range, qy_range, n_points=150):
    """Grid-scans |RDT| over a Qx/Qy window for each RDT in `rdts`.
    Returns (QX, QY, maps) where maps[rdt] is an (n_points, n_points) array
    of |f_pqrt(Qx, Qy)| evaluated on the meshgrid QX, QY.
    """
    qx_vals = np.linspace(*qx_range, n_points)
    qy_vals = np.linspace(*qy_range, n_points)
    QX, QY = np.meshgrid(qx_vals, qy_vals)

    maps = {}
    for rdt in rdts:
        precomp = precompute_rdt_elements(rdt, ref_twiss, line, line_table,
                                           observation_point, source_elements)
        grid = np.zeros_like(QX)
        for i in range(QX.shape[0]):
            for j in range(QX.shape[1]):
                grid[i, j] = evaluate_rdt_at(precomp, QX[i, j], QY[i, j])
        maps[rdt] = grid

    return QX, QY, maps


def _overlay_resonance_lines(ax, qx_range, qy_range, max_order=4):
    """Draws integer resonance lines a*Qx + b*Qy = c for |a|+|b| <= max_order.
    Solid = 1st/2nd order (usually must-avoid), dashed = 3rd, dotted = 4th."""
    qx_vals = np.array(qx_range)
    for a in range(-max_order, max_order + 1):
        for b in range(-max_order, max_order + 1):
            if a == 0 and b == 0:
                continue
            order = abs(a) + abs(b)
            if order > max_order:
                continue
            style = '-' if order <= 2 else ('--' if order == 3 else ':')
            alpha = 0.6 if order <= 2 else 0.3
            for c in range(-10, 11):
                if b != 0:
                    qy_vals = (c - a * qx_vals) / b
                    if np.any((qy_vals >= qy_range[0]) & (qy_vals <= qy_range[1])):
                        ax.plot(qx_vals, qy_vals, style, color='gray', alpha=alpha, lw=0.8)
                elif a != 0:
                    qx_line = c / a
                    if qx_range[0] <= qx_line <= qx_range[1]:
                        ax.axvline(qx_line, ls=style, color='gray', alpha=alpha, lw=0.8)


def plot_rdt_working_point_map(QX, QY, maps, qx_range, qy_range, max_resonance_order=4,
                                current_qx=None, current_qy=None):
    """One subplot per RDT: log10|f_pqrt| as a heatmap, resonance lines
    overlaid in gray, and (optionally) your current working point marked."""
    n = len(maps)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5), squeeze=False)
    axes = axes[0]

    for ax, (rdt, grid) in zip(axes, maps.items()):
        # Clip to a robust percentile range rather than raw min/max: a
        # single near-exact resonance hit (denom -> 0, magnitude -> huge
        # or inf) otherwise saturates the colour scale and makes every
        # other point look like a flat, solid block by comparison.
        log_grid = np.log10(grid + 1e-12)
        finite = log_grid[np.isfinite(log_grid)]
        if finite.size == 0:
            print(f"[plot_rdt_working_point_map] '{rdt}': no finite values to plot.")
            continue
        vmin, vmax = np.percentile(finite, [2, 98])
        if vmin == vmax:
            print(f"[plot_rdt_working_point_map] '{rdt}': all values identical "
                  f"({vmin:.3g}) -- see the precompute_rdt_elements warning above "
                  f"if h.size was 0 for this RDT.")
        im = ax.pcolormesh(QX, QY, log_grid, shading='auto', cmap='inferno',
                            vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, label=fr'$\log_{{10}}|f_{{{rdt[1:]}}}|$')
        _overlay_resonance_lines(ax, qx_range, qy_range, max_resonance_order)
        if current_qx is not None:
            ax.plot(current_qx, current_qy, 'o', color='cyan', ms=10,
                    mec='black', label='Current working point')
            ax.legend(loc='upper right', fontsize='small')
        ax.set_xlabel('$Q_x$')
        ax.set_ylabel('$Q_y$')
        ax.set_title(f'$f_{{{rdt[1:]}}}$')
        ax.set_xlim(qx_range)
        ax.set_ylim(qy_range)

    plt.tight_layout()
    return fig


def find_best_working_point(QX, QY, maps, weight=None):
    """Combines several RDT maps (log10 scale, so it behaves like combining
    dB) into one score and returns the (Qx, Qy) that minimizes it.
    `weight` lets you emphasize the RDTs you care about most, e.g.
    weight={'f3000': 1.0, 'f1200': 2.0} to prioritize avoiding f1200."""
    if weight is None:
        weight = {rdt: 1.0 for rdt in maps}
    combined = sum(weight[rdt] * np.log10(grid + 1e-12) for rdt, grid in maps.items())
    idx = np.unravel_index(np.argmin(combined), combined.shape)
    return QX[idx], QY[idx], combined

tt = ring.get_table()

# One-shot diagnostic on a real, confirmed-present sextupole -- run this
# once before the full scan so any failure shows up immediately with a
# real traceback/reason instead of being silently absorbed by the scan's
# broad except and just showing up as "found ZERO elements" later.
_debug_sext_candidates = tt.rows[tt.element_type == 'Sextupole'].name
if len(_debug_sext_candidates) > 0:
    debug_rdt_element(ref_twiss, ring, line_table, _debug_sext_candidates[0])
else:
    print("[debug] No elements with element_type=='Sextupole' found via "
          "tt.rows[tt.element_type == 'Sextupole'] -- unexpected given the "
          "lattice JSON has 96 of them; the table filter itself may be the issue.")

rdts_of_interest = ['f3000', 'f2100', 'f1020', 'f1011', 'f1002']  # 3rd-order, sextupole-driven
current_qx, current_qy = ref_twiss.qx, ref_twiss.qy

QX, QY, maps = scan_working_point(
rdts_of_interest, ref_twiss, ring, line_table,
    observation_point=tt.name[0], source_elements=line_table.name,
    qx_range=(current_qx - 0.15, current_qx + 0.15),
    qy_range=(current_qy - 0.15, current_qy + 0.15),
    n_points=120,
    )
    
plot_rdt_working_point_map(QX, QY, maps, (current_qx-0.15, current_qx+0.15),
                               (current_qy-0.15, current_qy+0.15),
                               current_qx=current_qx, current_qy=current_qy)
plt.show()
    
best_qx, best_qy, combined_score = find_best_working_point(QX, QY, maps)
print(f"Best working point in scanned window: Qx={best_qx:.4f}, Qy={best_qy:.4f}")

#%%
#----------------------------------
# Tune performance scan (min DA/MA)
#----------------------------------

"""
tune_scan.py

Performance tune scan: for each candidate (Qx, Qy) in a grid, actually
re-matches the ring's quadrupoles to hit that tune, then runs a
reduced-resolution DA study there and records the resulting minimum
dynamic aperture. This is the "real" counterpart to an RDT-based working
point scan -- it accounts for how the optics (beta functions, and
therefore the actual dynamic aperture) change as you move the tune,
instead of approximating with the reference lattice's optics held fixed.


Safety: line.match() mutates the line's knobs in place. This module always
scans on a line.copy() so your actual ring is never left at some
arbitrary grid point's tune when the scan finishes.

You will need to adjust for your lattice:
- tune_knobs: which quadrupole families actually control GLOBAL tune in
  your ring. This example project uses arc quad families like
  ['kQFarcM', 'kQDarcM'] for tune matching elsewhere (see
  LatticeBuild/linear_optics.py) -- confirm that's the right pair (or
  combination) for YOUR global tune knob before trusting scan results.
- start_element: wherever your DA study's particle grid should start from
  (matches study_params_DA['start_element'] convention already used in
  analysis.py).
"""

import numpy as np
import matplotlib.pyplot as plt
import xtrack as xt
import xobjects as xo
import xutil_DA_CC.xsuite_utilities as xutil

gamma0 = ring.particle_ref.gamma0[0]
beta0 = ring.particle_ref.beta0[0]

n_emittancex = 1.354177116369456e-6 * gamma0 * beta0
n_emittancey = 1.420755089827341e-6 * gamma0 * beta0


def get_DA_boundary(particles, num_r_steps, num_theta_steps, x_norm, y_norm):
    """Identical logic to the version already in analysis.py."""
    if isinstance(particles, dict):
        max_turns = np.shape(particles['x'])[1] - 1
        part_at_turn = np.nanmax(particles['at_turn'], axis=1)
    else:
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
 
 
def match_tune(line, qx_target, qy_target, tune_knobs, tol=1e-6):
    """Re-matches line's tunes to (qx_target, qy_target). Returns True if
    the match converged, False (with a printed reason) otherwise -- a
    failed match at some grid points is normal near resonances/edges of
    the achievable tune space, not something that should crash the scan.

    Disables radiation ENTIRELY before matching (not model='mean') --
    confirmed via debug_tune_match() that 'mean' radiation alone breaks
    compute_linear_normal_form() on this ring ("Invalid n1"), reproduced
    even with NO match() involved at all, on a plain line.twiss6d() call.
    'mean' radiation still adds a trajectory-dependent energy-loss term
    to the one-turn map, making it non-symplectic -- exactly the case
    _twiss_4d_radiation_safe() / scan_mux_muy_emittance() above already
    work around for method='4d' by disabling radiation completely rather
    than setting it to 'mean'. This mirrors that, restoring whatever
    model was active afterward."""
    had_radiation = getattr(line, '_radiation_model', None) is not None
    if had_radiation:
        SR_model = line._radiation_model
        BS_model = line._beamstrahlung_model
        line.configure_radiation(model=None, model_beamstrahlung=None)

    try:
        try:
            line.match(
                method='6d',
                vary=[xt.VaryList(tune_knobs, step=1e-4)],
                targets=[xt.TargetSet(qx=qx_target, qy=qy_target, tol=tol)],
            )
            return True
        except Exception as e:
            print(f"  Tune match FAILED at Qx={qx_target:.4f}, Qy={qy_target:.4f}: {e}")
            return False
    finally:
        if had_radiation:
            line.configure_radiation(model=SR_model, model_beamstrahlung=BS_model)


def debug_tune_match(line, qx_target, qy_target, tune_knobs, tol=1e-6):
    """Verifies the fix: disabling radiation ENTIRELY (not model='mean')
    before twiss/match, confirmed via the traceback this produced last
    time -- Step A alone reproduced "Invalid n1" with no match() involved,
    proving 'mean' radiation itself breaks compute_linear_normal_form()
    on this ring. Prints the FULL traceback if it's somehow still failing,
    rather than a one-line summary, so any remaining issue is visible."""
    import traceback
    had_radiation = getattr(line, '_radiation_model', None) is not None
    if had_radiation:
        SR_model = line._radiation_model
        BS_model = line._beamstrahlung_model
        line.configure_radiation(model=None, model_beamstrahlung=None)

    print(f"--- debug_tune_match(Qx={qx_target}, Qy={qy_target}) ---")

    print("--- Step A: plain line.twiss6d() with radiation OFF (no match) ---")
    try:
        tw_check = line.twiss6d()
        print(f"twiss6d() SUCCEEDED: qx={tw_check.qx:.6f}, qy={tw_check.qy:.6f}")
    except Exception:
        print("twiss6d() FAILED even with radiation off -- something else going on:")
        traceback.print_exc()
        if had_radiation:
            line.configure_radiation(model=SR_model, model_beamstrahlung=BS_model)
        return

    print("--- Step B: line.match(...) with radiation OFF ---")
    try:
        line.match(
            method='6d',
            vary=[xt.VaryList(tune_knobs, step=1e-4)],
            targets=[xt.TargetSet(qx=qx_target, qy=qy_target, tol=tol)],
        )
        print("--- match succeeded ---")
    except Exception:
        print("--- FULL traceback (not the one-line summary) ---")
        traceback.print_exc()
    finally:
        if had_radiation:
            line.configure_radiation(model=SR_model, model_beamstrahlung=BS_model)
 
 
def performance_at_tune(line, qx_target, qy_target, tune_knobs,
                         study_params_DA, context_tracking):
    """Matches to (qx_target, qy_target), runs a DA study, returns min_DA
    (np.nan if the tune match or tracking fails)."""
    if not match_tune(line, qx_target, qy_target, tune_knobs):
        return np.nan
 
    line.discard_tracker()
    line.build_tracker(_context=context_tracking)
    line.configure_radiation(model='mean')
 
    try:
        particles_DA, grid_DA = xutil.generate_particle_grid(line, study_params_DA)
        line.track(particles_DA, num_turns=study_params_DA['number_of_turns'],
                   turn_by_turn_monitor=False, time=True, with_progress=False)
        particles_DA.sort(interleave_lost_particles=True)
        _, _, min_DA = get_DA_boundary(particles_DA, grid_DA['num_r_y_points'],
                                        grid_DA['num_theta_x_points'],
                                        grid_DA['x_normalized'], grid_DA['y_normalized'])
        return min_DA
    except Exception as e:
        print(f"  DA tracking FAILED at Qx={qx_target:.4f}, Qy={qy_target:.4f}: {e}")
        return np.nan
 
 
def tune_scan(line, qx_range, qy_range, tune_knobs, n_emittancex, n_emittancey,
              start_element, n_points=10, scan_turns=1000, scan_particles=200):
    """Coarse performance tune scan over a grid of (Qx, Qy). Returns
    (QX, QY, DA_map). Operates on a COPY of `line` -- your original line
    object is never modified."""
    scan_line = line.copy()
    context_tracking = xo.ContextCpu(omp_num_threads=0)
    scan_line.build_tracker(_context=context_tracking)
 
    # Sanity check BEFORE burning time on the whole grid: try matching to
    # the center of the requested window. If even this fails, something
    # about method/knobs/targets is wrong -- better to find out in one
    # attempt than after 100 identical failures.
    qx_center = 0.5 * (qx_range[0] + qx_range[1])
    qy_center = 0.5 * (qy_range[0] + qy_range[1])
    if not match_tune(scan_line, qx_center, qy_center, tune_knobs):
        raise RuntimeError(
            "Tune match failed at the CENTER of the scan window -- this points to a "
            "setup problem (wrong tune_knobs, or method mismatch with how this ring "
            "is configured), not an unreachable tune. Fix this before scanning the "
            "full grid, or every point will fail the same way."
        )
 
    # Reduced-resolution study params -- this is a fast screening pass,
    # not a publication-quality DA study. Re-run at full resolution
    # (like study_params_DA elsewhere in analysis.py) for whichever
    # point(s) look best here.
    study_params_DA = {
        'ini_cond_type': 'grid_DA',
        'output_dir': 'out',
        'number_of_turns': scan_turns,
        'number_of_particles': scan_particles,
        'inv1': 0, 'inv2': 0,
        'start_element': start_element,
        'ini_cond_nemittance_x': n_emittancex,
        'ini_cond_nemittance_y': n_emittancey,
        'ini_cond_bunch_length': 4.8e-3,
        'ini_cond_energy_spread': 2e-3,
        'ini_cond_energy_offset': None,
        'new_closed_orbit': None,
        'covariance_dispertion_free': False,
    }
 
    qx_vals = np.linspace(*qx_range, n_points)
    qy_vals = np.linspace(*qy_range, n_points)
    QX, QY = np.meshgrid(qx_vals, qy_vals)
    DA_map = np.full_like(QX, np.nan)
 
    total = n_points * n_points
    done = 0
    for i in range(n_points):
        for j in range(n_points):
            DA_map[i, j] = performance_at_tune(
                scan_line, QX[i, j], QY[i, j], tune_knobs, study_params_DA, context_tracking)
            done += 1
            print(f"[{done}/{total}] Qx={QX[i, j]:.4f} Qy={QY[i, j]:.4f} "
                  f"-> min_DA={DA_map[i, j]}")
 
    return QX, QY, DA_map
 
 
def plot_tune_scan(QX, QY, DA_map, current_qx=None, current_qy=None):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.pcolormesh(QX, QY, DA_map, shading='auto', cmap='viridis')
    fig.colorbar(im, ax=ax, label=r'Minimum DA [$\sigma$]')
    if current_qx is not None:
        ax.plot(current_qx, current_qy, 'o', color='red', ms=10, mec='black',
                label='Current working point')
        ax.legend()
    ax.set_xlabel('$Q_x$')
    ax.set_ylabel('$Q_y$')
    ax.set_title('Performance Tune Scan: Minimum Dynamic Aperture')
    plt.tight_layout()
    return fig


def best_from_map(QX, QY, DA_map):
    """np.nanargmax raises on an all-NaN array instead of returning
    something useful -- surface a clear message pointing back at the scan
    output instead of a bare numpy traceback."""
    n_valid = np.sum(~np.isnan(DA_map))
    if n_valid == 0:
        raise RuntimeError(
            "Every point in the scan failed (DA_map is all NaN) -- there is no "
            "'best' point to report. Check the 'Tune match FAILED' / 'DA tracking "
            "FAILED' messages printed during the scan for the actual cause."
        )
    if n_valid < DA_map.size:
        print(f"Note: {DA_map.size - n_valid}/{DA_map.size} grid points failed "
              f"and are excluded from this result.")
    idx = np.unravel_index(np.nanargmax(DA_map), DA_map.shape)
    return QX[idx], QY[idx], DA_map[idx]
 


current_qx, current_qy = ref_twiss.qx, ref_twiss.qy

# Minimal isolation: 'mean' vs radiation-off both failed identically on a
# copy, and Step A doesn't even use a target tune -- so the only variable
# left is .copy() itself. Compare the ORIGINAL ring (unmodified, right
# here, right now) against a bare .copy() with NOTHING else changed.
print("--- minimal isolation: original ring vs a bare .copy(), nothing else changed ---")
try:
    tw_orig = ring.twiss6d()
    print(f"ring.twiss6d() (ORIGINAL, no copy) SUCCEEDED: qx={tw_orig.qx:.6f}")
except Exception as e:
    print(f"ring.twiss6d() (ORIGINAL, no copy) FAILED: {type(e).__name__}: {e}")

try:
    ring_copy_bare = ring.copy()
    tw_copy = ring_copy_bare.twiss6d()
    print(f"ring.copy().twiss6d() (bare copy, no other changes) SUCCEEDED: qx={tw_copy.qx:.6f}")
except Exception as e:
    print(f"ring.copy().twiss6d() (bare copy, no other changes) FAILED: {type(e).__name__}: {e}")
print("--- end minimal isolation ---")

# Run the debug version ONCE, targeting the line's OWN current tune, on a
# throwaway copy -- if matching to where it already is fails, that proves
# it's not about reachability and gives the real traceback instead of the
# "Invalid n1" one-liner.
debug_tune_match(ring.copy(), current_qx, current_qy, tune_knobs=['kQFarcM', 'kQDarcM'])

QX, QY, DA_map = tune_scan(
        ring,
        qx_range=(current_qx - 0.1, current_qx + 0.1),
        qy_range=(current_qy - 0.1, current_qy + 0.1),
        tune_knobs=['kQFarcM', 'kQDarcM'],   # CONFIRM this is your global tune knob
        n_emittancex=n_emittancex, n_emittancey=n_emittancey,
        start_element='QD1_R1',
        n_points=10, scan_turns=1000, scan_particles=200,
    )
plot_tune_scan(QX, QY, DA_map, current_qx=current_qx, current_qy=current_qy)
plt.show()

best_qx, best_qy, best_da = best_from_map(QX, QY, DA_map)
print(f"Best point in scanned window: Qx={best_qx:.4f}, Qy={best_qy:.4f}, "
      f"min_DA={best_da:.2f} sigma")