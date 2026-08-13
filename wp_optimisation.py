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
                        show_best=False, objectives=None):
    """
    Four-panel (mu_x, mu_y) scan plot:

        top-left     = horizontal equilibrium emittance
        top-right    = momentum compaction factor (alpha_c)
        bottom-left  = horizontal chromaticity (dqx)
        bottom-right = vertical chromaticity (dqy)

    Axes are cell phase advance only (mu_x, mu_y) -- no ring working
    point or secondary tune axes.

    show_best: if True, each panel gets its own textbox marking the grid
    point that's "best" FOR THAT SPECIFIC METRIC ALONE -- these four
    points will generally NOT coincide, which is the point: the plot
    exists to show the tradeoff between them, not to pick a single winner.
    objectives: optional dict overriding the default per-panel objective.
    Defaults: emit_x='min', alpha_c='abs_min', dqx='abs_min', dqy='abs_min'.
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


pdr = xt.Environment.from_json('/home/mwatson/Documents/laughing-octo-bassoon/JSON Files/D3/C0/pdr_perfect_90.json')
ring = pdr.lines['ring']
cell_arc = pdr.lines['cell_arc']
tw = cell_arc.twiss(method='4d')

scan = scan_mux_muy_emittance(
    cell_arc, mux_range=(tw.mux[-1]-0.1, tw.mux[-1]+0.1), muy_range=(tw.muy[-1]-0.1, tw.muy[-1]+0.1),
    n_mux=20, n_muy=20,
)
fig, axes = plot_mux_muy_4panel(scan)
fig.savefig('wp_plots.png')
