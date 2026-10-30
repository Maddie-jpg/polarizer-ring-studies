"""
Working-point optimisation studies for one lattice.

Sections (# %% cells):
  1. Arc-cell phase-advance scan (mux, muy) -> emittance and chromaticity maps
  2. Resonance driving terms (RDTs) over the tune diagram -> best working point
  3. Realistic tune scan: rematch the ring at each (Qx, Qy) and compute DA

Lattice chosen by DESIGN/CONFIG/PHASE/CHANGES (see paths.study_from_env).
"""
#%%
#---------------------------------------------------------------------------
# Setup -- load the lattice once, used by all three sections below
#---------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import xtrack as xt
import xobjects as xo
from math import factorial

import paths
from LatticeBuild.linear_optics import matchingWP, matchingBeta, _match_cells_3fold
import xutil_DA_CC.xsuite_utilities as xutil

study = paths.study_from_env()
# Every figure and data file from this script goes here, e.g.
# Results/D1/C9/90deg_FDF/WorkingPoint/
WP_DIR = paths.results_dir(5, study['config'], study['phase'],
                           study['changes'], metric='WorkingPoint')
pdr = paths.load_lattice(study['design'], study['config'], 'perfect',
                         study['phase'], study['changes'])
ring = pdr.lines['ring']
cell_arc = pdr.lines['cell_arc']
arc1R = pdr.lines['arc1R']
cell_tr = pdr.lines['cell_tr']
ref_twiss = ring.twiss6d()
line_table = ring.get_table()

n_emittancex, n_emittancey = paths.injected_beam_nemittance(
    study['design'], study['config'], study['phase'], study['changes'])
print(f"Normalised emittances from the macroparticle distribution: "
      f"x = {n_emittancex:.4e} m rad, y = {n_emittancey:.4e} m rad")


def two_integer_range(q):
    """Returns (lo, hi) spanning a full 2-integer window around q, e.g.
    q=15.46 -> (14, 16). The RDT scan and the DA tune scan both use this
    same helper so their windows always stay consistent with each other."""
    base = np.floor(q)
    return base - 1, base + 1


def one_integer_range(q, offset=0):
    """(lo, hi) of the integer square containing q, e.g. q=15.7 -> (15, 16).
    offset shifts it by whole integers: offset=-1 gives (14, 15)."""
    base = np.floor(q) + offset
    return base, base + 1


def centred_range(q, half_width=1.0):
    """(lo, hi) = (q - half_width, q + half_width), e.g. q=15.22 -> (14.22, 16.22)."""
    return q - half_width, q + half_width


def grid_points_aligned(lo, hi, per_unit):
    """Grid points spaced 1/per_unit, at the centres of cells aligned to the
    integers (k + (i + 0.5)/per_unit), keeping only those inside [lo, hi].
    The window can start anywhere (e.g. centred on the WP) and the points
    still never sit on an integer, and with per_unit even never on a
    half-integer. per_unit=10 -> 0.1 spacing, 20 points over a +-1 window."""
    if per_unit % 2:
        print(f'grid_points_aligned: per_unit={per_unit} is odd, so some points '
              f'are on the half-integer')
    k0, k1 = int(np.floor(lo)), int(np.ceil(hi))
    pts = np.concatenate([k + (np.arange(per_unit) + 0.5) / per_unit
                          for k in range(k0, k1)])
    return pts[(pts >= lo) & (pts <= hi)]


def grid_points(lo, hi, n):
    """n points at the centres of n equal cells between lo and hi, so the
    edges (integer resonances) are never sampled. With n even no point falls
    on the half-integer either, e.g. lo=15, hi=16, n=10 -> 15.05 ... 15.95."""
    if n % 2:
        print(f'grid_points: n={n} is odd, so the middle point is on the half-integer')
    return lo + (np.arange(n) + 0.5) * (hi - lo) / n


#%%
#---------------------
# phase advance scan
#---------------------

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

    Returns dict with 'mux_grid', 'muy_grid' (1D arrays) and four 2D
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
        # at. These knobs are shared with `ring` via the same pdr
        # environment, not local to cell_arc, so leaving them at a bad
        # value corrupts every twiss()/match() on the real ring for the
        # rest of the script, not just cell_arc.
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
                        show_best=False, objectives=None):
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


def plot_mux_muy_combined_chroma(scan_result, ax=None, cmap='viridis',
                                 show_best=False, objective='min'):
    """
    Plot mu_x vs mu_y with the heatmap being sqrt(dqx + dqy), computed
    from an existing scan_mux_muy_emittance() result (no re-scan needed).

    NOTE: dqx and dqy are typically both negative for uncorrected natural
    chromaticity, so dqx+dqy is usually negative -- sqrt() of that is NaN
    (numpy returns NaN rather than raising), so most of this plot may come
    back blank unless some region of your scanned (mu_x, mu_y) actually
    has dqx+dqy >= 0. That's expected given the formula as specified, not
    a bug. If you want a real-valued combined-chromaticity magnitude
    everywhere instead, use sqrt(dqx**2 + dqy**2) (quadrature sum).
    """
    mux_grid = scan_result['mux_grid']
    muy_grid = scan_result['muy_grid']
    dqx = scan_result['dqx']
    dqy = scan_result['dqy']

    with np.errstate(invalid='ignore'):
        combined = np.sqrt(dqx + dqy)

    n_valid = np.sum(~np.isnan(combined))
    print(f"sqrt(dqx+dqy): {n_valid}/{combined.size} grid points have "
         f"dqx+dqy >= 0 (real-valued); the rest are NaN.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.figure

    im = ax.pcolormesh(mux_grid, muy_grid, combined, shading='auto', cmap=cmap)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r'$\sqrt{\delta_{q_x}+\delta_{q_y}}$')

    ax.set_xlim(mux_grid.min(), mux_grid.max())
    ax.set_ylim(muy_grid.min(), muy_grid.max())

    if show_best:
        best = _find_best_point(combined, mux_grid, muy_grid, objective=objective)
        _annotate_best_point(ax, best, label=f'{objective} '+r'$\sqrt{\delta_{q_x}+\delta_{q_y}}$',
                             value_fmt='{:.4f}')

    ax.set_xlabel(r'Cell $\mu_x$ (fraction of $2\pi$)')
    ax.set_ylabel(r'Cell $\mu_y$ (fraction of $2\pi$)')
    ax.set_title(r'$\sqrt{\delta_{q_x}+\delta_{q_y}}$ vs cell phase advance')

    return fig, ax


scan = scan_mux_muy_emittance(
    cell_arc, mux_range=(0.30, 0.45), muy_range=(0.15, 0.35),
    n_mux=20, n_muy=20,
)
fig, axes = plot_mux_muy_4panel(scan)
fig.savefig(f'{WP_DIR}/phase_advance_scan.png', dpi=300, bbox_inches='tight')

fig2, ax2 = plot_mux_muy_combined_chroma(scan)
fig2.savefig(f'{WP_DIR}/combined_chroma_plot.png', dpi=300, bbox_inches='tight')

# Guard: kQFarc/kQDarc are shared globally between cell_arc and ring (both
# come from `pdr`), so if the scan above ever again leaves them at a bad
# grid point instead of restoring k0, this catches it here with a clear
# message instead of a confusing error much later in the script.
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

    Tries the scalar convention FIRST: dedicated element classes here
    (Sextupole etc.) carry BOTH a real .k2/.length AND a stale,
    always-zero .knl array simultaneously -- checking .knl first silently
    "succeeds" with 0.0 and never reaches the real value. The scalar
    attribute, when present, is authoritative for these dedicated
    classes; .knl is the fallback for generic Multipole-class elements
    that only have that array.
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
            continue

    h_arr = np.array(h_list, dtype=complex)
    if h_arr.size == 0:
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


def plot_rdt_lines_combined(QX, QY, maps, qx_range, qy_range,
                             current_qx=None, current_qy=None, cmap='inferno'):
    """Single combined diagram for all scanned RDTs:
      - background: log10 of the STRONGEST RDT at each grid point (max
        across whichever RDTs were scanned)
      - foreground: each RDT's own specific resonance line(s)
        (p-q)*Qx + (r-t)*Qy = integer, drawn in its own color with one
        legend entry
    """
    fig, ax = plt.subplots(figsize=(9, 8))

    combined = np.max([np.log10(grid + 1e-12) for grid in maps.values()], axis=0)
    finite = combined[np.isfinite(combined)]
    vmin, vmax = np.percentile(finite, [2, 98]) if finite.size > 0 else (None, None)
    im = ax.pcolormesh(QX, QY, combined, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, label=r'$\log_{10}|f|$ (strongest RDT at each point)')

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(maps), 2)))
    qx_vals = np.array(qx_range)

    for rdt, color in zip(maps.keys(), colors):
        p, q, r, t = _parse_rdt_key(rdt)
        a, b = p - q, r - t
        corner_vals = [a * cx + b * cy for cx in qx_range for cy in qy_range]
        c_lo, c_hi = int(np.floor(min(corner_vals))), int(np.ceil(max(corner_vals)))

        label_used = False
        for c in range(c_lo, c_hi + 1):
            if b != 0:
                qy_vals = (c - a * qx_vals) / b
                if np.any((qy_vals >= qy_range[0]) & (qy_vals <= qy_range[1])):
                    ax.plot(qx_vals, qy_vals, '-', color=color, lw=1.3,
                            label=(f'$f_{{{rdt[1:]}}}$' if not label_used else None))
                    label_used = True
            elif a != 0:
                qx_line = c / a
                if qx_range[0] <= qx_line <= qx_range[1]:
                    ax.axvline(qx_line, color=color, lw=1.3,
                               label=(f'$f_{{{rdt[1:]}}}$' if not label_used else None))
                    label_used = True

    if current_qx is not None:
        ax.plot(current_qx, current_qy, 'o', color='cyan', ms=11, mec='black',
                zorder=5, label='Current working point')

    ax.set_xlim(qx_range)
    ax.set_ylim(qy_range)
    ax.set_xlabel('$Q_x$')
    ax.set_ylabel('$Q_y$')
    ax.set_title('Resonance driving terms: combined working-point diagram')
    ax.legend(loc='upper right', fontsize='small', framealpha=0.9)
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


# NB: this map keeps the optics fixed and only moves the tune (like a phase
# trombone at s=0). With the observation point at s=0 the numerator of every
# RDT is the same at every (Qx, Qy), so the colours only show the distance
# to each RDT's resonance line -- use it as a resonance-line diagram. The DA
# scan below records the real RDTs (max around the ring), amplitude detuning
# and Q'' of each re-matched point; see plot_metric_maps there.
tt = ring.get_table()

rdts_of_interest = ['f3000', 'f2100', 'f1020', 'f1011', 'f1002']  # 3rd-order, sextupole-driven
current_qx, current_qy = ref_twiss.qx, ref_twiss.qy
qx_range = two_integer_range(current_qx)
qy_range = two_integer_range(current_qy)

QX, QY, maps = scan_working_point(
    rdts_of_interest, ref_twiss, ring, line_table,
    observation_point=tt.name[0], source_elements=line_table.name,
    qx_range=qx_range, qy_range=qy_range,
    n_points=400,
)

fig_rdt = plot_rdt_lines_combined(QX, QY, maps, qx_range, qy_range,
                                  current_qx=current_qx, current_qy=current_qy)
fig_rdt.savefig(f'{WP_DIR}/rdt_resonance_lines.png', dpi=300, bbox_inches='tight')
plt.show()

best_qx, best_qy, combined_score = find_best_working_point(QX, QY, maps)
print(f"Best working point in scanned window: Qx={best_qx:.4f}, Qy={best_qy:.4f}")


#%%
#----------------------------------------------
# Tune performance scan (min DA) -- REALISTIC
#----------------------------------------------
"""
For each candidate (Qx, Qy), re-matches the ring using the SAME
production matching process as linear_optics.py's _run_standard_matching
(matchingWP then matchingBeta on arc1R), instead of a generic placeholder
match on ring directly. Then runs a reduced-resolution DA study there.

Imports matchingWP / matchingBeta / _match_cells_3fold directly from
linear_optics.py (see the import at the top of this file) rather than
re-implementing them, so this always matches whatever the actual build
pipeline currently does.
"""


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


def match_tune_realistic(arc1R, cell_arc, cell_arc_opt, cell_tr, cell_tr_opt,
                         qx_target, qy_target, n_periods=6,
                         betay_DS_target=None):
    """
    Matches arc1R to (qx_target, qy_target) using the REAL production
    sequence -- matchingWP then matchingBeta, exactly as
    _run_standard_matching does in linear_optics.py -- instead of a
    simplified direct match on ring.

    Since ring and arc1R share the same pdr Environment variables, once
    arc1R's knobs are set here, ring's own twiss()/track() reflect the
    new tune automatically -- no separate match on ring itself is needed
    or correct (the real build pipeline never matches ring directly).

    Returns True if both matchingWP and matchingBeta completed without
    raising; False (with a printed reason) otherwise. A failure near
    resonances/the edge of the achievable tune space is normal, not
    something that should crash the scan.
    """
    try:
        matchingWP(qx_target, qy_target, cell_arc_opt, cell_arc, arc1R,
                  n_periods=n_periods)

        tw_tr = cell_tr.twiss(method='4d')
        mid = len(tw_tr.betx) // 2
        matchingBeta(tw_tr.betx[mid], tw_tr.bety[mid],
                    cell_arc_opt, cell_arc, cell_tr_opt, cell_tr, arc1R,
                    betay_DS_target=betay_DS_target)
        return True
    except Exception as e:
        print(f"  Realistic tune match FAILED at Qx={qx_target:.4f}, "
              f"Qy={qy_target:.4f}: {type(e).__name__}: {e}")
        return False


def performance_at_tune_realistic(ring, arc1R, cell_arc, cell_arc_opt,
                                  cell_tr, cell_tr_opt, qx_target, qy_target,
                                  n_periods, study_params_DA, context_tracking,
                                  betay_DS_target=None):
    """Matches to (qx_target, qy_target) via the real production process,
    then runs a DA study on `ring` (which picks up the new tune through
    its shared knobs with arc1R). Returns min_DA (np.nan if the tune
    match or tracking fails)."""
    if not match_tune_realistic(arc1R, cell_arc, cell_arc_opt, cell_tr,
                                cell_tr_opt, qx_target, qy_target,
                                n_periods=n_periods,
                                betay_DS_target=betay_DS_target):
        return np.nan

    ring.discard_tracker()
    ring.build_tracker(_context=context_tracking)
    ring.configure_radiation(model='mean')

    try:
        particles_DA, grid_DA = xutil.generate_particle_grid(ring, study_params_DA)
        ring.track(particles_DA, num_turns=study_params_DA['number_of_turns'],
                  turn_by_turn_monitor=False, time=True, with_progress=False)
        particles_DA.sort(interleave_lost_particles=True)
        _, _, min_DA = get_DA_boundary(particles_DA, grid_DA['num_r_y_points'],
                                       grid_DA['num_theta_x_points'],
                                       grid_DA['x_normalized'], grid_DA['y_normalized'])
        return min_DA
    except Exception as e:
        print(f"  DA tracking FAILED at Qx={qx_target:.4f}, Qy={qy_target:.4f}: {e}")
        return np.nan


def tune_scan_realistic(pdr, ring, arc1R, cell_arc, cell_tr, qx_range, qy_range,
                        n_emittancex, n_emittancey, start_element,
                        n_periods=6, mu_cell=0.25, betay_DS_target=None,
                        n_points=10, scan_turns=1000, scan_particles=200,
                        sanity_qx=None, sanity_qy=None):
    """
    Coarse performance tune scan over a grid of (Qx, Qy), matching via the
    REAL production matchingWP/matchingBeta sequence at every grid point.

    IMPORTANT -- safety / knob restoration: matchingWP/matchingBeta vary
    kQFarcM, kQDarcM, kQFDS, kQDDS, kQFDoub, kQDDoub, AND the physical
    length knobs l_trans/l_doub/l_trips. These are shared pdr-level
    variables between arc1R and ring (a Line.copy() does NOT give an
    independent Environment, so copying arc1R would NOT isolate this
    scan). This function saves every one of these knobs' values before
    scanning and restores them in a finally block, so your real
    ring/arc1R are left EXACTLY as they were regardless of how the scan
    goes -- do not remove this without re-adding equivalent protection.

    mu_cell: the design's regular-arc-cell phase advance (fraction of
    2*pi) -- used once, up front, to (re)build cell_arc_opt/cell_tr_opt
    via _match_cells_3fold exactly as the real pipeline does. This is NOT
    scanned here -- only the ring's global tune (via arc1R's insertion
    knobs) is scanned at FIXED cell phase advance.

    sanity_qx/sanity_qy: pass your REAL, currently-valid working point
    for the pre-flight check -- the ONLY check that can reliably tell
    "real setup bug" from "this grid point is genuinely unreachable".
    """
    restore_knobs = ['kQFarc', 'kQDarc', 'kQFarcM', 'kQDarcM', 'kQFDS', 'kQDDS',
                     'kQFDoub', 'kQDDoub', 'kQDDoubDS', 'kQFtr', 'kQDtr',
                     'l_trans', 'l_doub', 'l_trips']
    k0 = {kn: arc1R.vars[kn]._value for kn in restore_knobs
         if kn in arc1R.vars}

    context_tracking = xo.ContextCpu(omp_num_threads=0)

    try:
        cell_arc_opt, cell_tr_opt = _match_cells_3fold(
            pdr, cell_arc, cell_tr, mu_cell=mu_cell)

        if sanity_qx is not None and sanity_qy is not None:
            if not match_tune_realistic(arc1R, cell_arc, cell_arc_opt,
                                        cell_tr, cell_tr_opt,
                                        sanity_qx, sanity_qy,
                                        n_periods=n_periods,
                                        betay_DS_target=betay_DS_target):
                raise RuntimeError(
                    f"Realistic tune match failed at the KNOWN CURRENT "
                    f"WORKING POINT (Qx={sanity_qx:.4f}, Qy={sanity_qy:.4f}) -- "
                    f"since this is a real, already-valid tune, failing here "
                    f"means a genuine setup problem (wrong n_periods/mu_cell, "
                    f"or arc1R/cell_arc not the right objects), not an "
                    f"unreachable point. Fix this before trusting the scan."
                )
        else:
            qx_center = 0.5 * (qx_range[0] + qx_range[1])
            qy_center = 0.5 * (qy_range[0] + qy_range[1])
            if not match_tune_realistic(arc1R, cell_arc, cell_arc_opt,
                                        cell_tr, cell_tr_opt,
                                        qx_center, qy_center,
                                        n_periods=n_periods,
                                        betay_DS_target=betay_DS_target):
                print(f"[tune_scan_realistic] NOTE: sanity check at the "
                      f"window's geometric center (Qx={qx_center:.4f}, "
                      f"Qy={qy_center:.4f}) failed -- may be genuinely "
                      f"unreachable rather than a setup bug. Continuing with "
                      f"the full scan; pass sanity_qx/sanity_qy (your real "
                      f"current working point) for a pre-flight check that "
                      f"can tell the two apart.")

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
        n_failed = 0
        for i in range(n_points):
            for j in range(n_points):
                DA_map[i, j] = performance_at_tune_realistic(
                    ring, arc1R, cell_arc, cell_arc_opt, cell_tr, cell_tr_opt,
                    QX[i, j], QY[i, j], n_periods, study_params_DA,
                    context_tracking, betay_DS_target=betay_DS_target)
                done += 1
                if np.isnan(DA_map[i, j]):
                    n_failed += 1
                print(f"[{done}/{total}] Qx={QX[i, j]:.4f} Qy={QY[i, j]:.4f} "
                      f"-> min_DA={DA_map[i, j]}")

        print(f"[tune_scan_realistic] Done: {total - n_failed}/{total} points "
              f"succeeded, {n_failed} failed/unreachable.")

        return QX, QY, DA_map

    finally:
        # Restore every knob matchingWP/matchingBeta touched, regardless
        # of whether the scan finished cleanly or raised partway through.
        for kn, v in k0.items():
            arc1R.vars[kn] = v
        print(f"[tune_scan_realistic] Restored {len(k0)} knobs to their "
             f"pre-scan values: {list(k0.keys())}")


def trim_failed_edges(QX, QY, DA_map, maps=None):
    """Drop outer rows/columns of the grid where every point failed (DA NaN)."""
    ok = np.isfinite(DA_map)
    rows = np.where(ok.any(axis=1))[0]
    cols = np.where(ok.any(axis=0))[0]
    r = slice(rows[0], rows[-1] + 1)
    c = slice(cols[0], cols[-1] + 1)
    trimmed = None if maps is None else {k: v[r, c] for k, v in maps.items()}
    return QX[r, c], QY[r, c], DA_map[r, c], trimmed


def fill_metric_gaps(QX, QY, z, valid):
    """Fill NaN cells of a metric map that sit inside the region where the DA
    scan succeeded, by linear interpolation from the computed neighbours
    (nearest neighbour where linear can't reach). Cells where the scan itself
    failed stay NaN."""
    from scipy.interpolate import griddata
    known = np.isfinite(z)
    holes = valid & ~known
    if not holes.any() or known.sum() < 4:
        return z
    pts = np.column_stack([QX[known], QY[known]])
    out = z.copy()
    out[holes] = griddata(pts, z[known], (QX[holes], QY[holes]), method='linear')
    still = holes & ~np.isfinite(out)
    if still.any():
        out[still] = griddata(pts, z[known], (QX[still], QY[still]), method='nearest')
    return out


def plot_metric_maps(QX, QY, maps, DA_map=None, current_qx=None, current_qy=None,
                     keys=None):
    """One panel per metric from wp_da_scan.metric_maps (plus the DA first).
    RDTs (|f|) are shown as log10; detuning and Q'' on a linear scale."""
    if keys is None:
        order = lambda k: (0 if k.startswith('f') else 1 if k.startswith('det') else 2, k)
        keys = sorted((k for k in maps if not k.endswith('_rms')), key=order)
    panels = ([('DA', DA_map)] if DA_map is not None else []) + [(k, maps[k]) for k in keys]
    nc = 4
    nr = int(np.ceil(len(panels) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.6 * nr), squeeze=False)
    valid = np.isfinite(DA_map) if DA_map is not None else None
    for ax, (k, z) in zip(axs.flat, panels):
        if valid is not None and k != 'DA':
            z = fill_metric_gaps(QX, QY, z, valid)
        is_rdt = k.startswith('f') and k[1:5].isdigit()
        zz = np.log10(z) if is_rdt else z
        lab = {'DA': r'min DA [$\sigma$]', 'det_xx': r'$dQ_x/dJ_x$ [1/m]',
               'det_yy': r'$dQ_y/dJ_y$ [1/m]', 'det_xy': r'$dQ_x/dJ_y$ [1/m]',
               'det_yx': r'$dQ_y/dJ_x$ [1/m]', 'ddqx': r"$Q_x''$", 'ddqy': r"$Q_y''$"}.get(
               k, rf'$\log_{{10}}|f_{{{k[1:5]}}}|$ max' if is_rdt else k)
        cmap = 'viridis' if k == 'DA' else ('magma' if is_rdt else 'coolwarm')
        if cmap == 'coolwarm' and np.isfinite(zz).any():
            v = np.nanmax(np.abs(zz)); kw = dict(vmin=-v, vmax=v)
        else:
            kw = {}
        im = ax.pcolormesh(QX, QY, zz, shading='auto', cmap=cmap, **kw)
        fig.colorbar(im, ax=ax)
        if current_qx is not None:
            ax.plot(current_qx, current_qy, 'o', color='cyan', mec='k', ms=7)
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel('$Q_x$'); ax.set_ylabel('$Q_y$')
    for ax in list(axs.flat)[len(panels):]:
        ax.axis('off')
    fig.tight_layout()
    return fig


def print_metric_correlations(maps, DA_map):
    """Spearman rank correlation of each metric with the DA over the points
    where both exist. Uses |value| for detuning and Q'' (size matters, not
    sign)."""
    def rank(a):
        r = np.empty(len(a)); r[np.argsort(a)] = np.arange(len(a)); return r
    print(f"{'metric':12s} {'rank corr. with DA':>20s} {'n':>5s}")
    for k, z in maps.items():
        v = np.abs(z) if not k.startswith('f') else z
        ok = np.isfinite(v) & np.isfinite(DA_map)
        if ok.sum() < 5 or np.ptp(v[ok]) == 0:
            continue
        c = np.corrcoef(rank(v[ok]), rank(DA_map[ok]))[0, 1]
        print(f'{k:12s} {c:20.2f} {int(ok.sum()):5d}')


def plot_tune_scan(QX, QY, DA_map, current_qx=None, current_qy=None):
    plot_data = np.nan_to_num(DA_map, nan=0.0)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.pcolormesh(QX, QY, plot_data, shading='auto', cmap='viridis')
    fig.colorbar(im, ax=ax, label=r'Minimum DA [$\sigma$] (0 = failed/unreachable)')
    if current_qx is not None:
        ax.plot(current_qx, current_qy, 'o', color='red', ms=10, mec='black',
               label='Current working point')
        ax.legend()
    ax.set_xlabel('$Q_x$')
    ax.set_ylabel('$Q_y$')
    ax.set_title('Performance Tune Scan (realistic matching): Minimum Dynamic Aperture')
    plt.tight_layout()
    return fig


def best_from_map(QX, QY, DA_map):
    n_valid = np.sum(~np.isnan(DA_map))
    if n_valid == 0:
        raise RuntimeError(
            "Every point in the scan failed (DA_map is all NaN) -- there is no "
            "'best' point to report. Check the 'Realistic tune match FAILED' / "
            "'DA tracking FAILED' messages printed during the scan."
        )
    if n_valid < DA_map.size:
        print(f"Note: {DA_map.size - n_valid}/{DA_map.size} grid points failed "
             f"and are excluded from this result.")
    idx = np.unravel_index(np.nanargmax(DA_map), DA_map.shape)
    return QX[idx], QY[idx], DA_map[idx]


current_qx, current_qy = ref_twiss.qx, ref_twiss.qy
# DA scan window: +-1 around the current tunes
# (e.g. 15.22, 13.72 -> Qx 14.22-16.22, Qy 12.72-14.72).
# one_integer_range(q) gives the integer square instead, two_integer_range
# the old 2x2 integer-aligned window.
half_width = 0.9
qy_floor = 13.0
qx_range = centred_range(current_qx, half_width)
qy_range = (max(current_qy - half_width, qy_floor), current_qy + half_width)

import gc
import wp_da_scan
per_unit = 20
rows_per_chunk = 4

qx_scan = grid_points_aligned(*qx_range, per_unit)
qy_scan = grid_points_aligned(*qy_range, per_unit)
print(f"DA scan: {len(qx_scan)} x {len(qy_scan)} = {len(qx_scan)*len(qy_scan)} points "
      f"in chunks of {rows_per_chunk} rows ({rows_per_chunk*len(qx_scan)} points each)")

lattice_json = paths.lattice_json_path(study['design'], study['config'], 'perfect',
                                       study['phase'], study['changes'])
QX_parts, QY_parts, DA_parts, scan_records = [], [], [], None
for c0 in range(0, len(qy_scan), rows_per_chunk):
    qy_chunk = qy_scan[c0:c0 + rows_per_chunk]
    print(f"--- chunk Qy {qy_chunk[0]:.3f}-{qy_chunk[-1]:.3f} "
          f"(rows {c0 + 1}-{c0 + len(qy_chunk)} of {len(qy_scan)}) ---")
    QX_c, QY_c, DA_c, rec_c = wp_da_scan.run_parallel(
        lattice_json, qx_scan, qy_chunk,
        nemitt_x=n_emittancex, nemitt_y=n_emittancey,
        n_turns=530, n_particles=200, radiation='mean',
        sext_knobs=['k2XFarc', 'k2XDarc'], n_workers=None,
    )
    QX_parts.append(QX_c)
    QY_parts.append(QY_c)
    DA_parts.append(DA_c)
    if scan_records is None:
        scan_records = rec_c
    elif isinstance(scan_records, dict):
        scan_records.update(rec_c)
    else:
        scan_records.extend(rec_c)
    del QX_c, QY_c, DA_c, rec_c
    gc.collect()

QX = np.concatenate(QX_parts, axis=0)
QY = np.concatenate(QY_parts, axis=0)
DA_map = np.concatenate(DA_parts, axis=0)
del QX_parts, QY_parts, DA_parts
gc.collect()

QX_t, QY_t, DA_t, _ = trim_failed_edges(QX, QY, DA_map)
fig_da = plot_tune_scan(QX_t, QY_t, DA_t, current_qx=current_qx, current_qy=current_qy)
fig_da.savefig(f'{WP_DIR}/tune_scan_min_DA.png', dpi=300, bbox_inches='tight')
plt.show()

best_qx, best_qy, best_da = best_from_map(QX, QY, DA_map)

# RDTs (max around the ring), amplitude detuning and Q'' from each point's
# re-matched optics, on the same grid as the DA -- and how well each one
# tracks the DA (rank correlation; strongly negative = "bigger is worse").
metric = wp_da_scan.metric_maps(scan_records, QX[0], QY[:, 0])
QX_p, QY_p, DA_p, metric_p = trim_failed_edges(QX, QY, DA_map, metric)
fig_m = plot_metric_maps(QX_p, QY_p, metric_p, DA_p, current_qx=current_qx, current_qy=current_qy)
fig_m.savefig(f'{WP_DIR}/tune_scan_metric_maps.png', dpi=300, bbox_inches='tight')
plt.show()

# Correlation table: printed as before, and also saved as text
import contextlib, io, json
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    print_metric_correlations(metric, DA_map)
print(_buf.getvalue(), end='')
with open(f'{WP_DIR}/tune_scan_correlations.txt', 'w') as f:
    f.write(_buf.getvalue())

# Raw scan data, so the figures can be redrawn without rerunning the scan
np.savez(f'{WP_DIR}/tune_scan_data.npz', QX=QX, QY=QY, DA_map=DA_map,
         current_qx=current_qx, current_qy=current_qy, **metric)
with open(f'{WP_DIR}/tune_scan_records.json', 'w') as f:
    json.dump(scan_records, f, indent=1, default=float)
print(f'Working-point figures and data saved to {WP_DIR}')
print(f"Best point in scanned window: Qx={best_qx:.4f}, Qy={best_qy:.4f}, "
      f"min_DA={best_da:.2f} sigma")

# Serial version (kept for reference; ~n_cores times slower):
# QX, QY, DA_map = tune_scan_realistic(
#     pdr, ring, arc1R, cell_arc, cell_tr,
#     qx_range=qx_range, qy_range=qy_range,
#     n_emittancex=n_emittancex, n_emittancey=n_emittancey,
#     start_element='QD1_R1', n_periods=6, mu_cell=0.25,
#     n_points=25, scan_turns=1000, scan_particles=200,
#     sanity_qx=current_qx, sanity_qy=current_qy,
# )