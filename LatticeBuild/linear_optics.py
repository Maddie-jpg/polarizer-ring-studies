"""
Linear optics builders for the polarizer ring.

Each public builder (three_fold_periodicity, two_fold_1straight,
two_fold_racetrack_3straight, ...) creates an xtrack Environment `pdr` with
lines 'ring', 'period', 'cell_arc', 'cell_tr' and 'arc1R', matches the arc
and straight cells, matches the ring to the requested working point WP and
adds the RF cavity. Sextupoles are added afterwards by a config_D*_C*
function in sextupole_configs.py.

Builders (default working point from config.py in brackets):
    three_fold_periodicity           3 arcs + 3 straights, 90 deg cells   (WP_D1)
    three_fold_periodicity_long      as above, longer cells/straights     (WP_D1)
    three_fold_periodicity_*_120*    120 deg arc cells                     (WP_D1_120)
    two_fold_1straight               two-fold, one long straight           (WP_D2)
    two_fold_racetrack_3straight     two-fold racetrack                    (WP_D3)
Which builder + sextupole config each D/C lattice used is listed in README.md.

Module-level helpers prefixed with "_" are internal steps shared by the
builders (element creation, knob seeding, matching, RF insertion).

Default energies, RF voltage and working points come from config.py.
"""
import sys
from pathlib import Path

# Make the repository root importable (config.py) however this file is reached.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

import numpy as np
import xtrack as xt

import config
import matplotlib as plt

# ----------------------
# Module-level helpers
# ----------------------

def get_natural_WP(cell_arc, arc1R, n_periods=6, verbose=True):
    """Return (qx, qy) implied by current knobs: n_periods × sextant phase advance."""
    tw_cell = cell_arc.twiss(method='4d')
    tw = arc1R.twiss(method='4d',
                     betx=tw_cell.betx[0], alfx=tw_cell.alfx[0],
                     bety=tw_cell.bety[0], alfy=tw_cell.alfy[0],
                     dx=tw_cell.dx[0],     dpx=tw_cell.dpx[0])
    qx = n_periods * tw.mux[-1]
    qy = n_periods * tw.muy[-1]
    if verbose:
        print(f'Natural WP: qx={qx:.6f}, qy={qy:.6f}  '
              f'(sextant: mux={tw.mux[-1]:.6f}, muy={tw.muy[-1]:.6f})')
    return qx, qy



# Smallest beta bound that still gives a stable ring for the shipped
# geometry at WP=(15.8, 13.87). Lower it only alongside a stability check.
BETA_BOUND_FLOOR = 10.

KNOB_LIMITS_DEFAULT = (-15., 15.)
KNOB_LIMITS = {}
FIXED_KNOBS = set()

# Polarity of each matched quad family. Without these the optimiser is free
# to flip signs, and an F-D-F straight with kQFtr < 0, kQDtr > 0 is a D-F-D
# straight in disguise (seen on a Mac build: kQFtr=-6.39, kQDtr=+6.21, betx
# 45 m in the straight). kQDDS is left free: it runs positive in both the
# D-F-D and F-D-F solutions. Override per knob with KNOB_LIMITS.
POLARITY_LIMITS = {
    'kQFarcM': (0., 15.),  'kQDarcM': (-15., 0.),
    'kQFDS':   (0., 15.),
    'kQFDoub': (0., 15.),  'kQDDoub': (-15., 0.),  'kQDDoubDS': (-15., 0.),
    'kQFtr':   (0., 15.),  'kQDtr':   (-15., 0.),
}


def _knob_limits(k):
    """Limits for a matching knob: KNOB_LIMITS, else its polarity, else default."""
    return KNOB_LIMITS.get(k, POLARITY_LIMITS.get(k, KNOB_LIMITS_DEFAULT))


def matchingWP(qx, qy, cell_arc_opt, cell_arc, arc1R, n_periods=6,
               betay_DS_target=None, MakePlot=False, FDF=True,
               beta_weight=3.0, use_ctr_symmetry=False):
    import numpy as np

    cell_arc_opt.run_jacobian(10)
    cell_arc_tw = cell_arc.twiss(method='4d')
    bc = dict(betx=cell_arc_tw.betx[0], alfx=cell_arc_tw.alfx[0],
              bety=cell_arc_tw.bety[0], alfy=cell_arc_tw.alfy[0],
              dx=cell_arc_tw.dx[0],     dpx=cell_arc_tw.dpx[0])

    tw0 = arc1R.twiss(method='4d', **bc)
    mux0, muy0 = tw0.mux[-1], tw0.muy[-1]

    knob_names, mk_list = _beta_bound_setup(arc1R, FDF=FDF)
    vary = [xt.Vary(k, step=1e-5,
                    limits=_knob_limits(k))
            for k in knob_names if k not in FIXED_KNOBS]

    targets = [
        xt.TargetSet(dx=0, dpx=0, at=xt.END, tol=1e-9),
        xt.TargetSet(mux=mux0, muy=muy0, at=xt.END, tol=1e-8, tag='phase'),
        xt.TargetSet(alfx=0, alfy=0, at=xt.END, tol=1e-9),
    ] + _ds_waist_targets(arc1R)        # horizontal DS waist, if markers exist
    # alfx=alfy=0 at CtrS1 is a design preference (waist at the straight
    # centre), not a periodicity requirement -- alfx=alfy=0 at END already
    # closes the half-period. With it on, the problem is 8 hard targets at
    # tol=1e-9 against 6 knobs and never converges, leaving ~0.4 of Qx for
    # the global fine-tune to brute-force. Off by default.
    if use_ctr_symmetry:
        targets.append(
            xt.TargetSet(alfx=0, alfy=0, at='CtrS1_xR1', tol=1e-9, weight=10.))

    # Bound beta through the DS and the straight. These must carry real
    # weight: at the old weight=0.02 they were effectively ignored and bety
    # ran to 16 m in the dispersion suppressor. BETA_MAX below ~9 m collapses
    # the horizontal plane instead (betx -> 30 m), so 10 m is the knee.
    BETA_MAX = betay_DS_target if betay_DS_target else BETA_BOUND_FLOOR
    if BETA_MAX < BETA_BOUND_FLOOR:
        print(f'matchingWP: betay_DS_target={BETA_MAX:g} m is below the '
              f'{BETA_BOUND_FLOOR:g} m knee found for this geometry/working '
              f'point -- the match either misses it entirely or goes unstable. '
              f'Clamping. Lower lo.BETA_BOUND_FLOOR to override after '
              f're-checking stability at your working point.')
        BETA_MAX = BETA_BOUND_FLOOR
    for mk in mk_list:
        targets += [
            xt.Target('betx', xt.LessThan(BETA_MAX), at=mk, weight=beta_weight),
            xt.Target('bety', xt.LessThan(BETA_MAX), at=mk, weight=beta_weight),
        ]

    opt = arc1R.match(
        method='4d', solve=False, verbose=False, **bc,
        vary=vary, targets=targets)

    pt = [t for t in opt.targets if t.tag == 'phase']

    target_mux = qx / n_periods
    target_muy = qy / n_periods

    # Ramping loop using increment steps relative to starting tune
    for frac in np.linspace(1/16, 1, 16):
        pt[0].value = mux0 + frac * (target_mux - mux0)
        pt[1].value = muy0 + frac * (target_muy - muy0)
        try:
            opt.solve(n_steps=40)
        except Exception:
            opt.step(50, broyden=True, rcond=1e-4)

    if MakePlot:
        arc1R.twiss(method='4d', **bc).plot()
    return opt


def matchingBeta(betxS, betyS, cell_arc_opt, cell_arc,
                 cell_tr_opt, cell_tr, arc1R, qx_target=None, qy_target=None,
                 n_periods=6, betay_DS_target=None, MakePlot=False, FDF=True,
                 beta_weight=3.0, match_end_betas=False):
    """Match the sextant optics (no tune targets unless qx/qy_target given).

    Hard targets at the sextant END (the straight-section symmetry point):
        dx = dpx = 0      dispersion-free straight
        alfx = alfy = 0   mirror symmetry -> the half-period closes
    plus upper bounds beta <= BETA_MAX through the matching cell, DS and
    straight (same knobs, bound and weight as matchingWP, so the WP match
    can start from this solution).

    Done in two stages for robustness: (1) closure only, (2) the beta bound
    lowered in steps to BETA_MAX, re-closing each time; the last step that
    stays closed is kept. Quad polarities are held by POLARITY_LIMITS.

    The phase advance is left free, so the ring keeps whatever tunes these
    optics give; match them afterwards with matchingWP if needed.

    Previously this also required betx/bety at END to equal the straight
    cell's values, varied only 6 knobs (no triplet) and bounded beta at 5 m.
    That is over-constrained for the F-D-F straight: it ended with Dx = 22 mm
    and alfa != 0 at the straight, and the period had no stable solution
    ('Invalid n1/n2'). match_end_betas=True restores those END beta targets
    (betxS/betyS from the straight cell) as soft targets.
    """
    cell_arc_opt.run_jacobian(10)
    tw_cell = cell_arc.twiss(method='4d')
    bc = dict(betx=tw_cell.betx[0], alfx=tw_cell.alfx[0],
              bety=tw_cell.bety[0], alfy=tw_cell.alfy[0],
              dx=tw_cell.dx[0],     dpx=tw_cell.dpx[0])

    knob_names, mk_list = _beta_bound_setup(arc1R, FDF=FDF)
    vary = [xt.Vary(k, step=1e-5, limits=_knob_limits(k))
            for k in knob_names if k not in FIXED_KNOBS]

    closure = [
        xt.TargetSet(dx=0, dpx=0, at=xt.END, tol=1e-9),
        xt.TargetSet(alfx=0, alfy=0, at=xt.END, tol=1e-9),
    ] + _ds_waist_targets(arc1R)        # horizontal DS waist, if markers exist
    extra = []
    if match_end_betas:
        cell_tr_opt.targets[0].value = betxS
        cell_tr_opt.targets[1].value = betyS
        cell_tr_opt.run_jacobian(10)
        tw_tr = cell_tr.twiss(method='4d')
        extra.append(xt.TargetSet(betx=tw_tr.betx[0], bety=tw_tr.bety[0],
                                  at=xt.END, tol=1e-3, weight=0.1))
    if qx_target is not None and qy_target is not None:
        extra.append(xt.TargetSet(mux=qx_target / n_periods,
                                  muy=qy_target / n_periods, at=xt.END, tol=1e-8))

    def _solve(opt, n=60):
        try:
            opt.solve(n_steps=n)
        except Exception:
            try:
                opt.step(n, broyden=True, rcond=1e-4)
            except Exception:
                pass

    def _closed():
        try:
            t = arc1R.twiss(method='4d', **bc)
        except Exception:
            return False, None
        res = [abs(t.dx[-1]), abs(t.dpx[-1]), abs(t.alfx[-1]), abs(t.alfy[-1])]
        res += [abs(t['alfx', n]) for n in arc1R.element_names if n.startswith(DS_WAIST_PREFIX)]
        ok = max(res) < 1e-6
        return ok, t

    knob_vals = lambda: {k: pdr_vars[k]._value for k in knob_names if k not in FIXED_KNOBS}
    pdr_vars = arc1R.env.vars if hasattr(arc1R, 'env') else arc1R.vars

    # Stage 1: close the half-period (dispersion-free, mirror-symmetric
    # straight). 4 hard targets vs 8-9 knobs, no beta pull -> converges.
    opt = arc1R.match(method='4d', solve=False, verbose=False, **bc,
                      vary=vary, targets=closure + extra, assert_within_tol=False)
    _solve(opt)
    ok, tw = _closed()
    if not ok:
        print('matchingBeta: WARNING closure not reached in stage 1')
    last_good = knob_vals()

    # Stage 2: lower the beta bound in steps from where the optics sit now
    # to BETA_MAX, re-closing at each step; keep the last step that closes.
    BETA_MAX = max(betay_DS_target or 0, BETA_BOUND_FLOOR)
    t_now = tw if tw is not None else arc1R.twiss(method='4d', **bc)
    idx = [i for i, n in enumerate(t_now.name) if n in mk_list] + [len(t_now.name) - 1]  # + END
    b0 = max(t_now.betx[idx].max(), t_now.bety[idx].max())
    if b0 > BETA_MAX:
        for bound in np.geomspace(b0, BETA_MAX, 8)[1:]:
            bounds = []
            for mk in mk_list:
                bounds += [xt.Target('betx', xt.LessThan(bound), at=mk, weight=beta_weight),
                           xt.Target('bety', xt.LessThan(bound), at=mk, weight=beta_weight)]
            opt = arc1R.match(method='4d', solve=False, verbose=False, **bc, vary=vary,
                              targets=closure + extra + bounds, assert_within_tol=False)
            _solve(opt)
            ok, _ = _closed()
            if not ok:
                for k, v in last_good.items():
                    pdr_vars[k] = v
                print(f'matchingBeta: beta bound stopped at the previous step '
                      f'(closure lost at {bound:.1f} m)')
                break
            last_good = knob_vals()

    tw = arc1R.twiss(method='4d', **bc)
    print(f'matchingBeta   -> END dx={tw.dx[-1]:+.2e} alfx={tw.alfx[-1]:+.2e} '
          f'alfy={tw.alfy[-1]:+.2e}; sextant mux={tw.mux[-1]:.4f} muy={tw.muy[-1]:.4f} '
          f'(ring {n_periods * tw.mux[-1]:.3f}, {n_periods * tw.muy[-1]:.3f}); '
          f'betx<={tw.betx.max():.2f} bety<={tw.bety.max():.2f}')
    if MakePlot:
        tw.plot()
    return opt

DS_WAIST_PREFIX = 'DSW_'


def insert_DS_waist_markers(pdr, *lines):
    """Insert a marker DSW_<sector> at the centre of the long straight drift
    (DrTripl, between the doublet and the outer triplet quad -- where the RF
    cavity sits) in every sector of the given lines. When present,
    matchingBeta and matchingWP require a horizontal waist there (alfx = 0)
    -- see _ds_waist_targets.

    This is where the D-F-D lattice has its natural horizontal waist
    (betx ~1.4 m). In the F-D-F straight the x-waist otherwise moves to the
    straight centre, so it has to be asked for.

    Pass the matching line (arc1R) and period, NOT the ring: the RF cavity is
    inserted into this drift of the ring afterwards and needs it unsplit.
    Positions are taken from the line itself, so it works for both F-D-F
    (QFDoub next to the drift) and D-F-D (QDDoub next to it).
    """
    def insert_into_line(line, label):
        tt = line.get_table()
        names = list(tt.name)
        L = lambda n: line.element_dict[n].length
        plan = []
        for trip in (n for n in names if n.startswith(('QFTrip_', 'QDTrip_'))):
            sector = trip.split('_', 1)[1][:-1]           # QFTrip_1R1 -> 1R
            new = DS_WAIST_PREFIX + sector
            if new in names:
                continue
            doub = [n for n in (f'QFDoub_{sector}', f'QDDoub_{sector}') if n in names]
            if not doub:
                continue
            s_t0, s_t1 = tt['s', trip], tt['s', trip] + L(trip)
            # doublet quad nearest the triplet quad, then the gap between them
            d = min(doub, key=lambda n: abs(tt['s', n] - s_t0))
            s_d0, s_d1 = tt['s', d], tt['s', d] + L(d)
            mid = (s_d1 + s_t0) / 2 if s_d0 < s_t0 else (s_t1 + s_d0) / 2
            plan.append((new, mid))
        for new, mid in sorted(plan, key=lambda x: -x[1]):
            line.insert(pdr.new(new, xt.Marker), at=mid)
        print(f'insert_DS_waist_markers [{label}]: inserted {len(plan)} markers')

    for i, ln in enumerate(lines):
        insert_into_line(ln, f'line[{i}]')
    return pdr


def _ds_waist_targets(arc1R, tol=1e-9):
    """alfx = 0 at the DS waist marker(s) of the matched sextant, if any."""
    return [xt.Target('alfx', 0, at=n, tol=tol)
            for n in arc1R.element_names if n.startswith(DS_WAIST_PREFIX)]


def insert_DS_betay_quads(pdr, ring, period, *extra_lines,
                          l_qy=None, frac=0.9):
    """
    Insert one extra defocusing quad into the DrDSL drift of every DS region,
    identical to the existing doublet quad QDDoub: same length ('l_quad') and
    same shared strength knob 'kQDDoub'. Because it references kQDDoub it is powered 
    automatically together with the existing doublet quad; no new knob or match target is needed.

    """
    q_length = l_qy if l_qy is not None else 'l_quad'

    def insert_into_line(line, line_label):
        names = line.element_names

        def drift_length(nm):
            try:
                el = line.element_dict[nm]
                return el.length if el.__class__.__name__ == 'Drift' else None
            except Exception:
                return None

        LONG = 1.0  # DrDSL is 2.25 m; anything >1 m is "the long drift"

        plan, skipped = [], []
        for qfds in sorted(n for n in names if n.startswith('QFDS_')):
            sector = qfds[len('QFDS_'):]
            new_name = 'QDDoubDS_' + sector
            if new_name in names:
                continue
            idx = names.index(qfds)
            qfds_half = line.element_dict[qfds].length / 2.0

            prev_nm = names[idx - 1] if idx - 1 >= 0 else None
            next_nm = names[idx + 1] if idx + 1 < len(names) else None
            prev_len = drift_length(prev_nm)
            next_len = drift_length(next_nm)

            cand = []
            if next_len is not None and next_len > LONG:
                cand.append(('+', next_len))
            if prev_len is not None and prev_len > LONG:
                cand.append(('-', prev_len))

            if not cand:
                skipped.append((qfds, prev_nm, prev_len, next_nm, next_len))
                continue

            sign, dlen = cand[0]  # if both long, prefers downstream (symmetric)
            off = qfds_half + frac * dlen
            plan.append((new_name, qfds, f'{sign}{off}', sign, round(off, 3)))

        if 'kQDDoubDS' not in pdr.vars:
            # Own knob, seeded from the doublet quad it used to share.
            # Sharing kQDDoub left the DS with no independent handle on bety,
            # which is why betay_DS_target could never be implemented.
            pdr.vars['kQDDoubDS'] = pdr['kQDDoub']
        for new_name, qfds, at_expr, sign, off in plan:
            line.insert(
                pdr.new(new_name, xt.Quadrupole, length=q_length, k1='kQDDoubDS'),
                at=at_expr, from_=qfds, from_anchor='center')

        print(f"insert_DS_betay_quads [{line_label}]: inserted {len(plan)} quads "
              f"(copies of QDDoub, on kQDDoub)")
        for new_name, qfds, at_expr, sign, off in plan:
            print(f"   {new_name}: at {sign}{off} m from {qfds} centre")
        if skipped:
            print(f"   WARNING [{line_label}]: {len(skipped)} QFDS had no adjacent "
                  f"long drift -- not inserted:")
            for row in skipped:
                print(f"     {row}")

    insert_into_line(ring, 'ring')
    insert_into_line(period, 'period')
    for i, ln in enumerate(extra_lines):
        insert_into_line(ln, f'extra[{i}]')
    return pdr

def _check_triplet_basin(pdr, kf_max=3.0):
    """
    The F-D-F straight has two self-consistent solutions and the matcher can
    land in either. kQFtr is the tell:

      kQFtr ~ 1.8   weak triplet   -> QDTripC holds bety, bety <= ~10 m
      kQFtr ~ 5.2   strong triplet -> QDTripC cannot contain bety, which then
                                      peaks in DrTrips at the straight centre
                                      (14-23 m) while betx drops to ~5 m

    Pass triplet_seed=(1.8, -3.2) to force the weak basin.
    """
    kf, kd = pdr['kQFtr'], pdr['kQDtr']
    if abs(kf) > kf_max:
        print(f'  WARNING: kQFtr={kf:+.3f} (kQDtr={kd:+.3f}) is the '
              f'STRONG-triplet basin. bety will peak at the straight centre. '
              f'Rebuild with triplet_seed=(1.8, -3.2).')
    else:
        print(f'  triplet basin OK: kQFtr={kf:+.3f} kQDtr={kd:+.3f}')
    return kf


def _beta_bound_setup(arc1R, FDF=True):
    """
    Return (knob_names, marker_names) for the bounded sextant match.

    Both are derived from the line rather than hard-coded, so they follow
    N_cells_S and the triplet topology instead of silently going stale.
    """
    names = list(arc1R.element_names)
    knobs = ['kQFarcM', 'kQDarcM', 'kQFDS', 'kQDDS',
             'kQFDoub', 'kQDDoub']
    if any(n.startswith('QDDoubDS_') for n in names):
        knobs.append('kQDDoubDS')
    pref = ('QFA_M', 'QDA_M', 'QFDS_', 'QDDS_', 'QDDoubDS_',
            'QFDoub_', 'QDDoub_', 'QFTrip_' if FDF else 'QDTrip_')
    seen, mks = set(), []
    for n in names:
        if n.startswith(pref) and n not in seen:
            seen.add(n); mks.append(n)
    # The sextant END is the straight centre, inside the centre triplet quad
    # (QDTripC / QFTripC half-quad). beta peaks there in y for F-D-F (44 m
    # when left unbounded), so it must be bounded too.
    mks.append(xt.END)
    return knobs, mks


def _seed_arc_knobs(pdr, l_cell=None, l_quad=None, mu_cell=0.25):
    """
    Thin-lens FODO estimate for the arc quad strengths.
 
        sin(mu/2) = L_half / (2 f)   ->   f = L_half / (2 sin(mu/2))
        k1 = 1 / (f * l_quad)
 
    Sets kQFarc/kQDarc and seeds the matching quads to the same magnitude.
    Cheap, closed-form, and always lands inside the FODO stability region.
    """
    import numpy as np
    l_cell = pdr['l_cell'] if l_cell is None else l_cell
    l_quad = pdr['l_quad'] if l_quad is None else l_quad
 
    mu     = 2*np.pi*mu_cell
    L_half = l_cell/2.
    f      = L_half/(2*np.sin(mu/2.))
    k1     = 1./(f*l_quad)
 
    pdr.vars({'kQFarc':  k1,       'kQDarc':  -k1,
              'kQFarcM': k1*0.98,  'kQDarcM': -k1*0.94})
    print(f'_seed_arc_knobs: mu_cell={mu_cell:.4f} -> f={f:.4f} m, '
          f'kQFarc={k1:.4f}, kQDarc={-k1:.4f}')
    return k1
 
 
def _seed_triplet_knobs(pdr, cell_tr,
                        kf_lo=0.5, kf_hi=10.0, nf=40,
                        ratio_lo=1.2, ratio_hi=3.0, nr=25,
                        betx_target=2.5, bety_target=2.5, verbose=True):
    """
    Find a stable (kQFtr, kQDtr) starting point for the triplet cell.

    cell_tr is  QF/2 - QD - drift(l_tripl) - QD - QF/2, so the integrated
    strengths are  l_quad*kQFtr  focusing  against  2*l_quad*kQDtr
    defocusing. The cell is only stable near  kQFtr ~ 2*|kQDtr| ; on the
    antisymmetric line kQDtr = -kQFtr the net is always defocusing in x
    and no stable point exists. So the scan runs over kQFtr and the ratio
    r = kQFtr / |kQDtr| rather than over kQFtr alone.

    Picks the point whose beta at Mkr_cell_tr is closest to the targets.
    Needed whenever l_tripl changes, since the old strengths then fall
    outside the stability region ('Invalid n1' / 'Invalid n2').
    """
    import numpy as np

    kf_save, kd_save = pdr['kQFtr'], pdr['kQDtr']

    best, n_stable = None, 0
    kf_stable, r_stable = [], []

    for kf in np.linspace(kf_lo, kf_hi, nf):
        for r in np.linspace(ratio_lo, ratio_hi, nr):
            kd = -kf/r
            pdr.vars['kQFtr'] = kf
            pdr.vars['kQDtr'] = kd
            try:
                tw = cell_tr.twiss(method='4d')
                bx = tw['betx', 'Mkr_cell_tr']
                by = tw['bety', 'Mkr_cell_tr']
                if np.isnan(bx) or np.isnan(by):
                    continue
                n_stable += 1
                kf_stable.append(kf); r_stable.append(r)
                score = abs(bx - betx_target) + abs(by - bety_target)
                if best is None or score < best[0]:
                    best = (score, kf, kd, bx, by)
            except Exception:
                continue

    if best is None:
        pdr.vars['kQFtr'], pdr.vars['kQDtr'] = kf_save, kd_save
        raise RuntimeError(
            f'_seed_triplet_knobs: no stable point found for '
            f'l_tripl={pdr["l_tripl"]} over kQFtr in [{kf_lo}, {kf_hi}], '
            f'ratio in [{ratio_lo}, {ratio_hi}]. Widen the scan.')

    _, kf, kd, bx, by = best
    pdr.vars['kQFtr'], pdr.vars['kQDtr'] = kf, kd

    if verbose:
        print(f'_seed_triplet_knobs: {n_stable} stable points found')
        print(f'  kQFtr range {min(kf_stable):.2f}..{max(kf_stable):.2f}, '
              f'ratio range {min(r_stable):.2f}..{max(r_stable):.2f}')
        print(f'  chose kQFtr={kf:.4f}, kQDtr={kd:.4f} '
              f'(ratio {kf/abs(kd):.2f}) -> '
              f'betx={bx:.3f} m, bety={by:.3f} m at Mkr_cell_tr')
    return kf, kd
 
 
def _seed_transition_knobs(pdr, scale=1.0):
    """
    Seed the DS / doublet knobs from the arc strength. These only need to be
    in the right ballpark and sign; the matching does the rest. Seeding from
    kQFarc keeps them consistent when l_cell or the phase advance changes.
    """
    k = pdr['kQFarc']
    pdr.vars({'kQFDS':   k*0.95*scale, 'kQDDS':   -k*0.78*scale,
              'kQFDoub': k*1.33*scale, 'kQDDoub': -k*0.85*scale})
    print(f'_seed_transition_knobs: from kQFarc={k:.4f} -> '
          f'kQFDS={pdr["kQFDS"]:.4f}, kQDDS={pdr["kQDDS"]:.4f}, '
          f'kQFDoub={pdr["kQFDoub"]:.4f}, kQDDoub={pdr["kQDDoub"]:.4f}')

def _seed_triplet_knobs_FDF(pdr, cell_tr,
                            kf_lo=0.3, kf_hi=6.0, nf=40,
                            ratio_lo=1.2, ratio_hi=3.0, nr=25,
                            betx_target=2.5, bety_target=2.5, verbose=True):
    """
    Seeding scan for the F-D-F triplet cell.
 
    cell_tr is  QD/2 - QF - drift(l_tripl) - QF - QD/2, so the integrated
    strengths are  2*l_quad*kQFtr  focusing  against  l_quad*|kQDtr|
    defocusing. Stability requires roughly  |kQDtr| ~ 2*kQFtr, so the scan
    runs over kQFtr and the ratio  r = |kQDtr| / kQFtr.
 
    (This is the inverse of the D-F-D case, where the ratio was
    kQFtr / |kQDtr|. Using the wrong one finds no stable points.)
    """
    import numpy as np
 
    kf_save, kd_save = pdr['kQFtr'], pdr['kQDtr']
 
    best, n_stable = None, 0
    kf_stable, r_stable = [], []
 
    for kf in np.linspace(kf_lo, kf_hi, nf):
        for r in np.linspace(ratio_lo, ratio_hi, nr):
            kd = -kf*r                      # |kQDtr| = r * kQFtr
            pdr.vars['kQFtr'] = kf
            pdr.vars['kQDtr'] = kd
            try:
                tw = cell_tr.twiss(method='4d')
                bx = tw['betx', 'Mkr_cell_tr']
                by = tw['bety', 'Mkr_cell_tr']
                if np.isnan(bx) or np.isnan(by):
                    continue
                n_stable += 1
                kf_stable.append(kf); r_stable.append(r)
                score = abs(bx - betx_target) + abs(by - bety_target)
                if best is None or score < best[0]:
                    best = (score, kf, kd, bx, by)
            except Exception:
                continue
 
    if best is None:
        pdr.vars['kQFtr'], pdr.vars['kQDtr'] = kf_save, kd_save
        raise RuntimeError(
            f'_seed_triplet_knobs_FDF: no stable point for '
            f'l_tripl={pdr["l_tripl"]} over kQFtr in [{kf_lo}, {kf_hi}], '
            f'ratio |kQDtr|/kQFtr in [{ratio_lo}, {ratio_hi}]. Widen the scan.')
 
    _, kf, kd, bx, by = best
    pdr.vars['kQFtr'], pdr.vars['kQDtr'] = kf, kd
 
    if verbose:
        print(f'_seed_triplet_knobs_FDF: {n_stable} stable points')
        print(f'  kQFtr range {min(kf_stable):.2f}..{max(kf_stable):.2f}, '
              f'ratio range {min(r_stable):.2f}..{max(r_stable):.2f}')
        print(f'  chose kQFtr={kf:.4f}, kQDtr={kd:.4f} '
              f'(|kQDtr|/kQFtr = {abs(kd)/kf:.2f}) -> '
              f'betx={bx:.3f} m, bety={by:.3f} m at Mkr_cell_tr')
    return kf, kd
 
 
 
ARC_KNOBS = ('kQFarc', 'kQDarc', 'kQFarcM', 'kQDarcM')
 
 
def make_sext_knobs(pdr, ring, prefixes=None, verbose=True):
    """
    Put every sextupole in `ring` onto a per-family knob so xt.Vary can reach it.
 
    sextupole_configs sets k2 directly on the elements, so there is no knob to
    vary -- that is what 'Variable `kSF_n` not found' means. This groups the
    sextupoles, creates one knob per family seeded from the current k2, and
    re-points the elements at it. Returns the knob names.
 
    prefixes=None groups by name prefix (XF1, XF2, XD1, XD2, ...).
    prefixes=('XF','XD') lumps them into two families instead.
    """
    if prefixes is None:
        fams = {}
        for n in set(ring.element_names):
            if pdr.element_dict[n].__class__.__name__ != 'Sextupole':
                continue
            m = re.match(r'^([A-Za-z]+\d*)', n)
            fams.setdefault(m.group(1) if m else n, []).append(n)
    else:
        fams = {p: [n for n in set(ring.element_names)
                    if n.startswith(p)
                    and pdr.element_dict[n].__class__.__name__ == 'Sextupole']
                for p in prefixes}
 
    knobs = []
    for fam, names in sorted(fams.items()):
        if not names:
            continue
        knob = f'k_{fam}'
        if knob not in pdr.vars:
            pdr.vars[knob] = float(pdr.element_dict[sorted(names)[0]].k2)
        for n in names:
            pdr.element_refs[n].k2 = pdr.vars[knob]
        knobs.append(knob)
        if verbose:
            print(f'  {knob:10s} <- {len(names):3d} sextupoles, '
                  f'k2 = {pdr.varval[knob]:+9.3f}')
    return knobs
 
 
def retune_qx(ring, sext_knobs, qx_target, qy_target, method='6d',
              step=0.05, arc_knobs=ARC_KNOBS, verbose=True):
    """
    Walk Qx to qx_target in small steps, re-correcting chromaticity each step.
 
    One big jump does not work: in 6d the tune match fails outright with
    'Could not find point within tolerance', and in 4d it converges to a worse
    solution (betx 14.66 m vs 10.48 m for the same final Qx). Stepping also
    keeps the sextupoles tracking the changing optics instead of being corrected
    once at the end.
    """
    qx0 = ring.twiss(method=method).qx
    n = max(1, int(np.ceil(abs(qx_target - qx0) / step)))
    reached = qx0
    for qx in np.linspace(qx0, qx_target, n + 1)[1:]:
        try:
            ring.match(method=method,
                       vary=[xt.Vary(k, step=1e-5) for k in arc_knobs],
                       targets=[xt.Target('qx', float(qx), tol=1e-6),
                                xt.Target('qy', qy_target, tol=1e-6)])
            ring.match(method=method,
                       vary=[xt.Vary(k, step=1e-3) for k in sext_knobs],
                       targets=[xt.Target('dqx', 0., tol=1e-3),
                                xt.Target('dqy', 0., tol=1e-3)])
            reached = float(qx)
        except Exception as e:
            if verbose:
                print(f'  stopped at Qx={reached:.4f}: '
                      f'{type(e).__name__}: {str(e)[:50]}')
            break
        if verbose:
            tw = ring.twiss(method=method)
            print(f'  Qx={tw.qx:7.4f} Qy={tw.qy:7.4f} '
                  f'dqx={tw.dqx:+6.3f} dqy={tw.dqy:+6.3f} '
                  f'betx<={tw.betx.max():6.2f} bety<={tw.bety.max():6.2f}')
    return reached
 
 
def delta_range(ring, dmax=0.09, step=0.0025, method='4d'):
    """Largest +/- delta for which the periodic solution still exists."""
    hi = lo = 0.
    for d in np.arange(step, dmax, step):
        try:
            ring.twiss(method=method, delta0=float(d)); hi = d
        except Exception:
            break
    for d in np.arange(-step, -dmax, -step):
        try:
            ring.twiss(method=method, delta0=float(d)); lo = d
        except Exception:
            break
    return lo, hi

# ---------------------------------------------------
# Fringe Field Expansion - taken from silke's github
# ---------------------------------------------------


def _bshape4(L, bb0, bp0, bbL, bpL, bint):
    """
    Coefficients (lowest order first, in powers of s) of the 4th-order
    polynomial on [0, L] with value bb0 / slope bp0 at 0, value bbL / slope
    bpL at L, and integral bint over [0, L].
    """
    t1 = np.array([1, 0, -18,  32, -15])   # p(0)  = 1
    t2 = np.array([0, 1, -9/2,  6, -5/2])  # p'(0) = 1
    t3 = np.array([0, 0, -12,  28, -15])   # p(1)  = 1
    t4 = np.array([0, 0, 3/2,  -4,  5/2])  # p'(1) = 1
    t5 = np.array([0, 0, 30,  -60,  30])   # int_0^1 p = 1
    return ((bb0*t1 + L*bp0*t2 + bbL*t3 + L*bpL*t4 + bint/L*t5)
            * L**np.arange(0, -5, -1, dtype=float))


def _field_expansion_bend(env, name, length, angle, gap=0.05, edge_angle=None,
                          fringe_inside=True, num_integration_steps=5):
    """
    Rectangular dipole with soft fringe fields, as a line of 5
    xt.BFieldExpansion pieces (+ 2 thin edges) called `name`.

        [in-fringe 1st half | in-fringe 2nd half | body | out-fringe 1st half | out-fringe 2nd half]
          straight            curved               curved  curved                straight

    Each fringe is a 4th-order ramp over 3*gap, with zero slope at both ends
    and half the full field integrated over it, so the equivalent hard edge
    is at the centre of the ramp -- where the curved reference starts and
    ends. Total bending angle = `angle` exactly.

    fringe_inside=True  (default) the whole field fits inside `length`, so the
        element is a drop-in replacement with the same footprint (layout and
        circumference unchanged). The equivalent hard-edge length is
        length - 3*gap and the peak field is higher by length/(length - 3*gap).
    fringe_inside=False hard-edge length = `length` (same peak field as
        xt.Bend); the element is 3*gap longer, so shorten the drift on each
        side by 1.5*gap or the ring gets longer.

    edge_angle: pole-face rotation at each end (rectangular magnet:
        angle/2, which is what the hard-edge 'Bend' used). Added as linear
        xt.DipoleEdge kicks at the hard-edge positions; without them the
        magnet focuses like a sector bend (horizontal instead of vertical).

    NB: BFieldExpansion does not radiate yet (xtrack 0.115), so these bends
    give no synchrotron radiation / damping.
    """
    fringe = 3 * gap
    l_mag = length - fringe if fringe_inside else length   # hard-edge-equivalent length
    body = l_mag - fringe
    if body < 0:
        raise ValueError(f'{name}: fringe 3*gap = {fringe} m too long for a '
                         f'{length} m magnet; use a smaller gap')
    h = angle / l_mag                     # curvature = k0 of the body
    b_in = _bshape4(fringe, 0, 0, h, 0, fringe * h / 2)
    b_out = _bshape4(fringe, h, 0, 0, 0, fringe * h / 2)

    kw = dict(num_integration_steps=num_integration_steps)
    pieces = [
        (f'{name}_fe0', xt.BFieldExpansion(length=fringe/2, knc=[b_in], **kw)),
        (f'{name}_fe1', xt.BFieldExpansion(length=fringe/2, knc=[b_in], h=h,
                                           s_start=fringe/2, **kw)),
        (f'{name}_fe2', xt.BFieldExpansion(length=body, knc=[[h]], h=h, **kw)),
        (f'{name}_fe3', xt.BFieldExpansion(length=fringe/2, knc=[b_out], h=h, **kw)),
        (f'{name}_fe4', xt.BFieldExpansion(length=fringe/2, knc=[b_out],
                                           s_start=fringe/2, **kw)),
    ]
    for n, el in pieces:
        env.elements[n] = el
    names = [n for n, _ in pieces]

    if edge_angle:
        env.elements[f'{name}_edge_in'] = xt.DipoleEdge(
            k=h, e1=edge_angle, side='entry', model='linear')
        env.elements[f'{name}_edge_out'] = xt.DipoleEdge(
            k=h, e1=edge_angle, side='exit', model='linear')
        names = (names[:1] + [f'{name}_edge_in'] + names[1:4]
                 + [f'{name}_edge_out'] + names[4:])

    return env.new_line(name=name, components=names)


def plot_bend_profile(line):
    """Field profile k0(s) of a bend built by _field_expansion_bend."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    s0 = 0.0
    for el in line.elements:
        if isinstance(el, xt.BFieldExpansion):
            c = np.asarray(el.knc)[0]
            s = np.linspace(0, el.length, 50)
            ax.plot(s0 + s, np.polyval(c[::-1], s + el.s_start))
            s0 += el.length
    ax.set_xlabel('s [m]'); ax.set_ylabel('k0 [1/m]')
    return fig


# ----------------
# Shared Builders
# ----------------

def _make_env(fringe_fields):
    pdr = xt.Environment()
    pdr.particle_ref = xt.Particles(kinetic_energy0=2.86e9, mass0=xt.ELECTRON_MASS_EV)
    return pdr, fringe_fields, 'full' if fringe_fields else 'linear'


def _make_base_elements(pdr, quad_edge, bend_edge, field_expansion_bends=True,
                        gap=0.05, fringe_inside=True):
    # Arc dipoles. pdr.new(f'Bend1_...', 'Bend') in makesextant clones this
    # line for every magnet (elements become e.g. 'Bend_fe2.Bend1_1R8').
    # Field values are numbers taken from l_bend / hBarc NOW, so set those
    # vars before calling this.
    dipoles = {'Bend': 'l_bend', 'BendDS': 'l_bendDS'}

    for name, l_var in dipoles.items():
        if field_expansion_bends:
            l_b = pdr[l_var]
            ang = pdr['hBarc'] * l_b
            _field_expansion_bend(pdr, name, length=l_b, angle=ang,
                                  gap=gap, edge_angle=ang / 2,
                                  fringe_inside=fringe_inside)
        else:
            pdr.new(name, xt.Bend, length=l_var, angle=f'hBarc*{l_var}',
                    k0_from_h=True,
                    edge_entry_angle=f'hBarc*{l_var}/2',
                    edge_exit_angle=f'hBarc*{l_var}/2',
                    edge_entry_model=bend_edge, edge_exit_model=bend_edge)

    pdr.new('QFarc',  xt.Quadrupole, length='l_quad',    k1='kQFarc',
            edge_entry_active=quad_edge, edge_exit_active=quad_edge)
    pdr.new('QFarcH', xt.Quadrupole, length='l_quad/2.', k1='kQFarc',
            edge_entry_active=quad_edge, edge_exit_active=quad_edge)
    pdr.new('QDarc',  xt.Quadrupole, length='l_quad',    k1='kQDarc',
            edge_entry_active=quad_edge, edge_exit_active=quad_edge)
    pdr.new('QFarcM', xt.Quadrupole, length='l_quad',    k1='kQFarcM',
            edge_entry_active=quad_edge, edge_exit_active=quad_edge)
    pdr.new('QDarcM', xt.Quadrupole, length='l_quad',    k1='kQDarcM',
            edge_entry_active=quad_edge, edge_exit_active=quad_edge)
    pdr.new('QFDS',   xt.Quadrupole, length='l_quad',    k1='kQFDS',
            edge_entry_active=quad_edge, edge_exit_active=quad_edge)
    pdr.new('QDDS',   xt.Quadrupole, length='l_quad',    k1='kQDDS',
            edge_entry_active=quad_edge, edge_exit_active=quad_edge)
    pdr.new('QFDoub', xt.Quadrupole, length='l_quad',    k1='kQFDoub',
            edge_entry_active=quad_edge, edge_exit_active=quad_edge)
    pdr.new('QDDoub', xt.Quadrupole, length='l_quad',    k1='kQDDoub',
            edge_entry_active=quad_edge, edge_exit_active=quad_edge)
    pdr.new('QFtr',   xt.Quadrupole, length='l_quad',    k1='kQFtr',
            edge_entry_active=quad_edge, edge_exit_active=quad_edge)
    pdr.new('QFtrH',  xt.Quadrupole, length='l_quad/2.', k1='kQFtr',
            edge_entry_active=quad_edge, edge_exit_active=quad_edge)
    pdr.new('QDtr',   xt.Quadrupole, length='l_quad',    k1='kQDtr',
            edge_entry_active=quad_edge, edge_exit_active=quad_edge)
    pdr.new('QDtrH',  xt.Quadrupole, length='l_quad/2',    k1='kQDtr',
            edge_entry_active=quad_edge, edge_exit_active=quad_edge)
    pdr.new('Drarc',  xt.Drift, length='l_drift')
    pdr.new('DrarcS', xt.Drift, length='l_drift + dl_drift')
    pdr.new('DrDSL',  xt.Drift, length='2*l_drift + l_bend + dl_noben')
    pdr.new('DrTrans',xt.Drift, length='l_trans')
    pdr.new('DrDoub', xt.Drift, length='l_doub')
    pdr.new('DrTripl',xt.Drift, length='l_tripl')
    pdr.new('DrTrips',xt.Drift, length='l_trips')


def _make_reference_cells(pdr, FDF=True):
    cell_arc = pdr.new_line(components=[
        pdr.new('QF_cell_arcH1',   'QFarcH'), pdr.place('Drarc'),
        pdr.new('Bend1_cell_arcH', 'Bend'),   pdr.place('Drarc'),
        pdr.new('QD_cell_arcH',    'QDarc'),  pdr.place('Drarc'),
        pdr.new('Bend2_cell_arcH', 'Bend'),   pdr.place('Drarc'),
        pdr.new('QF_cell_arcH2',   'QFarcH'),
    ])
    if FDF==False:
        cell_tr = pdr.new_line(components=[
            pdr.new('QF_cell_trH1', 'QFtrH',
                    at='0*l_trips + 0.25*l_quad + 0.0*l_tripl'),
            pdr.new('QD_cell_tr1',  'QDtr',
                    at='1*l_trips + 1.00*l_quad + 0.0*l_tripl'),
            pdr.new('Mkr_cell_tr',  xt.Marker,
                    at='1*l_trips + 1.50*l_quad + 0.5*l_tripl'),
            pdr.new('QD_cell_tr2',  'QDtr',
                    at='1*l_trips + 2.00*l_quad + 1.0*l_tripl'),
            pdr.new('QF_cell_trH2', 'QFtrH',
                    at='2*l_trips + 2.75*l_quad + 1.0*l_tripl'),
        ])
    else:
        cell_tr=pdr.new_line(components=[
        pdr.new('QD_cell_trH1', 'QDtrH',
                at='0*l_trips + 0.25*l_quad + 0.0*l_tripl'),
        pdr.new('QF_cell_tr1',  'QFtr',
                at='1*l_trips + 1.00*l_quad + 0.0*l_tripl'),
        pdr.new('Mkr_cell_tr',  xt.Marker,
                at='1*l_trips + 1.50*l_quad + 0.5*l_tripl'),
        pdr.new('QF_cell_tr2',  'QFtr',
                at='1*l_trips + 2.00*l_quad + 1.0*l_tripl'),
        pdr.new('QD_cell_trH2', 'QDtrH',
                at='2*l_trips + 2.75*l_quad + 1.0*l_tripl'),
    ])
    return cell_arc, cell_tr

def _straight_anchor(line, drift_base='DrTripl'):
    """
    Return the quadrupole immediately upstream of the first <drift_base>* drift.

    This is QDDoub in a D-F-D straight and QFDoub in an F-D-F straight.
    Resolving it from the lattice instead of hard-coding it is what stops a
    change of triplet topology from silently misplacing the symmetry marker
    and the RF cavity.
    """
    names = list(line.element_names)
    d = next((n for n in names if n.startswith(drift_base)), None)
    if d is None:
        raise RuntimeError(f'_straight_anchor: no {drift_base}* drift in line')
    anchor = names[names.index(d) - 1]
    cls = line.element_dict[anchor].__class__.__name__
    if cls != 'Quadrupole':
        raise RuntimeError(
            f'_straight_anchor: {anchor} (element before {d}) is a {cls}, '
            f'not a Quadrupole. The at=(l_tripl+l_quad)/2 offset assumes the '
            f'anchor quad is adjacent to the long drift.')
    return anchor


def _straight_drift_after(line, anchor, drift_base='DrTripl'):
    """
    Verify <anchor> really is adjacent to the long straight drift and return
    (drift_name, drift_length). at='(l_tripl+l_quad)/2' lands on the drift
    centre only if this holds -- if it does not, the placement is silently
    wrong by the length of whatever sits in between.
    """
    names = list(line.element_names)
    if anchor not in names:
        raise RuntimeError(f'_straight_drift_after: {anchor} not in line')
    nxt = names[names.index(anchor) + 1]
    if not nxt.startswith(drift_base):
        raise RuntimeError(
            f'_straight_drift_after: element after {anchor} is {nxt}, not a '
            f'{drift_base}* drift. at=(l_tripl+l_quad)/2 will NOT land on the '
            f'drift centre -- update the anchor for this straight topology.')
    return nxt, line.element_dict[nxt].length


def _insert_rf(pdr, ring, U0, VRF, rf_from=None, rf_length=1.5):
    """
    Insert the RF cavity at the centre of the long straight drift.

    rf_from=None (recommended) resolves the anchor from the lattice. Passing a
    name explicitly is still honoured, but it is now validated: a cavity that
    would not fit, or an anchor that is not adjacent to the long drift, raises
    instead of letting xtrack resolve the overlap by slicing away part of a
    quadrupole.
    """
    if rf_from is None:
        rf_from = _straight_anchor(ring)
    d_name, drift_len = _straight_drift_after(ring, rf_from)

    if rf_length > drift_len:
        raise RuntimeError(
            f'_insert_rf: cavity is {rf_length:.3f} m but straight drift '
            f'{d_name} is only {drift_len:.3f} m. Increase l_tripl.')

    print(f'_insert_rf: RFCav_1 ({rf_length:.3f} m) at centre of {d_name} '
          f'({drift_len:.3f} m), anchored on {rf_from}')

    fRev = 1. / ring.twiss(method='4d').T_rev0
    fRF  = fRev * round(4.e8 / fRev)
    pdr.new('RFCav', xt.Cavity, length=rf_length, frequency=fRF, voltage=VRF,
            lag=(180/np.pi) * (np.pi - np.arcsin(U0/VRF)) - 1.8)
    ring.insert(pdr.new('RFCav_1', 'RFCav'),
                at='(l_tripl+l_quad)/2', from_=rf_from)


def _make_rf_and_finalise(pdr, ring, arc1R, cell_arc, cell_tr,
                           period, U0, VRF, bend_edge,
                           rf_from=None, rf_length=1.5):
    """Insert the RF cavity and then actually finalise. The old version never
    called _finalise, so configure_radiation / configure_bend_model were never
    applied to the ring."""
    _insert_rf(pdr, ring, U0, VRF, rf_from=rf_from, rf_length=rf_length)
    _finalise(pdr, ring, arc1R, cell_arc, cell_tr, period, bend_edge)


def _register_lines(pdr, **lines):
    """
    Register lines in pdr.lines, tolerating names that are already present.

    xtrack's EnvLines.__setitem__ raises ValueError('There is already a line
    with name ...') for any existing key -- even when re-assigning the very
    same object. Several lattice functions call _export_lines before matching,
    so finalising afterwards would otherwise blow up on the second write.
    """
    for nm, ln in lines.items():
        cur = pdr.lines.get(nm, None)
        if cur is ln:
            continue                      # already registered, nothing to do
        if cur is not None:
            del pdr.lines[nm]             # replace a stale object
        pdr.lines[nm] = ln


def _finalise(pdr, ring, arc1R, cell_arc, cell_tr, period, bend_edge):
    """Configure radiation/bend model and register exported lines."""
    ring.configure_radiation(model='mean')
    ring.configure_bend_model(edge=bend_edge)
    _register_lines(pdr, arc1R=arc1R, cell_arc=cell_arc, cell_tr=cell_tr,
                    period=period, ring=ring)

def _sliced(line):
    sl = line.select()
    sl.cut_at_s(np.linspace(.05, line.get_length() - .05,
                             int(line.get_length() / .05 - .5)))
    return sl


def _match_cells_3fold(pdr, cell_arc, cell_tr, mu_cell=0.25):
    cell_arc_opt = cell_arc.match( method='4d', solve=True, verbose=False,
        vary=[
            xt.VaryList(['kQFarc', 'kQDarc', 'kQFarcM'], step=1e-4),    ],
        targets=[
            xt.TargetSet(qx=mu_cell, qy=mu_cell, tol=1.0e-6, tag='end'), # just twice the same 
            xt.TargetSet(mux=mu_cell, muy=mu_cell, at=xt.END, tol=1.0e-6)]  )

    # Triplet cell to chosen betatron functions
    cell_tr_opt = cell_tr.match( method='4d', solve=True,
        vary=[ # use individual Vary commands instead of List for arc cell
            xt.Vary('kQFtr', step=1e-4 ),
            xt.Vary('kQDtr', step=1e-4),    ],
        targets=[
            xt.TargetSet(betx=2.50, bety=2.50, at='Mkr_cell_tr', tol=1.0e-6, tag='betas')] )
    return cell_arc_opt, cell_tr_opt

def _run_standard_matching(ring, cell_arc_opt, cell_arc, cell_tr_opt, cell_tr,
                           arc1R, wp_constants, n_periods=6,
                           betay_DS_target=None, FDF=True, bounded=True,
                           trim_knobs=('kQFarc', 'kQDarc', 'kQFarcM', 'kQDarcM'),
                           trim_warn_frac=0.05, match_WP=True):
    """
    Match the sextant, then trim the ring tunes globally.

    bounded=True (default) matches tune and beta TOGETHER inside matchingWP:
    tunes as hard equalities, beta as inequalities (xt.LessThan) through the
    matching cell, dispersion suppressor and straight. You rarely care what
    beta IS, only that it is not too large, so leaving it an inequality gives
    the solver slack to hit the tunes exactly, instead of beta and tune
    fighting over the same knobs as competing equalities.

    bounded=False falls back to the sequential scheme, and does BETAS FIRST.
    Tune is global and additive, so it can be trimmed at the end with small
    distributed gradient changes that barely move beta; beta is local, set by
    the boundary match, and correcting it needs large changes that move the
    tune a lot. The cheap, nearly beta-neutral step therefore goes last.
    Tune-first (the original order) over-constrains the second stage, does not
    converge, and leaves the residual tune error to the global trim -- which
    then buys tune by spending beta.

    Measured on the F-D-F ring at WP=(15.8, 13.87):
        simultaneous   (bounded=True)   betx<= 5.62  bety<=10.00  |dx|<=0.272
        betas -> tune  (bounded=False)  betx<= 8.01  bety<=11.50  |dx|<=0.278
        tune  -> betas (the old order)  betx<=18.94  bety<=11.96  |dx|<=0.604

    match_WP=False (only with bounded=False) stops after the beta match: no
    tune match and no global trim, so the ring keeps whatever tunes the
    beta-matched optics give. Those tunes are printed and returned in tw.
    """
    qx_t, qy_t = wp_constants

    if not bounded:
        tw_tr = cell_tr.twiss(method='4d')
        mid = len(tw_tr.betx) // 2
        matchingBeta(tw_tr.betx[mid], tw_tr.bety[mid],
                     cell_arc_opt, cell_arc, cell_tr_opt, cell_tr, arc1R,
                     qx_target=None, qy_target=None, n_periods=n_periods,
                     betay_DS_target=betay_DS_target, FDF=FDF)

    if not match_WP:
        if bounded:
            raise ValueError('match_WP=False needs bounded=False (betas matched first)')
        tw = ring.twiss(method='4d')
        print(f'beta match only (WP not matched) -> Qx={tw.qx:9.6f} Qy={tw.qy:9.6f}')
        print(f'                  betx<={tw.betx.max():6.2f} m  '
              f'bety<={tw.bety.max():6.2f} m  |dx|<={np.abs(tw.dx).max():.4f} m')
        return tw

    matchingWP(qx_t, qy_t, cell_arc_opt, cell_arc, arc1R,
               n_periods=n_periods, FDF=FDF, betay_DS_target=betay_DS_target)

    tw_cell = cell_arc.twiss(method='4d')
    tw_sx = arc1R.twiss(method='4d',
                        betx=tw_cell.betx[0], alfx=tw_cell.alfx[0],
                        bety=tw_cell.bety[0], alfy=tw_cell.alfy[0],
                        dx=tw_cell.dx[0],     dpx=tw_cell.dpx[0])
    qx0, qy0 = n_periods * tw_sx.mux[-1], n_periods * tw_sx.muy[-1]
    print(f'sextant match  -> Qx={qx0:9.6f} (miss {qx0 - qx_t:+.4f}), '
          f'Qy={qy0:9.6f} (miss {qy0 - qy_t:+.4f})')

    before = {k: ring.varval[k] for k in trim_knobs}
    ring.match(
        method='4d',
        vary=[xt.Vary(k, step=1e-5) for k in trim_knobs],
        targets=[xt.Target('qx', qx_t, tol=1e-6),
                 xt.Target('qy', qy_t, tol=1e-6)],
    )
    worst, worst_k = 0.0, 'no change'
    for k in trim_knobs:
        rel = abs(ring.varval[k] - before[k]) / max(abs(before[k]), 1e-12)
        if rel > worst:
            worst, worst_k = rel, k

    tw = ring.twiss(method='4d')
    print(f'global trim    -> Qx={tw.qx:9.6f} Qy={tw.qy:9.6f}   '
          f'largest gradient change {worst * 100:5.2f}% ({worst_k})')
    print(f'                  betx<={tw.betx.max():6.2f} m  '
          f'bety<={tw.bety.max():6.2f} m  |dx|<={np.abs(tw.dx).max():.4f} m')
    if worst > trim_warn_frac:
        print(f'  WARNING: the global trim moved {worst_k} by {worst * 100:.1f}%. '
              f'It is meant to be a nudge. A change this large means the '
              f'sextant match did not converge and the trim is buying tune by '
              f'spending beta -- check the matchingWP target status.')
    return tw


def _export_lines(pdr, arc1R, cell_arc, cell_tr, period, ring):
    """Register all lines in pdr.lines — same keys for all lattice functions."""
    _register_lines(pdr, arc1R=arc1R, cell_arc=cell_arc, cell_tr=cell_tr,
                    period=period, ring=ring)


# ---------------------------------------------------------------------------
# Design 1 (more variations as this was created before pipeline optimisation)
# ---------------------------------------------------------------------------

def three_fold_periodicity(fringe_fields=True, matched=True,WP=config.WP_D1,phase_advance=0.25,betay_DS_target=None,swap_DS=False,triplet_seed=(1.8,-3.2),
                           match_WP=True, ds_waist_x=True, field_expansion_bends=True):
    """Three-fold (D1) ring with F-D-F straights, entered through a D-F-D triplet.

    The straight entry is QDDoub - QFDoub - QDDoub2, a symmetric D-F-D on the
    shared kQDDoub/kQFDoub knobs, followed by the long drift and the F-D-F
    triplet. QDDoub2 costs l_quad + l_doub = 0.55 m per sextant, taken out of
    l_tripl (2.5 -> 1.95 m) so the circumference is unchanged.

    Matching order (always): arc and straight cells, then the betas through
    the matching cell, dispersion suppressor and straight, then -- only if
    match_WP=True -- the working point WP and a small global tune trim.

    match_WP : True (default) matches the ring to WP after the betas.
        False keeps the tunes the beta-matched optics give (WP is ignored);
        they are printed so you can see where the lattice sits.
    ds_waist_x : True (default) puts a marker DSW_<sector> at the centre of
        the long straight drift (doublet to triplet, where the RF cavity
        sits) and requires a horizontal waist there (alfx = 0) in both the
        beta and the WP match, like the natural waist of the D-F-D lattice.
    field_expansion_bends : True (default) builds every dipole (Bend and
        BendDS) as a soft-fringe BFieldExpansion magnet; False uses hard-edge
        xt.Bend. Use False for spin tracking or anything needing radiation:
        BFieldExpansion does not radiate and cannot be spin-tracked.
    """
    
    pdr, quad_edge, bend_edge = _make_env(fringe_fields)
    E0 = config.E0; VRF = config.VRF

    '''pdr.vars({
        'l_cell':   3.4,    'l_bend':   0.40,   'l_bendDS': 0.55,
        'dl_noben': 0.25,   'l_quad':   0.30,
        'l_drift':  '(l_cell - 2*l_bend - 2*l_quad)/4.',
        'dl_drift': -0.1,   'dl_trans': 0.00,
        'l_doub':   0.25,   'l_tripl':  2.7,    'l_trips':  0.40,
        'l_sext':   0.20,
    })'''
    pdr.vars({
            'l_cell':   3.4,    'l_bend':   0.40,   'l_bendDS': 0.4,
            'dl_noben': 0.95,   'l_quad':   0.30,
            'l_drift':  '(l_cell - 2*l_bend - 2*l_quad)/4.',
            'dl_drift': -0.0,   'dl_trans': 0.20,
            # l_tripl was 2.5. The new QDDoub2 and its DrDoub add
            # l_quad + l_doub = 0.55 m per sextant, taken back out of the long
            # straight drift so the circumference is unchanged (3.30 m over the
            # six sextants). The RF cavity is 1.5 m, so 0.45 m of clearance is
            # left and _insert_rf's check still passes -- but it is now the
            # binding constraint on l_tripl.
            'l_doub':   0.25,   'l_tripl':  1.95,   'l_trips':  0.40,
            'l_sext':   0.20,'l_trans':  'l_drift+dl_trans',
                    'l_DSL':    '2*l_drift + l_bend + dl_noben',
        })


    pdr.vars({
        'N_cells_S': 8,
        'hBarc': '6.283185307/(6*(2*N_cells_S*l_bend + l_bendDS))',
        'kQFarc':  2.9478,  'kQDarc':  -2.9231,
        'kQFarcM': 2.8846,  'kQDarcM': -2.7567,
        'kQFDS':   2.8042,  'kQDDS':   -2.2858,
        'kQFDoub': 3.9170,  'kQDDoub': -2.5190,
        'kQFtr':   4.4429,  'kQDtr':   -2.4723,
    })
    U0 = (0.88463e-31)*E0**4*(2.*np.pi) / (
        6*(2*pdr['N_cells_S']*pdr['l_bend'] + pdr['l_bendDS']))

    _make_base_elements(pdr, quad_edge, bend_edge,
                        field_expansion_bends=field_expansion_bends)
    cell_arc, cell_tr = _make_reference_cells(pdr)

    _seed_arc_knobs(pdr, mu_cell=phase_advance)
    _seed_transition_knobs(pdr)
    if triplet_seed is None:
        _seed_triplet_knobs_FDF(pdr, cell_tr)
    else:
        pdr.vars['kQFtr'], pdr.vars['kQDtr'] = triplet_seed
        print(f'triplet seed forced: kQFtr={triplet_seed[0]}, '
              f'kQDtr={triplet_seed[1]}')

    def makesextant(name, fall):
            comps = []
            for ind in range(int(pdr['N_cells_S']) - 1):
                comps += [pdr.place('Drarc'), pdr.new(f'Bend1_{name}{ind+1}', 'Bend'),
                          pdr.place('Drarc'), pdr.new(f'QDA_{name}{ind+1}',   'QDarc'),
                          pdr.place('Drarc'), pdr.new(f'Bend2_{name}{ind+1}', 'Bend'),
                          pdr.place('Drarc'), pdr.new(f'QFA_{name}{ind+1}',   'QFarc')]
            n = int(pdr['N_cells_S'])
            comps[-1] = pdr.new(f'QFA_M{name}{n-1}', 'QFarcM')
            comps += [pdr.place('Drarc'),  pdr.new(f'Bend1_{name}{n}', 'Bend'),
                      pdr.place('Drarc'),  pdr.new(f'QDA_M{name}{n}',  'QDarcM'),
                      pdr.place('Drarc'),  pdr.new(f'Bend2_{name}{n}', 'Bend'),
                      # swap_DS=True reverses the suppressor poles so the
                      # DS->straight boundary alternates (QFDS next to QDDoub)
                      # instead of putting QDDS beside QDDoub. MEASURED: worse.
                      # It only moves the like-pole adjacency to the arc side
                      # (QDA_M next to QDDS), where beta_x and D_x are both
                      # near their maxima, so x pays for what y gains:
                      #   swap_DS=False, bound 10 -> betx 5.62 bety 10.00 dx 0.272
                      #   swap_DS=True,  bound  8 -> betx 8.63 bety 10.75 dx 0.314
                      # and the swapped version is stable only at bound 8
                      # (bounds 7/9/10/12 and other working points all fail).
                      # Kept as a switch so it can be re-checked if the
                      # straight geometry changes; leave False otherwise.
                      pdr.place('DrarcS'),
                      pdr.new(f'{"QDDS" if swap_DS else "QFDS"}_{name}',
                              'QDDS' if swap_DS else 'QFDS'),
                      pdr.place('DrDSL'),
                      pdr.new(f'{"QFDS" if swap_DS else "QDDS"}_{name}',
                              'QFDS' if swap_DS else 'QDDS'),
                      pdr.place('Drarc'),  pdr.new(f'BendDS_{name}',   'BendDS'),
                      # ---- D-F-D entry triplet, then the F-D-F straight ----
                      # QDDoub2 turns the old QDDoub-QFDoub doublet into a
                      # D-F-D triplet sitting immediately before DrTripl and
                      # the F-D-F triplet. It is built from the 'QDDoub'
                      # element type, so it shares kQDDoub and is powered with
                      # the other outer quad automatically -- the triplet is
                      # symmetric and the match gains no new variable.
                      pdr.place('DrTrans'),pdr.new(f'QDDoub_{name}',   'QDDoub'),
                      pdr.place('DrDoub'), pdr.new(f'QFDoub_{name}',   'QFDoub'),
                      pdr.place('DrDoub'), pdr.new(f'QDDoub2_{name}',  'QDDoub'),
                      pdr.place('DrTripl'),pdr.new(f'QFTrip_{name}1',  'QFtr')]
            if fall == 'symm':
                comps = [pdr.new(f'QFA_{name}CH', 'QFarcH')] + comps + \
                        [pdr.place('DrTrips'),
                         pdr.new(f'QDTripC_{name}2H', 'QDtrH')]
            elif fall == 'right':
                comps = [pdr.new(f'QFA_{name}C', 'QFarc')] + comps + \
                        [pdr.place('DrTrips')]
            elif fall == 'left':
                comps += [pdr.place('DrTrips'),
                          pdr.new(f'QDTripC_{name}2', 'QDtr')]
                comps = list(reversed(comps))
            else:
                raise ValueError(f'Unknown fall value: {fall!r}')
            return pdr.new_line(components=comps)

    arc1R = makesextant('xR', 'symm')
    # Marker at the centre of the long drift. The anchor is resolved from the
    # lattice: QDDoub2 now that the straight is entered through a D-F-D triplet
    # (it was QFDoub with the plain doublet). Hard-coding
    # QDDoub here is what put the marker 0.55 m (l_doub + l_quad) upstream of
    # the straight centre when the triplet was reordered to F-D-F.
    _ctr_anchor = _straight_anchor(arc1R)
    _straight_drift_after(arc1R, _ctr_anchor)          # validates adjacency
    arc1R.insert(pdr.new('CtrS1_xR1', xt.Marker),
                 at='(l_tripl+l_quad)/2', from_=_ctr_anchor)
    arc1R_sliced = _sliced(arc1R)
    period       = makesextant('PR', 'symm') + (-makesextant('PL', 'symm'))
    period_sliced = _sliced(period)
    ring = (makesextant('1R', 'right') + makesextant('2L', 'left') +
            makesextant('2R', 'right') + makesextant('3L', 'left') +
            makesextant('3R', 'right') + makesextant('1L', 'left'))

    # arc1R included so the extra DS quad is present where matching runs.
    # It reuses the QDDoub element/kQDDoub knob, so matching powers it with
    # the existing doublet quad automatically.
    insert_DS_betay_quads(pdr, ring, period, arc1R)
    if ds_waist_x:
        insert_DS_waist_markers(pdr, arc1R, period)   # not ring: RF goes there

    _export_lines(pdr, arc1R, cell_arc, cell_tr, period, ring)

    if not matched:
        return pdr

    cell_arc_opt, cell_tr_opt = _match_cells_3fold(pdr, cell_arc, cell_tr,
                                                    mu_cell=phase_advance)
    _run_standard_matching(ring,cell_arc_opt, cell_arc, cell_tr_opt, cell_tr,
                           arc1R, WP, betay_DS_target=betay_DS_target,
                           bounded=False, match_WP=match_WP)
    _make_rf_and_finalise(pdr, ring, arc1R, cell_arc, cell_tr,
                          period, U0, VRF, bend_edge)
    _check_triplet_basin(pdr)
    return pdr


def three_fold_periodicity_long(fringe_fields=True, matched=True,
                                WP=config.WP_D1, phase_advance=0.25,
                                betay_DS_target=None):
 
    pdr, quad_edge, bend_edge = _make_env(fringe_fields)
    E0 = config.E0; VRF = config.VRF
 
    pdr.vars({
        'l_cell':   3.6,    'l_bend':   0.40,   'l_bendDS': 0.40,
        'dl_noben': 0.85,   'l_quad':   0.30,
        'l_drift':  '(l_cell - 2*l_bend - 2*l_quad)/4.',
        'dl_drift': -0.0,   'dl_trans': 0.20,
        'l_doub':   0.25,   'l_tripl':  7,
        'l_trips':  0.40,   'l_sext':   0.20,
        'l_trans':  'l_drift+dl_trans',
        'l_DSL':    '2*l_drift + l_bend + dl_noben',
    })
 
    pdr.vars({
        'N_cells_S': 12,
        'hBarc': '6.283185307/(6*(2*N_cells_S*l_bend + l_bendDS))',
        # placeholders -- overwritten by the seeding helpers below
        'kQFarc':  2.9478,  'kQDarc':  -2.9231,
        'kQFarcM': 2.8846,  'kQDarcM': -2.7567,
        'kQFDS':   2.8042,  'kQDDS':   -2.2858,
        'kQFDoub': 3.9170,  'kQDDoub': -2.5190,
        'kQFtr':   4.4429,  'kQDtr':   -2.4723,
    })
 
    U0 = (0.88463e-31)*E0**4*(2.*np.pi) / (
        6*(2*pdr['N_cells_S']*pdr['l_bend'] + pdr['l_bendDS']))
 
    _make_base_elements(pdr, quad_edge, bend_edge)
 
    # cell_arc from the shared helper; cell_tr rebuilt as F-D-F
    cell_arc, cell_tr = _make_reference_cells(pdr)
 
    # ---- seed knobs before anything is Twissed ----
    _seed_arc_knobs(pdr, mu_cell=phase_advance)
    _seed_transition_knobs(pdr)
    _seed_triplet_knobs_FDF(pdr, cell_tr)     # F-D-F ratio convention
 
    def makesextant(name, fall):
        comps = []
        for ind in range(int(pdr['N_cells_S']) - 1):
            comps += [pdr.place('Drarc'), pdr.new(f'Bend1_{name}{ind+1}', 'Bend'),
                      pdr.place('Drarc'), pdr.new(f'QDA_{name}{ind+1}',   'QDarc'),
                      pdr.place('Drarc'), pdr.new(f'Bend2_{name}{ind+1}', 'Bend'),
                      pdr.place('Drarc'), pdr.new(f'QFA_{name}{ind+1}',   'QFarc')]
        n = int(pdr['N_cells_S'])
        comps[-1] = pdr.new(f'QFA_M{name}{n-1}', 'QFarcM')
        comps += [pdr.place('Drarc'),  pdr.new(f'Bend1_{name}{n}', 'Bend'),
                  pdr.place('Drarc'),  pdr.new(f'QDA_M{name}{n}',  'QDarcM'),
                  pdr.place('Drarc'),  pdr.new(f'Bend2_{name}{n}', 'Bend'),
                  pdr.place('DrarcS'), pdr.new(f'QFDS_{name}',     'QFDS'),
                  pdr.place('DrDSL'),  pdr.new(f'QDDS_{name}',     'QDDS'),
                  pdr.place('Drarc'),  pdr.new(f'BendDS_{name}',   'BendDS'),
                  # ---- F-D-F straight: D outside, F either side of drift ----
                  pdr.place('DrTrans'),pdr.new(f'QDDoub_{name}',   'QDDoub'),
                  pdr.place('DrDoub'), pdr.new(f'QFDoub_{name}',   'QFDoub'),
                  pdr.place('DrTripl'),pdr.new(f'QFTrip_{name}1',  'QFtr')]
        if fall == 'symm':
            comps = [pdr.new(f'QFA_{name}CH', 'QFarcH')] + comps + \
                    [pdr.place('DrTrips'),
                     pdr.new(f'QDTripC_{name}2H', 'QDtrH')]
        elif fall == 'right':
            comps = [pdr.new(f'QFA_{name}C', 'QFarc')] + comps + \
                    [pdr.place('DrTrips')]
        elif fall == 'left':
            comps += [pdr.place('DrTrips'),
                      pdr.new(f'QDTripC_{name}2', 'QDtr')]
            comps = list(reversed(comps))
        else:
            raise ValueError(f'Unknown fall value: {fall!r}')
        return pdr.new_line(components=comps)
 
    arc1R = makesextant('xR', 'symm')
    # Marker at the centre of the long drift; anchor resolved from the lattice
    # (QFDoub here, since this straight is F-D-F).
    _ctr_anchor = _straight_anchor(arc1R)
    _straight_drift_after(arc1R, _ctr_anchor)          # validates adjacency
    arc1R.insert(pdr.new('CtrS1_xR1', xt.Marker),
                 at='(l_tripl+l_quad)/2', from_=_ctr_anchor)
    arc1R_sliced  = _sliced(arc1R)
    period        = makesextant('PR', 'symm') + (-makesextant('PL', 'symm'))
    period_sliced = _sliced(period)
    ring = (makesextant('1R', 'right') + makesextant('2L', 'left') +
            makesextant('2R', 'right') + makesextant('3L', 'left') +
            makesextant('3R', 'right') + makesextant('1L', 'left'))
 
    insert_DS_betay_quads(pdr, ring, period, arc1R)
 
    print(f'\nthree_fold_periodicity_long (F-D-F): C = {ring.get_length():.3f} m '
          f'(N_cells_S={int(pdr["N_cells_S"])}, l_tripl={pdr["l_tripl"]} m)')
 
    _export_lines(pdr, arc1R, cell_arc, cell_tr, period, ring)
 
    if not matched:
        return pdr
 
    cell_arc_opt, cell_tr_opt = _match_cells_3fold(pdr, cell_arc, cell_tr,
                                                    mu_cell=phase_advance)
    _run_standard_matching(ring,cell_arc_opt, cell_arc, cell_tr_opt, cell_tr,
                           arc1R, WP, betay_DS_target=betay_DS_target)
    _make_rf_and_finalise(pdr, ring, arc1R, cell_arc, cell_tr,
                          period, U0, VRF, bend_edge)
    return pdr






def three_fold_periodicity_90_deg_many_sext(fringe_fields=True, matched=True, WP=config.WP_D1):
   
    pdr, quad_edge, bend_edge = _make_env(fringe_fields)
    E0 = config.E0; VRF = config.VRF

    '''pdr.vars({
        'l_cell':   3.5,    'l_bend':   0.40,   'l_bendDS': 0.55,
        'dl_noben': 0.25,   'l_quad':   0.30,
        'l_drift':  0.525,  'dl_drift': -0.15,  'dl_trans': 0.00,
        'l_doub':   0.25,   'l_tripl':  3.0,    'l_trips':  0.40,
        'l_sext':   0.10,
    })
    '''
    pdr.vars({
                'l_cell':   3.4,    'l_bend':   0.40,   'l_bendDS': 0.4,
                'dl_noben': 0.85,   'l_quad':   0.30,
                'l_drift':  '(l_cell - 2*l_bend - 2*l_quad)/4.',
                'dl_drift': -0.0,   'dl_trans': 0.20,
                'l_doub':   0.25,   'l_tripl':  2.7,    'l_trips':  0.40,
                'l_sext':   0.20,
            })
    pdr.vars({
        'N_cells_S': 8,
        'hBarc': '6.283185307/(6*(2*N_cells_S*l_bend + l_bendDS))',
        'kQFarc':  2.9478,  'kQDarc':  -2.9231,
        'kQFarcM': 2.8846,  'kQDarcM': -2.7567,
        'kQFDS':   2.8042,  'kQDDS':   -2.2858,
        'kQFDoub': 3.9170,  'kQDDoub': -2.5190,
        'kQFtr':   4.4429,  'kQDtr':   -2.4723,
        'kSF':    80.5236,  'kSD':   -114.8259,
    })
    U0 = (0.88463e-31)*E0**4*(2.*np.pi) / (
        6*(2*pdr['N_cells_S']*pdr['l_bend'] + pdr['l_bendDS']))

    _make_base_elements(pdr, quad_edge, bend_edge)
    pdr.new('Drarc2', xt.Drift,     length='(l_drift-l_sext)/2')
    pdr.new('SFDS',   xt.Sextupole, length='l_sext', k2='kSF',
            edge_entry_active=quad_edge, edge_exit_active=quad_edge)
    pdr.new('SDDS',   xt.Sextupole, length='l_sext', k2='kSD',
            edge_entry_active=quad_edge, edge_exit_active=quad_edge)

    cell_arc = pdr.new_line(components=[
        pdr.new('QF_cell_arcH1',  'QFarcH'),  pdr.place('Drarc2'),
        pdr.new('SF_cell_arc1',   'SFDS'),     pdr.place('Drarc2'),
        pdr.new('Bend1_cell_arcH','Bend'),     pdr.place('Drarc'),
        pdr.new('QD_cell_arcH',   'QDarc'),    pdr.place('Drarc2'),
        pdr.new('SD_cell_arc1',   'SDDS'),     pdr.place('Drarc2'),
        pdr.new('Bend2_cell_arcH','Bend'),     pdr.place('Drarc'),
        pdr.new('QF_cell_arcH2',  'QFarcH'),
    ])
    cell_tr = pdr.new_line(components=[
        pdr.new('QF_cell_trH1', 'QFtrH',
                at='0*l_trips + 0.25*l_quad + 0.0*l_tripl'),
        pdr.new('QD_cell_tr1',  'QDtr',
                at='1*l_trips + 1.00*l_quad + 0.0*l_tripl'),
        pdr.new('Mkr_cell_tr',  xt.Marker,
                at='1*l_trips + 1.50*l_quad + 0.5*l_tripl'),
        pdr.new('QD_cell_tr2',  'QDtr',
                at='1*l_trips + 2.00*l_quad + 1.0*l_tripl'),
        pdr.new('QF_cell_trH2', 'QFtrH',
                at='2*l_trips + 2.75*l_quad + 1.0*l_tripl'),
    ])

    def makesextant(name, fall):
        n      = int(pdr['N_cells_S'])
        n_sext = 6
        comps  = [pdr.new(f'Drarc0_{name}0', 'Drarc2'),
                  pdr.new(f'SFA0_{name}0',   'SFDS')]
        for ind in range(n - 1):
            place_sext = (ind < n_sext - 1)
            comps += [pdr.new(f'Drarc1_{name}{ind+1}', 'Drarc2'),
                      pdr.new(f'Bend1_{name}{ind+1}',  'Bend'),
                      pdr.new(f'Drarc2_{name}{ind+1}', 'Drarc'),
                      pdr.new(f'QDA_{name}{ind+1}',    'QDarc')]
            if place_sext:
                comps += [pdr.new(f'Drarc3_{name}{ind+1}',  'Drarc2'),
                          pdr.new(f'SDA_{name}{ind+1}',     'SDDS'),
                          pdr.new(f'Drarc4_{name}{ind+1}',  'Drarc2')]
            else:
                comps += [pdr.new(f'Drarc34_{name}{ind+1}', 'Drarc')]
            comps += [pdr.new(f'Bend2_{name}{ind+1}',  'Bend'),
                      pdr.new(f'Drarc5_{name}{ind+1}', 'Drarc'),
                      pdr.new(f'QFA_{name}{ind}',      'QFarc')]
            if place_sext:
                comps += [pdr.new(f'Drarc6_{name}{ind+1}',  'Drarc2'),
                          pdr.new(f'SFA_{name}{ind+1}',     'SFDS'),
                          pdr.new(f'Drarc6b_{name}{ind+1}', 'Drarc2')]
            else:
                comps += [pdr.new(f'Drarc66_{name}{ind+1}', 'Drarc')]
        comps[-3] = pdr.new(f'QFA_M{name}{n}', 'QFarcM')
        comps += [pdr.new(f'DrarcS1_{name}',  'Drarc2'),
                  pdr.new(f'Bend1_{name}{n}', 'Bend'),
                  pdr.new(f'Drarc7_{name}',   'Drarc'),
                  pdr.new(f'QDA_M{name}{n}',  'QDarcM'),
                  pdr.new(f'Drarc8_{name}',   'Drarc'),
                  pdr.new(f'Bend2_{name}{n}', 'Bend'),
                  pdr.new(f'Drarc9_{name}',   'Drarc'),  pdr.new(f'QFDS_{name}',   'QFDS'),
                  pdr.new(f'DrDSL1_{name}',   'DrDSL'),  pdr.new(f'QDDS_{name}',   'QDDS'),
                  pdr.new(f'Drarc10_{name}',  'Drarc'),  pdr.new(f'BendDS_{name}', 'BendDS'),
                  pdr.new(f'DrTrans1_{name}', 'DrTrans'),pdr.new(f'QFDoub_{name}', 'QFDoub'),
                  pdr.new(f'DrDoub1_{name}',  'DrDoub'), pdr.new(f'QDDoub_{name}', 'QDDoub'),
                  pdr.new(f'DrTripl1_{name}', 'DrTripl'),pdr.new(f'QDTrip_{name}1','QDtr')]
        if fall == 'symm':
            comps = [pdr.new(f'QFA_{name}CH', 'QFarcH')] + comps + \
                    [pdr.place('DrTrips'), pdr.new(f'QFTripC_{name}2H', 'QFtrH')]
        elif fall == 'right':
            comps = [pdr.new(f'QFA_{name}C', 'QFarc')] + comps + \
                    [pdr.place('DrTrips')]
        elif fall == 'left':
            comps += [pdr.place('DrTrips'), pdr.new(f'QFTripC_{name}2', 'QFtr')]
            comps = list(reversed(comps))
        else:
            raise ValueError(f'Unknown fall value: {fall!r}')
        return pdr.new_line(components=comps)

    arc1R = makesextant('xR', 'symm')
    arc1R.insert(pdr.new('CtrS1_xR1', xt.Marker),
                 at='(l_tripl+l_quad)/2', from_='QDDoub_xR')
    arc1R_sliced = _sliced(arc1R)
    period       = makesextant('PR', 'symm') + (-makesextant('PL', 'symm'))
    period_sliced = _sliced(period)
    ring = (makesextant('1R', 'right') + makesextant('2L', 'left') +
            makesextant('2R', 'right') + makesextant('3L', 'left') +
            makesextant('3R', 'right') + makesextant('1L', 'left'))

    _export_lines(pdr, arc1R, cell_arc, cell_tr, period, ring)

    if not matched:
        return pdr

    cell_arc_opt, cell_tr_opt = _match_cells_3fold(pdr, cell_arc, cell_tr,
                                                    mu_cell=0.25)
    _run_standard_matching(ring,cell_arc_opt, cell_arc, cell_tr_opt, cell_tr,
                           arc1R, WP)
    _make_rf_and_finalise(pdr, ring, arc1R, cell_arc, cell_tr,
                          period, U0, VRF, bend_edge)
    return pdr




def three_fold_periodicity_120_deg_many_sext(fringe_fields=True, matched=True, WP=config.WP_D1_120):
  
    pdr, quad_edge, bend_edge = _make_env(fringe_fields)
    E0 = config.E0; VRF = config.VRF

    pdr.vars({
        'l_cell':   3.5,    'l_bend':   0.40,   'l_bendDS': 0.55,
        'dl_noben': 0.25,   'l_quad':   0.30,
        'l_drift':  0.525,  'dl_drift': -0.15,  'dl_trans': 0.00,
        'l_doub':   0.25,   'l_tripl':  3.0,    'l_trips':  0.40,
        'l_sext':   0.10,
    })
    pdr.vars({
        'N_cells_S': 8,
        'hBarc': '6.283185307/(6*(2*N_cells_S*l_bend + l_bendDS))',
        'kQFarc':  2.9478,  'kQDarc':  -2.9231,
        'kQFarcM': 2.8846,  'kQDarcM': -2.7567,
        'kQFDS':   2.8042,  'kQDDS':   -2.2858,
        'kQFDoub': 3.9170,  'kQDDoub': -2.5190,
        'kQFtr':   4.4429,  'kQDtr':   -2.4723,
        'kSF':    80.5236,  'kSD':   -114.8259,
    })
    U0 = (0.88463e-31)*E0**4*(2.*np.pi) / (
        6*(2*pdr['N_cells_S']*pdr['l_bend'] + pdr['l_bendDS']))

    _make_base_elements(pdr, quad_edge, bend_edge)
    # Drarc2: half-drift flanking a sextupole so that
    # Drarc2 + sext + Drarc2 = Drarc (cell length preserved)
    pdr.new('Drarc2', xt.Drift,     length='(l_drift-l_sext)/2')
    pdr.new('SFDS',   xt.Sextupole, length='l_sext', k2='kSF',
            edge_entry_active=quad_edge, edge_exit_active=quad_edge)
    pdr.new('SDDS',   xt.Sextupole, length='l_sext', k2='kSD',
            edge_entry_active=quad_edge, edge_exit_active=quad_edge)


    cell_arc = pdr.new_line(components=[
        pdr.new('QF_cell_arcH1',  'QFarcH'),  pdr.place('Drarc2'),
        pdr.new('SF_cell_arc1',   'SFDS'),     pdr.place('Drarc2'),
        pdr.new('Bend1_cell_arcH','Bend'),     pdr.place('Drarc2'),
        pdr.new('SD_cell_arc1',   'SDDS'),     pdr.place('Drarc2'),
        pdr.new('QD_cell_arcH',   'QDarc'),    pdr.place('Drarc'),
        pdr.new('Bend2_cell_arcH','Bend'),     pdr.place('Drarc'),
        pdr.new('QF_cell_arcH2',  'QFarcH'),
    ])

    cell_tr = pdr.new_line(components=[
        pdr.new('QF_cell_trH1', 'QFtrH',
                at='0*l_trips + 0.25*l_quad + 0.0*l_tripl'),
        pdr.new('QD_cell_tr1',  'QDtr',
                at='1*l_trips + 1.00*l_quad + 0.0*l_tripl'),
        pdr.new('Mkr_cell_tr',  xt.Marker,
                at='1*l_trips + 1.50*l_quad + 0.5*l_tripl'),
        pdr.new('QD_cell_tr2',  'QDtr',
                at='1*l_trips + 2.00*l_quad + 1.0*l_tripl'),
        pdr.new('QF_cell_trH2', 'QFtrH',
                at='2*l_trips + 2.75*l_quad + 1.0*l_tripl'),
    ])

 
    def makesextant(name, fall):
        n      = int(pdr['N_cells_S'])
        n_sext = 6
        comps  = []

        for ind in range(n - 1):
            place_sext = (ind < n_sext)
            if place_sext:
                comps += [
                    pdr.new(f'Drarc1_{name}{ind+1}', 'Drarc2'),
                    pdr.new(f'SFA_{name}{ind+1}',    'SFDS'),    # SF near QF
                    pdr.new(f'Drarc2_{name}{ind+1}', 'Drarc2'),
                    pdr.new(f'Bend1_{name}{ind+1}',  'Bend'),
                    pdr.new(f'Drarc3_{name}{ind+1}', 'Drarc2'),
                    pdr.new(f'SDA_{name}{ind+1}',    'SDDS'),    # SD near QD
                    pdr.new(f'Drarc4_{name}{ind+1}', 'Drarc2'),
                    pdr.new(f'QDA_{name}{ind+1}',    'QDarc'),
                    pdr.new(f'Drarc5_{name}{ind+1}', 'Drarc'),
                    pdr.new(f'Bend2_{name}{ind+1}',  'Bend'),
                    pdr.new(f'Drarc6_{name}{ind+1}', 'Drarc'),
                    pdr.new(f'QFA_{name}{ind+1}',    'QFarc'),
                ]
            else:
                comps += [
                    pdr.new(f'Drarc1_{name}{ind+1}', 'Drarc'),
                    pdr.new(f'Bend1_{name}{ind+1}',  'Bend'),
                    pdr.new(f'Drarc2_{name}{ind+1}', 'Drarc'),
                    pdr.new(f'QDA_{name}{ind+1}',    'QDarc'),
                    pdr.new(f'Drarc3_{name}{ind+1}', 'Drarc'),
                    pdr.new(f'Bend2_{name}{ind+1}',  'Bend'),
                    pdr.new(f'Drarc4_{name}{ind+1}', 'Drarc'),
                    pdr.new(f'QFA_{name}{ind+1}',    'QFarc'),
                ]

        # Matching cell — last QF becomes QFarcM, no sextupoles
        comps[-1] = pdr.new(f'QFA_M{name}{n}', 'QFarcM')
        comps += [
            pdr.new(f'DrarcM1_{name}',  'Drarc'),
            pdr.new(f'Bend1_{name}{n}', 'Bend'),
            pdr.new(f'DrarcM2_{name}',  'Drarc'),
            pdr.new(f'QDA_M{name}{n}',  'QDarcM'),
            pdr.new(f'DrarcM3_{name}',  'Drarc'),
            pdr.new(f'Bend2_{name}{n}', 'Bend'),
            # DS section
            pdr.new(f'DrarcDS_{name}',  'Drarc'),
            pdr.new(f'QFDS_{name}',     'QFDS'),
            pdr.new(f'DrDSL_{name}',    'DrDSL'),
            pdr.new(f'QDDS_{name}',     'QDDS'),
            pdr.new(f'DrBDS_{name}',    'Drarc'),
            pdr.new(f'BendDS_{name}',   'BendDS'),
            # Doublet + triplet
            pdr.new(f'DrTrans_{name}',  'DrTrans'),
            pdr.new(f'QFDoub_{name}',   'QFDoub'),
            pdr.new(f'DrDoub_{name}',   'DrDoub'),
            pdr.new(f'QDDoub_{name}',   'QDDoub'),
            pdr.new(f'DrTripl_{name}',  'DrTripl'),
            pdr.new(f'QDTrip_{name}1',  'QDtr'),
        ]

        if fall == 'symm':
            comps = [pdr.new(f'QFA_{name}CH', 'QFarcH')] + comps + \
                    [pdr.place('DrTrips'), pdr.new(f'QFTripC_{name}2H', 'QFtrH')]
        elif fall == 'right':
            comps = [pdr.new(f'QFA_{name}C', 'QFarc')] + comps + \
                    [pdr.place('DrTrips')]
        elif fall == 'left':
            comps += [pdr.place('DrTrips'), pdr.new(f'QFTripC_{name}2', 'QFtr')]
            comps = list(reversed(comps))
        else:
            raise ValueError(f'Unknown fall value: {fall!r}')
        return pdr.new_line(components=comps)

    arc1R = makesextant('xR', 'symm')
    arc1R.insert(pdr.new('CtrS1_xR1', xt.Marker),
                 at='(l_tripl+l_quad)/2', from_='QDDoub_xR')
    arc1R_sliced = _sliced(arc1R)
    period       = makesextant('PR', 'symm') + (-makesextant('PL', 'symm'))
    period_sliced = _sliced(period)
    ring = (makesextant('1R', 'right') + makesextant('2L', 'left') +
            makesextant('2R', 'right') + makesextant('3L', 'left') +
            makesextant('3R', 'right') + makesextant('1L', 'left'))

    if not matched:
        _export_lines(pdr, arc1R, cell_arc, cell_tr, period, ring)
        return pdr

    cell_arc_opt, cell_tr_opt = _match_cells_3fold(pdr, cell_arc, cell_tr,
                                                    mu_cell=1/3)
    _run_standard_matching(ring,cell_arc_opt, cell_arc, cell_tr_opt, cell_tr,
                           arc1R, WP)

    _insert_rf(pdr, ring, U0, VRF, rf_from='QDDoub_1R')

    ring.match(solve=True, method='6d',
               vary=xt.VaryList(['kSF', 'kSD'], step=1e-4),
               targets=[xt.Target('dqx', 0, tol=1e-4),
                        xt.Target('dqy', 0, tol=1e-4)])

    _finalise(pdr, ring, arc1R, cell_arc, cell_tr, period, bend_edge)
    return pdr

def three_fold_periodicity_120_deg(fringe_fields=True, matched=True, WP=config.WP_D1_120):

    pdr, quad_edge, bend_edge = _make_env(fringe_fields)
    E0 = config.E0; VRF = config.VRF

    pdr.vars({
        'l_cell':   3.5,    'l_bend':   0.40,   'l_bendDS': 0.55,
        'dl_noben': 0.25,   'l_quad':   0.30,
        'l_drift':  0.525,  'dl_drift': -0.15,  'dl_trans': 0.00,
        'l_doub':   0.25,   'l_tripl':  3.0,    'l_trips':  0.40,
        'l_sext':   0.10,   # kept for sextupole insertion compatibility
    })
    pdr.vars({
        'N_cells_S': 8,
        'hBarc': '6.283185307/(6*(2*N_cells_S*l_bend + l_bendDS))',
        'kQFarc':  2.9478,  'kQDarc':  -2.9231,
        'kQFarcM': 2.8846,  'kQDarcM': -2.7567,
        'kQFDS':   2.8042,  'kQDDS':   -2.2858,
        'kQFDoub': 3.9170,  'kQDDoub': -2.5190,
        'kQFtr':   4.4429,  'kQDtr':   -2.4723,
    })
    U0 = (0.88463e-31)*E0**4*(2.*np.pi) / (
        6*(2*pdr['N_cells_S']*pdr['l_bend'] + pdr['l_bendDS']))

    _make_base_elements(pdr, quad_edge, bend_edge)

    cell_arc, cell_tr = _make_reference_cells(pdr)

    def makesextant(name, fall):
        n     = int(pdr['N_cells_S'])
        comps = []

        for ind in range(n - 1):
            comps += [pdr.new(f'Drarc_{name}_{ind}_1', 'Drarc'),
                      pdr.new(f'Bend1_{name}{ind+1}',  'Bend'),
                      pdr.new(f'Drarc_{name}_{ind}_2', 'Drarc'),
                      pdr.new(f'QDA_{name}{ind+1}',    'QDarc'),
                      pdr.new(f'Drarc_{name}_{ind}_3', 'Drarc'),
                      pdr.new(f'Bend2_{name}{ind+1}',  'Bend'),
                      pdr.new(f'Drarc_{name}_{ind}_4', 'Drarc'),
                      pdr.new(f'QFA_{name}{ind+1}',    'QFarc')]

        # Matching cell — replace last QF/QD with matching quads
        n = int(pdr['N_cells_S'])
        comps[-1] = pdr.new(f'QFA_M{name}{n-1}', xt.Quadrupole,
                             length='l_quad', k1=pdr.vars['kQFarcM'],
                             edge_entry_active=quad_edge,
                             edge_exit_active=quad_edge)
        comps += [pdr.new(f'Drarc_{name}_m1', 'Drarc'),
                  pdr.new(f'Bend1_{name}{n}',  'Bend'),
                  pdr.new(f'Drarc_{name}_m2', 'Drarc'),
                  pdr.new(f'QDA_M{name}{n}',  xt.Quadrupole,
                          length='l_quad', k1=pdr.vars['kQDarcM'],
                          edge_entry_active=quad_edge,
                          edge_exit_active=quad_edge),
                  pdr.new(f'Drarc_{name}_m3', 'Drarc'),
                  pdr.new(f'Bend2_{name}{n}',  'Bend'),
                  pdr.new(f'DrarcS_{name}',   'DrarcS'),
                  pdr.new(f'QFDS_{name}',     xt.Quadrupole, length='l_quad',
                          k1=pdr.vars['kQFDS'],
                          edge_entry_active=quad_edge, edge_exit_active=quad_edge),
                  pdr.new(f'DrDSL_{name}',    'DrDSL'),
                  pdr.new(f'QDDS_{name}',     xt.Quadrupole, length='l_quad',
                          k1=pdr.vars['kQDDS'],
                          edge_entry_active=quad_edge, edge_exit_active=quad_edge),
                  pdr.new(f'Drarc_{name}_ds', 'Drarc'),
                  pdr.new(f'BendDS_{name}',   'BendDS'),
                  pdr.new(f'DrTrans_{name}',  'DrTrans'),
                  pdr.new(f'QFDoub_{name}',   xt.Quadrupole, length='l_quad',
                          k1=pdr.vars['kQFDoub'],
                          edge_entry_active=quad_edge, edge_exit_active=quad_edge),
                  pdr.new(f'DrDoub_{name}',   'DrDoub'),
                  pdr.new(f'QDDoub_{name}',   xt.Quadrupole, length='l_quad',
                          k1=pdr.vars['kQDDoub'],
                          edge_entry_active=quad_edge, edge_exit_active=quad_edge),
                  pdr.new(f'DrTripl_{name}',  'DrTripl'),
                  pdr.new(f'QDTrip_{name}1',  xt.Quadrupole, length='l_quad',
                          k1=pdr.vars['kQDtr'],
                          edge_entry_active=quad_edge, edge_exit_active=quad_edge)]

        if fall == 'symm':
            comps = [pdr.new(f'QFA_{name}CH', 'QFarcH')] + comps + \
                    [pdr.place('DrTrips'), pdr.new(f'QFTripC_{name}2H', 'QFtrH')]
        elif fall == 'right':
            comps = [pdr.new(f'QFA_{name}C', 'QFarc')] + comps + \
                    [pdr.place('DrTrips')]
        elif fall == 'left':
            comps += [pdr.place('DrTrips'), pdr.new(f'QFTripC_{name}2', 'QFtr')]
            comps = list(reversed(comps))
        else:
            raise ValueError(f'Unknown fall value: {fall!r}')
        return pdr.new_line(components=comps)

    arc1R = makesextant('xR', 'symm')
    arc1R.insert(pdr.new('CtrS1_xR1', xt.Marker),
                 at='(l_tripl+l_quad)/2', from_='QDDoub_xR')
    arc1R_sliced = _sliced(arc1R)
    period       = makesextant('PR', 'symm') + (-makesextant('PL', 'symm'))
    period_sliced = _sliced(period)
    ring = (makesextant('1R', 'right') + makesextant('2L', 'left') +
            makesextant('2R', 'right') + makesextant('3L', 'left') +
            makesextant('3R', 'right') + makesextant('1L', 'left'))

    if matched==False:
        _export_lines(pdr, arc1R, cell_arc, cell_tr, period, ring)
        return pdr
    else:
        cell_arc_opt, cell_tr_opt = _match_cells_3fold(pdr, cell_arc, cell_tr,
                                                        mu_cell=1/3)
        _run_standard_matching(ring,cell_arc_opt, cell_arc, cell_tr_opt, cell_tr,
                            arc1R, WP)

        _insert_rf(pdr, ring, U0, VRF, rf_from='QDDoub_1R')
        _finalise(pdr, ring, arc1R, cell_arc, cell_tr, period, bend_edge)
        return pdr

    
#------------------
# Design 2
#------------------

def two_fold_1straight(fringe_fields=True, matched=True,WP=config.WP_D2, phase_advance=0.25):

    pdr, quad_edge, bend_edge = _make_env(fringe_fields)
    E0 = config.E0            # shared values from config.py
    VRF = config.VRF
 
    pdr.vars({
        'l_cell':   3.0,    'l_bend':   0.40,   'l_bendDS': 0.4,
        'dl_noben': 0.25,   'l_quad':   0.30,
        'l_drift':  '(l_cell - 2*l_bend - 2*l_quad)/4.',
        'dl_drift': -0.1,   'dl_trans': 0.00,
        'l_doub':   0.25,   'l_tripl':  2.7,    'l_trips':  0.40,
        'l_sext':   0.20,
    })
    pdr.vars({
        'N_cells_S': 10,
        # One BendDS per sextant only (two-fold: factor 4).
        'hBarc': '6.283185307/(4*(2*N_cells_S*l_bend + l_bendDS))',
        # Starting values inherited from two_fold_periodicity_90_deg.
        'kQFarc':  2.9478,  'kQDarc':  -2.9231,
        'kQFarcM': 2.8846,  'kQDarcM': -2.7567,
        'kQFDS':   2.8042,  'kQDDS':   -2.2858,
        'kQFDoub': 3.9170,  'kQDDoub': -2.5190,
        'kQFtr':   4.4429,  'kQDtr':   -2.4723,
    })
    U0 = (0.88463e-31)*E0**4*(2.*np.pi) / (
        4*(2*pdr['N_cells_S']*pdr['l_bend'] + pdr['l_bendDS']))
 
    _make_base_elements(pdr, quad_edge, bend_edge)
    pdr.new('Bend_R', xt.Bend, length='l_bend', angle='hBarc*l_bend',
            k0_from_h=True,
            edge_entry_angle='hBarc*l_bend/2',
            edge_exit_angle='hBarc*l_bend/2',
            edge_entry_model=bend_edge, edge_exit_model=bend_edge)
    pdr.new('BendDS_R', xt.Bend, length='l_bendDS', angle='hBarc*l_bendDS',
            k0_from_h=True,
            edge_entry_angle='hBarc*l_bendDS/2',
            edge_exit_angle='hBarc*l_bendDS/2',
            edge_entry_model=bend_edge, edge_exit_model=bend_edge)
 
    cell_arc, cell_tr = _make_reference_cells(pdr)
 
    # ------------------------------------------------------------------
    # makesextant: arc cells + matching cell + single DS + ONE triplet.
    # ------------------------------------------------------------------
    def makesextant(name, fall):
        comps = []
        is_left  = (fall == 'left')
        b_type   = 'Bend_R'   if is_left else 'Bend'
        bds_type = 'BendDS_R' if is_left else 'BendDS'
 
        # Regular arc cells
        for ind in range(int(pdr['N_cells_S']) - 1):
            comps += [pdr.new(f'Drarc_{name}_{ind}_1', 'Drarc'),
                      pdr.new(f'Bend1_{name}{ind+1}',  b_type),
                      pdr.new(f'Drarc_{name}_{ind}_2', 'Drarc'),
                      pdr.new(f'QDA_{name}{ind+1}',    'QDarc'),
                      pdr.new(f'Drarc_{name}_{ind}_3', 'Drarc'),
                      pdr.new(f'Bend2_{name}{ind+1}',  b_type),
                      pdr.new(f'Drarc_{name}_{ind}_4', 'Drarc'),
                      pdr.new(f'QFA_{name}{ind+1}',    'QFarc')]
 
        # Matching cell
        n = int(pdr['N_cells_S'])
        comps[-1] = pdr.new(f'QFA_M{name}{n-1}', xt.Quadrupole,
                             length='l_quad', k1=pdr.vars['kQFarcM'],
                             edge_entry_active=quad_edge,
                             edge_exit_active=quad_edge)
        comps += [pdr.new(f'Drarc_{name}_m1', 'Drarc'),
                  pdr.new(f'Bend1_{name}{n}',  b_type),
                  pdr.new(f'Drarc_{name}_m2', 'Drarc'),
                  pdr.new(f'QDA_M{name}{n}',  xt.Quadrupole,
                          length='l_quad', k1=pdr.vars['kQDarcM'],
                          edge_entry_active=quad_edge,
                          edge_exit_active=quad_edge),
                  pdr.new(f'Drarc_{name}_m3', 'Drarc'),
                  pdr.new(f'Bend2_{name}{n}',  b_type)]
 
        # Single DS (arc -> straight boundary)
        comps += [pdr.new(f'DrarcS_{name}',   'DrarcS'),
                  pdr.new(f'QFDS_{name}',     xt.Quadrupole, length='l_quad',
                          k1=pdr.vars['kQFDS'],
                          edge_entry_active=quad_edge, edge_exit_active=quad_edge),
                  pdr.new(f'DrDSL_{name}',    'DrDSL'),
                  pdr.new(f'QDDS_{name}',     xt.Quadrupole, length='l_quad',
                          k1=pdr.vars['kQDDS'],
                          edge_entry_active=quad_edge, edge_exit_active=quad_edge),
                  pdr.new(f'Drarc_{name}_ds', 'Drarc'),
                  pdr.new(f'BendDS_{name}',    bds_type)]
 
        # ONE triplet (single straight) — no _{ti} suffix
        comps += [pdr.new(f'DrTrans_{name}',  'DrTrans'),
                  pdr.new(f'QFDoub_{name}',   xt.Quadrupole, length='l_quad',
                          k1=pdr.vars['kQFDoub'],
                          edge_entry_active=quad_edge, edge_exit_active=quad_edge),
                  pdr.new(f'DrDoub_{name}',   'DrDoub'),
                  pdr.new(f'QDDoub_{name}',   xt.Quadrupole, length='l_quad',
                          k1=pdr.vars['kQDDoub'],
                          edge_entry_active=quad_edge, edge_exit_active=quad_edge),
                  pdr.new(f'DrTripl_{name}',  'DrTripl'),
                  pdr.new(f'QDTrip_{name}1',  xt.Quadrupole, length='l_quad',
                          k1=pdr.vars['kQDtr'],
                          edge_entry_active=quad_edge, edge_exit_active=quad_edge)]
 
        # Fall-dependent boundary caps (DrTrips lives here in the 1-straight ring)
        if fall == 'symm':
            comps = [pdr.new(f'QFA_{name}CH', 'QFarcH')] + comps + \
                    [pdr.new(f'DrTrips_{name}', 'DrTrips'),
                     pdr.new(f'QFTripC_{name}2H', 'QFtrH')]
        elif fall == 'right':
            comps = [pdr.new(f'QFA_{name}C', 'QFarc')] + comps + \
                    [pdr.new(f'DrTrips_{name}', 'DrTrips')]
        elif fall == 'left':
            comps += [pdr.new(f'DrTrips_{name}', 'DrTrips'),
                      pdr.new(f'QFTripC_{name}2', xt.Quadrupole,
                              length='l_quad', k1=pdr.vars['kQFtr'],
                              edge_entry_active=quad_edge,
                              edge_exit_active=quad_edge)]
            comps = list(reversed(comps))
        else:
            raise ValueError(f'Unknown fall value: {fall!r}')
 
        return pdr.new_line(components=comps)
 
    arc1R = makesextant('xR', 'symm')
    arc1R.insert(pdr.new('CtrS1_xR1', xt.Marker),
                 at='(l_tripl+l_quad)/2', from_='QDDoub_xR')
    arc1R_sliced = _sliced(arc1R)
 
    period = makesextant('PR', 'symm') + (-makesextant('PL', 'symm'))
    period_sliced = _sliced(period)
 
    half_ring = makesextant('1R', 'right') + makesextant('2L', 'left')
    ring = half_ring + makesextant('2R', 'right') + makesextant('1L', 'left')
 
    if not matched:
        pdr.lines['arc1R']    = arc1R
        pdr.lines['cell_arc'] = cell_arc
        pdr.lines['cell_tr']  = cell_tr
        pdr.lines['period']   = period
        pdr.lines['ring']     = ring
        return pdr
 
    cell_arc_opt, cell_tr_opt = _match_cells_3fold(pdr, cell_arc, cell_tr,
                                                    mu_cell=phase_advance)
    _run_standard_matching(ring,cell_arc_opt, cell_arc, cell_tr_opt, cell_tr,
                           arc1R, WP, n_periods=4)
 
    _insert_rf(pdr, ring, U0, VRF, rf_from='QDDoub_1R')
    _finalise(pdr, ring, arc1R, cell_arc, cell_tr, period, bend_edge)
    return pdr

#-------------
# Design 3
#-------------

def two_fold_racetrack_3straight(fringe_fields=True, matched=True,WP=config.WP_D3, phase_advance=0.25,betay_DS_target=True):
    pdr, quad_edge, bend_edge = _make_env(fringe_fields)
    E0 = 2.86e9; VRF = 4.0e6

    pdr.vars({
        'l_cell':   3.0,    'l_bend':   0.40,   'l_bendDS': 0.40,
        'dl_noben': 0.27,   'l_quad':   0.30,
        'l_drift':  '(l_cell - 2*l_bend - 2*l_quad)/4.',
        'dl_drift': -0.1,   'dl_trans': 0.00,
        'l_doub':   0.25,   'l_tripl':  2.7,    'l_trips':  0.40,
        'l_sext':   0.20,
    })
    pdr.vars({
        'N_cells_S': 10,
        # One BendDS per sextant only
        'hBarc': '6.283185307/(4*(2*N_cells_S*l_bend + l_bendDS))',
        'kQFarc':   3.3710,  'kQDarc':  -3.3418,
        'kQFarcM':  2.8209,  'kQDarcM': -1.3405,
        'kQFDS':    3.0862,  'kQDDS':   -2.5894,
        'kQFDoub':  3.7553,  'kQDDoub': -2.1690,
        'kQFtr':    3.5695,  'kQDtr':   -1.5365,
    })
    U0 = (0.88463e-31)*E0**4*(2.*np.pi) / (
        4*(2*pdr['N_cells_S']*pdr['l_bend'] + pdr['l_bendDS']))

    _make_base_elements(pdr, quad_edge, bend_edge)
    pdr.new('Bend_R', xt.Bend, length='l_bend', angle='hBarc*l_bend',
            k0_from_h=True,
            edge_entry_angle='hBarc*l_bend/2',
            edge_exit_angle='hBarc*l_bend/2',
            edge_entry_model=bend_edge, edge_exit_model=bend_edge)
    pdr.new('BendDS_R', xt.Bend, length='l_bendDS', angle='hBarc*l_bendDS',
            k0_from_h=True,
            edge_entry_angle='hBarc*l_bendDS/2',
            edge_exit_angle='hBarc*l_bendDS/2',
            edge_entry_model=bend_edge, edge_exit_model=bend_edge)

    cell_arc, cell_tr = _make_reference_cells(pdr)

    def makesextant(name, fall):
        comps = []
        is_left  = (fall == 'left')
        b_type   = 'Bend_R'   if is_left else 'Bend'
        bds_type = 'BendDS_R' if is_left else 'BendDS'

        # Regular arc cells (identical to original)
        for ind in range(int(pdr['N_cells_S']) - 1):
            comps += [pdr.new(f'Drarc_{name}_{ind}_1', 'Drarc'),
                      pdr.new(f'Bend1_{name}{ind+1}',  b_type),
                      pdr.new(f'Drarc_{name}_{ind}_2', 'Drarc'),
                      pdr.new(f'QDA_{name}{ind+1}',    'QDarc'),
                      pdr.new(f'Drarc_{name}_{ind}_3', 'Drarc'),
                      pdr.new(f'Bend2_{name}{ind+1}',  b_type),
                      pdr.new(f'Drarc_{name}_{ind}_4', 'Drarc'),
                      pdr.new(f'QFA_{name}{ind+1}',    'QFarc')]

        # Matching cell (identical to original)
        n = int(pdr['N_cells_S'])
        comps[-1] = pdr.new(f'QFA_M{name}{n-1}', xt.Quadrupole,
                             length='l_quad', k1=pdr.vars['kQFarcM'],
                             edge_entry_active=quad_edge,
                             edge_exit_active=quad_edge)
        comps += [pdr.new(f'Drarc_{name}_m1', 'Drarc'),
                  pdr.new(f'Bend1_{name}{n}',  b_type),
                  pdr.new(f'Drarc_{name}_m2', 'Drarc'),
                  pdr.new(f'QDA_M{name}{n}',  xt.Quadrupole,
                          length='l_quad', k1=pdr.vars['kQDarcM'],
                          edge_entry_active=quad_edge,
                          edge_exit_active=quad_edge),
                  pdr.new(f'Drarc_{name}_m3', 'Drarc'),
                  pdr.new(f'Bend2_{name}{n}',  b_type)]

        # Single DS (arc -> straight boundary)
        comps += [pdr.new(f'DrarcS_{name}',   'DrarcS'),
                  pdr.new(f'QFDS_{name}',     xt.Quadrupole, length='l_quad',
                          k1=pdr.vars['kQFDS'],
                          edge_entry_active=quad_edge, edge_exit_active=quad_edge),
                  pdr.new(f'DrDSL_{name}',    'DrDSL'),
                  pdr.new(f'QDDS_{name}',     xt.Quadrupole, length='l_quad',
                          k1=pdr.vars['kQDDS'],
                          edge_entry_active=quad_edge, edge_exit_active=quad_edge),
                  pdr.new(f'Drarc_{name}_ds', 'Drarc'),
                  pdr.new(f'BendDS_{name}',    bds_type)]

        # Three back-to-back triplets — no DS between them
        for ti in range(3):
            comps += [pdr.new(f'DrTrans_{name}_{ti}',  'DrTrans'),
                      pdr.new(f'QFDoub_{name}_{ti}',   xt.Quadrupole, length='l_quad',
                              k1=pdr.vars['kQFDoub'],
                              edge_entry_active=quad_edge, edge_exit_active=quad_edge),
                      pdr.new(f'DrDoub_{name}_{ti}',   'DrDoub'),
                      pdr.new(f'QDDoub_{name}_{ti}',   xt.Quadrupole, length='l_quad',
                              k1=pdr.vars['kQDDoub'],
                              edge_entry_active=quad_edge, edge_exit_active=quad_edge),
                      pdr.new(f'DrTripl_{name}_{ti}',  'DrTripl'),
                      pdr.new(f'QDTrip_{name}_{ti}',   xt.Quadrupole, length='l_quad',
                              k1=pdr.vars['kQDtr'],
                              edge_entry_active=quad_edge, edge_exit_active=quad_edge),
                      pdr.new(f'DrTrips_{name}_{ti}',  'DrTrips')]

        # Fall-dependent boundary caps (identical logic to original)
        if fall == 'symm':
            comps = [pdr.new(f'QFA_{name}CH', 'QFarcH')] + comps + \
                    [pdr.new(f'QFTripC_{name}2H', 'QFtrH')]
        elif fall == 'right':
            comps = [pdr.new(f'QFA_{name}C', 'QFarc')] + comps
        elif fall == 'left':
            comps += [pdr.new(f'QFTripC_{name}2', xt.Quadrupole,
                              length='l_quad', k1=pdr.vars['kQFtr'],
                              edge_entry_active=quad_edge,
                              edge_exit_active=quad_edge)]
            comps = list(reversed(comps))
        else:
            raise ValueError(f'Unknown fall value: {fall!r}')

        return pdr.new_line(components=comps)

    arc1R = makesextant('xR', 'symm')
    arc1R.insert(pdr.new('CtrS1_xR1', xt.Marker),
                 at='(l_tripl+l_quad)/2', from_='QDDoub_xR_2')
    arc1R_sliced = _sliced(arc1R)

    period = makesextant('PR', 'symm') + (-makesextant('PL', 'symm'))
    period_sliced = _sliced(period)

    half_ring = makesextant('1R', 'right') + makesextant('2L', 'left')
    ring = half_ring + makesextant('2R', 'right') + makesextant('1L', 'left')

    #insert_DS_betay_quads(pdr, ring, period, arc1R)

    if not matched:
        pdr.lines['arc1R']    = arc1R
        pdr.lines['cell_arc'] = cell_arc
        pdr.lines['cell_tr']  = cell_tr
        pdr.lines['period']   = period
        pdr.lines['ring']     = ring
        return pdr

    cell_arc_opt, cell_tr_opt = _match_cells_3fold(pdr, cell_arc, cell_tr,
                                                    mu_cell=phase_advance)
    _run_standard_matching(ring,cell_arc_opt, cell_arc, cell_tr_opt, cell_tr,
                           arc1R, WP, n_periods=4,betay_DS_target=betay_DS_target)

    _insert_rf(pdr, ring, U0, VRF, rf_from='QDDoub_1R_2')
    _finalise(pdr, ring, arc1R, cell_arc, cell_tr, period, bend_edge)
    return pdr