import sys
import os

parent_dir = os.path.abspath('..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import xtrack as xt
import numpy as np
import constants


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
    vary = [xt.Vary(k, step=1e-5, limits=(-15, 15)) for k in knob_names]

    targets = [
        xt.TargetSet(dx=0, dpx=0, at=xt.END, tol=1e-9),
        xt.TargetSet(mux=mux0, muy=muy0, at=xt.END, tol=1e-8, tag='phase'),
        xt.TargetSet(alfx=0, alfy=0, at=xt.END, tol=1e-9),
    ]
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
    BETA_MAX = betay_DS_target if betay_DS_target else 10.
    if BETA_MAX < 10.:
        print(f'matchingWP: betay_DS_target={BETA_MAX:g} m is below the ~10 m '
              f'knee for this lattice -- the match either misses it entirely '
              f'(bety -> 16 m) or goes unstable. Clamping to 10.0 m; to go '
              f'lower, change the straight geometry (l_tripl / l_trips) first.')
        BETA_MAX = 10.
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
                 n_periods=6, betay_DS_target=None, MakePlot=False, FDF=True):
    cell_arc_opt.run_jacobian(10)
    tw_cell = cell_arc.twiss(method='4d')
    cell_tr_opt.targets[0].value = betxS
    cell_tr_opt.targets[1].value = betyS
    cell_tr_opt.run_jacobian(10)
    tw_tr = cell_tr.twiss(method='4d')

    vary = [
        xt.VaryList(['kQFarcM', 'kQDarcM'], step=1e-4),
        xt.VaryList(['kQFDS',   'kQDDS'],   step=1e-4),
        xt.VaryList(['kQFDoub', 'kQDDoub'], step=1e-4),
    ]

    targets = [
        xt.TargetSet(dx=0, dpx=0, at=xt.END, tol=1e-9),
        xt.TargetSet(alfx=0, alfy=0, at=xt.END, tol=1e-9),
        xt.TargetSet(betx=tw_tr.betx[0], bety=tw_tr.bety[0], at=xt.END, tol=1e-6),
    ]

    # Retain phase advance / WP matching constraint during beta matching
    if qx_target is not None and qy_target is not None:
        targets.extend([
            xt.Target('mux', qx_target / n_periods, at=xt.END, tol=1e-8),
            xt.Target('muy', qy_target / n_periods, at=xt.END, tol=1e-8),
        ])

    BETA_MAX = 5.
    soft_beta = []
    mk_list = ['QFDS_xR', 'QDDS_xR', 'QFDoub_xR', 'QDDoub_xR', 'QDDoubDS_xR', 
               'QFTrip_xR1' if FDF else 'QDTrip_xR1']
    for mk in mk_list:
        soft_beta += [
            xt.Target('betx', xt.LessThan(BETA_MAX), at=mk, weight=0.02),
            xt.Target('bety', xt.LessThan(BETA_MAX), at=mk, weight=0.02),
        ]
    targets += soft_beta

    opt = arc1R.match(
        method='4d', solve=True, assert_within_tol=False,
        betx=tw_cell.betx[0], alfx=tw_cell.alfx[0],
        bety=tw_cell.bety[0], alfy=tw_cell.alfy[0],
        dx=tw_cell.dx[0],     dpx=tw_cell.dpx[0],
        vary=vary, targets=targets)
    opt.run_jacobian(30)

    if MakePlot:
        arc1R.twiss(method='4d',
                    betx=tw_cell.betx[0], alfx=tw_cell.alfx[0],
                    bety=tw_cell.bety[0], alfy=tw_cell.alfy[0],
                    dx=tw_cell.dx[0],     dpx=tw_cell.dpx[0]).plot()
    return opt

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

def _beta_bound_setup(arc1R, FDF=True):
    """
    Return (knob_names, marker_names) for the bounded sextant match.

    Both are derived from the line rather than hard-coded, so they follow
    N_cells_S and the triplet topology instead of silently going stale.
    """
    names = list(arc1R.element_names)
    knobs = ['kQFarcM', 'kQDarcM', 'kQFDS', 'kQDDS',
             'kQFDoub', 'kQDDoub', 'kQFtr', 'kQDtr']
    if any(n.startswith('QDDoubDS_') for n in names):
        knobs.append('kQDDoubDS')
    pref = ('QFA_M', 'QDA_M', 'QFDS_', 'QDDS_', 'QDDoubDS_',
            'QFDoub_', 'QDDoub_', 'QFTrip_' if FDF else 'QDTrip_')
    seen, mks = set(), []
    for n in names:
        if n.startswith(pref) and n not in seen:
            seen.add(n); mks.append(n)
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
 
 


# ----------------
# Shared Builders
# ----------------

def _make_env(fringe_fields):
    pdr = xt.Environment()
    pdr.particle_ref = xt.Particles(kinetic_energy0=2.86e9,
                                     mass0=xt.ELECTRON_MASS_EV)
    return pdr, fringe_fields, 'full' if fringe_fields else 'linear'


def _make_base_elements(pdr, quad_edge, bend_edge):
    pdr.new('Bend', xt.Bend, length='l_bend', angle='hBarc*l_bend',
            k0_from_h=True,
            edge_entry_angle='hBarc*l_bend/2',
            edge_exit_angle='hBarc*l_bend/2',
            edge_entry_model=bend_edge, edge_exit_model=bend_edge)
    pdr.new('BendDS', xt.Bend, length='l_bendDS', angle='hBarc*l_bendDS',
            k0_from_h=True,
            edge_entry_angle='hBarc*l_bendDS/2',
            edge_exit_angle='hBarc*l_bendDS/2',
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

def _run_standard_matching(ring,cell_arc_opt, cell_arc, cell_tr_opt, cell_tr,
                            arc1R, wp_constants, n_periods=6,
                            betay_DS_target=None, FDF=True, bounded=True):
    # Step 1: Perform Working Point (Phase) matching
    matchingWP(*wp_constants, cell_arc_opt, cell_arc, arc1R,
               n_periods=n_periods, FDF=FDF,
               betay_DS_target=betay_DS_target)

    # Check status after matchingWP
    tw_cell = cell_arc.twiss(method='4d')
    tw_check = arc1R.twiss(method='4d',
                           betx=tw_cell.betx[0], alfx=tw_cell.alfx[0],
                           bety=tw_cell.bety[0], alfy=tw_cell.alfy[0],
                           dx=tw_cell.dx[0],     dpx=tw_cell.dpx[0])
    
    qx_ring = n_periods * tw_check.mux[-1]
    qy_ring = n_periods * tw_check.muy[-1]
    print(f'After matchingWP:')
    print(f'  mux at END = {tw_check.mux[-1]:.6f} (target {wp_constants[0]/n_periods:.6f})')
    print(f'  muy at END = {tw_check.muy[-1]:.6f} (target {wp_constants[1]/n_periods:.6f})')
    print(f'  qx_ring = {qx_ring:.6f} (target {wp_constants[0]:.6f})')
    print(f'  qy_ring = {qy_ring:.6f} (target {wp_constants[1]:.6f})')

    # Step 2: Perform Beta function matching while locking qx and qy targets
    tw_tr = cell_tr.twiss(method='4d')
    mid = len(tw_tr.betx) // 2

    # matchingBeta pins betx/bety at END to the reference triplet cell. In the
    # bounded scheme that over-constrains the problem and undoes the beta
    # bounds, so it is skipped unless explicitly re-enabled with bounded=False.
    if not bounded:
        matchingBeta(tw_tr.betx[mid], tw_tr.bety[mid],
                     cell_arc_opt, cell_arc, cell_tr_opt, cell_tr, arc1R,
                     qx_target=wp_constants[0], qy_target=wp_constants[1],
                     n_periods=n_periods, betay_DS_target=betay_DS_target,
                     FDF=FDF)

    

    # 1. Check ring Twiss
    tw_ring = ring.twiss(method='4d')
    print(f"Pre-fit Ring Tunes: Qx = {tw_ring.qx:.6f}, Qy = {tw_ring.qy:.6f}")

    # 2. Global fine-tune on full ring line.
    #    The arc-cell quads MUST be in the vary list. The sextant match
    #    typically lands ~0.4 short in Qx; forcing that onto kQFarcM/kQDarcM
    #    alone needs a ~23% gradient change, which breaks the arc->DS match
    #    and blows betx from 6.6 m to 19 m and |dx| from 0.29 m to 0.60 m.
    #    Spreading it over the arc cell as well costs ~3% and leaves the
    #    optics intact for the same tunes.
    ring.match(
        method='4d',
        vary=[
            xt.Vary('kQFarc',  step=1e-5),
            xt.Vary('kQDarc',  step=1e-5),
            xt.Vary('kQFarcM', step=1e-5),
            xt.Vary('kQDarcM', step=1e-5),
        ],
        targets=[
            xt.Target('qx', wp_constants[0], tol=1e-6),
            xt.Target('qy', wp_constants[1], tol=1e-6),
        ]
    )
            
    
def _export_lines(pdr, arc1R, cell_arc, cell_tr, period, ring):
    """Register all lines in pdr.lines — same keys for all lattice functions."""
    _register_lines(pdr, arc1R=arc1R, cell_arc=cell_arc, cell_tr=cell_tr,
                    period=period, ring=ring)


# ---------------------------------------------------------------------------
# Design 1 (more variations as this was created before pipeline optimisation)
# ---------------------------------------------------------------------------

def three_fold_periodicity(fringe_fields=True, matched=True,WP=constants.WP_D1,phase_advance=0.25,betay_DS_target=None):
    
    pdr, quad_edge, bend_edge = _make_env(fringe_fields)
    E0 = constants.E0; VRF = constants.VRF

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
            'l_doub':   0.25,   'l_tripl':  2.5,    'l_trips':  0.40,
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

    _make_base_elements(pdr, quad_edge, bend_edge)
    cell_arc, cell_tr = _make_reference_cells(pdr)

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
    # Marker at the centre of the long drift. The anchor is resolved from the
    # lattice: QFDoub in this F-D-F straight, QDDoub in a D-F-D one. Hard-coding
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


def three_fold_periodicity_long(fringe_fields=True, matched=True,
                                WP=constants.WP_D1, phase_advance=0.25,
                                betay_DS_target=None):
 
    pdr, quad_edge, bend_edge = _make_env(fringe_fields)
    E0 = constants.E0; VRF = constants.VRF
 
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






def three_fold_periodicity_90_deg_many_sext(fringe_fields=True, matched=True, WP=constants.WP_D1):
   
    pdr, quad_edge, bend_edge = _make_env(fringe_fields)
    E0 = constants.E0; VRF = constants.VRF

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




def three_fold_periodicity_120_deg_many_sext(fringe_fields=True, matched=True, WP=constants.WP_D1_120):
  
    pdr, quad_edge, bend_edge = _make_env(fringe_fields)
    E0 = constants.E0; VRF = constants.VRF

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

def three_fold_periodicity_120_deg(fringe_fields=True, matched=True, WP=constants.WP_D1_120):

    pdr, quad_edge, bend_edge = _make_env(fringe_fields)
    E0 = constants.E0; VRF = constants.VRF

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

def two_fold_1straight(fringe_fields=True, matched=True,WP=constants.WP_D2, phase_advance=0.25):

    pdr, quad_edge, bend_edge = _make_env(fringe_fields)
    E0 = constants.E0            # use the shared constants (16 MV RF), see note
    VRF = constants.VRF
 
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

def two_fold_racetrack_3straight(fringe_fields=True, matched=True,WP=constants.WP_D3, phase_advance=0.25,betay_DS_target=True):
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