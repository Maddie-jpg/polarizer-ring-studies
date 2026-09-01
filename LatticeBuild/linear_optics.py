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


def matchingWP(qx, qy, cell_arc_opt, cell_arc, arc1R, n_periods=6, MakePlot=False):
    cell_arc_opt.run_jacobian(10)
    cell_arc_tw = cell_arc.twiss(method='4d')
    bc = dict(betx=cell_arc_tw.betx[0], alfx=cell_arc_tw.alfx[0],
              bety=cell_arc_tw.bety[0], alfy=cell_arc_tw.alfy[0],
              dx=cell_arc_tw.dx[0],     dpx=cell_arc_tw.dpx[0])

    mux0 = arc1R.twiss(method='4d', **bc).mux[-1]      # current phase = start
    muy0 = arc1R.twiss(method='4d', **bc).muy[-1]

    opt = arc1R.match(
        method='4d', solve=False, verbose=False, **bc,
        vary=[
            xt.VaryList(['kQFarcM', 'kQDarcM'], step=1e-4, limits=(-10, 10)),
            xt.VaryList(['kQFDS',   'kQDDS'],   step=1e-4, limits=(-10, 10)),
            xt.VaryList(['kQFDoub', 'kQDDoub'], step=1e-4, limits=(-10, 10)),
            xt.VaryList(['kQFtr',   'kQDtr'],   step=1e-4, limits=(-10, 10)),
        ],
        targets=[
            xt.TargetSet(dx=0, dpx=0, at=xt.END, tol=1e-9),
            xt.TargetSet(mux=mux0, muy=muy0, at=xt.END, tol=1e-9, tag='phase'),
            xt.TargetSet(alfx=0, alfy=0, at=xt.END, tol=1e-9),
            xt.TargetSet(alfx=0, alfy=0, at='CtrS1_xR1', tol=1e-9, weight=10.),
        ])
    pt = [t for t in opt.targets if t.tag == 'phase']

    import numpy as np
    for frac in np.linspace(1/8, 1, 8):          # walk phase in, don't jump
        pt[0].value = mux0 + frac * (qx/n_periods - mux0)
        pt[1].value = muy0 + frac * (qy/n_periods - muy0)
        try:
            opt.solve(n_steps=40)
        except Exception:
            opt.step(30, broyden=True, rcond=1e-3)

    if MakePlot:
        arc1R.twiss(method='4d', **bc).plot()
    return opt


def matchingBeta(betxS, betyS, cell_arc_opt, cell_arc,
                 cell_tr_opt, cell_tr, arc1R, betay_DS_target=None,
                 MakePlot=False):
    cell_arc_opt.run_jacobian(10)
    tw_cell = cell_arc.twiss(method='4d')
    cell_tr_opt.targets[0].value = betxS
    cell_tr_opt.targets[1].value = betyS
    cell_tr_opt.run_jacobian(10)
    tw_tr = cell_tr.twiss(method='4d')

    vary = [xt.VaryList(['kQFarcM', 'kQDarcM'], step=1e-4),
            xt.VaryList(['kQFDS',   'kQDDS'],   step=1e-4),
            xt.VaryList(['kQFDoub', 'kQDDoub'], step=1e-4)]
            #xt.VaryList(['kQFtr',   'kQDtr'],   step=1e-4)]
    targets = [
        xt.TargetSet(dx=0, dpx=0, at=xt.END, tol=1e-9),
        xt.TargetSet(alfx=0, alfy=0, at=xt.END, tol=1e-9),
        xt.TargetSet(betx=tw_tr.betx[0], bety=tw_tr.bety[0],
                     at=xt.END, tol=1e-6),
    ]

    opt = arc1R.match(
        method='4d', solve=True, assert_within_tol=False,
        betx=tw_cell.betx[0], alfx=tw_cell.alfx[0],
        bety=tw_cell.bety[0], alfy=tw_cell.alfy[0],
        dx=tw_cell.dx[0],     dpx=tw_cell.dpx[0],
        vary=vary,
        targets=targets)
    opt.run_jacobian(30)
    if MakePlot:
        arc1R.twiss(method='4d',
                    betx=tw_cell.betx[0], alfx=tw_cell.alfx[0],
                    bety=tw_cell.bety[0], alfy=tw_cell.alfy[0],
                    dx=tw_cell.dx[0],     dpx=tw_cell.dpx[0]).plot()

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

        for new_name, qfds, at_expr, sign, off in plan:
            line.insert(
                pdr.new(new_name, xt.Quadrupole, length=q_length, k1='kQDDoub'),
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
    pdr.new('QDtrH',  xt.Quadrupole, length='l_quad',    k1='kQDtr',
            edge_entry_active=quad_edge, edge_exit_active=quad_edge)
    pdr.new('Drarc',  xt.Drift, length='l_drift')
    pdr.new('DrarcS', xt.Drift, length='l_drift + dl_drift')
    pdr.new('DrDSL',  xt.Drift, length='2*l_drift + l_bend + dl_noben')
    pdr.new('DrTrans',xt.Drift, length='l_drift + dl_trans')
    pdr.new('DrDoub', xt.Drift, length='l_doub')
    pdr.new('DrTripl',xt.Drift, length='l_tripl')
    pdr.new('DrTrips',xt.Drift, length='l_trips')


def _make_reference_cells(pdr):
    cell_arc = pdr.new_line(components=[
        pdr.new('QF_cell_arcH1',   'QFarcH'), pdr.place('Drarc'),
        pdr.new('Bend1_cell_arcH', 'Bend'),   pdr.place('Drarc'),
        pdr.new('QD_cell_arcH',    'QDarc'),  pdr.place('Drarc'),
        pdr.new('Bend2_cell_arcH', 'Bend'),   pdr.place('Drarc'),
        pdr.new('QF_cell_arcH2',   'QFarcH'),
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
    return cell_arc, cell_tr

def _insert_rf(pdr, ring, U0, VRF, rf_from='QDDoub_1R'):
    """Insert RF cavity into ring. Call before any 6d matching."""
    fRev = 1. / ring.twiss(method='4d').T_rev0
    fRF  = fRev * round(4.e8 / fRev)
    pdr.new('RFCav', xt.Cavity, length=1.5, frequency=fRF, voltage=VRF,
            lag=(180/np.pi) * (np.pi - np.arcsin(U0/VRF)) - 1.8)
    ring.insert(pdr.new('RFCav_1', 'RFCav'),
                at='(l_tripl+l_quad)/2', from_=rf_from)

def _make_rf_and_finalise(pdr, ring, arc1R, cell_arc, cell_tr,
                           period, U0, VRF, bend_edge,
                           rf_from='QDDoub_1R'):
    fRev = 1. / ring.twiss(method='4d').T_rev0
    fRF  = fRev * round(4.e8 / fRev)
    pdr.new('RFCav', xt.Cavity, length=1.5, frequency=fRF, voltage=VRF,
            lag=(180/np.pi) * (np.pi - np.arcsin(U0/VRF)) - 1.8)
    ring.insert(pdr.new('RFCav_1', 'RFCav'),
                at='(l_tripl+l_quad)/2', from_=rf_from)
    

def _finalise(pdr, ring, arc1R, cell_arc, cell_tr, period, bend_edge):
    """Configure radiation/bend model and register exported lines."""
    ring.configure_radiation(model='mean')
    ring.configure_bend_model(edge=bend_edge)
    pdr.lines['arc1R']    = arc1R
    pdr.lines['cell_arc'] = cell_arc
    pdr.lines['cell_tr']  = cell_tr
    pdr.lines['period']   = period
    pdr.lines['ring']     = ring

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

def _run_standard_matching(cell_arc_opt, cell_arc, cell_tr_opt, cell_tr,
                            arc1R, wp_constants, n_periods=6,
                            betay_DS_target=None):
    kQFtr_saved = arc1R.vars['kQFtr']._value
    kQDtr_saved = arc1R.vars['kQDtr']._value

    matchingWP(*wp_constants, cell_arc_opt, cell_arc, arc1R,n_periods=n_periods)
    
    # Check what we actually got
    tw_cell = cell_arc.twiss(method='4d')
    tw_check = arc1R.twiss(method='4d',
                            betx=tw_cell.betx[0], alfx=tw_cell.alfx[0],
                            bety=tw_cell.bety[0], alfy=tw_cell.alfy[0],
                            dx=tw_cell.dx[0],     dpx=tw_cell.dpx[0])
    print(f'After matchingWP:')
    print(f'  mux at END = {tw_check.mux[-1]:.6f}  '
          f'(target {wp_constants[0]/6:.6f})')
    print(f'  muy at END = {tw_check.muy[-1]:.6f}  '
          f'(target {wp_constants[1]/6:.6f})')
    print(f'  qx_ring = {n_periods*tw_check.mux[-1]:.6f}  (target {wp_constants[0]:.6f})')
    print(f'  qy_ring = {n_periods*tw_check.muy[-1]:.6f}  (target {wp_constants[1]:.6f})')

    '''arc1R.vars['kQFtr'] = kQFtr_saved
    arc1R.vars['kQDtr'] = kQDtr_saved
    cell_tr_opt.run_jacobian(20)'''

    tw_tr = cell_tr.twiss(method='4d')
    mid   = len(tw_tr.betx) // 2

    matchingBeta(tw_tr.betx[mid], tw_tr.bety[mid],
                 cell_arc_opt, cell_arc, cell_tr_opt, cell_tr, arc1R,
                 betay_DS_target=betay_DS_target)
    #matchingWP(*wp_constants, cell_arc_opt, cell_arc, arc1R,n_periods=n_periods)

def _export_lines(pdr, arc1R, cell_arc, cell_tr, period, ring):
    """Register all lines in pdr.lines — same keys for all lattice functions."""
    pdr.lines['arc1R']    = arc1R
    pdr.lines['cell_arc'] = cell_arc
    pdr.lines['cell_tr']  = cell_tr
    pdr.lines['period']   = period
    pdr.lines['ring']     = ring


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
    })
    U0 = (0.88463e-31)*E0**4*(2.*np.pi) / (
        6*(2*pdr['N_cells_S']*pdr['l_bend'] + pdr['l_bendDS']))

    _make_base_elements(pdr, quad_edge, bend_edge)
    cell_arc, cell_tr = _make_reference_cells(pdr)

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
                  pdr.place('DrTrans'),pdr.new(f'QFDoub_{name}',   'QFDoub'),
                  pdr.place('DrDoub'), pdr.new(f'QDDoub_{name}',   'QDDoub'),
                  pdr.place('DrTripl'),pdr.new(f'QDTrip_{name}1',  'QDtr')]
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

    # arc1R included so the extra DS quad is present where matching runs.
    # It reuses the QDDoub element/kQDDoub knob, so matching powers it with
    # the existing doublet quad automatically.
    insert_DS_betay_quads(pdr, ring, period, arc1R)

    _export_lines(pdr, arc1R, cell_arc, cell_tr, period, ring)

    if not matched:
        return pdr

    cell_arc_opt, cell_tr_opt = _match_cells_3fold(pdr, cell_arc, cell_tr,
                                                    mu_cell=phase_advance)
    _run_standard_matching(cell_arc_opt, cell_arc, cell_tr_opt, cell_tr,
                           arc1R, WP, betay_DS_target=betay_DS_target)
    _make_rf_and_finalise(pdr, ring, arc1R, cell_arc_opt, cell_tr_opt,
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
    _run_standard_matching(cell_arc_opt, cell_arc, cell_tr_opt, cell_tr,
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
    _run_standard_matching(cell_arc_opt, cell_arc, cell_tr_opt, cell_tr,
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
        _run_standard_matching(cell_arc_opt, cell_arc, cell_tr_opt, cell_tr,
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
    _run_standard_matching(cell_arc_opt, cell_arc, cell_tr_opt, cell_tr,
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
    _run_standard_matching(cell_arc_opt, cell_arc, cell_tr_opt, cell_tr,
                           arc1R, WP, n_periods=4,betay_DS_target=betay_DS_target)

    _insert_rf(pdr, ring, U0, VRF, rf_from='QDDoub_1R_2')
    _finalise(pdr, ring, arc1R, cell_arc, cell_tr, period, bend_edge)
    return pdr