
import sys
import os

# Adds the parent directory to the search path
parent_dir = os.path.abspath('..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import xtrack as xt
import numpy as np
import constants

def matchingWP( qx, qy, cell_arc_opt, cell_arc, arc1R, MakePlot=False ):
    cell_arc_opt.run_jacobian(10)
    cell_arc_tw = cell_arc.twiss( method='4d' )
    
    knobs_to_vary = ['kQFarcM', 'kQDarcM', 'kQFDS', 'kQDDS', 'kQFDoub', 'kQDDoub', 'kQFtr', 'kQDtr']
    
    arc1R_opttune = arc1R.match( method='4d', solve=True, verbose=False,
                 betx=cell_arc_tw.betx[0], alfx=cell_arc_tw.alfx[0],
                 bety=cell_arc_tw.bety[0], alfy=cell_arc_tw.alfy[0],
                 dx=cell_arc_tw.dx[0],     dpx=cell_arc_tw.dpx[0],
            vary=[ xt.VaryList(knobs_to_vary[:2], step=1e-4),
                   xt.VaryList(knobs_to_vary[2:4], step=1e-4),
                   xt.VaryList(knobs_to_vary[4:6], step=1e-4),
                   xt.VaryList(knobs_to_vary[6:], step=1e-4), ],
            targets=[ xt.TargetSet(dx=0, dpx=0, at=xt.END, tol=1.0e-9),
                      xt.TargetSet(mux=qx/6, muy=qy/6, at=xt.END, 
                                   tol=1.0e-9, weight=.1, tag='phase'),
                      xt.TargetSet(alfx=0, alfy=0, at=xt.END, tol=1.0e-9),
                      xt.TargetSet(alfx=0, alfy=0, at='CtrS1_xR1', 
                                   tol=1.0e-9, weight=10.) ])
    arc1R_opttune.run_jacobian(50)
    
    if MakePlot:
       arc1R.twiss( method='4d',
                    betx=cell_arc_tw.betx[0], alfx=cell_arc_tw.alfx[0],
                    bety=cell_arc_tw.bety[0], alfy=cell_arc_tw.alfy[0],
                    dx=cell_arc_tw.dx[0],     dpx=cell_arc_tw.dpx[0],).plot()
       
    optimized_knobs = {knob: arc1R.vars[knob]._value for knob in knobs_to_vary}
    return optimized_knobs

# Matching sequence for sextant with beta at center of straights fixed
def matchingBeta( betxS, betyS, cell_arc_opt, cell_arc, cell_tr_opt, cell_tr, arc1R, MakePlot=False ): #, **kwargs ):
    cell_arc_opt.run_jacobian(10)
    cell_arc_tw = cell_arc.twiss( method='4d' )
    cell_tr_opt.targets[0].value, cell_tr_opt.targets[1].value = betxS, betyS
    cell_tr_opt.run_jacobian(10)
    cell_tr_tw = cell_tr.twiss( method='4d' )
    arc1R_opt = arc1R.match( method='4d', solve=True,
                 betx=cell_arc_tw.betx[0], alfx=cell_arc_tw.alfx[0],
                 bety=cell_arc_tw.bety[0], alfy=cell_arc_tw.alfy[0],
                 dx=cell_arc_tw.dx[0],     dpx=cell_arc_tw.dpx[0],
            vary=[ xt.VaryList(['kQFarcM', 'kQDarcM'], step=1e-4),
                   xt.VaryList(['kQFDS', 'kQDDS'], step=1e-4),
                   xt.VaryList(['kQFDoub', 'kQDDoub'], step=1e-4) ],
            targets=[ xt.TargetSet(dx=0, dpx=0, at=xt.END, tol=1.0e-9),
                      xt.TargetSet(alfx=0, alfy=0, at=xt.END, tol=1.0e-9),
                      xt.TargetSet(betx=cell_tr_tw.betx[0], bety=cell_tr_tw.bety[0],
                                   at=xt.END, tol=1.0e-9) ])
    arc1R_opt.run_jacobian(10)
    if MakePlot:
       arc1R.twiss( method='4d',
                    betx=cell_arc_tw.betx[0], alfx=cell_arc_tw.alfx[0],
                    bety=cell_arc_tw.bety[0], alfy=cell_arc_tw.alfy[0],
                    dx=cell_arc_tw.dx[0],     dpx=cell_arc_tw.dpx[0],).plot() 
       
def three_fold_periodicity_90_deg():
    # %% create environment, parameters and first simple cells & structures

    pdr = xt.Environment()
    pdr.particle_ref = xt.Particles(kinetic_energy0=2.86e9, mass0 = xt.ELECTRON_MASS_EV)

    E0 = constants.E0; VRF = constants.VRF # beam energy in eV and RF voltage (1 cavity) in V

    #Uncomment vars depending on config

    pdr.vars({'l_cell': 3.4,'l_bend': 0.40,'l_bendDS': 0.55,'dl_noben': 0.25, 'l_quad': 0.30,
    'l_drift': '(l_cell - 2*l_bend - 2*l_quad)/4.','dl_drift': -0.1, 'dl_trans': 0.00,
    'l_doub':  0.25, 'l_tripl': 2.7, 'l_trips': 0.40, 'l_sext': 0.20, })

    #Currently for config 1
    '''pdr.vars({ 
    'l_cell':   3.5000,
    'dl_drift':  -0.1500,
    'l_tripl':   2.7000,
    })'''

    pdr.vars({'N_cells_S': 8,'hBarc' : '6.283185307/(6*(2*N_cells_S*l_bend + l_bendDS))',
    'kQFarc': 2.9478, 'kQDarc':-2.9231,'kQFarcM': 2.8846, 'kQDarcM':-2.7567, 
    'kQFDS': 2.8042, 'kQDDS':-2.2858, 'kQFDoub': 3.9170, 'kQDDoub':-2.5190, 
    'kQFtr': 4.4429, 'kQDtr':-2.4723,  } )

    U0 = (0.88463e-31)*E0**4*(2.*np.pi)/(6*(2*pdr['N_cells_S']*pdr['l_bend'] + pdr['l_bendDS']) )

    pdr.new('Bend', xt.Bend, length='l_bend', angle='hBarc*l_bend', k0_from_h=True, 
            edge_entry_angle='hBarc*l_bend/2', edge_exit_angle='hBarc*l_bend/2', edge_entry_model='full', edge_exit_model='full')
    pdr.new('QFarc',  xt.Quadrupole, length='l_quad',    k1='kQFarc', edge_entry_active=True, edge_exit_active=True)
    pdr.new('QFarcH', xt.Quadrupole, length='l_quad/2.', k1='kQFarc', edge_entry_active=True, edge_exit_active=True)
    pdr.new('Drarc',  xt.Drift,      length='l_drift')
    pdr.new('DrarcS', xt.Drift,      length='l_drift + dl_drift')
    pdr.new('QDarc',  xt.Quadrupole, length='l_quad',    k1='kQDarc', edge_entry_active=True, edge_exit_active=True)
    pdr.new('QFarcM', xt.Quadrupole, length='l_quad',    k1='kQFarcM', edge_entry_active=True, edge_exit_active=True)
    pdr.new('QDarcM', xt.Quadrupole, length='l_quad',    k1='kQDarcM', edge_entry_active=True, edge_exit_active=True)
    pdr.new('BendDS', xt.Bend, length='l_bendDS', angle='hBarc*l_bendDS', k0_from_h=True, 
            edge_entry_angle='hBarc*l_bendDS/2', edge_exit_angle='hBarc*l_bendDS/2', edge_entry_model='full', edge_exit_model='full')
    pdr.new('QFDS',   xt.Quadrupole, length='l_quad',    k1='kQFDS', edge_entry_active=True, edge_exit_active=True)
    pdr.new('QDDS',   xt.Quadrupole, length='l_quad',    k1='kQDDS', edge_entry_active=True, edge_exit_active=True)
    pdr.new('DrDSL',  xt.Drift,      length='2*l_drift + l_bend + dl_noben')
    pdr.new('DrTrans',xt.Drift,      length='l_drift + dl_trans')
    pdr.new('QFDoub', xt.Quadrupole, length='l_quad',    k1='kQFDoub', edge_entry_active=True, edge_exit_active=True)
    pdr.new('QDDoub', xt.Quadrupole, length='l_quad',    k1='kQDDoub', edge_entry_active=True, edge_exit_active=True)
    pdr.new('DrDoub', xt.Drift,      length='l_doub')

    pdr.new('QFtr',    xt.Quadrupole, length='l_quad',    k1='kQFtr', edge_entry_active=True, edge_exit_active=True)
    pdr.new('QFtrH',   xt.Quadrupole, length='l_quad/2.', k1='kQFtr', edge_entry_active=True, edge_exit_active=True)
    pdr.new('QDtr',    xt.Quadrupole, length='l_quad',    k1='kQDtr', edge_entry_active=True, edge_exit_active=True)
    pdr.new('QDtrH',   xt.Quadrupole, length='l_quad',    k1='kQDtr', edge_entry_active=True, edge_exit_active=True)
    pdr.new('DrTripl', xt.Drift,      length='l_tripl')
    pdr.new('DrTrips', xt.Drift,      length='l_trips')

    #%%

    # just an arc cell
    cell_arcS = pdr.new_line( length='l_cell', 
        components =[ 
        pdr.new('QF_cell_arc',    'QFarc',at='0*l_drift + 0.5*l_quad + 0.0*l_bend'),
        pdr.new('Bend1_cell_arc', 'Bend', at='1*l_drift + 1.0*l_quad + 0.5*l_bend'),
        pdr.new('QD_cell_arc',    'QDarc',at='2*l_drift + 1.5*l_quad + 1.0*l_bend'),
        pdr.new('Bend2_cell_arc', 'Bend', at='3*l_drift + 2.0*l_quad + 1.5*l_bend'),
        ])
    cell_arcS.survey().plot()
    print(cell_arcS.get_length())
    cell_arc = pdr.new_line( components = [ 
    pdr.new('QF_cell_arcH1',  'QFarcH' ), pdr.place( 'Drarc' ),
    pdr.new('Bend1_cell_arcH','Bend' ),   pdr.place( 'Drarc' ),
    pdr.new('QD_cell_arcH',   'QDarc' ),  pdr.place( 'Drarc' ),
    pdr.new('Bend2_cell_arcH','Bend' ),   pdr.place( 'Drarc' ),
    pdr.new('QF_cell_arcH2',  'QFarcH' ), 
    ])
    cell_arc.survey().plot()
    print(cell_arc.get_length())
    # triplet starting with 
    cell_tr = pdr.new_line(
        components = [
        pdr.new('QF_cell_trH1', 'QFtrH',    at='0*l_trips + 0.25*l_quad + 0.0*l_tripl'),
        pdr.new('QD_cell_tr1',  'QDtr',     at='1*l_trips + 1.00*l_quad + 0.0*l_tripl'),
        pdr.new('Mkr_cell_tr',   xt.Marker, at='1*l_trips + 1.50*l_quad + 0.5*l_tripl'),
        pdr.new('QD_cell_tr2',  'QDtr',     at='1*l_trips + 2.00*l_quad + 1.0*l_tripl'),
        pdr.new('QF_cell_trH2', 'QFtrH',    at='2*l_trips + 2.75*l_quad + 1.0*l_tripl')
        ])

    print(cell_tr.get_length())

    #%%

    def makesextant ( name, fall ):
        comps = [ ]
        for ind in range( pdr['N_cells_S'] - 1 ):  # regular arc cells
            comps = comps + [ pdr.place('Drarc'), pdr.new('Bend1_' + name + str(ind+1), 'Bend')]
            comps = comps + [ pdr.place('Drarc'), pdr.new('QDA_' + name + str(ind+1), 'QDarc')]
            comps = comps + [ pdr.place('Drarc'), pdr.new('Bend2_' + name + str(ind+1), 'Bend')]
            comps = comps + [ pdr.place('Drarc'), pdr.new('QFA_' + name + str(ind+1), 'QFarc')]

        # Replace last QF and add last arc cell (matching quads) "by hand"

        comps[-1] = pdr.new('QFA_M' + name + str(pdr['N_cells_S']-1), 'QFarcM' )
        comps = comps + [pdr.place('Drarc'),pdr.new('Bend1_' + name + str(pdr['N_cells_S']), 'Bend')]
        comps = comps + [pdr.place('Drarc'), pdr.new('QDA_M' + name + str(pdr['N_cells_S']), 'QDarcM')]
        comps = comps + [pdr.place('Drarc'), pdr.new('Bend2_' + name + str(pdr['N_cells_S']), 'Bend')]

        # Add dispersion suppressor
        comps = comps + [pdr.place('DrarcS'), pdr.new('QFDS_' + name, 'QFDS') ]
        comps = comps + [pdr.place('DrDSL'), pdr.new('QDDS_' + name, 'QDDS') ]
        comps = comps + [pdr.place('Drarc'), pdr.new('BendDS_' + name, 'BendDS')]

        # Add doublet matching 
        comps = comps + [pdr.place('DrTrans'), pdr.new('QFDoub_' + name, 'QFDoub')]
        comps = comps + [pdr.place('DrDoub'), pdr.new('QDDoub_' + name, 'QDDoub')]   
        comps = comps + [pdr.place('DrTripl'), pdr.new('QDTrip_' + name + '1', 'QDtr')]  

        if fall == 'symm':
            comps = [ pdr.new('QFA_' + name + 'CH', 'QFarcH' ) ] + comps
            comps = comps + [ pdr.place('DrTrips'), pdr.new('QFTripC_' + name + '2H', 'QFtrH')]

        elif fall == 'right':
            comps = [ pdr.new('QFA_' + name + 'C', 'QFarc'  ) ] + comps + [ pdr.place('DrTrips') ]
        elif fall == 'left':
            comps = comps + [pdr.place('DrTrips'), pdr.new('QFTripC_' + name + '2', 'QFtr') ]
            comps = list( reversed(comps) )

        else:
            print( '=====> choice for sextant version incorrect <=====' )
            quit()

        return pdr.new_line( components = comps )

    #%%

    arc1R = makesextant( 'xR', 'symm' )
    arc1R_sliced = arc1R.select()
    arc1R_sliced.cut_at_s( np.linspace(.05, arc1R.get_length()-.05, int(arc1R.get_length()/.05-.5)) )
    arc1R.insert( pdr.new('CtrS1_xR1', xt.Marker ), at='(l_tripl+l_quad)/2', from_='QDDoub_xR' )

    #%%

    # One period from center of arc to center of arc
    period = makesextant( 'PR', 'symm') + ( -makesextant( 'PL', 'symm') )
    period_sliced = period.select()
    period_sliced.cut_at_s( np.linspace(.05, period.get_length()-.05, int(period.get_length()/.05-.5)) )

    # Full ring without X-poles, RF and wigglers
    ring  = makesextant( '1R', 'right') + makesextant( '2L', 'left') + makesextant( '2R', 'right') + makesextant( '3L', 'left') + makesextant( '3R', 'right') + makesextant( '1L', 'left')
    fRev = 1./(ring.twiss(method='4d').T_rev0)
    fRF  = fRev*round(4.e8/fRev)  # at integer harmonics and close to 400 MHz
    pdr.new('RFCav',  xt.Cavity, length=1.5, frequency=fRF, voltage=VRF, 
            lag=(180/np.pi)*(np.pi - np.arcsin(U0/VRF)) - 1.8 ) 
    ring.insert( pdr.new('RFCav_1', 'RFCav'), at='(l_tripl+l_quad)/2', from_='QDDoub_1R' )
    ring.configure_radiation(model='mean')
    ring.configure_bend_model(edge='full')


    # %% Routine for several matchings in a row for 90 degrees arc cells and given tunes

    # Matching of arc cell
    cell_arc_opt = cell_arc.match( method='4d', solve=True, verbose=False,
        vary=[
            xt.VaryList(['kQFarc', 'kQDarc', 'kQFarcM'], step=1e-4),    ],
        targets=[
            xt.TargetSet(qx=0.25, qy=0.25, tol=1.0e-6, tag='end'), # just twice the same 
            xt.TargetSet(mux=0.25, muy=0.25, at=xt.END, tol=1.0e-6)]  )

    # Triplet cell to chosen betatron functions
    cell_tr_opt = cell_tr.match( method='4d', solve=True,
        vary=[ # use individual Vary commands instead of List for arc cell
            xt.Vary('kQFtr', step=1e-4 ),
            xt.Vary('kQDtr', step=1e-4),    ],
        targets=[
            xt.TargetSet(betx=2.50, bety=2.50, at='Mkr_cell_tr', tol=1.0e-6, tag='betas')] )
        

    matchingWP(*constants.WP_D1,cell_arc_opt, cell_arc, arc1R)

    cell_tr_tw = cell_tr.twiss(method='4d')
    mid_idx = len(cell_tr_tw.betx) // 2
    betxS = cell_tr_tw.betx[mid_idx]
    betyS = cell_tr_tw.bety[mid_idx]
    matchingBeta( betxS, betyS,cell_arc_opt, cell_arc, cell_tr_opt, cell_tr, arc1R, MakePlot=False )

    pdr.lines['arc1R'] = arc1R
    pdr.lines['cell_arc'] = cell_arc
    pdr.lines['cell_tr'] = cell_tr
    pdr.lines['period'] = period
    pdr.lines['ring'] = ring

    return pdr

def three_fold_periodicity_90_deg_many_sext():
    #Pre-defined params
    # create environment, parameters and first simple cells & structures
    #reupload

    pdr = xt.Environment()
    pdr.particle_ref = xt.Particles(kinetic_energy0=2.86e9, mass0 = xt.ELECTRON_MASS_EV)

    E0 = constants.E0; VRF = constants.VRF # beam energy in eV and RF voltage (1 cavity) in V

    pdr.vars({ 
    'l_cell':   3.5000,
    'l_bend':   0.4000,
    'l_bendDS':   0.5500,
    'dl_noben':   0.2500,
    'l_quad':   0.3000,
    'l_drift':   0.5250,
    'dl_drift':  -0.1500,
    'dl_trans':   0.0000,
    'l_doub':   0.2500,
    'l_tripl':   3.0000,
    'l_trips':   0.4000,
    'N_cells_S':   8.0000,
    'hBarc':   0.1507,
    'kQFarc':   2.8579,
    'kQDarc':  -2.8333,
    'kQFarcM':   2.7633,
    'kQDarcM':  -3.1990,
    'kQFDS':   2.8473,
    'kQDDS':  -2.5869,
    'kQFDoub':   3.7011,
    'kQDDoub':  -2.3242,
    'kQFtr':   4.1683,
    'kQDtr':  -2.5106,
    'l_sext':   0.1000,
    'kSF':  80.5236,
    'kSD': -114.8259,})

    pdr.vars({'N_cells_S': 8,'hBarc' : '6.283185307/(6*(2*N_cells_S*l_bend + l_bendDS))',
    'kQFarc': 2.9478, 'kQDarc':-2.9231,'kQFarcM': 2.8846, 'kQDarcM':-2.7567, 
    'kQFDS': 2.8042, 'kQDDS':-2.2858, 'kQFDoub': 3.9170, 'kQDDoub':-2.5190, 
    'kQFtr': 4.4429, 'kQDtr':-2.4723 } )

    U0 = (0.88463e-31)*E0**4*(2.*np.pi)/(6*(2*pdr['N_cells_S']*pdr['l_bend'] + pdr['l_bendDS']) )

    pdr.new('Bend', xt.Bend, length='l_bend', angle='hBarc*l_bend', k0_from_h=True, 
            edge_entry_angle='hBarc*l_bend/2', edge_exit_angle='hBarc*l_bend/2')
    pdr.new('QFarc',  xt.Quadrupole, length='l_quad',    k1='kQFarc' )
    pdr.new('QFarcH', xt.Quadrupole, length='l_quad/2.', k1='kQFarc' )
    pdr.new('Drarc',  xt.Drift,      length='l_drift' )
    pdr.new('DrarcS', xt.Drift,      length='l_drift + dl_drift' )
    pdr.new('QDarc',  xt.Quadrupole, length='l_quad',    k1='kQDarc' )
    pdr.new('QFarcM', xt.Quadrupole, length='l_quad',    k1='kQFarcM' )
    pdr.new('QDarcM', xt.Quadrupole, length='l_quad',    k1='kQDarcM' )
    pdr.new('BendDS', xt.Bend, length='l_bendDS', angle='hBarc*l_bendDS', k0_from_h=True, 
            edge_entry_angle='hBarc*l_bendDS/2', edge_exit_angle='hBarc*l_bendDS/2')
    pdr.new('QFDS',   xt.Quadrupole, length='l_quad',    k1='kQFDS' )
    pdr.new('QDDS',   xt.Quadrupole, length='l_quad',    k1='kQDDS' )
    pdr.new('DrDSL',  xt.Drift,      length='2*l_drift + l_bend + dl_noben' )
    pdr.new('DrTrans',xt.Drift,      length='l_drift + dl_trans' )
    pdr.new('QFDoub', xt.Quadrupole, length='l_quad',    k1='kQFDoub' )
    pdr.new('QDDoub', xt.Quadrupole, length='l_quad',    k1='kQDDoub' )
    pdr.new('DrDoub', xt.Drift,      length='l_doub' )

    pdr.new('QFtr',    xt.Quadrupole, length='l_quad',    k1='kQFtr' )
    pdr.new('QFtrH',   xt.Quadrupole, length='l_quad/2.', k1='kQFtr' )
    pdr.new('QDtr',    xt.Quadrupole, length='l_quad',    k1='kQDtr' )
    pdr.new('QDtrH',   xt.Quadrupole, length='l_quad',    k1='kQDtr' )
    pdr.new('DrTripl', xt.Drift,      length='l_tripl' )
    pdr.new('DrTrips', xt.Drift,      length='l_trips' )

    #Add any new variables and structures here


    pdr.new('Drarc2', xt.Drift, length='(l_drift-l_sext)/2')
    pdr.new('Drarc3', xt.Drift, length='((l_drift-l_sext)/2)+l_sext')
    pdr.new('SFDS', xt.Sextupole, length='l_sext',k2='kSF')
    pdr.new('SDDS', xt.Sextupole, length='l_sext',k2='kSD')


    # just an arc cell
    cell_arcS = pdr.new_line( length='l_cell', 
        components =[ 
        pdr.new('QF_cell_arc',    'QFarc',at='0*l_drift + 0.5*l_quad + 0.0*l_bend'),
        pdr.new('SF_cell_arc','SFDS', at='l_quad+(l_drift-l_sext)/2+0.5*l_sext'),
        pdr.new('Bend1_cell_arc', 'Bend', at='1*l_drift + 1.0*l_quad + 0.5*l_bend'),
        pdr.new('QD_cell_arc',    'QDarc',at='2*l_drift + 1.5*l_quad + 1.0*l_bend'),
        pdr.new('SD_cell_arc','SDDS',at='2*l_drift+2*l_quad+l_bend+l_drift/2'),
        pdr.new('Bend2_cell_arc', 'Bend', at='3*l_drift + 2.0*l_quad + 1.5*l_bend'),
        ])
    cell_arcS.survey().plot()
    print(cell_arcS.get_length())
    cell_arc = pdr.new_line( components = [ 
    pdr.new('QF_cell_arcH1',  'QFarcH' ), pdr.place( 'Drarc2' ),
    pdr.new('SF_cell_arc1', 'SFDS'), pdr.place('Drarc2'),
    pdr.new('Bend1_cell_arcH','Bend' ),   pdr.place( 'Drarc' ),
    pdr.new('QD_cell_arcH',   'QDarc' ),  pdr.place( 'Drarc2' ),
    pdr.new('SD_cell_arc1', 'SDDS'), pdr.place('Drarc2'),
    pdr.new('Bend2_cell_arcH','Bend' ),   pdr.place( 'Drarc' ),
    pdr.new('QF_cell_arcH2',  'QFarcH' ), 
    ])
    cell_arc.survey().plot()
    print(cell_arc.get_length())
    # triplet starting with 
    cell_tr = pdr.new_line(
        components = [
        pdr.new('QF_cell_trH1', 'QFtrH',    at='0*l_trips + 0.25*l_quad + 0.0*l_tripl'),
        pdr.new('QD_cell_tr1',  'QDtr',     at='1*l_trips + 1.00*l_quad + 0.0*l_tripl'),
        pdr.new('Mkr_cell_tr',   xt.Marker, at='1*l_trips + 1.50*l_quad + 0.5*l_tripl'),
        pdr.new('QD_cell_tr2',  'QDtr',     at='1*l_trips + 2.00*l_quad + 1.0*l_tripl'),
        pdr.new('QF_cell_trH2', 'QFtrH',    at='2*l_trips + 2.75*l_quad + 1.0*l_tripl')
        ])

    print(cell_tr.get_length())
    #makesextant function
    def makesextant ( name, fall ):
        comps = [ ]
        comps = comps + [pdr.new('Drarc0_'+name+str(0),'Drarc2'),pdr.new('SFA0_'+name+str(0),'SFDS')]
        for ind in range( pdr['N_cells_S'] - 1 ):  # regular arc cells
            comps = comps + [ pdr.new('Drarc1_'+ name + str(ind+1), 'Drarc2'), pdr.new('Bend1_' + name + str(ind+1), 'Bend')]
            comps = comps + [ pdr.new('Drarc2_'+ name + str(ind+1), 'Drarc'), pdr.new('QDA_' + name + str(ind+1), 'QDarc')]
            comps = comps + [pdr.new('Drarc3_'+name+str(ind+1),'Drarc2'),pdr.new('SDA_'+name+str(ind+1),'SDDS')]
            comps = comps + [ pdr.new('Drarc4_'+ name + str(ind+1), 'Drarc2'), pdr.new('Bend2_' + name + str(ind+1), 'Bend')]
            comps = comps + [ pdr.new('Drarc5_'+ name + str(ind+1), 'Drarc'), pdr.new('QFA_' + name + str(ind), 'QFarc')]
            comps = comps + [pdr.new('Drarc6_'+name+str(ind+1),'Drarc2'),pdr.new('SFA_'+name+str(ind+1),'SFDS')]
        # Replace last QF and add last arc cell (matching quads) "by hand"
        comps[-3] = pdr.new('QFA_M' + name + str(pdr['N_cells_S']), 'QFarcM' )
        comps = comps + [pdr.new('DrarcS1_'+ name + str(ind+1), 'Drarc2'),pdr.new('Bend1_' + name + str(pdr['N_cells_S']), 'Bend')]
        comps = comps + [pdr.new('Drarc7_'+ name + str(ind+1), 'Drarc'), pdr.new('QDA_M' + name + str(pdr['N_cells_S']), 'QDarcM')]
        comps = comps + [pdr.new('Drarc75_'+name+str(ind+1),'Drarc2'),pdr.new('SDAM_'+name+str(ind+1),'SDDS')]
        comps = comps + [pdr.new('Drarc8_'+ name + str(ind+1), 'Drarc2'), pdr.new('Bend2_' + name + str(pdr['N_cells_S']), 'Bend')]
        # Add dispersion suppressor
        comps = comps + [pdr.new('Drarc9_'+ name + str(ind+1), 'Drarc'), pdr.new('QFDS_' + name, 'QFDS') ]
        comps = comps + [pdr.new('DrDSL1_'+ name + str(ind+1), 'DrDSL'), pdr.new('QDDS_' + name, 'QDDS') ]
        comps = comps + [pdr.new('Drarc10_'+ name + str(ind+1), 'Drarc'), pdr.new('BendDS_' + name, 'BendDS')]
        # Add doublet matching 
        comps = comps + [pdr.new('DrTrans1_'+ name + str(ind+1), 'DrTrans'), pdr.new('QFDoub_' + name, 'QFDoub')]
        comps = comps + [pdr.new('DrDoub1_'+ name + str(ind+1), 'DrDoub'), pdr.new('QDDoub_' + name, 'QDDoub')]   
        comps = comps + [pdr.new('DrTripl1_'+ name + str(ind+1), 'DrTripl'), pdr.new('QDTrip_' + name + '1', 'QDtr')]  
        if fall == 'symm':
            comps = [ pdr.new('QFA_' + name + 'CH', 'QFarcH' ) ] + comps
            comps = comps + [ pdr.place('DrTrips'), pdr.new('QFTripC_' + name + '2H', 'QFtrH')]
        elif fall == 'right':
            comps = [ pdr.new('QFA_' + name + 'C', 'QFarc'  ) ] + comps + [ pdr.place('DrTrips') ]
        elif fall == 'left':
            comps = comps + [pdr.place('DrTrips'), pdr.new('QFTripC_' + name + '2', 'QFtr') ]
            comps = list( reversed(comps) )
        else:
            print( '=====> choice for sextant version incorrect <=====' )
            quit()
        return pdr.new_line( components = comps )
    arc1R = makesextant( 'xR', 'symm' )
    arc1R_sliced = arc1R.select()
    arc1R_sliced.cut_at_s( np.linspace(.05, arc1R.get_length()-.05, int(arc1R.get_length()/.05-.5)) )
    arc1R.insert( pdr.new('CtrS1_xR1', xt.Marker ), at='(l_tripl+l_quad)/2', from_='QDDoub_xR' )
    arc1R.survey().plot()

    cell_arc_opt = cell_arc.match( method='4d', solve=True, verbose=False,
        vary=[
            xt.VaryList(['kQFarc', 'kQDarc', 'kQFarcM'], step=1e-4),    ],
        targets=[
            xt.TargetSet(qx=0.25, qy=0.25, tol=1.0e-6, tag='end'), # just twice the same 
            xt.TargetSet(mux=0.25, muy=0.25, at=xt.END, tol=1.0e-6)]  )

    # Triplet cell to chosen betatron functions
    cell_tr_opt = cell_tr.match( method='4d', solve=True,
        vary=[ # use individual Vary commands instead of List for arc cell
            xt.Vary('kQFtr', step=1e-4 ),
            xt.Vary('kQDtr', step=1e-4),    ],
        targets=[
            xt.TargetSet(betx=2.50, bety=2.50, at='Mkr_cell_tr', tol=1.0e-6, tag='betas')] )

    def matchingWP( qx, qy, MakePlot=False ):
        cell_arc_opt.run_jacobian(10)
        cell_arc_tw = cell_arc.twiss( method='4d' )
        arc1R_opttune = arc1R.match( method='4d', solve=True, verbose=False,
                    betx=cell_arc_tw.betx[0], alfx=cell_arc_tw.alfx[0],
                    bety=cell_arc_tw.bety[0], alfy=cell_arc_tw.alfy[0],
                    dx=cell_arc_tw.dx[0],     dpx=cell_arc_tw.dpx[0],
                vary=[ xt.VaryList(['kQFarcM', 'kQDarcM'], step=1e-4),
                    xt.VaryList(['kQFDS', 'kQDDS'], step=1e-4),
                    xt.VaryList(['kQFDoub', 'kQDDoub'], step=1e-4),
                    xt.VaryList(['kQFtr', 'kQDtr'], step=1e-4), ],
                targets=[ xt.TargetSet(dx=0, dpx=0, at=xt.END, tol=1.0e-9),
                        xt.TargetSet(mux=qx/6, muy=qy/6, at=xt.END, 
                                    tol=1.0e-9, weight=.1, tag='phase'),
                        xt.TargetSet(alfx=0, alfy=0, at=xt.END, tol=1.0e-9),
                        xt.TargetSet(alfx=0, alfy=0, at='CtrS1_xR1', 
                                    tol=1.0e-9, weight=10.) ])
        arc1R_opttune.run_jacobian(50)
        if MakePlot:
            arc1R.twiss( method='4d',
                            betx=cell_arc_tw.betx[0], alfx=cell_arc_tw.alfx[0],
                            bety=cell_arc_tw.bety[0], alfy=cell_arc_tw.alfy[0],
                            dx=cell_arc_tw.dx[0],     dpx=cell_arc_tw.dpx[0],).plot()

            
    # Matching sequence for sextant with beta at center of straights fixed
    def matchingBeta( betxS, betyS, MakePlot=False ): #, **kwargs ):
        cell_arc_opt.run_jacobian(10)
        cell_arc_tw = cell_arc.twiss( method='4d' )
        cell_tr_opt.targets[0].value, cell_tr_opt.targets[1].value = betxS, betyS
        cell_tr_opt.run_jacobian(10)
        cell_tr_tw = cell_tr.twiss( method='4d' )
        arc1R_opt = arc1R.match( method='4d', solve=True,
                    betx=cell_arc_tw.betx[0], alfx=cell_arc_tw.alfx[0],
                    bety=cell_arc_tw.bety[0], alfy=cell_arc_tw.alfy[0],
                    dx=cell_arc_tw.dx[0],     dpx=cell_arc_tw.dpx[0],
                vary=[ xt.VaryList(['kQFarcM', 'kQDarcM'], step=1e-4),
                    xt.VaryList(['kQFDS', 'kQDDS'], step=1e-4),
                    xt.VaryList(['kQFDoub', 'kQDDoub'], step=1e-4) ],
                targets=[ xt.TargetSet(dx=0, dpx=0, at=xt.END, tol=1.0e-9),
                        xt.TargetSet(alfx=0, alfy=0, at=xt.END, tol=1.0e-9),
                        xt.TargetSet(betx=cell_tr_tw.betx[0], bety=cell_tr_tw.bety[0],
                                    at=xt.END, tol=1.0e-9) ])
        arc1R_opt.run_jacobian(10)
        if MakePlot:
            arc1R.twiss( method='4d',
                            betx=cell_arc_tw.betx[0], alfx=cell_arc_tw.alfx[0],
                            bety=cell_arc_tw.bety[0], alfy=cell_arc_tw.alfy[0],
                            dx=cell_arc_tw.dx[0],     dpx=cell_arc_tw.dpx[0],).plot() 
        

    # Execute the stabilized sequence
    matchingWP(*constants.WP_D1)

    cell_tr_tw = cell_tr.twiss(method='4d')
    mid_idx = len(cell_tr_tw.betx) // 2
    betxS = cell_tr_tw.betx[mid_idx]
    betyS = cell_tr_tw.bety[mid_idx]
    matchingBeta( betxS, betyS, MakePlot=False )
        

    # One period from center of arc to center of arc

    period = makesextant( 'PR', 'symm') + ( -makesextant( 'PL', 'symm') )

    period_sliced = period.select()
    period_sliced.cut_at_s( np.linspace(.05, period.get_length()-.05, int(period.get_length()/.05-.5)) )

    ring  = makesextant( '1R', 'right') + makesextant( '2L', 'left') + makesextant( '2R', 'right') + makesextant( '3L', 'left') + makesextant( '3R', 'right') + makesextant( '1L', 'left')

    fRev = 1./(ring.twiss(method='4d').T_rev0)
    fRF  = fRev*round(4.e8/fRev)  # at integer harmonics and close to 400 MHz
    pdr.new('RFCav',  xt.Cavity, length=1.5, frequency=fRF, voltage=VRF, 
            lag=(180/np.pi)*(np.pi - np.arcsin(U0/VRF)) - 1.8 ) 
    ring.insert( pdr.new('RFCav_1', 'RFCav'), at='(l_tripl+l_quad)/2', from_='QDDoub_1R' )
    ring.configure_radiation(model='mean')
    ring.configure_bend_model(edge='full')

    ring.match(
        solve=True,
        method='6d',
        vary=xt.VaryList(['kSF', 'kSD'], step=1e-4),
        targets=[
            xt.Target('dqx', 0, tol=1e-4),
            xt.Target('dqy', 0, tol=1e-4),
        ])

    pdr.lines['arc1R'] = arc1R
    pdr.lines['cell_arc'] = cell_arc
    pdr.lines['cell_tr'] = cell_tr
    pdr.lines['period'] = period
    pdr.lines['ring'] = ring

    return pdr


def two_fold_periodicity_90_deg():
    pdr = xt.Environment()
    pdr.particle_ref = xt.Particles(kinetic_energy0=2.86e9, mass0 = xt.ELECTRON_MASS_EV)

    E0 = 2.86e9  
    VRF = 4.0e6  

    pdr.vars({'l_cell': 3,'l_bend': 0.40,'l_bendDS': 0.55,'dl_noben': 0.25, 'l_quad': 0.30,
    'l_drift': '(l_cell - 2*l_bend - 2*l_quad)/4.','dl_drift': -0.1, 'dl_trans': 0.00,
    'l_doub':  0.25, 'l_tripl': 2.7, 'l_trips': 0.40, 'l_sext': 0.20, })

    pdr.vars({'N_cells_S': 10,'hBarc' : '6.283185307/(4*(2*N_cells_S*l_bend + l_bendDS))',
    'kQFarc': 2.9478, 'kQDarc':-2.9231,'kQFarcM': 2.8846, 'kQDarcM':-2.7567, 
    'kQFDS': 2.8042, 'kQDDS':-2.2858, 'kQFDoub': 3.9170, 'kQDDoub':-2.5190, 
    'kQFtr': 4.4429, 'kQDtr':-2.4723,  } )

    U0 = (0.88463e-31)*E0**4*(2.*np.pi)/(4*(2*pdr['N_cells_S']*pdr['l_bend'] + pdr['l_bendDS']) )

    pdr.new('Bend', xt.Bend, length='l_bend', angle='hBarc*l_bend', k0_from_h=True, 
            edge_entry_angle='hBarc*l_bend/2', edge_exit_angle='hBarc*l_bend/2', edge_entry_model='full', edge_exit_model='full')
    pdr.new('Bend_R', xt.Bend, length='l_bend', angle='hBarc*l_bend', k0_from_h=True, 
            edge_entry_angle='hBarc*l_bend/2', edge_exit_angle='hBarc*l_bend/2', edge_entry_model='full', edge_exit_model='full')

    pdr.new('QFarc',  xt.Quadrupole, length='l_quad',    k1='kQFarc', edge_entry_active=True, edge_exit_active=True)
    pdr.new('QFarcH', xt.Quadrupole, length='l_quad/2.', k1='kQFarc', edge_entry_active=True, edge_exit_active=True)
    pdr.new('Drarc',  xt.Drift,      length='l_drift')
    pdr.new('DrarcS', xt.Drift,      length='l_drift + dl_drift')
    pdr.new('QDarc',  xt.Quadrupole, length='l_quad',    k1='kQDarc', edge_entry_active=True, edge_exit_active=True)
    pdr.new('QFarcM', xt.Quadrupole, length='l_quad',    k1='kQFarcM', edge_entry_active=True, edge_exit_active=True)
    pdr.new('QDarcM', xt.Quadrupole, length='l_quad',    k1='kQDarcM', edge_entry_active=True, edge_exit_active=True)

    pdr.new('BendDS', xt.Bend, length='l_bendDS', angle='hBarc*l_bendDS', k0_from_h=True, 
            edge_entry_angle='hBarc*l_bendDS/2', edge_exit_angle='hBarc*l_bendDS/2', edge_entry_model='full', edge_exit_model='full')
    pdr.new('BendDS_R', xt.Bend, length='l_bendDS', angle='hBarc*l_bendDS', k0_from_h=True, 
            edge_entry_angle='hBarc*l_bendDS/2', edge_exit_angle='hBarc*l_bendDS/2', edge_entry_model='full', edge_exit_model='full')

    pdr.new('QFDS',   xt.Quadrupole, length='l_quad',    k1='kQFDS', edge_entry_active=True, edge_exit_active=True)
    pdr.new('QDDS',   xt.Quadrupole, length='l_quad',    k1='kQDDS', edge_entry_active=True, edge_exit_active=True)
    pdr.new('DrDSL',  xt.Drift,      length='2*l_drift + l_bend + dl_noben')
    pdr.new('DrTrans',xt.Drift,      length='l_drift + dl_trans')
    pdr.new('QFDoub', xt.Quadrupole, length='l_quad',    k1='kQFDoub', edge_entry_active=True, edge_exit_active=True)
    pdr.new('QDDoub', xt.Quadrupole, length='l_quad',    k1='kQDDoub', edge_entry_active=True, edge_exit_active=True)
    pdr.new('DrDoub', xt.Drift,      length='l_doub')

    pdr.new('QFtr',    xt.Quadrupole, length='l_quad',    k1='kQFtr', edge_entry_active=True, edge_exit_active=True)
    pdr.new('QFtrH',   xt.Quadrupole, length='l_quad/2.', k1='kQFtr', edge_entry_active=True, edge_exit_active=True)
    pdr.new('QDtr',    xt.Quadrupole, length='l_quad',    k1='kQDtr', edge_entry_active=True, edge_exit_active=True)
    pdr.new('QDtrH',   xt.Quadrupole, length='l_quad',    k1='kQDtr', edge_entry_active=True, edge_exit_active=True)
    pdr.new('DrTripl', xt.Drift,      length='l_tripl')
    pdr.new('DrTrips', xt.Drift,      length='l_trips')

    cell_arcS = pdr.new_line( length='l_cell', 
        components =[ 
        pdr.new('QF_cell_arc',    'QFarc',at='0*l_drift + 0.5*l_quad + 0.0*l_bend'),
        pdr.new('Bend1_cell_arc', 'Bend', at='1*l_drift + 1.0*l_quad + 0.5*l_bend'),
        pdr.new('QD_cell_arc',    'QDarc',at='2*l_drift + 1.5*l_quad + 1.0*l_bend'),
        pdr.new('Bend2_cell_arc', 'Bend', at='3*l_drift + 2.0*l_quad + 1.5*l_bend'),
        ])

    cell_arc = pdr.new_line( components = [ 
    pdr.new('QF_cell_arcH1',  'QFarcH' ), pdr.new('Drarc_cell1', 'Drarc'),
    pdr.new('Bend1_cell_arcH','Bend' ),   pdr.new('Drarc_cell2', 'Drarc'),
    pdr.new('QD_cell_arcH',   'QDarc' ),  pdr.new('Drarc_cell3', 'Drarc'),
    pdr.new('Bend2_cell_arcH','Bend' ),   pdr.new('Drarc_cell4', 'Drarc'),
    pdr.new('QF_cell_arcH2',  'QFarcH' ), 
    ])

    cell_tr = pdr.new_line(
        components = [
        pdr.new('QF_cell_trH1', 'QFtrH',    at='0*l_trips + 0.25*l_quad + 0.0*l_tripl'),
        pdr.new('QD_cell_tr1',  'QDtr',     at='1*l_trips + 1.00*l_quad + 0.0*l_tripl'),
        pdr.new('Mkr_cell_tr',   xt.Marker, at='1*l_trips + 1.50*l_quad + 0.5*l_tripl'),
        pdr.new('QD_cell_tr2',  'QDtr',     at='1*l_trips + 2.00*l_quad + 1.0*l_tripl'),
        pdr.new('QF_cell_trH2', 'QFtrH',    at='2*l_trips + 2.75*l_quad + 1.0*l_tripl')
        ])

    def makesextant ( name, fall ):
        comps = [ ]
        is_left = (fall == 'left')
        b_type = 'Bend_R' if is_left else 'Bend'
        bds_type = 'BendDS_R' if is_left else 'BendDS'

        for ind in range( pdr['N_cells_S'] - 1 ):
            comps = comps + [ pdr.new(f'Drarc_{name}_{ind}_1', 'Drarc'), pdr.new('Bend1_' + name + str(ind+1), b_type)]
            comps = comps + [ pdr.new(f'Drarc_{name}_{ind}_2', 'Drarc'), pdr.new('QDA_' + name + str(ind+1), 'QDarc')]
            comps = comps + [ pdr.new(f'Drarc_{name}_{ind}_3', 'Drarc'), pdr.new('Bend2_' + name + str(ind+1), b_type)]
            comps = comps + [ pdr.new(f'Drarc_{name}_{ind}_4', 'Drarc'), pdr.new('QFA_' + name + str(ind+1), 'QFarc')]

        comps[-1] = pdr.new('QFA_M' + name + str(pdr['N_cells_S']-1), xt.Quadrupole, length='l_quad', k1=pdr.vars['kQFarcM'])
        comps = comps + [pdr.new(f'Drarc_{name}_m1', 'Drarc'), pdr.new('Bend1_' + name + str(pdr['N_cells_S']), b_type)]
        comps = comps + [pdr.new(f'Drarc_{name}_m2', 'Drarc'), pdr.new('QDA_M' + name + str(pdr['N_cells_S']), xt.Quadrupole, length='l_quad', k1=pdr.vars['kQDarcM'])]
        comps = comps + [pdr.new(f'Drarc_{name}_m3', 'Drarc'), pdr.new('Bend2_' + name + str(pdr['N_cells_S']), b_type)]

        comps = comps + [pdr.new(f'DrarcS_{name}', 'DrarcS'), pdr.new('QFDS_' + name, xt.Quadrupole, length='l_quad', k1=pdr.vars['kQFDS']) ]
        comps = comps + [pdr.new(f'DrDSL_{name}', 'DrDSL'), pdr.new('QDDS_' + name, xt.Quadrupole, length='l_quad', k1=pdr.vars['kQDDS']) ]
        comps = comps + [pdr.new(f'Drarc_{name}_ds', 'Drarc'), pdr.new('BendDS_' + name, bds_type)]

        comps = comps + [pdr.new(f'DrTrans_{name}', 'DrTrans'), pdr.new('QFDoub_' + name, xt.Quadrupole, length='l_quad', k1=pdr.vars['kQFDoub'])]
        comps = comps + [pdr.new(f'DrDoub_{name}', 'DrDoub'), pdr.new('QDDoub_' + name, xt.Quadrupole, length='l_quad', k1=pdr.vars['kQDDoub'])]   
        comps = comps + [pdr.new(f'DrTripl_{name}', 'DrTripl'), pdr.new('QDTrip_' + name + '1', xt.Quadrupole, length='l_quad', k1=pdr.vars['kQDtr'])]  

        if fall == 'symm':
            comps = [ pdr.new('QFA_' + name + 'CH', 'QFarcH' ) ] + comps
            comps = comps + [ pdr.new(f'DrTrips_{name}', 'DrTrips'), pdr.new('QFTripC_' + name + '2H', 'QFtrH')]
        elif fall == 'right':
            comps = [ pdr.new('QFA_' + name + 'C', 'QFarc'  ) ] + comps + [ pdr.new(f'DrTrips_{name}', 'DrTrips') ]
        elif fall == 'left':
            comps = comps + [pdr.new(f'DrTrips_{name}', 'DrTrips'), pdr.new('QFTripC_' + name + '2', xt.Quadrupole, length='l_quad', k1=pdr.vars['kQFtr']) ]
            comps = list( reversed(comps) )
        return pdr.new_line( components = comps )

    arc1R = makesextant( 'xR', 'symm' )
    arc1R.insert( pdr.new('CtrS1_xR1', xt.Marker ), at='(l_tripl+l_quad)/2', from_='QDDoub_xR' )
    arc1R_sliced = arc1R.select()
    arc1R_sliced.cut_at_s( np.linspace(.05, arc1R.get_length()-.05, int(arc1R.get_length()/.05-.5)) )

    period = makesextant( 'PR', 'symm') + ( -makesextant( 'PL', 'symm') )
    period_sliced = period.select()
    period_sliced.cut_at_s( np.linspace(.05, period.get_length()-.05, int(period.get_length()/.05-.5)) )

    half_ring = makesextant('1R', 'right') + (makesextant('2L', 'left'))
    ring = half_ring + makesextant('2R', 'right') + makesextant('1L', 'left')

    cell_arc_opt = cell_arc.match( method='4d', solve=True, verbose=False,
        vary=[xt.VaryList(['kQFarc', 'kQDarc', 'kQFarcM'], step=1e-4)],
        targets=[xt.TargetSet(qx=0.25, qy=0.25, tol=1.0e-6, tag='end'), 
                xt.TargetSet(mux=0.25, muy=0.25, at=xt.END, tol=1.0e-6)]  )

    cell_tr_opt = cell_tr.match( method='4d', solve=True,
        vary=[xt.Vary('kQFtr', step=1e-4), xt.Vary('kQDtr', step=1e-4)],
        targets=[xt.TargetSet(betx=2.50, bety=2.50, at='Mkr_cell_tr', tol=1.0e-6, tag='betas')] )

    def matchingWP( qx, qy ):
        cell_arc_opt.run_jacobian(10)
        cell_arc_tw = cell_arc.twiss( method='4d' )
        knobs_to_vary = ['kQFarcM', 'kQDarcM', 'kQFDS', 'kQDDS', 'kQFDoub', 'kQDDoub', 'kQFtr', 'kQDtr']
        
        arc1R_opttune = arc1R.match( method='4d', solve=True, verbose=False,
                    betx=cell_arc_tw.betx[0], alfx=cell_arc_tw.alfx[0],
                    bety=cell_arc_tw.bety[0], alfy=cell_arc_tw.alfy[0],
                    dx=cell_arc_tw.dx[0],     dpx=cell_arc_tw.dpx[0],
                vary=[ xt.VaryList(knobs_to_vary[:2], step=1e-4),
                    xt.VaryList(knobs_to_vary[2:4], step=1e-4),
                    xt.VaryList(knobs_to_vary[4:6], step=1e-4),
                    xt.VaryList(knobs_to_vary[6:], step=1e-4), ],
                targets=[ xt.TargetSet(dx=0, dpx=0, at=xt.END, tol=1.0e-9),
                        xt.TargetSet(mux=qx/4, muy=qy/4, at=xt.END, tol=1.0e-9, weight=.1, tag='phase'),
                        xt.TargetSet(alfx=0, alfy=0, at=xt.END, tol=1.0e-9),
                        xt.TargetSet(alfx=0, alfy=0, at='CtrS1_xR1', tol=1.0e-9, weight=10.) ])
        arc1R_opttune.run_jacobian(50)
        return {knob: pdr.vars[knob]._value for knob in knobs_to_vary}
    
    def matchingBeta( betxS, betyS ):
        cell_arc_opt.run_jacobian(10)
        cell_arc_tw = cell_arc.twiss( method='4d' )
        cell_tr_opt.targets[0].value, cell_tr_opt.targets[1].value = betxS, betyS
        cell_tr_opt.run_jacobian(10)
        cell_tr_tw = cell_tr.twiss( method='4d' )
        arc1R_opt = arc1R.match( method='4d', solve=True,
                    betx=cell_arc_tw.betx[0], alfx=cell_arc_tw.alfx[0],
                    bety=cell_arc_tw.bety[0], alfy=cell_arc_tw.alfy[0],
                    dx=cell_arc_tw.dx[0],     dpx=cell_arc_tw.dpx[0],
                vary=[ xt.VaryList(['kQFarcM', 'kQDarcM'], step=1e-4),
                    xt.VaryList(['kQFDS', 'kQDDS'], step=1e-4),
                    xt.VaryList(['kQFDoub', 'kQDDoub'], step=1e-4) ],
                targets=[ xt.TargetSet(dx=0, dpx=0, at=xt.END, tol=1.0e-9),
                        xt.TargetSet(alfx=0, alfy=0, at=xt.END, tol=1.0e-9),
                        xt.TargetSet(betx=cell_tr_tw.betx[0], bety=cell_tr_tw.bety[0], at=xt.END, tol=1.0e-9) ])
        arc1R_opt.run_jacobian(10)

    matchingWP(*constants.WP_D2)

    cell_tr_tw = cell_tr.twiss(method='4d')
    mid_idx = len(cell_tr_tw.betx) // 2
    matchingBeta(cell_tr_tw.betx[mid_idx], cell_tr_tw.bety[mid_idx])

    fRev = 1./(ring.twiss(method='4d').T_rev0)
    fRF  = fRev*round(4.e8/fRev)
    pdr.new('RFCav',  xt.Cavity, length=1.5, frequency=fRF, voltage=VRF, lag=(180/np.pi)*(np.pi - np.arcsin(U0/VRF)) - 1.8 ) 
    ring.insert( pdr.new('RFCav_1', 'RFCav'), at='(l_tripl+l_quad)/2', from_='QDDoub_1R' )
    ring.configure_radiation(model='mean')
    ring.configure_bend_model(edge='full')

    pdr.lines['arc1R'] = arc1R
    pdr.lines['cell_arc'] = cell_arc
    pdr.lines['cell_tr'] = cell_tr
    pdr.lines['period'] = period
    pdr.lines['ring'] = ring

    return pdr