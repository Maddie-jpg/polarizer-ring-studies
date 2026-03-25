#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Worked on starting around January 2025



Another polarizer ring design attempt based 90 degrees phase FODO cells in arcs

 - Again three fold periodicity (similar geometry than damping ring in same 

   tunnel?)

 - Adjustement of DS bend length to improve optics

 - "Only" two "triplet waists" per straight section

 - No wigglers (and routines to generate them removed)

 - Three different schemes for chromaticiy correction - the two based on 

   sextupole pairs spaced by 180 degrees phase advance lead to large 

   off-momentum beta-beating!

"""



import xtrack             as xt

import numpy              as np

import os

import matplotlib.pyplot  as plt

import matplotlib.patches as patches



# %% create environment, parameters and first simple cells & structures



pdr = xt.Environment()

pdr.particle_ref = xt.Particles(kinetic_energy0=2.86e9, mass0 = xt.ELECTRON_MASS_EV)



E0 = 2.86e9; VRF = 2.000e6 # beam energy in eV and RF voltage (1 cavity) in V



pdr.vars({'l_cell': 3.40,'l_bend': 0.40,'l_bendDS': 0.55,'dl_noben': 0.25, 'l_quad': 0.30,

  'l_drift': '(l_cell - 2*l_bend - 2*l_quad)/4.','dl_drift': -0.15, 'dl_trans': 0.00,

  'l_doub':  0.25, 'l_tripl': 3.00, 'l_trips': 0.40, 'l_sext': 0.10, })



pdr.vars({'N_cells_S': 8,'hBarc' : '6.283185307/(6*(2*N_cells_S*l_bend + l_bendDS))',

  'kQFarc': 2.9478, 'kQDarc':-2.9231,'kQFarcM': 2.8846, 'kQDarcM':-2.7567, 

  'kQFDS': 2.8042, 'kQDDS':-2.2858, 'kQFDoub': 3.9170, 'kQDDoub':-2.5190, 

  'kQFtr': 4.4429, 'kQDtr':-2.4723,  } )



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



# just an arc cell

cell_arcS = pdr.new_line( length='l_cell', 

    components =[ 

    pdr.new('QF_cell_arc',    'QFarc',at='0*l_drift + 0.5*l_quad + 0.0*l_bend'),

    pdr.new('Bend1_cell_arc', 'Bend', at='1*l_drift + 1.0*l_quad + 0.5*l_bend'),

    pdr.new('QD_cell_arc',    'QDarc',at='2*l_drift + 1.5*l_quad + 1.0*l_bend'),

    pdr.new('Bend2_cell_arc', 'Bend', at='3*l_drift + 2.0*l_quad + 1.5*l_bend'),

    ])



cell_arc = pdr.new_line( components = [ 

   pdr.new('QF_cell_arcH1',  'QFarcH' ), pdr.place( 'Drarc' ),

   pdr.new('Bend1_cell_arcH','Bend' ),   pdr.place( 'Drarc' ),

   pdr.new('QD_cell_arcH',   'QDarc' ),  pdr.place( 'Drarc' ),

   pdr.new('Bend2_cell_arcH','Bend' ),   pdr.place( 'Drarc' ),

   pdr.new('QF_cell_arcH2',  'QFarcH' ), 

   ])



# triplet starting with 

cell_tr = pdr.new_line(

    components = [

    pdr.new('QF_cell_trH1', 'QFtrH',    at='0*l_trips + 0.25*l_quad + 0.0*l_tripl'),

    pdr.new('QD_cell_tr1',  'QDtr',     at='1*l_trips + 1.00*l_quad + 0.0*l_tripl'),

    pdr.new('Mkr_cell_tr',   xt.Marker, at='1*l_trips + 1.50*l_quad + 0.5*l_tripl'),

    pdr.new('QD_cell_tr2',  'QDtr',     at='1*l_trips + 2.00*l_quad + 1.0*l_tripl'),

    pdr.new('QF_cell_trH2', 'QFtrH',    at='2*l_trips + 2.75*l_quad + 1.0*l_tripl')

    ])



# %% define a couple of routines

# routine for a sextant starting with F quad at arc center

def makesextant ( name, fall ):

    comps = [ ]

    for ind in range( pdr['N_cells_S'] - 1 ):  # regular arc cells

       comps = comps + [ pdr.place('Drarc'), pdr.new('Bend1_' + name + str(ind+1), 'Bend')]

       comps = comps + [ pdr.place('Drarc'), pdr.new('QDA_' + name + str(ind+1), 'QDarc')]

       comps = comps + [ pdr.place('Drarc'), pdr.new('Bend2_' + name + str(ind+1), 'Bend')]

       comps = comps + [ pdr.place('Drarc'), pdr.new('QFA_' + name + str(ind+1), 'QFarc')]

    # Replace last QF and add last arc cell (matching quads) "by hand"

    comps[-1] = pdr.new('QFA_M' + name + str(pdr['N_cells_S']-1), 'QFarcM' )

    comps = comps + [pdr.place('DrarcS'),pdr.new('Bend1_' + name + str(pdr['N_cells_S']), 'Bend')]

    comps = comps + [pdr.place('Drarc'), pdr.new('QDA_M' + name + str(pdr['N_cells_S']), 'QDarcM')]

    comps = comps + [pdr.place('Drarc'), pdr.new('Bend2_' + name + str(pdr['N_cells_S']), 'Bend')]

    # Add dispersion suppressor

    comps = comps + [pdr.place('Drarc'), pdr.new('QFDS_' + name, 'QFDS') ]

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



# Routine may evolve to generic lattice plotr routine

def addSketchBL( acc_tw, acc, lims, limy1, limy2, scales, scWs=1. ):

    fig, ax = plt.subplots( 3, 1, figsize=(12, 8), height_ratios=(1.8, 4, 3) )

    axl, axp, axt = ax

    axp.sharex( axl )

    fig.subplots_adjust( hspace=0.0, top=1.0, bottom=0.0, left=.1 )



    axp.set_xlabel( 'Position s [m]' )

    axp.set_ylim( limy1 )

    axp.set_ylabel( r'$\beta_x$, $\beta_y$ [m]' )

      

    indm = int( np.array([(np.sign(lims[-1]-x) + 1)/2 for x in acc_tw.s ]).sum() )

    axp.plot( acc_tw.s[:indm], acc_tw.betx[:indm], color='black', linestyle=(0, (10, 0)) )

    axp.plot( acc_tw.s[:indm], acc_tw.bety[:indm], color='black', linestyle=(0, (6, 4)) )

    axp.plot( acc_tw.s[:indm], scWs*acc_tw.wx_chrom[:indm], color='grey', linestyle=(0, (10, 0)) )

    axp.plot( acc_tw.s[:indm], scWs*acc_tw.wy_chrom[:indm], color='grey', linestyle=(0, (6, 4)) )

    axp2 = axp.twinx()

    axp2.set_ylim( limy2 )

    axp2.set_ylabel( r'$D_x$, $D_y$ [m]' )

    axp2.plot( acc_tw.s[:indm], acc_tw.dx[:indm], color='black', linestyle=(0, (5, 3, 1, 3)) )

    axp2.plot( acc_tw.s[:indm], acc_tw.dy[:indm] )

    axp.set_xlim( lims )

    

    axl.set_ylim( -.30, 1.00 )

    axl.axis( 'off' )

    tab_pan = acc.get_table(attr=True).to_pandas()

    for ind in range( len(tab_pan.T.columns) -1 ):

        if tab_pan['s'][ind] < lims[-1]:

           if tab_pan['element_type'][ind].find('Drift') >= 0:

              axl.plot( [tab_pan['s'][ind], tab_pan['s'][ind+1]], [0, 0], color='black' )

           if tab_pan['element_type'][ind].find('Bend') >= 0:

              axl.add_patch( patches.Rectangle( (tab_pan['s'][ind], -0.08), 

                  tab_pan['s'][ind+1] - tab_pan['s'][ind], 0.16, fill=True, color='tab:blue' ) ) 

           if tab_pan['element_type'][ind].find('Quad') >= 0:

              kstr = scales[0]*tab_pan['k1l'][ind]/tab_pan['length'][ind]

              axl.add_patch( patches.Rectangle( (tab_pan['s'][ind], min(0.0, kstr)),

                  tab_pan['s'][ind+1] - tab_pan['s'][ind], abs(kstr), fill=True, color='tab:orange' ) )        

           if tab_pan['element_type'][ind].find('Sextu') >= 0:

              kstr = scales[1]*tab_pan['k2l'][ind]/tab_pan['length'][ind]

              axl.add_patch( patches.Rectangle( (tab_pan['s'][ind], min(0.0, kstr)),

                  tab_pan['s'][ind+1] - tab_pan['s'][ind], abs(kstr), fill=True, color='tab:green' ) )

    axp.plot( [acc.get_length(), acc.get_length()], limy1, color='black', 

              linestyle=(0, (8, 8)), linewidth=.5 )



    axt.axis( 'off' )

    axt.set_xlim( -.3, 3.8 ); axt.set_ylim( 0, 1 )

    return axt



# Short routine listing line elements with start and end position and expression for Q strength

def MyElsList(acc):

    tab_pan = acc.get_table(attr=True).to_pandas()

    print('  Name        Type      L(m)     sin(m)   sout(m)  driftl  driftr    k1(1/m^2)')

    for ind in range( len(tab_pan.T.columns) - 1 ):

       eltyp = tab_pan['element_type'][ind]

       if eltyp.find('Drift') < 0:

          elnam = tab_pan['name'][ind]

          if ind < 1: sin = '  0.0000'

          else: sin = f"{tab_pan['length'][ind-1]:8.4f}"         

          k1expr = pdr.element_refs[elnam].k1._expr

          if k1expr == None: k1expr = ' None'

          else: k1expr = str(k1expr)[5:-1]

          print(  "  " + elnam.ljust( 12 ) + eltyp.ljust(12)[:8] + f" {tab_pan['length'][ind]:7.4f} " + 

               f"{tab_pan['s'][ind]:8.4f} {tab_pan['s'][ind+1]:8.4f} " + sin +

               f"{tab_pan['length'][ind+1]:8.4f} " + k1expr.ljust(9) + 

               f"= {tab_pan['k1l'][ind]/max(tab_pan['length'][ind],1e-6):7.4f}  " )





# Routine generating graphical and text output describing the ring    

def SpuckParsAus( acc_tw, acc, lims, limy1, limy2, scK1, grname='NoGraph', scWs=1. ):

    axt = addSketchBL( acc_tw, acc, lims, limy1, limy2, scK1, scWs )

    axt.text( 0.0, .7, f'C ={3*acc_tw.circumference:8.4f} m', horizontalalignment='left' )

    axt.text( 1.0, .7, f'T ={3e6*acc_tw.T_rev0:8.4f} us', horizontalalignment='left')

    axt.text( 2.0, .7, r'($Q_x$, $Q_y$)' + f' = ({3*acc_tw.qx:9.5f}, {3*acc_tw.qy:9.5f})', 

        horizontalalignment='left')

    pos=0

    for item in pdr.vars.keys()[2:]:

       print( "'" + item + f"': {pdr[item]:8.4f}," )

       axt.text( pos%4 , .6 - .1*int( pos/4 ),

          "'" + item + f"': {pdr[item]:8.4f},", horizontalalignment='left' )

       pos += 1

    if grname != 'NoGraph': 

       if grname in os.listdir( '/Users/ccarli/Documents/FCC/PolarizerRing/Short90DegPer3'):

          print( ' Error: file ' + grname + ' exists <<<<<<<<<<<================' )

       else:

          fig.savefig( '/Users/ccarli/Documents/FCC/PolarizerRing/Short90DegPer3/' + grname )

    MyElsList(acc)



''' # remainder from testings of the routines

try: # error handling structure to generate arc1R and period only when not yet done

    arc1RFlag

except NameError:

    arc1R = makesextant( 'xR', 'symm' )

    period = makesextant( 'PR', 'symm') + (-makesextant( 'PL', 'symm'))

    period_sliced = period.select()

    print( np.linspace(.05, period_sliced.get_length()-.05, 

                            int( period_sliced.get_length()/.05 - .5 ) ) )

    period_sliced.cut_at_s( np.linspace(.05, period_sliced.get_length()-.05, 

                            int( period_sliced.get_length()/.05 - .5 ) ) )

    arc1RFlag = True



addSketchBL( period_sliced.twiss(method='4d'), arc1R, (0., 38.), (0., 10.), (0., 1.), .08 )

SpuckParsAus( period_sliced.twiss(method='4d'), arc1R, (0., 38.), (0., 10.), (0., 1.), .08, 'NoGraph' )

'''

# %% Generate structures - execute only once otherwise resulting in errors

# Half-period from the arc center (middle of quad) to the straight (middle of arc) center

arc1R = makesextant( 'xR', 'symm' )

arc1R_sliced = arc1R.select()

arc1R_sliced.cut_at_s( np.linspace(.05, arc1R.get_length()-.05, int(arc1R.get_length()/.05-.5)) )

arc1R.insert( pdr.new('CtrS1_xR1', xt.Marker ), at='(l_tripl+l_quad)/2', from_='QDDoub_xR' )



# One period from center of arc to center of arc

period = makesextant( 'PR', 'symm') + ( -makesextant( 'PL', 'symm') )

period_sliced = period.select()

period_sliced.cut_at_s( np.linspace(.05, period.get_length()-.05, int(period.get_length()/.05-.5)) )

SpuckParsAus( period_sliced.twiss(method='4d'), arc1R, (0., 38.), (0., 10.), (0., 1.), [.08, .01], "Standard.pdf" )



# Full ring without X-poles, RF and wigglers

ring  = makesextant( '1R', 'right') + makesextant( '2L', 'left') + makesextant( '2R', 'right') + makesextant( '3L', 'left') + makesextant( '3R', 'right') + makesextant( '1L', 'left')

fRev = 1./(ring.twiss(method='4d').T_rev0)

fRF  = fRev*round(4.e8/fRev)  # at integer harmonics and close to 400 MHz

pdr.new('RFCav',  xt.Cavity, length=1.5, frequency=fRF, voltage=VRF, 

        lag=(180/np.pi)*(np.pi - np.arcsin(U0/VRF)) - 1.8 ) 

ring.insert( pdr.new('RFCav_1', 'RFCav'), at='(l_tripl+l_quad)/2', from_='QDDoub_1R' )

ring.configure_radiation(model='mean')

ring.configure_bend_model(edge='full')

ring_tw=ring.twiss(method='6d', radiation_integrals=True, eneloss_and_damping=True,

                   spin=True, polarization=True )



ring.survey().plot()

ring_tw.plot()

ring_tw.plot('delta 3*(zeta+0.00373) x')



print( f'\nApproximatif harmonics {fRF/fRev:10.5f} and fRF ={fRF:8.5f}' )

# print( ring_tw.keys() )

print( f"  Length of short drifts: {pdr['l_drift']:7.3f} m" )

# print( ring_tw.keys() )

print( f'=>Circumference {ring_tw.circumference:8.4f} and revolution {1e6*ring_tw.T_rev0:8.5f} us' )

print( f'  Energy loss per turn from me {U0:10.2f} and twiss' + 

       f'  {ring_tw.eneloss_turn:10.2f} and diff {ring_tw.eneloss_turn-U0:10.2f}' )

print( f'  Vertical damping time {1/ring_tw.damping_constants_turns[1]:10.3f} and {2*E0/ring_tw.eneloss_turn:10.3f} turns' )

print( f'  Working point{ring_tw.qx:10.5f} and{ring_tw.qy:10.5f}')

print( f'  Equilibrium emmittances{ring_tw.eq_gemitt_x:10.3e},{ring_tw.eq_gemitt_y:10.3e} and{ring_tw.eq_gemitt_zeta:10.3e}'  )

print( f'  Equilibrium polarization{ring_tw.spin_polarization_eq:8.5f} and time {ring_tw.spin_t_pol_buildup_s:10.2f} s')



# %% Routine for several matchings in a row for 90 degrees arc cells and given tunes



# Matching of arc cell

cell_arc_opt = cell_arc.match( method='4d', solve=False, verbose=False,

    vary=[

        xt.VaryList(['kQFarc', 'kQDarc'], step=1e-4),    ],

    targets=[

        xt.TargetSet(qx=0.25, qy=0.25, tol=1.0e-6, tag='end'), # just twice the same 

        xt.TargetSet(mux=0.25, muy=0.25, at=xt.END, tol=1.0e-6)]  )



# Triplet cell to chosen betatron functions

cell_tr_opt = cell_tr.match( method='4d', solve=False,
    vary=[ # use individual Vary commands instead of List for arc cell
        xt.Vary('kQFtr', step=1e-4 ),
        xt.Vary('kQDtr', step=1e-4),    ],
    targets=[
        xt.TargetSet(betx=2.50, bety=2.50, at='Mkr_cell_tr', tol=1.0e-6, tag='betas')] )



# Matching sequence with ring working as (one of the) target(s)

def matchingWP( qx, qy, MakePlot=False ):
    cell_arc_opt.run_jacobian(10)
    cell_arc_tw = cell_arc.twiss( method='4d' )
    arc1R_opttune = arc1R.match( method='4d', solve=False, verbose=False,
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

    arc1R_opt = arc1R.match( method='4d', solve=False,

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



# %% (un-)comment to (not) invert triplet polarity

#pdr.vars( {'kQFarcM': 2.8883, 'kQDarcM':-2.7934, 'kQFDS': 2.8119, 'kQDDS':-2.3188,

#           'kQFDoub': 3.8745, 'kQDDoub':-2.6207, 'kQFtr':-4.4392,'kQDtr': 2.8319 } )

#SpuckParsAus( period_sliced.twiss(method='4d'), arc1R, (0., 38.), (0., 10.), (0., 1.), .08 )



# %% (un-)comment lines below to test (or not) moving working point!!

#pdr.vars['dl_noben'] = 0.20

#matchingWP( 15.60, 14.50 )

#SpuckParsAus( period_sliced.twiss(method='4d'), arc1R, (0., 38.), (0., 10.), (0., 1.), .08 )



# %% (un-)comment to test matchings with slighly changed parameters (geometry)

#pdr.vars( {'kQFtr': 4.4387, 'kQDtr': -2.8317, 'dl_noben': 0.30, 

#           'dl_drift': -0.15, 'dl_trans': -0.10} )

#matchingBeta(2., 2.)

#SpuckParsAus( period_sliced.twiss(method='4d'), arc1R, (0., 38.), (0., 10.), (0., 1.), .08 )

#pdr.vars( {'dl_noben': 0.30, 'dl_drift': -0.0, 'dl_trans': 0.0, 'l_doub': 0.30 } )

#matchingBeta(2., 2.)

#SpuckParsAus( period_sliced.twiss(method='4d'), arc1R, (0., 38.), (0., 10.), (0., 1.), .08 )

#pdr.vars( {'kQFtr': -4.6615, 'kQDtr': 2.9381} )

#SpuckParsAus( period_sliced.twiss(method='4d'), arc1R, (0., 38.), (0., 10.), (0., 1.), .08 )



# %% Start inserting sextupoles - in pairs space by 180 degrees to be different 

# first matching quad is QF in cell 8 about 23.80 m = 7*3.4 m from arc center

pdr.vars( { 'k2XF1arc': 0.00, 'k2XD1arc': 0.00, } )  # Sextupoles - two families defined here

pdr.new('XF1arc',  xt.Sextupole, length='l_sext',    k2='k2XF1arc' )

pdr.new('XD1arc',  xt.Sextupole, length='l_sext',    k2='k2XD1arc' )

 

ringS1 = ring.select()   # should make a copy?

for elem in ([ [el, '-'] for el in ['1R1', '1R3', '2R1', '2R3', '3R1', '3R3'] ] +

             [ [el, '+'] for el in ['1L1', '1L3', '2L1', '2L3', '3L1', '3L3'] ]):

  ringS1.insert( pdr.new('XD1arc_'+elem[0], 'XD1arc'), 

                at=elem[1] + '(l_drift+l_quad)/2', from_='QDA_' + elem[0] )



for elem in ([ [el, '-'] for el in ['1R3', '1R5', '2R3', '2R5', '3R3', '3R5'] ] +

             [ [el, '+'] for el in ['1L3', '1L5', '2L3', '2L5', '3L3', '3L5'] ]):

  ringS1.insert( pdr.new('XF1arc_'+elem[0], 'XF1arc'), 

                at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_' + elem[0] )

MyElsList( ringS1 )

ringS1.survey().plot()



periodS1 = period.select()

for elem in [ ['PR1', '-'], ['PR3', '-'], ['PL1', '+'], ['PL3', '+'] ]:

  periodS1.insert( pdr.new('XD1arc_'+elem[0], 'XD1arc'), 

                at=elem[1] + '(l_drift+l_quad)/2', from_='QDA_' + elem[0] )

for elem in [ ['PR3', '-'], ['PR5', '-'], ['PL3', '+'], ['PL5', '+'] ]:

  periodS1.insert( pdr.new('XF1arc_'+elem[0], 'XF1arc'), 

                at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_' + elem[0] )

MyElsList( periodS1 )

periodS1.survey().plot()



#  Just generate some output

periodS1_sliced = periodS1.select()

periodS1_sliced.cut_at_s( np.linspace(.05, period.get_length()-.05, int(period.get_length()/.05-.5)) )



ringS1_chroma = ringS1.match( method='4d', solve=False,

    vary = [ xt.VaryList(['k2XF1arc', 'k2XD1arc'], step=1e-4 )],

    targets = [ xt.TargetSet(dqx=0, dqy=0, tol=1e-5 ) ]  )



SpuckParsAus( periodS1_sliced.twiss(method='4d', delta0=-.000), periodS1, (0., 38.), (0., 10.), (0., 1.), [.08, .0005], 'NoGraph', 0.05 )

ringS1_chroma.run_jacobian(10)

SpuckParsAus( periodS1_sliced.twiss(method='4d', delta0=-.000), periodS1, (0., 38.), (0., 10.), (0., 1.), [.08, .0005], 'NoGraph', 0.05 )



# %% Sextupole pairs with XF moved by one FODOD on one side 

# first matching quad is QF in cell 8 about 23.80 m = 7*3.4 m from arc center

pdr.vars( { 'k2XF2arc': 0.00, 'k2XD2arc': 0.00, } )  # Sextupoles - two families defined here

pdr.new('XF2arc',  xt.Sextupole, length='l_sext',    k2='k2XF2arc' )

pdr.new('XD2arc',  xt.Sextupole, length='l_sext',    k2='k2XD2arc' )



ringS2 = ring.select()   # should make a copy?

for elem in ([ [el, '-'] for el in ['1R1', '1R3', '2R1', '2R3', '3R1', '3R3'] ] +

             [ [el, '+'] for el in ['1L1', '1L3', '2L1', '2L3', '3L1', '3L3'] ]):

  ringS2.insert( pdr.new('XD2arc_'+elem[0], 'XD2arc'), 

                at=elem[1] + '(l_drift+l_quad)/2', from_='QDA_' + elem[0] )



for elem in ([ [el, '-'] for el in ['1R4', '1R6', '2R4', '2R6', '3R4', '3R6'] ] +

             [ [el, '+'] for el in ['1L3', '1L5', '2L3', '2L5', '3L3', '3L5'] ]):

  ringS2.insert( pdr.new('XF2arc_'+elem[0], 'XF2arc'), 

                at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_' + elem[0] )

MyElsList( ringS2 )

ringS2.survey().plot()



periodS2 = period.select()

for elem in [ ['PR1', '-'], ['PR3', '-'], ['PL1', '+'], ['PL3', '+'] ]:

  periodS2.insert( pdr.new('XD2arc_'+elem[0], 'XD2arc'), 

                at=elem[1] + '(l_drift+l_quad)/2', from_='QDA_' + elem[0] )

for elem in [ ['PR4', '-'], ['PR6', '-'], ['PL3', '+'], ['PL5', '+'] ]:

  periodS2.insert( pdr.new('XF2arc_'+elem[0], 'XF2arc'), 

                at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_' + elem[0] )

MyElsList( periodS2 )

periodS2.survey().plot()



#  Just generate some output

periodS2_sliced = periodS2.select()

periodS2_sliced.cut_at_s( np.linspace(.05, period.get_length()-.05, int(period.get_length()/.05-.5)) )



ringS2_chroma = ringS2.match( method='4d', solve=False,

    vary = [ xt.VaryList(['k2XF2arc', 'k2XD2arc'], step=1e-4 )],

    targets = [ xt.TargetSet(dqx=0, dqy=0, tol=1e-5 ) ]  )



SpuckParsAus( periodS2_sliced.twiss(method='4d', delta0=-.000), periodS2, (0., 38.), (0., 10.), (0., 1.), [.08, .0005], 'NoGraph', 0.05 )

ringS2_chroma.run_jacobian(10)

SpuckParsAus( periodS2_sliced.twiss(method='4d', delta0=-.000), periodS2, (0., 38.), (0., 10.), (0., 1.), [.08, .0005], 'NoGraph', 0.05 )



# %% 12 Sextupoles per family space by 90 degrees 

# first matching quad is QF in cell 8 about 23.80 m = 7*3.4 m from arc center

pdr.vars( { 'k2XF3arc': 0.00, 'k2XD3arc': 0.00, } )  # Sextupoles - two families defined here

pdr.new('XF3arc',  xt.Sextupole, length='l_sext',    k2='k2XF3arc' )

pdr.new('XD3arc',  xt.Sextupole, length='l_sext',    k2='k2XD3arc' )



ringS3 = ring.select()   # should make a copy?

for elem in (['1L6', '1L5', '1L4', '1L3', '1L2', '1L1', '1R1', '1R2', '1R3', '1R4', '1R5', '1R6',

              '2L6', '2L5', '2L4', '2L3', '2L2', '2L1', '2R1', '2R2', '2R3', '2R4', '2R5', '2R6',

              '3L6', '3L5', '3L4', '3L3', '3L2', '3L1', '3R1', '3R2', '3R3', '3R4', '3R5', '3R6',]):

  ringS3.insert( pdr.new('XD3arc_' + elem, 'XD3arc'), 

            at = '(l_drift+l_quad)/2', from_='QDA_' + elem  )



for elem in (['1L6', '1L5', '1L4', '1L3', '1L2', '1L1', '1RC', '1R1', '1R2', '1R3', '1R4', '1R5',

              '2L6', '2L5', '2L4', '2L3', '2L2', '2L1', '2RC', '2R1', '2R2', '2R3', '2R4', '2R5',

              '3L6', '3L5', '3L4', '3L3', '3L2', '3L1', '3RC', '3R1', '3R2', '3R3', '3R4', '3R5' ]):

  ringS3.insert( pdr.new('XF3arc_' + elem, 'XF3arc'), 

            at = '(l_drift+l_quad)/2', from_='QFA_' + elem )

MyElsList( ringS3 )

ringS3.survey().plot()



periodS3 = period.select()

for elem in ['PL6', 'PL5', 'PL4', 'PL3', 'PL2', 'PL1', 'PR1', 'PR2', 'PR3', 'PR4', 'PR5', 'PR6']:

  periodS3.insert( pdr.new('XD3arc_' + elem, 'XD3arc'), 

              at='(l_drift+l_quad)/2', from_='QDA_' + elem )

for elem in ['PL6', 'PL5', 'PL4', 'PL3', 'PL2', 'PL1', 'PRCH', 'PR1', 'PR2', 'PR3', 'PR4', 'PR5']:

  periodS3.insert( pdr.new('XF3arc_' + elem, 'XF3arc'), 

              at ='(l_drift+l_quad)/2', from_='QFA_' + elem )

MyElsList( periodS3 )

periodS3.survey().plot()



#  Just generate some output

periodS3_sliced = periodS3.select()

periodS3_sliced.cut_at_s( np.linspace(.05, period.get_length()-.05, int(period.get_length()/.05-.5)) )



ringS3_chroma = ringS3.match( method='4d', solve=False,

    vary = [ xt.VaryList(['k2XF3arc', 'k2XD3arc'], step=1e-4 )],

    targets = [ xt.TargetSet(dqx=0, dqy=0, tol=1e-8 ) ]  )



SpuckParsAus( periodS3_sliced.twiss(method='4d', delta0=-.000), periodS3, (0., 38.), (0., 10.), (0., 1.), [.08, .0005], 'NoGraph', 0.05 )

ringS3_chroma.run_jacobian(10)

SpuckParsAus( periodS3_sliced.twiss(method='4d', delta0=-.000), periodS3, (0., 38.), (0., 10.), (0., 1.), [.08, .0005], 'NoGraph', 0.05 )



# %%

ringS3_chroma.run_jacobian(10)

ringS3_tw = ringS3.twiss(method='4d')

print( f'Chromaticities {ringS3_tw.dqx:8.5f} and {ringS3_tw.dqx:8.5f}')

pdr.vars( {'k2XF3arc': 0.,} )

ringS3_tw = ringS3.twiss(method='4d')

print( f'Chromaticities {ringS3_tw.dqx:8.5f} and {ringS3_tw.dqy:8.5f}')



quit()

print(--version)

# %% Simple test of tracking with synchrotron radiation - takes some time



ring.configure_radiation(model='mean')

ring_tw=ring.twiss(method='6d', radiation_integrals=True, eneloss_and_damping=True)

#ring_tw.plot()

#print( ring_tw.keys() )

print( f' Energy loss per turn from twiss {ring_tw.eneloss_turn:10.2f} and me {U0:10.2f}')

print( 1/ring_tw.damping_constants_turns )

print( 2*E0/ring_tw.eneloss_turn )



npts = 200

epsx, epsy, epsl = 1.0e-7, 1.0e-7, 1.0e-4  # starting with somewhat large emittances

betxin, alfxin  = ring_tw.betx[0], ring_tw.alfx[0] 

xin, pxin       = ring_tw.x[0], ring_tw.px[0]

dxin, dpxin     = ring_tw.dx[0], ring_tw.dpx[0]

betyin, alfyin  = ring_tw.bety[0], ring_tw.alfy[0] 

yin, pyin       = ring_tw.y[0], ring_tw.py[0]   # should be zero for perfect case

#epsx, epsl      = 2.0*ring_tw.eq_gemitt_x, 2.0*ring_tw.eq_gemitt_zeta

betlin          = ring_tw.bets0

zetain, deltain = ring_tw.zeta[0], ring_tw.delta[0]

print( f'At reference location: D ={dxin:7.4f} m, Dp ={dpxin:7.4f},' +

       f' betx ={betxin:7.4f} m, alfx ={alfxin:7.4f},' + 

       f' betl ={betlin:7.4f} m ')



normat = np.array([

    [(epsx*betxin)**.5,         0,                 0,  0, 0, dxin*(epsl/betlin)**.5],

    [-alfxin*(epsx/betxin)**.5, (epsx/betxin)**.5, 0,  0, 0, dpxin*(epsl/betlin)**.5],

    [0,  0,  (epsx*betyin)**.5,         0,                 0, 0],

    [0,  0,  -alfyin*(epsx/betyin)**.5, (epsx/betyin)**.5, 0, 0],

    [0, 0, 0, 0, (epsl*betlin)**.5, 0                ],

    [0, 0, 0, 0, 0,                 (epsl/betlin)**.5]               

    ])



parts = np.array([

     [xin, pxin, yin, pyin, zetain, deltain] + normat@np.random.randn(6) 

         for ind in range(npts) ]).T

# quit()

p2 = xt.Particles(kinetic_energy0 = 2860.e6, mass0 = xt.ELECTRON_MASS_EV,

                  x    = parts[0], px   = parts[1],

                  y    = parts[2], py   = parts[3],

                  zeta = parts[4], delta = parts[5] )

ring.configure_radiation(model='quantum')

ring.track( p2, num_turns = 6100, turn_by_turn_monitor = True,

            with_progress = True )



data = ring.record_last_track

trnplt = [1, 2000, 4000, 6000]

#trnplt = range( len(data.x.T) ) # superimpose all turns, rendering is slowish

fig = plt.figure( figsize=(14., 4.) )

fig.suptitle( f'Phase Space Plots - particle survival {int(((p2.state + 1)/2).sum()):5d} out of {npts:5d}' )

ax = fig.subplots(1, 3)

fig.subplots_adjust( wspace=0.4 )

ax[0].set_xlabel('x (mm)')

ax[0].set_ylabel("x' (mrad)")

ax[1].set_xlabel('y (mm)')

ax[1].set_ylabel("y' (mrad)")

ax[2].set_xlabel('z (mm)')

ax[2].set_ylabel(r'rel. mom. offset ($10^{-3}$)')

for ind in range( len(trnplt) ):  # data.x.T[ind] is nparray (behaviour of multiplication)!

   print( f' ==> Plotting turn"{trnplt[ind]:5d}')

   ax[0].scatter( 1000*(data.x.T[trnplt[ind]] - 1.*ring_tw.dx[0]*data.delta.T[trnplt[ind]]), 

            1000*(data.px.T[trnplt[ind]] - 1.*ring_tw.dpx[0]*data.delta.T[trnplt[ind]]),

            s=.4, c = f'C{ind:1d}')

   ax[1].scatter( 1000*(data.y.T[trnplt[ind]]), 1000*(data.py.T[trnplt[ind]]), 

            s=.2, c = f'C{ind:1d}' )

   ax[2].scatter(1000*data.zeta.T[trnplt[ind]], 1000*data.delta.T[trnplt[ind]], 

            s=.2, color=f'C{ind:1d}' )



print( [pdr['RFCav_1'].lag, pdr['RFCav_1'].voltage] )



print( f' Particle survival: {int(((p2.state + 1)/2).sum()):5d} out of {npts:5d}' )



# %%

'''

print( ring_tw.keys() )

print( f' Circumf ={ring_tw.circumference:7.4f} m, rev. time ={ring_tw.T_rev0:9.4g} s' )

print( f' Tunes: Qx ={ring_tw.qx:7.4f}, Qy ={ring_tw.qy:7.4f}' )

print(  ' Synchrotron radiation constants:' )

print( f'   Horizontal: \u03B5 ={ring_tw.rad_int_eq_gemitt_x*1e6:10.5f}' +

       f' \u03BCm, \u03C4 ={ring_tw.rad_int_damping_constant_x_s:9.4g} s' ) #+

#       f' = {ring_tw.damping_constants_turns[0]:9.4g} turns')

print( f'   Vertical:   \u03B5 ={ring_tw.rad_int_eq_gemitt_y*1e6:10.5f}' +

       f' \u03BCm, \u03C4 ={ring_tw.rad_int_damping_constant_y_s:9.4g} s' ) #+

#       f' ={ring_tw.damping_constants_turns[1]:9.4g} turns' )



#ring.insert_element( index='',

#      element=monitor,

#      name='monitor' )



# %%

pdr.vars({

   'l_cell' : 5.00,  'l_bend' :1.00,'l_quad' : 0.40,

   'l_drift' : '(l_cell - 2*l_bend - 2*l_quad)/4.', 'l_trans' : 0.60,

   'l_doub' : 0.25,'l_tripl' : 3.00, 'l_trips' : 0.40 })  # drift (short) between quads of a triplet



print( f" kQFarc ={pdr['kQFarc']:7.4f}, kQDarc ={pdr['kQDarc']:7.4f}")

print( f" kQFarcM ={pdr['kQFarcM']:7.4f}, kDarcM ={pdr['kQDarcM']:7.4f}")

print( f" kQFDS ={pdr['kQFDS']:7.4f}, kQDDS ={pdr['kQDDS']:7.4f}")

print( f" kQFDoub ={pdr['kQFDoub']:7.4f}, kQDDoub ={pdr['kQDDoub']:7.4f}")

print( f" kQFtr ={pdr['kQFtr']:7.4f}, kQDtr ={pdr['kQDtr']:7.4f}")



#pdr.vars({'alfBarc' : 2.*np.pi/(6*(2*N_cells_S + 1)),

#   'kQFarc' : 1.30, 'kQDarc' : -1.30, 'kQFarcM' : 1.30, 'kQDarcM' : -1.30,

#   'kQFDS' :  1.30,'kQDDS' : -1.30, 'kQFDoub' : 1.30, 'kQDDoub' : -1.30,

#   'kQFtr' : 1.60, 'kQDtr' : -0.90    })

pdr.vars({'alfBarc' : 2.*np.pi/(6*(2*N_cells_S + 1)),

    'kQFarc': 1.4952, 'kQDarc':-1.4871, 'kQFarcM': 1.5327, 'kDarcM' :-1.2669,

    'kQFDS' : 1.5094, 'kQDDS' :-1.2640, 'kQFDoub': 2.1870, 'kQDDoub':-1.5436,

    'kQFtr' : 3.0406, 'kQDtr' :-1.8991})



ring.twiss(method = '4d').plot()

'''