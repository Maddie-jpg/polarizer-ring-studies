#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified version with sppin tracking of file
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
import xobjects           as xo
import numpy              as np
import os
import matplotlib.pyplot  as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit

# %% create environment, parameters and first simple cells & structures

E0 = 2.86e9; VRF = 2.000e6 # beam energy in eV and RF voltage (1 cavity) in V

pdr = xt.Environment()
pdr.particle_ref = xt.Particles(kinetic_energy0=E0, mass0 = xt.ELECTRON_MASS_EV,
                                anomalous_magnetic_moment = 0.001159652181)

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
    
# Routine to generate truncated Gaussians with numpy with rgen a generator
def truncnorm(cut, rgen):
    var = rgen.normal( )
    if abs(var) > cut:
        var = truncnorm(cut, rgen)
    return var

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
fRev = 1./(ring.twiss(method='4d').t_rev0)
fRF  = fRev*round(4.e8/fRev)  # at integer harmonics and close to 400 MHz
pdr.new('RFCav',  xt.Cavity, length=1.5, frequency=fRF, voltage=VRF, 
        lag=(180/np.pi)*(np.pi - np.arcsin(U0/VRF)) - 1.8 ) 
ring.insert( pdr.new('RFCav_1', 'RFCav'), at='(l_tripl+l_quad)/2', from_='QDDoub_1R' )
ring.configure_radiation(model='mean')
ring.configure_bend_model(edge='full')
ring_tw=ring.twiss(method='6d', radiation_integrals=True, eneloss_and_damping=True,
                   spin=True, polarization=True )

# Had to install package pint (with pip) to get rid of an error message
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
# %% Add orbit perturbations to lattice with 12 X-poles per family and per period

ringS3imp = ringS3.select()
ringS3imp.configure_radiation(model='mean')
print(ringS3imp.particle_ref.anomalous_magnetic_moment)
rgen  = np.random.default_rng( 4 )

tab = ringS3imp.get_table(attr=True)
tab_els = tab.rows[tab.element_type == 'Bend'] + tab.rows[tab.element_type == 'Quadrupole'] + tab.rows[tab.element_type == 'Sextupole']

MisAlignrms = .26e-3 # larger gives unstable optics in vertical in some cases!!
for iter in range(10):
    for elem in tab_els.name:
       ringS3imp[elem].shift_x   = MisAlignrms*truncnorm(2.5, rgen)
       ringS3imp[elem].shift_y   = MisAlignrms*truncnorm(2.5, rgen)
       ringS3imp[elem].rot_s_rad = MisAlignrms*truncnorm(2.5, rgen)
       
#   ringS3imp_tw = ringS3imp.twiss(spin=True, polarization_analysis=True, radiation_analysis=True,)
    ringS3imp_tw = ringS3imp.twiss(method='6d', radiation_integrals=True, eneloss_and_damping=True,
                       spin=True, polarization=True )
    print( f'===> Tunes{ringS3imp_tw.qx:8.3f},{ringS3imp_tw.qy:8.3f} and{ringS3imp_tw.spin_tune_fractional:8.4f}' )
    print( f'     Equilibrium polarization {100*ringS3imp_tw.spin_polarization_eq:5.2f} % and {100*ringS3imp_tw.spin_polarization_inf_no_depol:6.2f} with time constant {ringS3imp_tw.spin_t_pol_buildup_s:7.2f} s')
    print( f'     Pol rates:{1000/ringS3imp_tw.spin_t_pol_component_s:7.4f}/ms +{1000/ringS3imp_tw.spin_t_depol_component_s:7.4f}/ms ={1000/ringS3imp_tw.spin_t_pol_buildup_s:7.4f}/ms')
    print( f'     Emittances{ringS3imp_tw.rad_int_eq_gemitt_x:10.3e} m and{ringS3imp_tw.rad_int_eq_gemitt_y:10.3e} m' )
    print( f'     Spin vector:({ringS3imp_tw.spin_x[0]:8.5f},{ringS3imp_tw.spin_y[0]:8.5f}, {ringS3imp_tw.spin_z[0]:8.5f})' )
    print( f'   --> Check {ringS3imp_tw.spin_t_pol_buildup_s:9.5f} = {1/(1/ringS3imp_tw.spin_t_pol_component_s+1/ringS3imp_tw.spin_t_depol_component_s):9.5f}')
    ringS3imp_tw.plot('x  y')
    
# use for the moment only the last version for spin tracking tests!!

print([ringS3imp_tw.method, ringS3imp_tw.radiation_method])
# print(ringS3imp_tw.keys())

# %% looping over several cases of misalignment to get spin build-up properties

MisAlignrms = .26e-3 # larger gives unstable optics in vertical in some cases!!
nPts, nTrns, icTrns, nMchs, fct = 300, 20000, 8000, 10, 0.
rgen  = np.random.default_rng( 17 )

ringS3imp = ringS3.select()
tab = ringS3imp.get_table(attr=True)
tab_els = tab.rows[tab.element_type == 'Bend'] + tab.rows[tab.element_type == 'Quadrupole'] + tab.rows[tab.element_type == 'Sextupole']

for iMch in range(1, nMchs+1):
    for elem in tab_els.name:
        ringS3imp[elem].shift_x   = MisAlignrms*truncnorm(2.5, rgen)
        ringS3imp[elem].shift_y   = MisAlignrms*truncnorm(2.5, rgen)
        ringS3imp[elem].rot_s_rad = MisAlignrms*truncnorm(2.5, rgen)
    ringS3imp.configure_radiation(model='mean')
    ringS3imp_tw = ringS3imp.twiss(method='6d', radiation_integrals=True, eneloss_and_damping=True,
                       spin=True, polarization=True )

    epsx, epsy, epsl    = ringS3imp_tw.eq_gemitt_x, ringS3imp_tw.eq_gemitt_y, ringS3imp_tw.eq_gemitt_zeta  # starting with somewhat large emittances
    betxin, alfxin      = ringS3imp_tw.betx[0], ringS3imp_tw.alfx[0] 
    xin, pxin           = ringS3imp_tw.x[0], ringS3imp_tw.px[0]
    dxin, dpxin         = ringS3imp_tw.dx[0], ringS3imp_tw.dpx[0]
    betyin, alfyin      = ringS3imp_tw.bety[0], ringS3imp_tw.alfy[0] 
    yin, pyin           = ringS3imp_tw.y[0], ringS3imp_tw.py[0]   # should be zero for perfect case
    betlin              = ringS3imp_tw.bets0
    zetain, deltain     = ringS3imp_tw.zeta[0], ringS3imp_tw.delta[0]
    sPxin, sPyin, sPzin = ringS3imp_tw.spin_x[0], ringS3imp_tw.spin_y[0], ringS3imp_tw.spin_z[0]
    dsPddx, dsPddy, dsPddz = ringS3imp_tw.spin_dn_ddelta_x[0], ringS3imp_tw.spin_dn_ddelta_y[0], ringS3imp_tw.spin_dn_ddelta_z[0], 
    Qx, Qy, qSp = ringS3imp_tw.qx, ringS3imp_tw.qy, ringS3imp_tw.spin_tune_fractional
#   From Baier-Katkov-Strakhovenko equation - not taking radiative depolarization into account
    PeqBKS, tauBKS = ringS3imp_tw.spin_polarization_inf_no_depol, ringS3imp_tw.spin_t_pol_component_s
#   From Derbenev-Kondratenko-Mane equation - taking radiative depolarization into account
    PeqDKM, tauDKM = ringS3imp_tw.spin_polarization_eq, ringS3imp_tw.spin_t_pol_buildup_s
    taudepDKM      = ringS3imp_tw.spin_t_depol_component_s  #
    print( f'Check for DKM polarization, depolarization and total time constants: {tauDKM:9.4f} s and {1/(1/tauBKS+1/taudepDKM):9.4f} s')
    
    normat = np.array([
        [(epsx*betxin)**.5,         0,                 0,  0, 0, dxin*(epsl/betlin)**.5],
        [-alfxin*(epsx/betxin)**.5, (epsx/betxin)**.5, 0,  0, 0, dpxin*(epsl/betlin)**.5],
        [0,  0,  (epsx*betyin)**.5,         0,                 0, 0],
        [0,  0,  -alfyin*(epsx/betyin)**.5, (epsx/betyin)**.5, 0, 0],
        [0, 0, 0, 0, (epsl*betlin)**.5, 0                ],
        [0, 0, 0, 0, 0,                 (epsl/betlin)**.5] ])
    
    def makeMacroPart(fct): # not sure wheter this really improves - ignore with fct = 0.
        x, px, y, py, zeta, delta = normat@np.random.randn(6)
        sPx = sPxin + fct*(6.5*px*sPzin + delta*dsPddx)
        sPy = sPyin + fct*(6.5*py*sPzin + delta*dsPddy)
        sPz = sPzin + fct*(-6.5*(px*sPxin + py*sPyin) + delta*dsPddz)
        return x + xin, px + pxin, y + yin, py + pyin, zeta + zetain, delta + deltain, sPx, sPy, sPz

    parts = np.array( [makeMacroPart(fct) for _ in range(nPts) ]).T

    p2 = xt.Particles(kinetic_energy0 = 2860.e6, mass0 = xt.ELECTRON_MASS_EV,
            x      = parts[0], px   = parts[1],
            y      = parts[2], py   = parts[3],
            zeta   = parts[4], delta = parts[5],
            spin_x = parts[6], spin_y = parts[7], spin_z = parts[8] )
    ringS3imp.configure_radiation(model='quantum')
    
#   ringS3imp.discard_tracker()
#   ringS3imp.build_tracker(_context=xo.ContextCpu(omp_num_threads=4) )#, use_prebuilt_kernels=False)
    ringS3imp.configure_spin('auto')
    ringS3imp.track( p2, num_turns = nTrns, turn_by_turn_monitor = True,
               with_progress = True )
    dattr = ringS3imp.record_last_track
 
    fig = plt.figure( figsize = (12, 12) )
#   ax = fig.subplots(2, 2)
    ax = fig.add_subplot(2, 2, 1)
    ax.set_xlabel( 'position (m)' ); ax.set_ylabel( 'closed orbit (mm)' )
    ax.plot(ringS3imp_tw.s, 1000*ringS3imp_tw.x, c='C1')
    ax.plot(ringS3imp_tw.s, 1000*ringS3imp_tw.y, c='C2')
    ax.text( 0.1, 0.92, f'Qx ={ringS3imp_tw.qx:8.3f}, Qy ={ringS3imp_tw.qy:8.3f}, qs ={ringS3imp_tw.spin_tune_fractional:8.4f}', transform=ax.transAxes )
    axb = ax.twinx() # second axis on right side
    axb.set_ylabel( r'$\beta_x$, $\beta_y$ (m)' )
    axb.plot(ringS3imp_tw.s, ringS3imp_tw.betx, c='C0')
    axb.plot(ringS3imp_tw.s, ringS3imp_tw.bety, c='C0')
    ax = fig.add_subplot(2, 2, 2)
    ax.set_xlabel( 'turn number' ); ax.set_ylabel( 'rms size (mm, mrad ..)' )
    ax.plot([iturn for iturn in range(nTrns)], [1000.*np.var(dattr.x.T[iturn])**.5 for iturn in range(nTrns)], c='C0')
    ax.plot([iturn for iturn in range(nTrns)], [1000.*np.var(dattr.px.T[iturn])**.5 for iturn in range(nTrns)], c='C0')
    ax.plot([iturn for iturn in range(nTrns)], [1000.*np.var(dattr.y.T[iturn])**.5 for iturn in range(nTrns)], c='C1')
    ax.plot([iturn for iturn in range(nTrns)], [1000.*np.var(dattr.py.T[iturn])**.5 for iturn in range(nTrns)], c='C1')
    ax.plot([iturn for iturn in range(nTrns)], [40.*np.var(dattr.zeta.T[iturn])**.5 for iturn in range(nTrns)], c='C2')
    ax.plot([iturn for iturn in range(nTrns)], [500.*np.var(dattr.delta.T[iturn])**.5 for iturn in range(nTrns)], c='C2')
    axb = ax.twinx()  # second axis on right side
    axb.set_ylim(0., 1.1)
    axb.set_ylabel('Surviving fraction', color='C5')
    axb.plot( [iTrn for iTrn in range(nTrns)], [dattr.state.T[iTrn].sum()/nPts for iTrn in range(nTrns)], c='C3' )

    polVs = [[dattr.spin_x.T[iturn].sum()/nPts, dattr.spin_y.T[iturn].sum()/nPts, dattr.spin_z.T[iturn].sum()/nPts] 
             for iturn in range(nTrns)]
    pols = [(polV[0]**2 + polV[1]**2 + polV[2]**2)**.5 for polV in polVs ]
    ax = fig.add_subplot(2, 2, (3, 4) )
    ax.set_xlabel( 'turn number' ); ax.set_ylabel( 'Polarization - 1' )
    ax.plot( [iturn for iturn in range(nTrns)], [pol - 1 for pol in pols] )

    def fitfu(x, a, tauinv):
        return a*np.exp(-tauinv*x)
    popt0, pcov0 = curve_fit(fitfu, [iTrn for iTrn in range( nTrns)], pols, bounds=([0.9, 0.], [1.1, .001]) )
    popt1, pcov1 = curve_fit(fitfu, [iTrn for iTrn in range(icTrns, nTrns)], pols[icTrns:], bounds=([0.9, 0.], [1.1, .001]) )
    tauDep0, tauDep1 = ringS3imp_tw.t_rev0/popt0[1], ringS3imp_tw.t_rev0/popt1[1]
    ax.plot( [iturn for iturn in range(nTrns)], [popt0[0]*np.exp(-popt0[1]*iturn) - 1 for iturn in range(nTrns)], c='C1' )
    ax.plot( [iturn for iturn in range(icTrns,nTrns)], [popt1[0]*np.exp(-popt1[1]*iturn) - 1 for iturn in range(icTrns,nTrns)], c='C1' )

    ax.text(0.67, 0.92, r'$P_{BKS}$'+f' ={100.*PeqBKS:6.3f} % and '+r'$\tau_{BKS}$'+f' ={tauBKS:7.2f} s', transform=ax.transAxes)
    ax.text(0.67, 0.85, r'$P_{DKM}$'+f' ={100.*PeqDKM:6.3f} % and '+r'$\tau_{DKM}$'+f' ={tauDKM:7.2f} s', transform=ax.transAxes )
    ax.text(0.64, 0.78, r'with $\tau_{Dep}$' + f' ={tauDep0:7.2f} s:', transform=ax.transAxes )
    ax.text(0.67, 0.73, r'$P_{eq}$'+f' ={100.*PeqBKS/(1+tauBKS/tauDep0):6.3f} % and '+r'$\tau_{Pol}$'+f' ={tauBKS/(1+tauBKS/tauDep0):7.2f} s', transform=ax.transAxes)
    ax.text(0.64, 0.66, r'with $\tau_{Dep}$' + f' ={tauDep1:7.2f} s:', transform=ax.transAxes )
    ax.text(0.67, 0.61, r'$P_{eq}$'+f' ={100.*PeqBKS/(1+tauBKS/tauDep1):6.3f} % and '+r'$\tau_{Pol}$'+f' ={tauBKS/(1+tauBKS/tauDep1):7.2f} s', transform=ax.transAxes)
    fig.savefig( '/Users/ccarli/Documents/FCC/PolarizerRing/Short90DegPer3/DepolSim' + f'{iMch:3d}'.replace(' ','0') )
    
# %%

fig = plt.figure( figsize = (12, 12) )
#   ax = fig.subplots(2, 2)
ax = fig.add_subplot(2, 2, 1)
ax.set_xlabel( 'position (m)' ); ax.set_ylabel( 'closed orbit (mm)' )
ax.plot(ringS3imp_tw.s, 1000*ringS3imp_tw.x, c='C1')
ax.plot(ringS3imp_tw.s, 1000*ringS3imp_tw.y, c='C2')
ax = fig.add_subplot(2, 2, 2)
ax.set_xlabel( 'turn number' ); ax.set_ylabel( 'rms size (mm, mrad ..)' )
ax.plot([iturn for iturn in range(nTrns)], [1000.*np.var(dattr.x.T[iturn])**.5 for iturn in range(nTrns)], c='C0')
ax.plot([iturn for iturn in range(nTrns)], [1000.*np.var(dattr.px.T[iturn])**.5 for iturn in range(nTrns)], c='C0')
ax.plot([iturn for iturn in range(nTrns)], [1000.*np.var(dattr.y.T[iturn])**.5 for iturn in range(nTrns)], c='C1')
ax.plot([iturn for iturn in range(nTrns)], [1000.*np.var(dattr.py.T[iturn])**.5 for iturn in range(nTrns)], c='C1')
ax.plot([iturn for iturn in range(nTrns)], [40.*np.var(dattr.zeta.T[iturn])**.5 for iturn in range(nTrns)], c='C2')
ax.plot([iturn for iturn in range(nTrns)], [500.*np.var(dattr.delta.T[iturn])**.5 for iturn in range(nTrns)], c='C2')
axb = ax.twinx()  # second axis on right side
axb.set_ylim(0., 1.1)
axb.set_ylabel('Surviving fraction', color='C5')
axb.plot( [iTrn for iTrn in range(nTrns)], [dattr.state.T[iTrn].sum()/nPts for iTrn in range(nTrns)], c='C3' )


polVs = [[dattr.spin_x.T[iturn].sum()/nPts, dattr.spin_y.T[iturn].sum()/nPts, dattr.spin_z.T[iturn].sum()/nPts] 
             for iturn in range(nTrns)]
pols = [(polV[0]**2 + polV[1]**2 + polV[2]**2)**.5 for polV in polVs ]
ax = fig.add_subplot(2, 2, (3, 4) )
ax.set_xlabel( 'turn number' ); ax.set_ylabel( 'Polarization - 1' )
ax.plot( [iturn for iturn in range(nTrns)], [pol - 1 for pol in pols] )

def fitfu(x, a, tauinv):
        return a*np.exp(-tauinv*x)
popt0, pcov0 = curve_fit(fitfu, [iTrn for iTrn in range( nTrns)], pols, bounds=([0.9, 0.], [1.1, .001]) )
popt1, pcov1 = curve_fit(fitfu, [iTrn for iTrn in range(4000, nTrns)], pols[4000:], bounds=([0.9, 0.], [1.1, .001]) )
tauDep0, tauDep1 = ringS3imp_tw.t_rev0/popt0[1], ringS3imp_tw.t_rev0/popt1[1]

ax.plot( [iturn for iturn in range(nTrns)], [popt0[0]*np.exp(-popt0[1]*iturn) - 1 for iturn in range(nTrns)], c='C1' )
ax.plot( [iturn for iturn in range(4000,nTrns)], [popt1[0]*np.exp(-popt1[1]*iturn) - 1 for iturn in range(4000,nTrns)], c='C1' )

ax.text(0.65, 0.92, r'$P_{BKS}$'+f' ={100.*PeqBKS:6.3f} % and '+r'$\tau_{BKS}$'+f' ={tauBKS:7.2f} s', transform=ax.transAxes)
ax.text(0.65, 0.85, r'$P_{DKM}$'+f' ={100.*PeqDKM:6.3f}', transform=ax.transAxes)
ax.text(0.62, 0.78, r'with $\tau_{Dep}$' + f' ={tauDep0:7.2f} s:', transform=ax.transAxes )
ax.text(0.65, 0.73, r'$P_{eq}$'+f' ={100.*PeqBKS/(1+tauBKS/tauDep0):6.3f} % and '+r'$\tau_{Pol}$'+f' ={tauBKS/(1+tauBKS/tauDep0):7.2f} s', transform=ax.transAxes)
ax.text(0.62, 0.66, r'with $\tau_{Dep}$' + f' ={tauDep1:7.2f} s:', transform=ax.transAxes )
ax.text(0.65, 0.61, r'$P_{eq}$'+f' ={100.*PeqBKS/(1+tauBKS/tauDep1):6.3f} % and '+r'$\tau_{Pol}$'+f' ={tauBKS/(1+tauBKS/tauDep1):7.2f} s', transform=ax.transAxes)

