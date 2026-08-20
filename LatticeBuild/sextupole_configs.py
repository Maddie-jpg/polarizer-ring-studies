"""
Sextupole configs to be paired with linear optics script
Each function should include a description of the configuration


"""

import numpy as np
import xtrack as xt

#-----------------
# Helper functions
#-----------------

def _quad_name(prefix, sext, ref_cell):
    if ref_cell == '8':
        return f'{prefix}_M{sext}{ref_cell}'
    else:
        return f'{prefix}_{sext}{ref_cell}'

def ChromCorrect(ring, pdr, variables, MakePlot=False):
    ring.configure_radiation(model='mean')
    
    
    print("Correcting linear chromaticity to zero...")
    ring.match(
        solve=True,
        method='6d',
        vary=xt.VaryList(variables, step=1e-4),
        targets=[
            xt.Target('dqx', 0, tol=1e-4),
            xt.Target('dqy', 0, tol=1e-4),
        ])

    tw = ring.twiss(method='6d')
    if len(variables)>2:
        curr_x, curr_y = tw.ddqx, tw.ddqy
        last_success_vars = {v: pdr.vars[v]._get_value() for v in variables}

        print(f"Starting iterative correction. Initial ddqx: {curr_x:.4f}, ddqy: {curr_y:.4f}")

        for i in range(1):
            target_x = curr_x * 0.8
            target_y = curr_y * 0.8
            
            if abs(target_x) < 0.5 and abs(target_y) < 0.5:
                target_x, target_y = 0, 0

            try:
                print(f"\nIteration {i+1}: Targeting ddqx={target_x:.4f}, ddqy={target_y:.4f}")   

                opt = ring.match(
                    solve=False,
                    method='6d',
                    vary=xt.VaryList(variables, step=1e-4),
                    targets=[
                        xt.Target('dqx', 0, tol=1e-3),
                        xt.Target('dqy', 0, tol=1e-3),
                        xt.Target('ddqx', target_x, tol=5),
                        xt.Target('ddqy', target_y, tol=5),
                    ])
                
                opt.run_jacobian(n_steps=100)
                tw_check = ring.twiss(method='6d')
                
                if abs(tw_check.ddqx) < abs(curr_x) or abs(tw_check.ddqy) < abs(curr_y):
                
                    last_success_vars = {v: pdr.vars[v]._get_value() for v in variables}
                    curr_x, curr_y = tw_check.ddqx, tw_check.ddqy
                    print(f"Progress made. New ddqx: {curr_x:.4f}, ddqy: {curr_y:.4f}")
                    
                    if target_x == 0 and target_y == 0:
                        break
                else:
                    print("No progress made in this step.")
                    raise ValueError("Stalled")

            except Exception:
                print(f"Convergence failed. Reverting to last successful values and exiting.")
                for var, val in last_success_vars.items():
                    pdr.vars[var] = val
                break
                
    for v in variables:
        print(f"Final {v}: {pdr.vars[v]._get_value():.6f}")
        
    return pdr

def ChromCorrect_ddq(ring, pdr, ksf, ksd,ksf2,ksd2,ddqx_val,ddqy_val,tol_val, MakePlot=False):
        
    opt_chrom = ring.match(
    solve=True,
    method='6d',
    vary=xt.VaryList([ksf, ksd, ksf2, ksd2], step=1e-3),
    targets=[
        xt.Target(dqx=0, tol=1e-4),
        xt.Target(dqy=0, tol=1e-4),
        xt.Target(ddqx=ddqx_val, tol=tol_val),
        xt.Target(ddqy=ddqy_val, tol=tol_val),
    ])
    opt_chrom.target_status()
    opt_chrom.run_jacobian(n_steps=100)
    opt_chrom.target_status()
    # Print the final matched strengths

    print(f"Matched kSF: {pdr.vars[f'{ksf}']._get_value():.6f}")
    print(f"Matched kSD: {pdr.vars[f'{ksd}']._get_value():.6f}")   
    print(f"Matched kSF: {pdr.vars[f'{ksf2}']._get_value():.6f}")
    print(f"Matched kSD: {pdr.vars[f'{ksd2}']._get_value():.6f}")  

#-------------
# Design 1
#-------------

def config_D1_C1(pdr, fringe_fields=True):

    """
    8 SF's and 8 SD's in each half-arc

    """

    ring=pdr.lines['ring']
    print(ring.element_names)
    period=pdr.lines['period']

    pdr.vars( {'l_sext':0.1, 'k2XF2arc': 0.00, 'k2XD2arc': 0.00} )  # Sextupoles - 3 families defined here
    pdr.new('XF2arc',  xt.Sextupole, length='l_sext',    k2='k2XF2arc' , edge_entry_active=fringe_fields, edge_exit_active=fringe_fields)
    pdr.new('XD2arc',  xt.Sextupole, length='l_sext',    k2='k2XD2arc' , edge_entry_active=fringe_fields, edge_exit_active=fringe_fields)


    # Defocusing (QDA)
    for elem in ([ [el, '+'] for el in ['1R1','1R2','1R3','1R4','1R5','1R6','1R7','2R1','2R2','2R3','2R4','2R5','2R6','2R7','3R1','3R2','3R3','3R4','3R5','3R6','3R7'] ] +
                 [ [el, '-'] for el in ['1L1','1L2','1L3','1L4','1L5','1L6','1L7','2L1','2L2','2L3','2L4','2L5','2L6','2L7','3L1','3L2','3L3','3L4','3L5','3L6','3L7'] ]):

        ring.insert( pdr.new('XD2arc_'+elem[0], 'XD2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QDA_' + elem[0] )
        
    for elem in ([[el, '+'] for el in ['1R8','2R8','3R8']]+
                 [[el, '-'] for el in ['1L8','2L8','3L8']]):
        ring.insert( pdr.new('XD2arc_'+elem[0], 'XD2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QDA_M' + elem[0] )
        

    # Focusing (QFA)
    for elem in ([ [el, '+'] for el in ['1R1','1R2','1R3','1R4','1R5','1R6','2R1','2R2','2R3','2R4','2R5','2R6','3R1','3R2','3R3','3R4','3R5','3R6'] ] +
                 [ [el, '-'] for el in ['1L1','1L2','1L3','1L4','1L5','1L6','2L1','2L2','2L3','2L4','2L5','2L6','3L1','3L2','3L3','3L4','3L5','3L6'] ]):

        ring.insert( pdr.new('XF2arc_'+elem[0], 'XF2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_' + elem[0] )


    # Matching quads
    for elem in [['1R7','+'], ['2R7','+'], ['3R7','+'],['1L7','-'], ['2L7','-'], ['3L7','-']]:
        ring.insert( pdr.new('XF2arc_'+elem[0], 'XF2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_M' + elem[0] )


    # Second focusing family
    for elem in [['1RC','+'], ['2RC','+'], ['3RC','+']]:
        ring.insert( pdr.new('XF2arc_'+elem[0], 'XF2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_' + elem[0] )

    print(period.element_names)
    # Defocusing
    for elem in [['PR1','+'], ['PR2','+'],['PR3','+'], ['PR4','+'],['PR5','+'], ['PR6','+'],['PR7','+'], ['PL1','-'], ['PL2','-'], ['PL3','-'], ['PL4','-'], ['PL5','-'], ['PL6','-'], ['PL7','-']]:
            period.insert( pdr.new('XD2arc_'+elem[0], 'XD2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QDA_' + elem[0] )
    for elem in ([[el, '+'] for el in ['PR8']]+
                 [[el, '-'] for el in ['PL8']]):
        period.insert( pdr.new('XD2arc_'+elem[0], 'XD2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QDA_M' + elem[0] )

    # Focusing
    for elem in [['PR1','+'], ['PR2','+'],['PR3','+'], ['PR4','+'],['PR5','+'], ['PR6','+'], ['PL1','-'], ['PL2','-'], ['PL3','-'], ['PL4','-'], ['PL5','-'], ['PL6','-']]:
            period.insert( pdr.new('XF2arc_'+elem[0], 'XF2arc'), 
                    at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_' + elem[0] )
    

    # Matching
    for elem in [['PR7','+'],['PL7','-']]:
            period.insert( pdr.new('XF2arc_'+elem[0], 'XF2arc'), 
                    at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_M' + elem[0] )
    
    for elem in [['PRCH','+'], ['PLCH','-']]:
            period.insert( pdr.new('XF2arc_'+elem[0], 'XF2arc'), 
                    at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_' + elem[0] )
    
    variables=['k2XF2arc', 'k2XD2arc']
    ChromCorrect(ring,pdr, variables, MakePlot=False)

    return pdr

def config_D1_C2(pdr):

    """
    within one period there are 2 pairs of focusing sextupoles, of different families, and 1 pair of defocusing sextupoles

    they are at 180 degree phase advance
    
    """

    ring=pdr.lines['ring']
    print(ring.element_names)
    period=pdr.lines['period']

    pdr.vars( { 'l_sext':0.2, 'k2XF2arc': 0.00, 'k2XD2arc': 0.00, 'k2XF2arc2': 0.00 } )  # Sextupoles - 3 families defined here
    pdr.new('XF2arc',  xt.Sextupole, length='l_sext',    k2='k2XF2arc' , edge_entry_active=True, edge_exit_active=True)
    pdr.new('XD2arc',  xt.Sextupole, length='l_sext',    k2='k2XD2arc' , edge_entry_active=True, edge_exit_active=True)
    pdr.new('XF2arc2',  xt.Sextupole, length='l_sext',    k2='k2XF2arc2' , edge_entry_active=True, edge_exit_active=True)

    for elem in ([ [el, '+'] for el in ['1R2', '1R4', '2R2', '2R4', '3R2', '3R4'] ] +
                [ [el, '-'] for el in ['1L2', '1L4', '2L2', '2L4', '3L2', '3L4'] ]):
        ring.insert( pdr.new('XD2arc_'+elem[0], 'XD2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QDA_' + elem[0] )



    for elem in ([ [el, '+'] for el in ['1R5', '2R5','3R5'] ] +
                [ [el, '-'] for el in ['1L4', '1L6', '2L4', '2L6', '3L4', '3L6'] ]):

        ring.insert( pdr.new('XF2arc_'+elem[0], 'XF2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_' + elem[0] )

    for elem in ([ [el, '+'] for el in ['1R7','2R7', '3R7'] ]):
        ring.insert( pdr.new('XF2arc_'+elem[0], 'XF2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_M' + elem[0] )
    
    for elem in ([[el, '+'] for el in ['1R1','2R1','3R1']]):
        ring.insert( pdr.new('XF2arc2_'+elem[0], 'XF2arc2'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_' + elem[0] )

    for elem in ([[el, '-'] for el in ['1L1','2L1','3L1']]):
        ring.insert( pdr.new('XF2arc2_'+elem[0], 'XF2arc2'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_' + elem[0] )
    
    ring.survey().plot(figsize=(12,12))

    for elem in [ ['PR2', '+'], ['PR4', '+'], ['PL2', '-'], ['PL4', '-'] ]:
        period.insert( pdr.new('XD2arc_'+elem[0], 'XD2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QDA_' + elem[0] )
    for elem in [ ['PR5', '+'], ['PL4', '-'], ['PL6', '-'] ]:
        period.insert( pdr.new('XF2arc_'+elem[0], 'XF2arc'), 
                    at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_' + elem[0] )
    for elem in [ ['PR7', '+']]:
        period.insert( pdr.new('XF2arc_'+elem[0], 'XF2arc'), 
                    at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_M' + elem[0] )

    for elem in [ ['PR1', '+'] ,['PL1', '-']]:
        period.insert( pdr.new('XF2arc2_'+elem[0], 'XF2arc2'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_' + elem[0] )

    '''ringS1_chroma = ring.match( method='6d', solve=True,
    vary = [ xt.VaryList(['k2XF2arc', 'k2XD2arc','k2XF2arc2'], step=1e-4 )],
    targets = [ xt.TargetSet(dqx=0, dqy=0,tol=1e-5 ) ]  )

    ringS1_chroma.run_jacobian(10)'''

    variables=['k2XF2arc', 'k2XD2arc','k2XF2arc2']
    ChromCorrect(ring,pdr, variables, MakePlot=False)

    return pdr

def config_D1_C3(pdr):

    """
    2 sextupoles from each family
    180 degree phase advance

    """
    ring=pdr.lines['ring']
    print(ring.element_names)
    period=pdr.lines['period']



    pdr.vars( {'l_sext':0.1, 'k2XF1arc': 0.00, 'k2XD1arc': 0.00, } )  # Sextupoles - two families defined here
    pdr.new('XF1arc',  xt.Sextupole, length='l_sext',    k2='k2XF1arc' )
    pdr.new('XD1arc',  xt.Sextupole, length='l_sext',    k2='k2XD1arc' )

    for elem in ([ [el, '-'] for el in ['1R1', '1R3', '2R1', '2R3', '3R1', '3R3'] ] +
                [ [el, '+'] for el in ['1L1', '1L3', '2L1', '2L3', '3L1', '3L3'] ]):
        ring.insert( pdr.new('XD1arc_'+elem[0], 'XD1arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QDA_' + elem[0] )


    for elem in ([ [el, '-'] for el in ['1R3', '1R5', '2R3', '2R5', '3R3', '3R5'] ] +
                [ [el, '+'] for el in ['1L3', '1L5', '2L3', '2L5', '3L3', '3L5'] ]):

        ring.insert( pdr.new('XF1arc_'+elem[0], 'XF1arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_' + elem[0] )


    for elem in [ ['PR1', '-'], ['PR3', '-'], ['PL1', '+'], ['PL3', '+'] ]:
        period.insert( pdr.new('XD1arc_'+elem[0], 'XD1arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QDA_' + elem[0] )

    for elem in [ ['PR3', '-'], ['PR5', '-'], ['PL3', '+'], ['PL5', '+'] ]:
        period.insert( pdr.new('XF1arc_'+elem[0], 'XF1arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_' + elem[0] )

    variables=['k2XF1arc', 'k2XD1arc']
    ChromCorrect(ring, pdr,variables, MakePlot=False)

    return pdr
    
def config_D1_C4(pdr):
    """
    Adding 6 sextupoles instead of 8 to each half-arc
    """
    ring=pdr.lines['ring']
    print(ring.element_names)
    period=pdr.lines['period']

    pdr.vars( { 'l_sext':0.1,'k2XF3arc': 0.00, 'k2XD3arc': 0.00, } )  # Sextupoles - two families defined here
    pdr.new('XF3arc',  xt.Sextupole, length='l_sext',    k2='k2XF3arc' )
    pdr.new('XD3arc',  xt.Sextupole, length='l_sext',    k2='k2XD3arc' )
   # should make a copy?
    for elem in (['1L6', '1L5', '1L4', '1L3', '1L2', '1L1', '1R1', '1R2', '1R3', '1R4', '1R5', '1R6',
                '2L6', '2L5', '2L4', '2L3', '2L2', '2L1', '2R1', '2R2', '2R3', '2R4', '2R5', '2R6',
                '3L6', '3L5', '3L4', '3L3', '3L2', '3L1', '3R1', '3R2', '3R3', '3R4', '3R5', '3R6',]):

        ring.insert( pdr.new('XD3arc_' + elem, 'XD3arc'), 
                    at = '(l_drift+l_quad)/2', from_='QDA_' + elem  )

    for elem in (['1L6', '1L5', '1L4', '1L3', '1L2', '1L1', '1RC', '1R1', '1R2', '1R3', '1R4', '1R5',
                '2L6', '2L5', '2L4', '2L3', '2L2', '2L1', '2RC', '2R1', '2R2', '2R3', '2R4', '2R5',
                '3L6', '3L5', '3L4', '3L3', '3L2', '3L1', '3RC', '3R1', '3R2', '3R3', '3R4', '3R5' ]):

        ring.insert( pdr.new('XF3arc_' + elem, 'XF3arc'), 
                    at = '(l_drift+l_quad)/2', from_='QFA_' + elem )

    for elem in ['PL6', 'PL5', 'PL4', 'PL3', 'PL2', 'PL1', 'PR1', 'PR2', 'PR3', 'PR4', 'PR5', 'PR6']:
        period.insert( pdr.new('XD3arc_' + elem, 'XD3arc'), 
                    at='(l_drift+l_quad)/2', from_='QDA_' + elem )

    for elem in ['PL6', 'PL5', 'PL4', 'PL3', 'PL2', 'PL1', 'PRCH', 'PR1', 'PR2', 'PR3', 'PR4', 'PR5']:
        period.insert( pdr.new('XF3arc_' + elem, 'XF3arc'), 
                    at ='(l_drift+l_quad)/2', from_='QFA_' + elem )

    variables=['k2XF3arc', 'k2XD3arc']
    ChromCorrect(ring, pdr, variables, MakePlot=False)

    return pdr

def config_D1_C5(pdr):
    '''
    4 SF 4 SD per half-arc
    '''
    ring=pdr.lines['ring']
    print(ring.element_names)
    period=pdr.lines['period']

    pdr.vars( {'l_sext':0.1, 'k2XF2arc': 0.00, 'k2XD2arc': 0.00, } )  # Sextupoles - two families defined here
    pdr.new('XF2arc',  xt.Sextupole, length='l_sext',    k2='k2XF2arc' )
    pdr.new('XD2arc',  xt.Sextupole, length='l_sext',    k2='k2XD2arc' )

    for elem in ([ [el, '-'] for el in ['1R1', '1R5', '2R1', '2R5', '3R1', '3R5'] ] +
                [ [el, '+'] for el in ['1L1', '1L5', '2L1', '2L5', '3L1', '3L5'] ]):
        ring.insert( pdr.new('XD2arc_'+elem[0], 'XD2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QDA_' + elem[0] )

    for elem in ([ [el, '-'] for el in ['1R3', '1R7', '2R3', '2R7', '3R3', '3R7'] ] +
                [ [el, '+'] for el in ['1L3', '1L7', '2L3', '2L7', '3L3', '3L7'] ]):
        ring.insert( pdr.new('XD2arc_'+elem[0], 'XD2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QDA_' + elem[0] )

    for elem in ([ [el, '-'] for el in ['1R3', '2R3', '3R3'] ] +
                [ [el, '+'] for el in ['1L3', '2L3', '3L3'] ]):

        ring.insert( pdr.new('XF2arc_'+elem[0], 'XF2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_' + elem[0] )
        
    for elem in ([ [el, '-'] for el in ['1R7', '2R7', '3R7'] ] +
                [ [el, '+'] for el in ['1L7', '2L7', '3L7'] ]):

        ring.insert( pdr.new('XF2arc_M'+elem[0], 'XF2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_M' + elem[0] )
        
    for elem in ([ [el, '-'] for el in ['1R1', '1R5', '2R1', '2R5', '3R1', '3R5'] ] +
                [ [el, '+'] for el in ['1L1', '1L5', '2L1', '2L5', '3L1', '3L5'] ]):
        ring.insert( pdr.new('XF2arc_'+elem[0], 'XF2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_' + elem[0] )

   
    for elem in [ ['PR1', '-'], ['PR5', '-'], ['PL1', '+'], ['PL5', '+'] ]:
        period.insert( pdr.new('XD2arc_'+elem[0], 'XD2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QDA_' + elem[0] )
    for elem in [ ['PR3', '-'], ['PR7', '-'], ['PL3', '+'], ['PL7', '+'] ]:
        period.insert( pdr.new('XD2arc_'+elem[0], 'XD2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QDA_' + elem[0] )
    for elem in [ ['PR3', '-'],['PL3', '+'] ]:
        period.insert( pdr.new('XF2arc_'+elem[0], 'XF2arc'), 
                    at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_' + elem[0] )
    for elem in [ ['PR7', '-'],['PL7', '+'] ]:
        period.insert( pdr.new('XF2arc_'+elem[0], 'XF2arc'), 
                    at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_M' + elem[0] )
    for elem in [ ['PR1', '-'], ['PR5', '-'], ['PL1', '+'], ['PL5', '+'] ]:
        period.insert( pdr.new('XF2arc_'+elem[0], 'XF2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_' + elem[0] )

    variables=['k2XF2arc', 'k2XD2arc']
    ChromCorrect(ring,pdr, variables, MakePlot=False)

    return pdr


def config_D1_C6(pdr):

    """
    8 SF's and 8 SD's in each half-arc, 4 families rather than 2

    """

    ring=pdr.lines['ring']
    print(ring.element_names)
    period=pdr.lines['period']

    pdr.vars( {'l_sext':0.1, 'k2XF2arc': 0.00, 'k2XD2arc': 0.00,  'k2XF2arc2': 0.00, 'k2XD2arc2': 0.00} )  # Sextupoles - 3 families defined here
    pdr.new('XF2arc',  xt.Sextupole, length='l_sext',    k2='k2XF2arc' , edge_entry_active=True, edge_exit_active=True)
    pdr.new('XD2arc',  xt.Sextupole, length='l_sext',    k2='k2XD2arc' , edge_entry_active=True, edge_exit_active=True)
    pdr.new('XF2arc2',  xt.Sextupole, length='l_sext',    k2='k2XF2arc2' , edge_entry_active=True, edge_exit_active=True)
    pdr.new('XD2arc2',  xt.Sextupole, length='l_sext',    k2='k2XD2arc2' , edge_entry_active=True, edge_exit_active=True)


    # Defocusing (QDA)
    for elem in ([ [el, '+'] for el in ['1R2','1R4','1R6','2R2','2R4','2R6','3R2','3R4','3R6',] ] +
                 [ [el, '-'] for el in ['1L2','1L4','1L6','2L2','2L4','2L6','3L2','3L4','3L6',] ]):

        ring.insert( pdr.new('XD2arc_'+elem[0], 'XD2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QDA_' + elem[0] )
        
    for elem in ([[el, '+'] for el in ['1R8','2R8','3R8']]+
                 [[el, '-'] for el in ['1L8','2L8','3L8']]):
        ring.insert( pdr.new('XD2arc_'+elem[0], 'XD2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QDA_M' + elem[0] )
        
    for elem in ([ [el, '+'] for el in ['1R1','1R3','1R5','1R7','2R1','2R3','2R5','2R7','3R1','3R3','3R5','3R7'] ] +
                 [ [el, '-'] for el in ['1L1','1L3','1L5','1L7','2L1','2L3','2L5','2L7','3L1','3L3','3L5','3L7'] ]):

        ring.insert( pdr.new('XD2arc2_'+elem[0], 'XD2arc2'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QDA_' + elem[0] )
        
        

    # Focusing (QFA)
    for elem in ([ [el, '+'] for el in ['1R1','1R3','1R5','2R1','2R3','2R5','3R1','3R3','3R5'] ] +
                 [ [el, '-'] for el in ['1L1','1L3','1L5','2L1','2L3','2L5','3L1','3L3','3L5'] ]):

        ring.insert( pdr.new('XF2arc_'+elem[0], 'XF2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_' + elem[0] )


    # Matching quads
    for elem in [['1R7','+'], ['2R7','+'], ['3R7','+'],['1L7','-'], ['2L7','-'], ['3L7','-']]:
        ring.insert( pdr.new('XF2arc_'+elem[0], 'XF2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_M' + elem[0] )

    for elem in ([ [el, '+'] for el in ['1R2','1R3','1R6','2R2','2R4','2R6','3R2','3R4','3R6'] ] +
                 [ [el, '-'] for el in ['1L2','1L4','1L6','2L2','2L4','2L6','3L2','3L4','3L6'] ]):

        ring.insert( pdr.new('XF2arc2_'+elem[0], 'XF2arc2'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_' + elem[0] )
    # Second focusing family
    for elem in [['1RC','+'], ['2RC','+'], ['3RC','+']]:
        ring.insert( pdr.new('XF2arc2_'+elem[0], 'XF2arc2'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_' + elem[0] )

    print(period.element_names)
    # Defocusing
    for elem in [['PR1','+'], ['PR2','+'],['PR3','+'], ['PR4','+'],['PR5','+'], ['PR6','+'],['PR7','+'], ['PL1','-'], ['PL2','-'], ['PL3','-'], ['PL4','-'], ['PL5','-'], ['PL6','-'], ['PL7','-']]:
            period.insert( pdr.new('XD2arc_'+elem[0], 'XD2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QDA_' + elem[0] )
    for elem in ([[el, '+'] for el in ['PR8']]+
                 [[el, '-'] for el in ['PL8']]):
        period.insert( pdr.new('XD2arc_'+elem[0], 'XD2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QDA_M' + elem[0] )

    # Focusing
    for elem in [['PR1','+'], ['PR2','+'],['PR3','+'], ['PR4','+'],['PR5','+'], ['PR6','+'], ['PL1','-'], ['PL2','-'], ['PL3','-'], ['PL4','-'], ['PL5','-'], ['PL6','-']]:
            period.insert( pdr.new('XF2arc_'+elem[0], 'XF2arc'), 
                    at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_' + elem[0] )
    

    # Matching
    for elem in [['PR7','+'],['PL7','-']]:
            period.insert( pdr.new('XF2arc_'+elem[0], 'XF2arc'), 
                    at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_M' + elem[0] )
    
    for elem in [['PRCH','+'], ['PLCH','-']]:
            period.insert( pdr.new('XF2arc_'+elem[0], 'XF2arc'), 
                    at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_' + elem[0] )
    

    variables=['k2XF2arc', 'k2XD2arc', 'k2XF2arc2', 'k2XD2arc2']
    ChromCorrect(ring, pdr,variables, MakePlot=False)

    return pdr




def config_D1_C1_120(pdr, fringe_fields=True):

    def _ordered_quads(tt, prefix, sext):
        """Real, cell-order list of {prefix}_{sext}* quad names for one
        sextant, taken directly from the line's element table -- not
        reconstructed from a guessed numbering scheme. QDA runs 1..7,M8.
        QFA runs C,1..6,M8 (no '7' -- confirmed absent from this lattice).
        L sextants have no 'C' entry; R sextants do."""
        cand = [n for n in tt.name
                if n.startswith(f'{prefix}_{sext}') or n.startswith(f'{prefix}_M{sext}')]
        def key(n):
            tag = n.split('_')[1].replace(sext, '', 1)
            if tag == 'C':
                return -1
            if tag.startswith('M'):
                return 100
            return int(tag)
        return sorted(cand, key=key)

    def _ordered_quads_period(tt, prefix, side):
        """Same idea as _ordered_quads, but for the `period` line's own
        naming: {prefix}_P{side}n for n=1..7, {prefix}_MP{side}8 for the
        outer boundary quad, and {prefix}_P{side}CH for the inner boundary
        quad (QFA only -- QDA has no CH-type entry in this line)."""
        cand = [n for n in tt.name
                if n.startswith(f'{prefix}_P{side}') or n.startswith(f'{prefix}_MP{side}')]
        def key(n):
            tag = n.split('_')[1].replace(f'P{side}', '', 1).replace(f'MP{side}', '', 1)
            if tag == 'CH':
                return -1
            if tag == '':   # the MP{side}8 case: after stripping P{side}, 'M8' remains -> handled below
                return 100
            if tag.startswith('M'):
                return 100
            return int(tag)
        return sorted(cand, key=key)
    
    ring   = pdr.lines['ring']
    period = pdr.lines['period']
    tt = ring.get_table()

    pdr.vars({'l_sext': 0.1, 'k2XF2arc': 0.0, 'k2XD2arc': 0.0})
    pdr.new('XF2arc', xt.Sextupole, length='l_sext', k2='k2XF2arc',
            edge_entry_active=fringe_fields, edge_exit_active=fringe_fields)
    pdr.new('XD2arc', xt.Sextupole, length='l_sext', k2='k2XD2arc',
            edge_entry_active=fringe_fields, edge_exit_active=fringe_fields)

    sextants = [('1R','+'), ('2R','+'), ('3R','+'),
                ('1L','-'), ('2L','-'), ('3L','-')]

    for sext, sign in sextants:
        qda_list = _ordered_quads(tt, 'QDA', sext)   # 8 entries: 1..7, M8
        qfa_list = _ordered_quads(tt, 'QFA', sext)   # 7 or 8 entries

        # Extend to 7 consecutive quads starting at the FIRST position,
        # then shift that window up by one -> positions 1..7.
        qda_window = qda_list[1:8]   # e.g. ['2','3','4','5','6','7','M8'] worth
        qfa_window = qfa_list[1:8]   # for R: ['1',...,'6','M8']; for L: only 6 long (no 'C' to skip)

        for name in qda_window:
            cell = name.split(sext)[-1]
            ring.insert(pdr.new(f'XD2arc_{sext}{cell}', 'XD2arc'),
                        at=sign + '(l_drift+l_quad)/2', from_=name)

        for name in qfa_window:
            cell = name.split(sext)[-1]
            ring.insert(pdr.new(f'XF2arc_{sext}{cell}', 'XF2arc'),
                        at=sign + '(l_drift+l_quad)/2', from_=name)

        tt_period = period.get_table()

    for side, sign in [('R', '+'), ('L', '-')]:
        qda_list = _ordered_quads_period(tt_period, 'QDA', side)   # 8: 1..7, M8
        qfa_list = _ordered_quads_period(tt_period, 'QFA', side)   # 8: CH,1..6,M8

        qda_window = qda_list[1:8]
        qfa_window = qfa_list[1:8]

        for name in qda_window:
            cell = name.replace(f'QDA_P{side}', '').replace(f'QDA_MP{side}', 'M')
            period.insert(pdr.new(f'XD2arc_P{side}{cell}', 'XD2arc'),
                          at=sign + '(l_drift+l_quad)/2', from_=name)

        for name in qfa_window:
            cell = name.replace(f'QFA_P{side}', '').replace(f'QFA_MP{side}', 'M')
            period.insert(pdr.new(f'XF2arc_P{side}{cell}', 'XF2arc'),
                          at=sign + '(l_drift+l_quad)/2', from_=name)

    # Both boundary quads are genuine, independent elements in `period`
    # (unlike `ring`, where the equivalent quad is shared between two
    # sextants) -- so a 15th sextupole per side is unambiguous here.
    period.insert(pdr.new('XF2arc_PRCH', 'XF2arc'),
                  at='+(l_drift+l_quad)/2', from_='QFA_PRCH')
    period.insert(pdr.new('XF2arc_PLCH', 'XF2arc'),
                  at='-(l_drift+l_quad)/2', from_='QFA_PLCH')

    ChromCorrect(ring, pdr, ['k2XF2arc', 'k2XD2arc'])
    return pdr

def config_D1_C7(pdr):

    """
    7 SF's and 7 SD's in each half-arc

    """

    ring=pdr.lines['ring']
    # print(ring.element_names)
    period=pdr.lines['period']

    pdr.vars( {'l_sext':0.1, 'k2XF2arc': 0.00, 'k2XD2arc': 0.00} )  # Sextupoles - 3 families defined here
    pdr.new('XF2arc',  xt.Sextupole, length='l_sext',    k2='k2XF2arc' , edge_entry_active=True, edge_exit_active=True)
    pdr.new('XD2arc',  xt.Sextupole, length='l_sext',    k2='k2XD2arc' , edge_entry_active=True, edge_exit_active=True)


    
    for per in ['1','2','3']:
        
        # Defocusing (QDA)
        for elem in ([ [per+'R'+el, '+'] for el in ['1','2','3','4','5','6','7'] ] +
                     [ [per+'L'+el, '+'] for el in ['1','2','3','4','5','6','7'] ]):
            ring.insert( pdr.new('XD2arc_'+elem[0], 'XD2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QDA_' + elem[0] )
        
        for elem in ([[per+'R'+el, '+'] for el in ['8']]+
                     [[per+'L'+el, '+'] for el in ['8']]):
            ring.insert( pdr.new('XD2arc_'+elem[0], 'XD2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QDA_M' + elem[0] )
        
        # Focusing (QFA)
        for elem in ([ [per+'R'+el, '+'] for el in ['1','2','3','4','5','6'] ] +
                     [ [per+'L'+el, '+'] for el in ['1','2','3','4','5','6'] ]):
            ring.insert( pdr.new('XF2arc_'+elem[0], 'XF2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_' + elem[0] )

        # Matching quads
        for elem in [[per+'R7','+'], [per+'L7','+']]:
            ring.insert( pdr.new('XF2arc_'+elem[0], 'XF2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_M' + elem[0] )

    
    ##################
    print(period.element_names)
    # Defocusing
    for elem in ([ ['PR'+el, '+'] for el in ['1','2','3','4','5','6','7'] ] +
                 [ ['PL'+el, '+'] for el in ['1','2','3','4','5','6','7'] ]):
        print(elem)
        period.insert( pdr.new('XD2arc_'+elem[0], 'XD2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QDA_' + elem[0] )
    # for elem in [ ['PR1','+'], ['PR2','+'],['PR3','+'], ['PR4','+'],['PR5','+'], ['PR6','+'],['PR7','+'], ['PL1','+'], ['PL2','+'], ['PL3','+'], ['PL4','+'], ['PL5','+'], ['PL6','+'], ['PL7','+']]:
    #         period.insert( pdr.new('XD2arc_'+elem[0], 'XD2arc'), 
    #                     at=elem[1] + '(l_drift+l_quad)/2', from_='QDA_' + elem[0] )
    for elem in ([[el, '+'] for el in ['PR8']]+
                 [[el, '+'] for el in ['PL8']]):
        period.insert( pdr.new('XD2arc_'+elem[0], 'XD2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QDA_M' + elem[0] )

    # Focusing
    for elem in ([ ['PR'+el, '+'] for el in ['1','2','3','4','5','6'] ] +
                 [ ['PL'+el, '+'] for el in ['1','2','3','4','5','6'] ]):
    # for elem in [['PR1','+'], ['PR2','+'],['PR3','+'], ['PR4','+'],['PR5','+'], ['PR6','+'], ['PL1','+'], ['PL2','+'], ['PL3','+'], ['PL4','+'], ['PL5','+'], ['PL6','+']]:
        period.insert( pdr.new('XF2arc_'+elem[0], 'XF2arc'), 
                    at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_' + elem[0] )
    

    # Matching
    for elem in [['PR7','+'],['PL7','+']]:
            period.insert( pdr.new('XF2arc_'+elem[0], 'XF2arc'), 
                    at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_M' + elem[0] )
    
  
    ChromCorrect(ring, pdr, ['k2XF2arc', 'k2XD2arc'], MakePlot=False)

    return pdr


def config_D1_C8(pdr):

    """
    within one period there are 2 pairs of defocusing sextupoles, and 1 pair of focusing sextupoles

    they are at 180 degree phase advance
    
    """

    ring=pdr.lines['ring']
    print(ring.element_names)
    period=pdr.lines['period']

    pdr.vars( { 'l_sext':0.2, 'k2XF2arc': 0.00, 'k2XD2arc': 0.00, 'k2XF2arc2': 0.00 } )  # Sextupoles - 3 families defined here
    pdr.new('XF2arc',  xt.Sextupole, length='l_sext',    k2='k2XF2arc' , edge_entry_active=True, edge_exit_active=True)
    pdr.new('XD2arc',  xt.Sextupole, length='l_sext',    k2='k2XD2arc' , edge_entry_active=True, edge_exit_active=True)
    pdr.new('XF2arc2',  xt.Sextupole, length='l_sext',    k2='k2XF2arc2' , edge_entry_active=True, edge_exit_active=True)

    for elem in ([ [el, '+'] for el in ['1R2', '1R4', '2R2', '2R4', '3R2', '3R4'] ] +
                 [ [el, '+'] for el in ['1L2', '1L4', '2L2', '2L4', '3L2', '3L4'] ]):
        ring.insert( pdr.new('XD2arc_'+elem[0], 'XD2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QDA_' + elem[0] )

    for elem in ([ [el, '+'] for el in ['1R5', '2R5', '3R5'] ] +
                 [ [el, '+'] for el in ['1L5', '2L5', '3L5'] ]):
        ring.insert( pdr.new('XF2arc_'+elem[0], 'XF2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_' + elem[0] )

    for elem in ([ [el, '+'] for el in ['1R7', '2R7', '3R7'] ]+
                 [ [el, '+'] for el in ['1L7', '2L7', '3L7'] ]):
        ring.insert( pdr.new('XF2arc_'+elem[0], 'XF2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_M' + elem[0] )
    
    for elem in ([[el, '+'] for el in ['1R1','2R1','3R1']]):
        ring.insert( pdr.new('XF2arc_'+elem[0], 'XF2arc2'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_' + elem[0] )
    for elem in ([[el, '+'] for el in ['1L1','2L1','3L1']]):
        ring.insert( pdr.new('XF2arc_'+elem[0], 'XF2arc2'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_' + elem[0] )
    
    # ring.survey().plot(figsize=(12,12))

    ''
    for elem in [ ['PR2', '+'], ['PR4', '+'], ['PL2', '+'], ['PL4', '+'] ]:
        period.insert( pdr.new('XD2arc_'+elem[0], 'XD2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QDA_' + elem[0] )
    for elem in [ ['PR5', '+'], ['PL4', '+'], ['PL6', '+'] ]:
        period.insert( pdr.new('XF2arc_'+elem[0], 'XF2arc'), 
                    at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_' + elem[0] )
    for elem in [ ['PR7', '+']]:
        period.insert( pdr.new('XF2arc_'+elem[0], 'XF2arc'), 
                    at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_M' + elem[0] )

    for elem in [ ['PR1', '+'] ,['PL1', '+']]:
        period.insert( pdr.new('XF2arc2_'+elem[0], 'XF2arc2'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_' + elem[0] )
    ''


    ringS1_chroma = ring.match( method='6d', solve=True,
    vary = [ xt.VaryList(['k2XF2arc', 'k2XD2arc','k2XF2arc2'], step=1e-4 )],
    targets = [ xt.TargetSet(dqx=0, dqy=0,tol=1e-5 ) ]  )

    ringS1_chroma.run_jacobian(10)


    return pdr


def config_D1_C9(pdr):
    """
    8 SF's and 8 SD's in each half-arc
    """

    ring=pdr.lines['ring']
    period=pdr.lines['period']

    pdr.vars( {'l_sext':0.1, 'k2XFarc': 0.00, 'k2XDarc': 0.00, 'k2XFarc_delta':0.0, 'k2XDarc_delta':0.0} )  # Sextupoles - 3 families defined here
    pdr.new('XF1arc',  xt.Sextupole, length='l_sext',    k2='k2XFarc+k2XFarc_delta' , edge_entry_active=True, edge_exit_active=True)
    pdr.new('XF2arc',  xt.Sextupole, length='l_sext',    k2='k2XFarc-k2XFarc_delta' , edge_entry_active=True, edge_exit_active=True)
    pdr.new('XD1arc',  xt.Sextupole, length='l_sext',    k2='k2XDarc+k2XDarc_delta' , edge_entry_active=True, edge_exit_active=True)
    pdr.new('XD2arc',  xt.Sextupole, length='l_sext',    k2='k2XDarc-+k2XDarc_delta' , edge_entry_active=True, edge_exit_active=True)


    
    for per in ['1','2','3']:
        
        # Defocusing (QDA)
        for elem in ([ [per+'R'+el, '-'] for el in ['1','3','5','7'] ] +
                     [ [per+'L'+el, '+'] for el in ['1','3','5','7'] ]):
            ring.insert( pdr.new('XD1arc_'+elem[0], 'XD1arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QDA_' + elem[0] )
        for elem in ([ [per+'R'+el, '-'] for el in ['2','4','6'] ] +
                     [ [per+'L'+el, '+'] for el in ['2','4','6'] ]):
            ring.insert( pdr.new('XD2arc_'+elem[0], 'XD2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QDA_' + elem[0] )
        for elem in ([[per+'R'+el, '-'] for el in ['8']]+
                     [[per+'L'+el, '+'] for el in ['8']]):
            ring.insert( pdr.new('XD2arc_'+elem[0], 'XD2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QDA_M' + elem[0] )

        # Focusing (QFA)
        for elem in ([ [per+'R'+el, '-'] for el in ['1','3','5'] ] +
                     [ [per+'L'+el, '+'] for el in ['1','3','5'] ]):
            ring.insert( pdr.new('XF1arc_'+elem[0], 'XF1arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_' + elem[0] )
        for elem in ([ [per+'R'+el, '-'] for el in ['2','4','6'] ] +
                     [ [per+'L'+el, '+'] for el in ['2','4','6'] ]):
            ring.insert( pdr.new('XF2arc_'+elem[0], 'XF2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_' + elem[0] )


        # Matching quads
        for elem in [[per+'R7','-'], [per+'L7','+']]:
            ring.insert( pdr.new('XF1arc_'+elem[0], 'XF1arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_M' + elem[0] )

        for elem in [[per+'R','-'], [per+'L','+']]:
            ring.insert( pdr.new('XF2arc_'+elem[0], 'XF2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QFDS_' + elem[0] )
         
    
    ##################
    # Defocusing
    for elem in ([ ['PR'+el, '-'] for el in ['1','3','5','7'] ] +
                 [ ['PL'+el, '+'] for el in ['1','3','5','7'] ]):
        period.insert( pdr.new('XD1arc_'+elem[0], 'XD1arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QDA_' + elem[0] )
    for elem in ([ ['PR'+el, '-'] for el in ['2','4','6'] ] +
                 [ ['PL'+el, '+'] for el in ['2','4','6'] ]):
        period.insert( pdr.new('XD2arc_'+elem[0], 'XD2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QDA_' + elem[0] )
    for elem in ([[el, '-'] for el in ['PR8']]+
                 [[el, '+'] for el in ['PL8']]):
        period.insert( pdr.new('XD2arc_'+elem[0], 'XD2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QDA_M' + elem[0] )

    # Focusing
    for elem in ([ ['PR'+el, '-'] for el in ['1','3','5'] ] +
                 [ ['PL'+el, '+'] for el in ['1','3','5'] ]):
        period.insert( pdr.new('XF1arc_'+elem[0], 'XF1arc'), 
                    at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_' + elem[0] )
    for elem in ([ ['PR'+el, '-'] for el in ['2','4','6'] ] +
                 [ ['PL'+el, '+'] for el in ['2','4','6'] ]):
        period.insert( pdr.new('XF2arc_'+elem[0], 'XF2arc'), 
                    at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_' + elem[0] )
    

    # Matching
    for elem in [['PR7','-'],['PL7','+']]:
            period.insert( pdr.new('XF1arc_'+elem[0], 'XF1arc'), 
                    at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_M' + elem[0] )
    for elem in [['PR','-'],['PL','+']]:
            period.insert( pdr.new('XF2arc_'+elem[0], 'XF2arc'), 
                    at=elem[1] + '(l_drift+l_quad)/2', from_='QFDS_' + elem[0] )
    

    ChromCorrect(ring, pdr, ['k2XFarc', 'k2XDarc'], MakePlot=False)

    
    return pdr



def config_D1_C10(pdr):

    """
    7 SF's and 8 SD's in each half-arc

    """

    ring=pdr.lines['ring']
    period=pdr.lines['period']

    pdr.vars( {'l_sext':0.1, 'k2XF2arc': 0.00, 'k2XD2arc': 0.00} )  # Sextupoles - 3 families defined here
    pdr.new('XF2arc',  xt.Sextupole, length='l_sext',    k2='k2XF2arc' , edge_entry_active=True, edge_exit_active=True)
    pdr.new('XD2arc',  xt.Sextupole, length='l_sext',    k2='k2XD2arc' , edge_entry_active=True, edge_exit_active=True)


    
    for per in ['1','2','3']:
        
        # Defocusing (QDA)
        for elem in ([ [per+'R'+el, '+'] for el in ['1','2','3','4','5','6','7'] ] +
                     [ [per+'L'+el, '+'] for el in ['1','2','3','4','5','6','7'] ]):
            ring.insert( pdr.new('XD2arc_'+elem[0], 'XD2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QDA_' + elem[0] )
        
        for elem in ([[per+'R'+el, '+'] for el in ['8']]+
                     [[per+'L'+el, '+'] for el in ['8']]):
            ring.insert( pdr.new('XD2arc_'+elem[0], 'XD2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QDA_M' + elem[0] )
        
        # Focusing (QFA)
        for elem in ([ [per+'R'+el, '+'] for el in ['1','2','3','4','5','6'] ] +
                     [ [per+'L'+el, '+'] for el in ['1','2','3','4','5','6'] ]):
            ring.insert( pdr.new('XF2arc_'+elem[0], 'XF2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_' + elem[0] )

        # Matching quads
        for elem in [[per+'R7','+'], [per+'L7','+']]:
            ring.insert( pdr.new('XF2arc_'+elem[0], 'XF2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_M' + elem[0] )

        # Central quadrupole
        elem = [per+'RC','+']
        ring.insert( pdr.new('XF2arc_'+elem[0], 'XF2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_' + elem[0] )
 
    
    ##################
    # Defocusing
    for elem in ([ ['PR'+el, '+'] for el in ['1','2','3','4','5','6','7'] ] +
                 [ ['PL'+el, '+'] for el in ['1','2','3','4','5','6','7'] ]):
        period.insert( pdr.new('XD2arc_'+elem[0], 'XD2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QDA_' + elem[0] )
    for elem in ([[el, '+'] for el in ['PR8']]+
                 [[el, '+'] for el in ['PL8']]):
        period.insert( pdr.new('XD2arc_'+elem[0], 'XD2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QDA_M' + elem[0] )

    # Focusing
    for elem in ([ ['PR'+el, '+'] for el in ['1','2','3','4','5','6'] ] +
                 [ ['PL'+el, '+'] for el in ['1','2','3','4','5','6'] ]):
        period.insert( pdr.new('XF2arc_'+elem[0], 'XF2arc'), 
                    at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_' + elem[0] )
    

    # Matching
    for elem in [['PR7','+'],['PL7','+']]:
            period.insert( pdr.new('XF2arc_'+elem[0], 'XF2arc'), 
                    at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_M' + elem[0] )
    
    for elem in [['PRCH','+'], ['PLCH','-']]:
            period.insert( pdr.new('XF2arc_'+elem[0], 'XF2arc'), 
                    at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_' + elem[0] )
    

    ChromCorrect(ring, pdr, ['k2XF2arc', 'k2XD2arc'], MakePlot=False)

    return pdr

#---------------
# Design 2
#---------------

def config_D2_C1(pdr):
    """
    Dynamically maps SF's and SD's to match the exact structural layout
    of the 10-cell ring configuration by tracking boundary quads correctly.
    """
    ring = pdr.lines['ring']
    period = pdr.lines['period']
    n_cells = pdr.vars['N_cells_S']._value 

    pdr.vars({'l_sext': 0.1, 'k2XF2arc': 0.00, 'k2XD2arc': 0.00})  
    pdr.new('XF2arc', xt.Sextupole, length='l_sext', k2='k2XF2arc', edge_entry_active=True, edge_exit_active=True)
    pdr.new('XD2arc', xt.Sextupole, length='l_sext', k2='k2XD2arc', edge_entry_active=True, edge_exit_active=True)

    # Correct dynamic indexing for N_cells_S = 10:
    # Standard FODO cells: 1 -> 8
    sda_mid_cells = [str(i) for i in range(1, n_cells )]       # 1 to 8
    # The last cell is the matching block element: 10
    sda_last_cell = str(n_cells)                                 # 10

    # Standard Focusing cells: 1 -> 8
    sfa_mid_cells = [str(i) for i in range(1, n_cells -1)]       # 1 to 8
    # The penultimate focusing element gets the matching suffix '_M': 9
    sfa_match_cells = [str(n_cells - 1)]                         # 9


    # Right-handed sectors (Standard processing orientation)
    for prefix in ['1R', '2R']:
        for cell in sda_mid_cells:
            ring.insert(pdr.new(f'XD2arc_{prefix}{cell}', 'XD2arc'), at='+(l_drift+l_quad)/2', from_=f'QDA_{prefix}{cell}')
        ring.insert(pdr.new(f'XD2arc_{prefix}{sda_last_cell}', 'XD2arc'), at='+(l_drift+l_quad)/2', from_=f'QDA_M{prefix}{sda_last_cell}')

        for cell in sfa_mid_cells:
            ring.insert(pdr.new(f'XF2arc_{prefix}{cell}', 'XF2arc'), at='+(l_drift+l_quad)/2', from_=f'QFA_{prefix}{cell}')
        for cell in sfa_match_cells:
            ring.insert(pdr.new(f'XF2arc_{prefix}{cell}', 'XF2arc'), at='+(l_drift+l_quad)/2', from_=f'QFA_M{prefix}{cell}')

    # Left-handed sectors (Reversed structural orientation)
    for prefix in ['2L', '1L']:
        for cell in sda_mid_cells:
            ring.insert(pdr.new(f'XD2arc_{prefix}{cell}', 'XD2arc'), at='-(l_drift+l_quad)/2', from_=f'QDA_{prefix}{cell}')
        ring.insert(pdr.new(f'XD2arc_{prefix}{sda_last_cell}', 'XD2arc'), at='-(l_drift+l_quad)/2', from_=f'QDA_M{prefix}{sda_last_cell}')

        for cell in sfa_mid_cells:
            ring.insert(pdr.new(f'XF2arc_{prefix}{cell}', 'XF2arc'), at='-(l_drift+l_quad)/2', from_=f'QFA_{prefix}{cell}')
        for cell in sfa_match_cells:
            ring.insert(pdr.new(f'XF2arc_{prefix}{cell}', 'XF2arc'), at='-(l_drift+l_quad)/2', from_=f'QFA_M{prefix}{cell}')

    # Second focusing family (Center entries)
    for prefix in ['1RC', '2RC']:
        ring.insert(pdr.new(f'XF2arc_{prefix}', 'XF2arc'), at='+(l_drift+l_quad)/2', from_=f'QFA_{prefix}')

    
    # Symmetric Right Segment
    for cell in sda_mid_cells:
        period.insert(pdr.new(f'XD2arc_PR{cell}', 'XD2arc'), at='+(l_drift+l_quad)/2', from_=f'QDA_PR{cell}')
    period.insert(pdr.new(f'XD2arc_PR{sda_last_cell}', 'XD2arc'), at='+(l_drift+l_quad)/2', from_=f'QDA_MPR{sda_last_cell}')

    for cell in sfa_mid_cells:
        period.insert(pdr.new(f'XF2arc_PR{cell}', 'XF2arc'), at='+(l_drift+l_quad)/2', from_=f'QFA_PR{cell}')
    for cell in sfa_match_cells:
        period.insert(pdr.new(f'XF2arc_PR{cell}', 'XF2arc'), at='+(l_drift+l_quad)/2', from_=f'QFA_MPR{cell}')

    # Symmetric Inverted Left Segment
    for cell in sda_mid_cells:
        period.insert(pdr.new(f'XD2arc_PL{cell}', 'XD2arc'), at='-(l_drift+l_quad)/2', from_=f'QDA_PL{cell}')
    period.insert(pdr.new(f'XD2arc_PL{sda_last_cell}', 'XD2arc'), at='-(l_drift+l_quad)/2', from_=f'QDA_MPL{sda_last_cell}')

    for cell in sfa_mid_cells:
        period.insert(pdr.new(f'XF2arc_PL{cell}', 'XF2arc'), at='-(l_drift+l_quad)/2', from_=f'QFA_PL{cell}')
    for cell in sfa_match_cells:
        period.insert(pdr.new(f'XF2arc_PL{cell}', 'XF2arc'), at='-(l_drift+l_quad)/2', from_=f'QFA_MPL{cell}')

    # Half quad central markers
    period.insert(pdr.new('XF2arc_PRCH', 'XF2arc'), at='+(l_drift+l_quad)/2', from_='QFA_PRCH')
    period.insert(pdr.new('XF2arc_PLCH', 'XF2arc'), at='-(l_drift+l_quad)/2', from_='QFA_PLCH')

    variables = ['k2XF2arc', 'k2XD2arc']
    ChromCorrect(ring, pdr, variables, MakePlot=False) # Uncomment when function is available

    return pdr

def config_D2_C2(pdr):
    """
    Configures 3 families of sextupoles by dynamically matching the true, 
    reversed element names of your 10-cell FODO line layout.
    """
    ring = pdr.lines['ring']
    period = pdr.lines['period']
    n_cells = pdr.vars['N_cells_S']._value # 10

    pdr.vars({ 'l_sext': 0.2, 'k2XF2arc': 0.00, 'k2XD2arc': 0.00, 'k2XF2arc2': 0.00 })  
    pdr.new('XF2arc',   xt.Sextupole, length='l_sext', k2='k2XF2arc' ,  edge_entry_active=True, edge_exit_active=True)
    pdr.new('XD2arc',   xt.Sextupole, length='l_sext', k2='k2XD2arc' ,  edge_entry_active=True, edge_exit_active=True)
    pdr.new('XF2arc2',  xt.Sextupole, length='l_sext', k2='k2XF2arc2' , edge_entry_active=True, edge_exit_active=True)

    match_cell_idx = str(n_cells - 1)  # '9'


    for prefix in ['1R', '2R']:
        # Defocusing cells 2 and 4
        for cell in ['2', '4']:
            ring.insert(pdr.new(f'XD2arc_{prefix}{cell}', 'XD2arc'), at='+(l_drift+l_quad)/2', from_=f'QDA_{prefix}{cell}')
        # Focusing cell 5
        ring.insert(pdr.new(f'XF2arc_{prefix}5', 'XF2arc'), at='+(l_drift+l_quad)/2', from_=f'QFA_{prefix}5')
        # Matching Focusing cell 9 (QFA_M1R9 / QFA_M2R9)
        ring.insert(pdr.new(f'XF2arc_{prefix}{match_cell_idx}', 'XF2arc'), at='+(l_drift+l_quad)/2', from_=f'QFA_M{prefix}{match_cell_idx}')
        # Focusing family 2 cell 1
        ring.insert(pdr.new(f'XF2arc2_{prefix}1', 'XF2arc2'), at='+(l_drift+l_quad)/2', from_=f'QFA_{prefix}1')
        

    for prefix in ['2L', '1L']:
        # Defocusing cells 2 and 4 (Note the negative direction sign since the line sequence is reversed!)
        for cell in ['2', '4']:
            ring.insert(pdr.new(f'XD2arc_{prefix}{cell}', 'XD2arc'), at='-(l_drift+l_quad)/2', from_=f'QDA_{prefix}{cell}')
        # Focusing cells 4 and 6
        for cell in ['4', '6']:
            ring.insert(pdr.new(f'XF2arc_{prefix}{cell}', 'XF2arc'), at='-(l_drift+l_quad)/2', from_=f'QFA_{prefix}{cell}')
        # Focusing family 2 cell 1
        ring.insert(pdr.new(f'XF2arc2_{prefix}1', 'XF2arc2'), at='-(l_drift+l_quad)/2', from_=f'QFA_{prefix}1')


    for cell in ['2', '4']:
        period.insert(pdr.new(f'XD2arc_PR{cell}', 'XD2arc'), at='+(l_drift+l_quad)/2', from_=f'QDA_PR{cell}')
    period.insert(pdr.new(f'XF2arc_PR5', 'XF2arc'), at='+(l_drift+l_quad)/2', from_=f'QFA_PR5')
    period.insert(pdr.new(f'XF2arc_PR{match_cell_idx}', 'XF2arc'), at='+(l_drift+l_quad)/2', from_=f'QFA_MPR{match_cell_idx}')
    period.insert(pdr.new(f'XF2arc2_PR1', 'XF2arc2'), at='+(l_drift+l_quad)/2', from_=f'QFA_PR1')

    # --- PL (Symmetric Left Section) ---
    for cell in ['2', '4']:
        period.insert(pdr.new(f'XD2arc_PL{cell}', 'XD2arc'), at='-(l_drift+l_quad)/2', from_=f'QDA_PL{cell}')
    for cell in ['4', '6']:
        period.insert(pdr.new(f'XF2arc_PL{cell}', 'XF2arc'), at='-(l_drift+l_quad)/2', from_=f'QFA_PL{cell}')
    period.insert(pdr.new(f'XF2arc2_PL1', 'XF2arc2'), at='-(l_drift+l_quad)/2', from_=f'QFA_PL1')

    

    variables = ['k2XF2arc', 'k2XD2arc','k2XF2arc2']
    ChromCorrect(ring, pdr, variables, MakePlot=False) # Uncomment when function is available

    return pdr


def config_D2_C3(pdr):

    """
    180 degree phase advance sextupoles for D2

    """

    ring=pdr.lines['ring']
    print(ring.element_names)
    period=pdr.lines['period']

  

    pdr.vars( {'l_sext':0.1, 'k2XF2arc': 0.00, 'k2XD2arc': 0.00,  'k2XF2arc2': 0.00,  'k2XD2arc2': 0.00} ) 
    pdr.new('XF2arc',  xt.Sextupole, length='l_sext',    k2='k2XF2arc' , edge_entry_active=True, edge_exit_active=True)
    pdr.new('XD2arc',  xt.Sextupole, length='l_sext',    k2='k2XD2arc' , edge_entry_active=True, edge_exit_active=True)
    pdr.new('XF2arc2',  xt.Sextupole, length='l_sext',    k2='k2XF2arc2' , edge_entry_active=True, edge_exit_active=True)
    pdr.new('XD2arc2',  xt.Sextupole, length='l_sext',    k2='k2XD2arc2' , edge_entry_active=True, edge_exit_active=True)



    # Defocusing (QDA)
    for elem in ([ [el, '+'] for el in ['1R4','1R6','2R4','2R6',] ] +
                 [ [el, '-'] for el in ['1L4','1L6','2L4','2L6',] ]):

        ring.insert( pdr.new('XD2arc_'+elem[0], 'XD2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QDA_' + elem[0] )

    '''for elem in [ ['1R8', '+'], ['1L8', '-'],['2R8', '+'], ['2L8', '-']]:
        ring.insert( pdr.new('XD2arc2_'+elem[0], 'XD2arc2'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QDA_' + elem[0] )   
        
    for elem in [ ['1R10', '+'], ['1L10', '-'],['2R10', '+'], ['2L10', '-']]:
        ring.insert( pdr.new('XD2arc2_'+elem[0], 'XD2arc2'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QDA_M' + elem[0] )'''
        
    # Focusing (QFA)
    for elem in ([ [el, '+'] for el in ['1R1','1R3','2R1','2R3',] ] +
                 [ [el, '-'] for el in ['1L1','1L3','2L1','2L3',] ]):

        ring.insert( pdr.new('XF2arc_'+elem[0], 'XF2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_' + elem[0] )
        
    for elem in ([ [el, '+'] for el in ['1R6','1R8','2R6','2R8',] ] +
                 [ [el, '-'] for el in ['1L6','1L8','2L6','2L8',] ]):

        ring.insert( pdr.new('XF2arc2_'+elem[0], 'XF2arc2'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_' + elem[0] )
        
    for elem in [ ['PR4', '+'], ['PR6', '+'], ['PL4', '-'], ['PL6', '-'] ]:
        period.insert( pdr.new('XD2arc_'+elem[0], 'XD2arc'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QDA_' + elem[0] )
        
    '''for elem in [ ['PR8', '+'],['PL8', '-'] ]:
        period.insert( pdr.new('XD2arc2_'+elem[0], 'XD2arc2'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QDA_' + elem[0] ) 

    for elem in [ ['PR10', '+'],['PL10', '-'] ]:
        period.insert( pdr.new('XD2arc2_'+elem[0], 'XD2arc2'), 
                        at=elem[1] + '(l_drift+l_quad)/2', from_='QDA_M' + elem[0] ) '''
    
    for elem in [ ['PR1', '+'], ['PR3', '+'],['PL1', '-'], ['PL3', '-']]:
        period.insert( pdr.new('XF2arc_'+elem[0], 'XF2arc'), 
                    at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_' + elem[0] )
        
    for elem in [['PR6', '+'], ['PR8', '+'], ['PL6', '-'], ['PL8', '-'] ]:
        period.insert( pdr.new('XF2arc2_'+elem[0], 'XF2arc2'), 
                    at=elem[1] + '(l_drift+l_quad)/2', from_='QFA_' + elem[0] )
        
    variables = ['k2XF2arc', 'k2XD2arc','k2XF2arc2']
    ChromCorrect(ring, pdr, variables, MakePlot=False)

    return pdr
        
    