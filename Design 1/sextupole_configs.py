"""
Sextupole configs to be paired with linear optics script
Each function should include a description of the configuration

"""
import xtrack as xt

def ChromCorrect_dq(ring, pdr, ksf, ksd, MakePlot=False):
        
    opt_chrom = ring.match(
        solve=True,
        verbose=False,
        method='6d',
        vary=xt.VaryList([ksf, ksd], step=1e-3),
        targets=[
            # Global goals
            xt.Target(dqx=0, tol=1e-4),
            xt.Target(dqy=0, tol=1e-4),
            
            
        ])
    opt_chrom.target_status()
    opt_chrom.run_jacobian(n_steps=100)
    opt_chrom.target_status()
    # Print the final matched strengths

    print(f"Matched kSF: {pdr.vars[f'{ksf}']._get_value():.6f}")
    print(f"Matched kSD: {pdr.vars[f'{ksd}']._get_value():.6f}")   

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

def config_D1(pdr):

    """
    8 SF's and 8 SD's in each half-arc

    """

    ring=pdr.lines['ring']
    print(ring.element_names)
    period=pdr.lines['period']

    pdr.vars( {'l_sext':0.1, 'k2XF2arc': 0.00, 'k2XD2arc': 0.00} )  # Sextupoles - 3 families defined here
    pdr.new('XF2arc',  xt.Sextupole, length='l_sext',    k2='k2XF2arc' , edge_entry_active=True, edge_exit_active=True)
    pdr.new('XD2arc',  xt.Sextupole, length='l_sext',    k2='k2XD2arc' , edge_entry_active=True, edge_exit_active=True)


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
    

    ChromCorrect_dq(ring, pdr, 'k2XF2arc', 'k2XD2arc', MakePlot=False)

    return pdr

def config_D2(pdr):

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

    ringS1_chroma = ring.match( method='6d', solve=True,
    vary = [ xt.VaryList(['k2XF2arc', 'k2XD2arc','k2XF2arc2'], step=1e-4 )],
    targets = [ xt.TargetSet(dqx=0, dqy=0,tol=1e-5 ) ]  )

    ringS1_chroma.run_jacobian(10)

    return pdr

def config_D3(pdr):

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
        
    ChromCorrect_dq(ring, pdr, 'k2XF1arc', 'k2XD1arc', MakePlot=False)

    return pdr
    
def config_D4(pdr):
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

    ChromCorrect_dq(ring, pdr, 'k2XF3arc', 'k2XD3arc', MakePlot=False)

    return pdr

def config_D5(pdr):
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

    ChromCorrect_dq(ring, pdr, 'k2XF2arc', 'k2XD2arc', MakePlot=False)

    return pdr


def config_D6(pdr):

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
    

    ChromCorrect_ddq(ring, pdr, 'k2XF2arc', 'k2XD2arc', MakePlot=False)

    return pdr


def insert_BPMs(pdr, start_at_turn, stop_at_turn, fRev):

   offset=0.0

   bpm = xt.BeamPositionMonitor(
    start_at_turn=start_at_turn, 
    stop_at_turn=stop_at_turn, 
    frev=fRev
   )
    
   ring=pdr.lines['ring']
     # Defocusing (QDA)
   for elem in ([ [el, '-'] for el in ['1R1','1R2','1R3','1R4','1R5','1R6','1R7','2R1','2R2','2R3','2R4','2R5','2R6','2R7','3R1','3R2','3R3','3R4','3R5','3R6','3R7'] ] +
                 [ [el, '+'] for el in ['1L1','1L2','1L3','1L4','1L5','1L6','1L7','2L1','2L2','2L3','2L4','2L5','2L6','2L7','3L1','3L2','3L3','3L4','3L5','3L6','3L7'] ]):

        ring.insert( 'BPMy_'+elem[0], bpm, 
                        at=elem[1] + str(offset), from_='QDA_' + elem[0], from_anchor='start' )
        
   for elem in ([[el, '-'] for el in ['1R8','2R8','3R8']]+
                 [[el, '+'] for el in ['1L8','2L8','3L8']]):
        ring.insert( 'BPMy_'+elem[0],bpm,  
                        at=elem[1] + str(offset), from_='QDA_M' + elem[0] , from_anchor='start')
        
   for elem in ([[el, '+'] for el in ['1R','2R','3R']] +
        [[el, '-'] for el in ['1L','2L','3L']]):

        ring.insert( 'BPMy_'+elem[0],bpm,
                        at=elem[1] + str(offset), from_='QDDoub_' + elem[0], from_anchor='start' )
        
   for elem in ([ [el, '-'] for el in ['1R1','1R2','1R3','1R4','1R5','1R6','1RC','2R1','2R2','2R3','2R4','2R5','2R6','2RC','3R1','3R2','3R3','3R4','3R5','3R6','3RC'] ] +
                 [ [el, '+'] for el in ['1L1','1L2','1L3','1L4','1L5','1L6','2L1','2L2','2L3','2L4','2L5','2L6','3L1','3L2','3L3','3L4','3L5','3L6'] ]):

        ring.insert( 'BPMx_'+elem[0],bpm,
                        at=elem[1] + str(offset), from_='QFA_' + elem[0], from_anchor='start' )
        
   for elem in ([[el, '+'] for el in ['1R','2R','3R']] +
        [[el, '-'] for el in ['1L','2L','3L']]):

        ring.insert( 'BPMx_'+elem[0],bpm,
                        at=elem[1] + str(offset), from_='QFDoub_' + elem[0], from_anchor='start' )



    # Matching quads
   for elem in [['1R7','-'], ['2R7','-'], ['3R7','-'],['1L7','+'], ['2L7','+'], ['3L7','+']]:
        ring.insert( 'BPMx_'+elem[0],bpm, 
                        at=elem[1] + str(offset), from_='QFA_M' + elem[0] , from_anchor='start')

   period=pdr.lines['period']
    
   for elem in [['PR1','-'], ['PR2','-'],['PR3','-'], ['PR4','-'],['PR5','-'], ['PR6','-'],['PR7','-'], ['PL1','+'], ['PL2','+'], ['PL3','+'], ['PL4','+'], ['PL5','+'], ['PL6','+'], ['PL7','+']]:
            period.insert( 'BPMy_'+elem[0],  bpm,
                        at=elem[1] + str(offset), from_='QDA_' + elem[0] , from_anchor='start')
            
   for elem in ([[el, '+'] for el in ['PR8']]+
                 [[el, '-'] for el in ['PL8']]):
        period.insert(  'BPMy_'+elem[0], bpm,
                        at=elem[1] + str(offset), from_='QDA_M' + elem[0] , from_anchor='start')
   
   for elem in [['PR','-'],['PL','+']]:
            period.insert( 'BPMy_'+elem[0],bpm,
                        at=elem[1] + str(offset), from_='QDDoub_' + elem[0], from_anchor='start' )
    

    # Focusing
   for elem in [['PR1','-'], ['PR2','-'],['PR3','-'], ['PR4','-'],['PR5','-'], ['PR6','-'],['PRCH','-'], ['PL1','+'], ['PL2','+'], ['PL3','+'], ['PL4','+'], ['PL5','+'], ['PL6','+'], ['PLCH','+']]:
            period.insert( 'BPMx_'+elem[0],bpm,
                        at=elem[1] + str(offset), from_='QFA_' + elem[0], from_anchor='start' )
    
   for elem in [['PR','-'],['PL','+']]:
            period.insert( 'BPMx_'+elem[0],bpm,
                        at=elem[1] + str(offset), from_='QFDoub_' + elem[0], from_anchor='start' )
            
    # Matching
   for elem in [['PR7','-'],['PL7','+']]:
            period.insert( 'BPMx_'+elem[0],bpm,
                        at=elem[1] + str(offset), from_='QFA_M' + elem[0] , from_anchor='start')  

   return pdr



def insert_correctors(pdr):
    offset =(pdr['l_quad']+pdr['l_drift'])/2
    pdr['l_kick']=0.1
    ring = pdr.lines['ring']
    drift_map = {
        'QDA_':    'l_drift',
        'QFA_':    'l_drift',
        'QDA_M':   'l_drift',
        'QFA_M':   'l_drift',
        'QDDS_':   'l_drift + dl_drift', # Using the 'DrarcS' logic
        'QFDS_':   'l_drift + dl_drift',
        'QDDoub_': 'l_doub',             # Doublet drift
        'QFDoub_': 'l_doub',
        'QDTrip_': 'l_tripl',            # Triplet drift
    }

    #vertical correctors
    qy_ring_list = (
        [[el, '-', 'QDA_'] for el in ['1R1','1R2','1R3','1R4','1R5','1R6','1R7','2R1','2R2','2R3','2R4','2R5','2R6','2R7','3R1','3R2','3R3','3R4','3R5','3R6','3R7']] +
        [[el, '+', 'QDA_'] for el in ['1L1','1L2','1L3','1L4','1L5','1L6','1L7','2L1','2L2','2L3','2L4','2L5','2L6','2L7','3L1','3L2','3L3','3L4','3L5','3L6','3L7']] +
        [[el, '-', 'QDA_M'] for el in ['1R8','2R8','3R8']] +
        [[el, '+', 'QDA_M'] for el in ['1L8','2L8','3L8']] +
        [[el, '-', 'QDDoub_'] for el in ['1R','2R','3R']] +
        [[el, '+', 'QDDoub_'] for el in ['1L','2L','3L']] +
        [[el, '-', 'QDDS_'] for el in ['1R','2R','3R']] +
        [[el, '+', 'QDDS_'] for el in ['1L','2L','3L']] +
        [[el, '-', 'QDTrip_'] for el in ['1R1','2R1','3R1']] +
        [[el, '+', 'QDTrip_'] for el in ['1L1','2L1','3L1']] 
    )

    for elem, sign, prefix in qy_ring_list:
        current_drift = drift_map.get(prefix, 'l_drift')
        dynamic_offset = f'({current_drift}+l_quad) / 2'
        v_name = f'vk_ring_{elem}'
        pdr[v_name] = 0.0  # Unique vertical kick variable
        ring.insert(pdr.new('My_'+prefix+elem, xt.Multipole, ksl=[pdr.ref[v_name]], length='l_kick'), #edge_entry_active=True, edge_exit_active=True), 
                    at=sign + dynamic_offset, from_=prefix + elem, from_anchor='center')

    # horizontal correctors
    qx_ring_list = (
        [[el, '-', 'QFA_'] for el in ['1R1','1R2','1R3','1R4','1R5','1R6','2R1','2R2','2R3','2R4','2R5','2R6','3R1','3R2','3R3','3R4','3R5','3R6']] +
        [[el, '+', 'QFA_'] for el in ['1L1','1L2','1L3','1L4','1L5','1L6','2L1','2L2','2L3','2L4','2L5','2L6','3L1','3L2','3L3','3L4','3L5','3L6','1RC','2RC','3RC']] +
        [[el, '-', 'QFA_M'] for el in ['1R7','2R7','3R7']] +
        [[el, '+', 'QFA_M'] for el in ['1L7','2L7','3L7']] +
        [[el, '-', 'QFDoub_'] for el in ['1R','2R','3R']] +
        [[el, '+', 'QFDoub_'] for el in ['1L','2L','3L']] +
        [[el, '-', 'QFDS_'] for el in ['1R','2R','3R']] +
        [[el, '+', 'QFDS_'] for el in ['1L','2L','3L']] +
        [[el, '+', 'QFTripC_'] for el in ['1L2','2L2','3L2']]
    )

    for elem, sign, prefix in qx_ring_list:
        current_drift = drift_map.get(prefix, 'l_drift')
        dynamic_offset = f'({current_drift}+l_quad) / 2'
        h_name = f'hk_ring_{elem}'
        pdr[h_name] = 0.0  # Unique horizontal kick variable
        ring.insert(pdr.new('Mx_'+prefix+elem, xt.Multipole, knl=[pdr.ref[h_name]], length='l_kick'),# edge_entry_active=True, edge_exit_active=True), 
                    at=sign + dynamic_offset, from_=prefix + elem, from_anchor='center')
        
    return pdr

def orbit_correction(pdr,twiss,threading=False):   
    ring = pdr.lines['ring']
    period=pdr.lines['period']
    
    tt = ring.get_table()
    bpm_names_x = tt.rows['BPMx.*'].name
    bpm_names_y = tt.rows['BPMy.*'].name
    #bpm_names = tt.rows['BPM.*'].name
    corr_x_names = tt.rows['Mx.*'].name
    corr_y_names = tt.rows['My.*'].name

    ring.steering_monitors_x = bpm_names_x
    ring.steering_monitors_y = bpm_names_y
    ring.steering_correctors_x = corr_x_names
    ring.steering_correctors_y = corr_y_names

    if threading is False:
       ring.correct_trajectory(twiss_table=twiss)
    else:
        tw0 = twiss

        corr_handler = ring.correct_trajectory(twiss_table=tw0, run=False)
        corr_handler.thread(ds_thread=10., rcond_long=1e-2)

        tw1 = ring.twiss(method='6d')
        ring.correct_trajectory(twiss_table=tw1)

    return pdr

def insert_BPMs_all(pdr, start_at_turn, stop_at_turn, fRev):
   
   ring=pdr.lines['ring']

   bpm = xt.BeamPositionMonitor(
    start_at_turn=start_at_turn, 
    stop_at_turn=stop_at_turn, 
    frev=fRev
   )

   tab_r=ring.get_table()
   
   quads_ring = tab_r.rows[tab_r.element_type == 'Quadrupole'].name

   for elem_name in quads_ring:
        element = ring[elem_name]
        
        if element.k1 > 0:
            ring.insert(
                'BPMx_'+elem_name,
                bpm,
                at=0.0,
                from_=elem_name,
                from_anchor='end'
            )
        
        if element.k1 < 0:
            ring.insert(
                'BPMy_'+elem_name,
                bpm,
                at=0.0,
                from_=elem_name,
                from_anchor='end'
            )

   period=pdr.lines['period']
   tab_p=period.get_table()
   quads_period = tab_p.rows[tab_p.element_type == 'Quadrupole'].name

   for elem_name in quads_period:
        element = period[elem_name]
        
        if element.k1 > 0:
            period.insert(
                'BPMx_'+elem_name,
                bpm,
                at=0.0,
                from_=elem_name,
                from_anchor='end'
            )
        
        if element.k1 < 0:
            period.insert(
                'BPMy_'+elem_name,
                bpm,
                at=0.0,
                from_=elem_name,
                from_anchor='end'
            )

def insert_correctors_no_QDAM(pdr):
    offset =(pdr['l_quad']+pdr['l_drift'])/2
    pdr['l_kick']=0.1
    ring = pdr.lines['ring']
    drift_map = {
        'QDA_':    'l_drift',
        'QFA_':    'l_drift',
        'QDA_M':   'l_drift',
        'QFA_M':   'l_drift',
        'QDDS_':   'l_drift + dl_drift', # Using the 'DrarcS' logic
        'QFDS_':   'l_drift + dl_drift',
        'QDDoub_': 'l_doub',             # Doublet drift
        'QFDoub_': 'l_doub',
        'QDTrip_': 'l_tripl',            # Triplet drift
    }

    #vertical correctors
    qy_ring_list = (
        [[el, '-', 'QDA_'] for el in ['1R1','1R2','1R3','1R4','1R5','1R6','1R7','2R1','2R2','2R3','2R4','2R5','2R6','2R7','3R1','3R2','3R3','3R4','3R5','3R6','3R7','1R8','2R8','3R8']] +
        [[el, '+', 'QDA_'] for el in ['1L1','1L2','1L3','1L4','1L5','1L6','1L7','2L1','2L2','2L3','2L4','2L5','2L6','2L7','3L1','3L2','3L3','3L4','3L5','3L6','3L7','1L8','2L8','3L8']] +
        [[el, '-', 'QDDoub_'] for el in ['1R','2R','3R']] +
        [[el, '+', 'QDDoub_'] for el in ['1L','2L','3L']] +
        [[el, '-', 'QDDS_'] for el in ['1R','2R','3R']] +
        [[el, '+', 'QDDS_'] for el in ['1L','2L','3L']] +
        [[el, '-', 'QDTrip_'] for el in ['1R1','2R1','3R1']] +
        [[el, '+', 'QDTrip_'] for el in ['1L1','2L1','3L1']] 
    )

    for elem, sign, prefix in qy_ring_list:
        current_drift = drift_map.get(prefix, 'l_drift')
        dynamic_offset = f'({current_drift}+l_quad) / 2'
        v_name = f'vk_ring_{elem}'
        pdr[v_name] = 0.0  # Unique vertical kick variable
        ring.insert(pdr.new('My_'+prefix+elem, xt.Multipole, ksl=[pdr.ref[v_name]], length='l_kick'), #edge_entry_active=True, edge_exit_active=True), 
                    at=sign + dynamic_offset, from_=prefix + elem, from_anchor='center')

    # horizontal correctors
    qx_ring_list = (
        [[el, '-', 'QFA_'] for el in ['1R1','1R2','1R3','1R4','1R5','1R6','2R1','2R2','2R3','2R4','2R5','2R6','3R1','3R2','3R3','3R4','3R5','3R6']] +
        [[el, '+', 'QFA_'] for el in ['1L1','1L2','1L3','1L4','1L5','1L6','2L1','2L2','2L3','2L4','2L5','2L6','3L1','3L2','3L3','3L4','3L5','3L6','1RC','2RC','3RC']] +
        [[el, '-', 'QFA_M'] for el in ['1R7','2R7','3R7']] +
        [[el, '+', 'QFA_M'] for el in ['1L7','2L7','3L7']] +
        [[el, '-', 'QFDoub_'] for el in ['1R','2R','3R']] +
        [[el, '+', 'QFDoub_'] for el in ['1L','2L','3L']] +
        [[el, '-', 'QFDS_'] for el in ['1R','2R','3R']] +
        [[el, '+', 'QFDS_'] for el in ['1L','2L','3L']] +
        [[el, '+', 'QFTripC_'] for el in ['1L2','2L2','3L2']]
    )

    for elem, sign, prefix in qx_ring_list:
        current_drift = drift_map.get(prefix, 'l_drift')
        dynamic_offset = f'({current_drift}+l_quad) / 2'
        h_name = f'hk_ring_{elem}'
        pdr[h_name] = 0.0  # Unique horizontal kick variable
        ring.insert(pdr.new('Mx_'+prefix+elem, xt.Multipole, knl=[pdr.ref[h_name]], length='l_kick'),# edge_entry_active=True, edge_exit_active=True), 
                    at=sign + dynamic_offset, from_=prefix + elem, from_anchor='center')
        
    return pdr