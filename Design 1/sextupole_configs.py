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
    targets = [ xt.TargetSet(dqx=0, dqy=0, tol=1e-5 ) ]  )

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