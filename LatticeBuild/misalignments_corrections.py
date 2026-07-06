import xtrack as xt
import numpy as np

def get_safe_insertion(ring,prefix, elem, sign, default_offset):
        # Find the target element name we are referencing from
        ref_element = prefix + elem
        if ref_element not in ring.element_names:
            return ref_element, 'center', sign + default_offset

        # Check elements immediately neighboring or sitting at the intended s position
        # If the standard calculation causes a collision, we alter anchors to the parent quadrupole edges
        try:
            # We can change the anchor from 'center' to 'entry' or 'exit' of the quad
            # to push the corrector completely out of the neighboring drift space if an RF cavity occupies it.
            if sign == '-':
                # Wants to go upstream of the quad: anchor to 'entry' and move backward by half a kick length
                return ref_element, 'entry', f'-l_kick/2'
            else:
                # Wants to go downstream of the quad: anchor to 'exit' and move forward by half a kick length
                return ref_element, 'exit', f'+l_kick/2'
        except:
            # Fallback to defaults if layout queries fail
            return ref_element, 'center', sign + default_offset

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
        'QFA_1RC': 'l_drift-l_sext',     # RC centre quads (many-sext lattice)
        'QFA_2RC': 'l_drift-l_sext',
        'QFA_3RC': 'l_drift-l_sext',
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

        from_elem, anchor, final_offset = get_safe_insertion(ring, prefix, elem, sign, dynamic_offset)

        ring.insert(pdr.new('My_'+prefix+elem, xt.Multipole, ksl=[pdr.ref[v_name]], length='l_kick'),
                    at=final_offset, from_=prefix + elem, from_anchor='center')

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
        full_name = prefix + elem
        h_name = f'hk_ring_{elem}'
        pdr[h_name] = 0.0  # Unique horizontal kick variable

        if full_name in ['QFA_1RC', 'QFA_2RC', 'QFA_3RC']:
            dynamic_offset = '(((l_drift/2)+l_quad)/2)-l_sext'
        else:
            current_drift = drift_map.get(prefix, 'l_drift')
            dynamic_offset = f'({current_drift}+l_quad) / 2'

        from_elem, anchor, final_offset = get_safe_insertion(ring, prefix, elem, sign, dynamic_offset)

        ring.insert(pdr.new('Mx_'+prefix+elem, xt.Multipole, knl=[pdr.ref[h_name]], length='l_kick'),
                    at=final_offset, from_=prefix + elem, from_anchor='center')
        
    return pdr

def orbit_correction(ring, twiss, threading=False, rcond_x=1e-4, rcond_y=1e-2):   
    
    
    tt = ring.get_table()
    bpm_names_x = tt.rows['BPMx.*'].name
    bpm_names_y = tt.rows['BPMy.*'].name
    corr_x_names = tt.rows['Mx.*'].name
    corr_y_names = tt.rows['My.*'].name

    ring.steering_monitors_x = bpm_names_x
    ring.steering_monitors_y = bpm_names_y
    ring.steering_correctors_x = corr_x_names
    ring.steering_correctors_y = corr_y_names

    if threading is False:
        corr_handler = ring.correct_trajectory(twiss_table=twiss, run=False)
        corr_handler.x_correction.rcond = rcond_x
        corr_handler.y_correction.rcond = rcond_y
        corr_handler.correct()
    else:
        tw0 = twiss
        corr_handler = ring.correct_trajectory(twiss_table=tw0, run=False)
        corr_handler.thread(ds_thread=10., rcond_long=1e-2)

        tw1 = ring.twiss(method='6d')
        corr_handler_final = ring.correct_trajectory(twiss_table=tw1, run=False)
        corr_handler_final.x_correction.rcond = rcond_x
        corr_handler_final.y_correction.rcond = rcond_y
        corr_handler_final.correct()

    return ring

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

def insert_correctors_var2(pdr):

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
        'QFA_1RC': 'l_drift-l_sext',
        'QFA_2RC': 'l_drift-l_sext',
        'QFA_3RC': 'l_drift-l_sext',
    }

    #vertical correctors
    qy_ring_list = (
    # Standard Arc Defocusing Quads (R1-R7 and L1-L7)
    [[el, '-', 'QDA_'] for el in ['1R1','1R2','1R3','1R4','1R5','1R6','1R7','2R1','2R2','2R3','2R4','2R5','2R6','2R7','3R1','3R2','3R3','3R4','3R5','3R6','3R7']] +
    [[el, '+', 'QDA_'] for el in ['1L1','1L2','1L3','1L4','1L5','1L6','1L7','2L1','2L2','2L3','2L4','2L5','2L6','2L7','3L1','3L2','3L3','3L4','3L5','3L6','3L7']] +
    # Matching and DS Quads
    [[el, '-', 'QDA_M'] for el in ['1R8','2R8','3R8']] +
    [[el, '+', 'QDA_M'] for el in ['1L8','2L8','3L8']] +
    [[el, '-', 'QDDoub_'] for el in ['1R','2R','3R']] +
    [[el, '+', 'QDDoub_'] for el in ['1L','2L','3L']] +
    [[el, '-', 'QDDS_'] for el in ['1R','2R','3R']] +
    [[el, '+', 'QDDS_'] for el in ['1L','2L','3L']] +
    # Interaction Region Triplets (Adding missing R side)
    [[el, '-', 'QDTrip_'] for el in ['1R1','2R1','3R1']] +
    [[el, '+', 'QDTrip_'] for el in ['1L1','2L1','3L1']] 
)

    for elem, sign, prefix in qy_ring_list:
        current_drift = drift_map.get(prefix, 'l_drift')
        dynamic_offset = f'({current_drift}+l_quad) / 2'
        v_name = f'vk_ring_{elem}'
        pdr[v_name] = 0.0  

        # Check layout to prevent slicing an RF Cavity
        from_elem, anchor, final_offset = get_safe_insertion(ring,prefix, elem, sign, dynamic_offset)
        
        ring.insert(pdr.new('My_'+prefix+elem, xt.Multipole, ksl=[pdr.ref[v_name]], length='l_kick'), #edge_entry_active=True, edge_exit_active=True), 
                    at=final_offset, from_=prefix + elem, from_anchor='center')


    # horizontal correctors
    qx_ring_list = (
    # QFA Arc body (Added R0/L0 and R1-R5)
    [[el, '-', 'QFA_'] for el in ['1R0','2R0','3R0', '1R1','1R2','1R3','1R4','1R5','2R1','2R2','2R3','2R4','2R5','3R1','3R2','3R3','3R4','3R5']] +
    [[el, '+', 'QFA_'] for el in ['1L0','2L0','3L0', '1L1','1L2','1L3','1L4','1L5','2L1','2L2','2L3','2L4','2L5','3L1','3L2','3L3','3L4','3L5']] +
    # QFA Center Quads (Added R side and corrected L side with 'H' suffix to match ring names)
    [[el, '+', 'QFA_'] for el in ['1RC','2RC','3RC']] +
    #[[el, '+', 'QFA_'] for el in ['1LCH','2LCH','3LCH']] +
    # Matching and DS Quads
    [[el, '-', 'QFA_M'] for el in ['1R8','2R8','3R8']] +
    [[el, '+', 'QFA_M'] for el in ['1L8','2L8','3L8']] +
    [[el, '-', 'QFDoub_'] for el in ['1R','2R','3R']] +
    [[el, '+', 'QFDoub_'] for el in ['1L','2L','3L']] +
    [[el, '-', 'QFDS_'] for el in ['1R','2R','3R']] +
    [[el, '+', 'QFDS_'] for el in ['1L','2L','3L']] +
    #[[el, '-', 'QFTripC_'] for el in ['1R2','2R2','3R2']] +
    [[el, '+', 'QFTripC_'] for el in ['1L2','2L2','3L2']]
)
    
    for elem, sign, prefix in qx_ring_list:
        full_name = prefix + elem
        h_name = f'hk_ring_{elem}'
        pdr[h_name] = 0.0  
        
        if full_name in ['QFA_1RC', 'QFA_2RC', 'QFA_3RC']:
            dynamic_offset = '(((l_drift/2)+l_quad)/2)-l_sext'
        else:
            current_drift = drift_map.get(prefix, 'l_drift')
            dynamic_offset = f'({current_drift}+l_quad) / 2'
            
        from_elem, anchor, final_offset = get_safe_insertion(ring,prefix, elem, sign, dynamic_offset)

        ring.insert(pdr.new('Mx_'+prefix+elem, xt.Multipole, knl=[pdr.ref[h_name]], length='l_kick'), #edge_entry_active=True, edge_exit_active=True), 
                    at=final_offset, from_=prefix + elem, from_anchor='center')

        
    return pdr

def insert_BPMs_all_as_markers(pdr):
   
   ring=pdr.lines['ring']

   bpm = xt.Marker()


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
   return pdr

def misalignments(line,sigma, seed=None):
   

   rng=np.random.default_rng(seed)

   actual_seed = rng.bit_generator.seed_seq.entropy
   print(f"Applying misalignments with seed: {actual_seed}")

   tab=line.get_table()
   #Quad and sextupole misalignments
   quads = list(tab.rows[tab.element_type == 'Quadrupole'].name)
   sexts = list(tab.rows[tab.element_type == 'Sextupole'].name)
   bends = list(tab.rows[tab.element_type == 'Bend'].name)

   for name in quads + sexts + bends:
        # Access the element via the reference manager
        ref = line.element_refs[name]
        
        # Apply translations
        ref.shift_x = rng.normal(0, sigma)
        ref.shift_y = rng.normal(0, sigma)
        ref.shift_s = rng.normal(0, sigma)
        
        # Apply rotations (in radians)
        ref.rot_s_rad = rng.normal(0, sigma)
        ref.rot_x_rad = rng.normal(0, sigma)
        ref.rot_y_rad = rng.normal(0, sigma)

        relative_error = 1 + rng.normal(0, 1e-3)
        ref.knl *= relative_error
        ref.ksl *= relative_error

   return line