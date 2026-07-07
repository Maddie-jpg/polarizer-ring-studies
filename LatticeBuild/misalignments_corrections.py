import xtrack as xt
import numpy as np

def get_safe_insertion(ring, prefix, elem, sign, default_offset):
    ref_element = prefix + elem
    names = ring.element_names

    if ref_element not in names:
        return ref_element, 'center', sign + default_offset

    idx = names.index(ref_element)

    def is_open_drift(i):
        if not (0 <= i < len(names)):
            return False
        cls = ring.element_dict[names[i]].__class__.__name__  # adjust if your API differs
        return cls == 'Drift'   # a full, un-split drift -- safe to center in

    prev_open = is_open_drift(idx - 1)
    next_open = is_open_drift(idx + 1)
    want_prev = (sign == '-')

    if want_prev and prev_open:
        return ref_element, 'center', sign + default_offset
    if (not want_prev) and next_open:
        return ref_element, 'center', sign + default_offset

    # Preferred side wasn't actually open -- use whichever side is
    if prev_open:
        return ref_element, 'center', '-' + default_offset
    if next_open:
        return ref_element, 'center', '+' + default_offset

    # Neither side is a plain open drift -> hug the quad edge
    if sign == '-':
        return ref_element, 'start', '-l_kick/2'
    else:
        return ref_element, 'end', '+l_kick/2'
        
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



import re

def insert_correctors(pdr):
    offset = (pdr['l_quad'] + pdr['l_drift']) / 2
    pdr['l_kick'] = 0.1
    ring = pdr.lines['ring']

    drift_map = {
        'QDA_':     'l_drift',
        'QFA_':     'l_drift',
        'QDA_M':    'l_drift',
        'QFA_M':    'l_drift',
        'QDDS_':    'l_drift + dl_drift',
        'QFDS_':    'l_drift + dl_drift',
        'QDDoub_':  'l_doub',
        'QFDoub_':  'l_doub',
        'QDTrip_':  'l_tripl',
        'QFTripC_': 'l_tripl',
    }

    def elems_for_prefix(prefix):
        longer = [p for p in drift_map if p != prefix and p.startswith(prefix)]
        out = []
        for name in ring.element_names:
            if name.startswith(prefix) and not any(name.startswith(lp) for lp in longer):
                out.append(name[len(prefix):])
        return sorted(out)

    def sign_and_offset(prefix, suffix, plane):
        is_center = bool(re.search(r'[RL]C$', suffix))
        if is_center:
            sign = '+'
        elif 'R' in suffix:
            sign = '-'
        elif 'L' in suffix:
            sign = '+'
        else:
            raise ValueError(f"Can't infer side (R/L) for {prefix}{suffix}")

        if plane == 'h' and prefix == 'QFA_' and is_center:
            dynamic_offset = '(((l_drift/2)+l_quad)/2)-l_sext'
        else:
            current_drift = drift_map.get(prefix, 'l_drift')
            dynamic_offset = f'({current_drift}+l_quad) / 2'
        return sign, dynamic_offset

    def safe_insert(new_element, ref_element, sign, default_offset):
        """Try to center the corrector in the open drift next to
        ref_element. If xtrack can't resolve that position (occupied,
        already sliced, whatever the reason), fall back to hugging
        the quad's edge on the requested side instead of crashing."""
        try:
            ring.insert(new_element, at=sign + default_offset,
                        from_=ref_element, from_anchor='center')
            return
        except (ValueError, AssertionError):
            pass

        if sign == '-':
            ring.insert(new_element, at='-l_kick/2',
                        from_=ref_element, from_anchor='start')
        else:
            ring.insert(new_element, at='+l_kick/2',
                        from_=ref_element, from_anchor='end')

    # ---- vertical correctors on QD-type quads ----
    for prefix in ['QDA_', 'QDA_M', 'QDDoub_', 'QDDS_', 'QDTrip_']:
        for elem in elems_for_prefix(prefix):
            sign, dynamic_offset = sign_and_offset(prefix, elem, 'v')
            v_name = f'vk_ring_{prefix}{elem}'
            pdr[v_name] = 0.0
            ref_element = prefix + elem

            new_element = pdr.new('My_' + ref_element, xt.Multipole,
                                   ksl=[pdr.ref[v_name]], length='l_kick')
            safe_insert(new_element, ref_element, sign, dynamic_offset)

    # ---- horizontal correctors on QF-type quads ----
    for prefix in ['QFA_', 'QFA_M', 'QFDoub_', 'QFDS_', 'QFTripC_']:
        for elem in elems_for_prefix(prefix):
            sign, dynamic_offset = sign_and_offset(prefix, elem, 'h')
            h_name = f'hk_ring_{prefix}{elem}'
            pdr[h_name] = 0.0
            ref_element = prefix + elem

            new_element = pdr.new('Mx_' + ref_element, xt.Multipole,
                                   knl=[pdr.ref[h_name]], length='l_kick')
            safe_insert(new_element, ref_element, sign, dynamic_offset)

    return pdr

def orbit_correction(ring, twiss, threading=False,
                      rcond_x=1e-4, rcond_y=1e-2,
                      n_eig_x=None, n_eig_y=None):
    """
    n_eig_x / n_eig_y: number of singular values (eigenvectors) to keep
    for the x / y correction, respectively. Can be an int, or a (low, high)
    tuple to select a specific window of singular values. If None (default),
    falls back to the existing rcond-based cutoff.
    """
    tt = ring.get_table()
    bpm_names_x = tt.rows['BPMx.*'].name
    bpm_names_y = tt.rows['BPMy.*'].name
    corr_x_names = tt.rows['Mx.*'].name
    corr_y_names = tt.rows['My.*'].name

    ring.steering_monitors_x = bpm_names_x
    ring.steering_monitors_y = bpm_names_y
    ring.steering_correctors_x = corr_x_names
    ring.steering_correctors_y = corr_y_names

    def run_correction(corr_handler):
        if n_eig_x is not None or n_eig_y is not None:
            corr_handler.x_correction.rcond = rcond_x
            corr_handler.y_correction.rcond = rcond_y
            corr_handler.correct(n_singular_values=(n_eig_x, n_eig_y))
        else:
            corr_handler.x_correction.rcond = rcond_x
            corr_handler.y_correction.rcond = rcond_y
            corr_handler.correct()

    if threading is False:
        corr_handler = ring.correct_trajectory(twiss_table=twiss, run=False)
        run_correction(corr_handler)
    else:
        tw0 = twiss
        corr_handler = ring.correct_trajectory(twiss_table=tw0, run=False)
        corr_handler.thread(ds_thread=10., rcond_long=1e-2)

        tw1 = ring.twiss(method='6d')
        corr_handler_final = ring.correct_trajectory(twiss_table=tw1, run=False)
        run_correction(corr_handler_final)

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

def misalignments(line, sigma, seed=None, cut=2.5):
    rng = np.random.default_rng(seed)

    actual_seed = rng.bit_generator.seed_seq.entropy
    print(f"Applying misalignments with seed: {actual_seed}")

    def truncnorm(cut, rgen):
        var = rgen.normal()
        if abs(var) > cut:
            var = truncnorm(cut, rgen)
        return var

    tab = line.get_table()
    quads = list(tab.rows[tab.element_type == 'Quadrupole'].name)
    sexts = list(tab.rows[tab.element_type == 'Sextupole'].name)
    bends = list(tab.rows[tab.element_type == 'Bend'].name)

    for name in quads + sexts + bends:
        ref = line.element_refs[name]

        ref.shift_x   = sigma * truncnorm(cut, rng)
        ref.shift_y   = sigma * truncnorm(cut, rng)
        ref.shift_s   = sigma * truncnorm(cut, rng)
        ref.rot_s_rad = sigma * truncnorm(cut, rng)
        ref.rot_x_rad = sigma * truncnorm(cut, rng)
        ref.rot_y_rad = sigma * truncnorm(cut, rng)

        relative_error = 1 + truncnorm(cut, rng) * 1e-3
        ref.knl *= relative_error
        ref.ksl *= relative_error

    return line