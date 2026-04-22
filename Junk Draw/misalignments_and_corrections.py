#functions used for misalignments and corrections
import numpy as np
import xtrack as xt

def misalignments(line):
   sigma=0.3e-3

   tab=line.get_table()
   #Quad and sextupole misalignments
   quads = list(tab.rows[tab.element_type == 'Quadrupole'].name)
   sexts = list(tab.rows[tab.element_type == 'Sextupole'].name)
   bends = list(tab.rows[tab.element_type == 'Bend'].name)

   for name in quads + sexts + bends:
        # Access the element via the reference manager
        ref = line.element_refs[name]
        
        # Apply translations
        ref.shift_x = np.random.normal(0, sigma)
        ref.shift_y = np.random.normal(0, sigma)
        ref.shift_s = np.random.normal(0, sigma)
        
        # Apply rotations (in radians)
        ref.rot_s_rad = np.random.normal(0, sigma)
        ref.rot_x_rad = np.random.normal(0, sigma)
        ref.rot_y_rad = np.random.normal(0, sigma)

        relative_error = 1 + np.random.normal(0, 1e-3)
        ref.knl *= relative_error
        ref.ksl *= relative_error

   return line

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

def insert_correctors(pdr):
    offset = 0
    pdr['l_kick']=0.1
    ring = pdr.lines['ring']
    period = pdr.lines['period']

    #vertical correctors
    qy_ring_list = (
        [[el, '+', 'QDA_'] for el in ['1R1','1R2','1R3','1R4','1R5','1R6','1R7','2R1','2R2','2R3','2R4','2R5','2R6','2R7','3R1','3R2','3R3','3R4','3R5','3R6','3R7']] +
        [[el, '-', 'QDA_'] for el in ['1L1','1L2','1L3','1L4','1L5','1L6','1L7','2L1','2L2','2L3','2L4','2L5','2L6','2L7','3L1','3L2','3L3','3L4','3L5','3L6','3L7']] +
        [[el, '+', 'QDA_M'] for el in ['1R8','2R8','3R8']] +
        [[el, '-', 'QDA_M'] for el in ['1L8','2L8','3L8']] +
        [[el, '+', 'QDDoub_'] for el in ['1R','2R','3R']] +
        [[el, '-', 'QDDoub_'] for el in ['1L','2L','3L']] +
        [[el, '+', 'QDDS_'] for el in ['1R','2R','3R']] +
        [[el, '-', 'QDDS_'] for el in ['1L','2L','3L']] +
        [[el, '+', 'QDTrip_'] for el in ['1R1','2R1','3R1']] +
        [[el, '-', 'QDTrip_'] for el in ['1L1','2L1','3L1']] 
    )

    for elem, sign, prefix in qy_ring_list:
        v_name = f'vk_ring_{elem}'
        pdr[v_name] = 0.0  # Unique vertical kick variable
        ring.insert(pdr.new('My_'+prefix+elem, xt.Multipole, ksl=[pdr.ref[v_name]], length='l_kick'), #edge_entry_active=True, edge_exit_active=True), 
                    at=sign + str(offset), from_=prefix + elem, from_anchor='end')

    # horizontal correctors
    qx_ring_list = (
        [[el, '+', 'QFA_'] for el in ['1R1','1R2','1R3','1R4','1R5','1R6','1RC','2R1','2R2','2R3','2R4','2R5','2R6','2RC','3R1','3R2','3R3','3R4','3R5','3R6','3RC']] +
        [[el, '-', 'QFA_'] for el in ['1L1','1L2','1L3','1L4','1L5','1L6','2L1','2L2','2L3','2L4','2L5','2L6','3L1','3L2','3L3','3L4','3L5','3L6']] +
        [[el, '+', 'QFA_M'] for el in ['1R7','2R7','3R7']] +
        [[el, '-', 'QFA_M'] for el in ['1L7','2L7','3L7']] +
        [[el, '+', 'QFDoub_'] for el in ['1R','2R','3R']] +
        [[el, '-', 'QFDoub_'] for el in ['1L','2L','3L']] +
        [[el, '+', 'QFDS_'] for el in ['1R','2R','3R']] +
        [[el, '-', 'QFDS_'] for el in ['1L','2L','3L']] 
    )

    for elem, sign, prefix in qx_ring_list:
        h_name = f'hk_ring_{elem}'
        pdr[h_name] = 0.0  # Unique horizontal kick variable
        ring.insert(pdr.new('Mx_'+prefix+elem, xt.Multipole, knl=[pdr.ref[h_name]], length='l_kick'),# edge_entry_active=True, edge_exit_active=True), 
                    at=sign + str(offset), from_=prefix + elem, from_anchor='end')

    # vertical correctors
    qy_period_list = (
        [[el, '+', 'QDA_'] for el in ['PR1','PR2','PR3','PR4','PR5','PR6','PR7']] +
        [[el, '-', 'QDA_'] for el in ['PL1','PL2','PL3','PL4','PL5','PL6','PL7']] +
        [[el, '+', 'QDA_M'] for el in ['PR8']] +
        [[el, '-', 'QDA_M'] for el in ['PL8']] +
        [[el, '+', 'QDDoub_'] for el in ['PR']] +
        [[el, '-', 'QDDoub_'] for el in ['PL']] +
        [[el, '+', 'QDDS_'] for el in ['PR']] +
        [[el, '-', 'QDDS_'] for el in ['PL']] +
        [[el, '+', 'QDTrip_'] for el in ['PR1']] +
        [[el, '-', 'QDTrip_'] for el in ['PL1']]
    )

    for elem, sign, prefix in qy_period_list:
        v_name = f'vk_per_{elem}'
        pdr[v_name] = 0.0
        period.insert(pdr.new('My_'+prefix+elem, xt.Multipole, ksl=[pdr.ref[v_name]], length='l_kick'),# edge_entry_active=True, edge_exit_active=True), 
                      at=sign + str(offset), from_=prefix + elem, from_anchor='end')

    # horizontal correctors
    qx_period_list = (
        [[el, '+', 'QFA_'] for el in ['PR1','PR2','PR3','PR4','PR5','PR6','PRCH']] +
        [[el, '-', 'QFA_'] for el in ['PL1','PL2','PL3','PL4','PL5','PL6','PLCH']] +
        [[el, '+', 'QFA_M'] for el in ['PR7']] +
        [[el, '-', 'QFA_M'] for el in ['PL7']]+
        [[el, '+', 'QFDoub_'] for el in ['PR']] +
        [[el, '-', 'QFDoub_'] for el in ['PL']] +
        [[el, '+', 'QFDS_'] for el in ['PR']] +
        [[el, '-', 'QFDS_'] for el in ['PL']] 
    )

    for elem, sign, prefix in qx_period_list:
        h_name = f'hk_per_{elem}'
        pdr[h_name] = 0.0
        period.insert(pdr.new('Mx_'+prefix+elem, xt.Multipole, knl=[pdr.ref[h_name]], length='l_kick'),# edge_entry_active=True, edge_exit_active=True), 
                      at=sign + str(offset), from_=prefix + elem, from_anchor='end')
        
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