import xtrack as xt
import numpy as np

def get_safe_insertion(ring, prefix, elem, sign, default_offset):
    """
    default_offset is kept for backward compatibility (existing callers
    still pass it) but is no longer used to compute the placement -- see
    below for why.

    Why: default_offset is built by the caller from a generic drift_map
    (e.g. prefix -> 'l_tripl'), which assumes a specific nominal drift
    length for that quad TYPE. That assumption can be wrong at specific
    locations -- e.g. triplet quads that sit at an IP/symmetry boundary,
    where the 'L1' quad of one sextant and the 'R1' quad of a DIFFERENT
    sextant share a much shorter drift than the generic 'l_tripl' value
    implies. When that happens, is_open_drift correctly identifies a real
    Drift-class neighbor, but centering using the caller's oversized
    assumed offset overshoots straight through that drift and into
    whatever quad sits beyond it -- with no exception raised (see
    insert_correctors' safe_insert docstring). This showed up as
    corrector insertions for one triplet quad silently damaging an
    unrelated triplet quad elsewhere in the ring.

    Fix: measure the actual neighboring element's length directly and use
    that to compute the centering offset, and explicitly check the kicker
    (length l_kick) actually fits inside it before treating that side as
    usable. If it doesn't fit, this falls through exactly like the
    already-existing "neither side is a plain open drift" case.
    """
    ref_element = prefix + elem
    names = ring.element_names

    if ref_element not in names:
        # This should not legitimately happen: callers only ever pass
        # ref_element values discovered by scanning ring.element_names in
        # the first place (see elems_for_prefix / qy_ring_list etc). If
        # we get here, ref_element existed when that scan ran but is gone
        # NOW -- almost certainly because an earlier insertion in the same
        # call silently sliced through and removed it. Line.insert() does
        # this with no exception of its own (see cut_at_s + the
        # remove-overlap logic), so failing silently here would surface as
        # a confusing deep ValueError from xtrack several frames down,
        # pointing at the wrong place. Fail loudly here instead, at the
        # actual missing element.
        try:
            l_kick_val = ring.vars['l_kick']._value
        except Exception:
            l_kick_val = '?'
        raise RuntimeError(
            f"get_safe_insertion: '{ref_element}' is not in ring.element_names "
            f"anymore. It was presumably present when the target list was "
            f"built, so this most likely means an EARLIER corrector "
            f"insertion in this same call silently sliced through and "
            f"removed it. Run snapshot_quad_strengths()/"
            f"check_quad_strength_conserved() around each insertion in "
            f"this loop to find which one did it, and check whether "
            f"'{prefix}' quads are spaced closely enough that l_kick="
            f"{l_kick_val} doesn't fit."
        )

    idx = names.index(ref_element)
    ref_length = getattr(ring.element_dict[ref_element], 'length', 0.0)

    try:
        l_kick = ring.vars['l_kick']._value
    except Exception:
        l_kick = 0.1  # matches the default set in insert_correctors

    def neighbor(i):
        if not (0 <= i < len(names)):
            return None, 0.0
        el = ring.element_dict[names[i]]
        return el.__class__.__name__, getattr(el, 'length', 0.0)

    prev_cls, prev_len = neighbor(idx - 1)
    next_cls, next_len = neighbor(idx + 1)

    margin = 1e-6  # small clearance beyond the kicker's own length

    def usable(cls, length):
        # a plain, un-split Drift AND actually long enough to hold the
        # kicker with a little room to spare -- being class=='Drift' alone
        # isn't sufficient, as the IR-triplet case above showed.
        return cls == 'Drift' and length > l_kick + margin

    def centered_offset(length):
        # distance from ref_element's center to the center of a neighbor
        # of this REAL, measured length -- not the caller's guess.
        return (length + ref_length) / 2

    prev_usable = usable(prev_cls, prev_len)
    next_usable = usable(next_cls, next_len)
    want_prev = (sign == '-')

    if want_prev and prev_usable:
        return ref_element, 'center', f'-({centered_offset(prev_len)})'
    if (not want_prev) and next_usable:
        return ref_element, 'center', f'+({centered_offset(next_len)})'

    # Preferred side wasn't actually usable -- use whichever side is
    if prev_usable:
        return ref_element, 'center', f'-({centered_offset(prev_len)})'
    if next_usable:
        return ref_element, 'center', f'+({centered_offset(next_len)})'

    # Neither side is a plain, long-enough-to-fit open drift -> hug the
    # quad's own edge instead (small, fixed offset, cannot overshoot).
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

def insert_correctors(pdr, debug_check=False):
    """
    debug_check=True wraps EVERY single corrector insertion with a
    snapshot_quad_strengths()/check_quad_strength_conserved() pair, so if
    an insertion silently slices/removes part of some OTHER quad (the
    failure mode get_safe_insertion can't detect ahead of time, since it
    only knows about the target quad's immediate neighbors), you get an
    immediate, per-insertion warning naming exactly which quad it damaged
    and which corrector insertion did it -- instead of finding out several
    iterations later as a confusing 'element not found' error when
    something ELSE tries to reference the now-missing quad.

    This is slower (one extra get_table() call per insertion) so it's off
    by default; turn it on when chasing exactly this kind of bug.
    """
    offset = (pdr['l_quad'] + pdr['l_drift']) / 2
    pdr['l_kick'] = 0.1
    ring = pdr.lines['ring']

    if debug_check:
        all_quad_snapshot = snapshot_quad_strengths(ring)

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

    def safe_insert(new_element, prefix, elem, sign, default_offset):
        """
        Uses the module-level get_safe_insertion() pre-check (same one
        insert_correctors_var2 is meant to use) instead of trying an
        insertion and catching exceptions.

        This changed from an earlier version that wrapped ring.insert() in
        a try/except (ValueError, AssertionError) and fell back only if
        that raised. xtrack's Line.insert() does NOT raise in the failure
        case that matters here: it calls cut_at_s() to slice the line at
        the insertion boundaries -- which slices straight through a thick
        element like a quad if the target position lands inside one -- and
        then silently removes whatever ends up fully inside the insertion
        span. No exception is raised. So the old except-branch fallback
        could never actually trigger for the case it was written for, and
        a corrector landing inside a quad's span would silently delete
        part of that quad with no warning. get_safe_insertion looks at the
        real neighboring element types before choosing where to insert, so
        it doesn't depend on an exception that doesn't happen.

        NOTE: get_safe_insertion only checks the TARGET quad's own
        immediate neighbors. It has no way to know that inserting THIS
        corrector might reach far enough to damage some OTHER, unrelated
        quad elsewhere named later in the loop -- that's what debug_check
        is for.
        """
        if debug_check:
            before = snapshot_quad_strengths(ring, quad_prefixes=list(all_quad_snapshot.keys()))

        from_elem, anchor, final_offset = get_safe_insertion(
            ring, prefix, elem, sign, default_offset)
        ring.insert(new_element, at=final_offset,
                    from_=from_elem, from_anchor=anchor)

        if debug_check:
            flagged = check_quad_strength_conserved(ring, before, verbose=False)
            if flagged:
                print(f"[insert_correctors] inserting corrector for "
                      f"'{prefix}{elem}' damaged: "
                      f"{[f[0] for f in flagged]}")

    # ---- vertical correctors on QD-type quads ----
    for prefix in ['QDA_', 'QDA_M', 'QDDoub_', 'QDDS_', 'QDTrip_']:
        for elem in elems_for_prefix(prefix):
            sign, dynamic_offset = sign_and_offset(prefix, elem, 'v')
            v_name = f'vk_ring_{prefix}{elem}'
            pdr[v_name] = 0.0
            ref_element = prefix + elem

            new_element = pdr.new('My_' + ref_element, xt.Multipole,
                                   ksl=[pdr.ref[v_name]], length='l_kick')
            safe_insert(new_element, prefix, elem, sign, dynamic_offset)

    # ---- horizontal correctors on QF-type quads ----
    for prefix in ['QFA_', 'QFA_M', 'QFDoub_', 'QFDS_', 'QFTripC_']:
        for elem in elems_for_prefix(prefix):
            sign, dynamic_offset = sign_and_offset(prefix, elem, 'h')
            h_name = f'hk_ring_{prefix}{elem}'
            pdr[h_name] = 0.0
            ref_element = prefix + elem

            new_element = pdr.new('Mx_' + ref_element, xt.Multipole,
                                   knl=[pdr.ref[h_name]], length='l_kick')
            safe_insert(new_element, prefix, elem, sign, dynamic_offset)

    return pdr

def report_eigenvector_counts(ring, twiss, rcond_x=1e-4, rcond_y=1e-2):
    """
    Report, for each plane:
      - n_available: total singular values = min(n_monitors, n_correctors)
      - n_kept_by_rcond: how many actually survive the CURRENT rcond cutoff
        (S_inv[S < rcond * S[0]] = 0 inside xtrack's correction code -- this
        is what orbit_correction() uses whenever n_eig_x/n_eig_y is None,
        i.e. right now unless you've explicitly set them)

    Call this with the same rcond_x/rcond_y you're passing (or plan to
    pass) to orbit_correction(), so the "currently kept" number matches
    what that call would actually do.
    """
    corr_handler = ring.correct_trajectory(twiss_table=twiss, run=False)

    S_x = corr_handler.x_correction.singular_values
    S_y = corr_handler.y_correction.singular_values

    n_kept_x = int(np.sum(S_x >= rcond_x * S_x[0])) if len(S_x) else 0
    n_kept_y = int(np.sum(S_y >= rcond_y * S_y[0])) if len(S_y) else 0

    print(f"x-plane: {len(S_x)} singular values available, "
          f"{n_kept_x} kept at rcond_x={rcond_x}")
    print(f"y-plane: {len(S_y)} singular values available, "
          f"{n_kept_y} kept at rcond_y={rcond_y}")

    return dict(n_available_x=len(S_x), n_kept_x=n_kept_x,
               n_available_y=len(S_y), n_kept_y=n_kept_y,
               singular_values_x=S_x, singular_values_y=S_y)


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
        
        # NOTE: previously this always used from_anchor='center', discarding
        # the `anchor` value get_safe_insertion returns. That silently broke
        # the one case get_safe_insertion exists to handle -- when neither
        # neighbor is a plain open drift, it returns anchor='start'/'end'
        # with an edge-hugging offset (e.g. '-l_kick/2'), which only means
        # what it's supposed to mean when interpreted from that anchor, not
        # from 'center'. Now passes the returned anchor through.
        ring.insert(pdr.new('My_'+prefix+elem, xt.Multipole, ksl=[pdr.ref[v_name]], length='l_kick'), #edge_entry_active=True, edge_exit_active=True), 
                    at=final_offset, from_=from_elem, from_anchor=anchor)


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

        # See note above -- respect the returned anchor rather than
        # hardcoding 'center'.
        ring.insert(pdr.new('Mx_'+prefix+elem, xt.Multipole, knl=[pdr.ref[h_name]], length='l_kick'), #edge_entry_active=True, edge_exit_active=True), 
                    at=final_offset, from_=from_elem, from_anchor=anchor)

        
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


def snapshot_quad_strengths(ring, quad_prefixes=None):
    """
    Record each quad's integrated strength (sum of k1*length across all
    current element(s) matching its name) before an insertion step, for
    later comparison via check_quad_strength_conserved().

    Grouping is done by name PREFIX rather than exact name, so this is
    robust to however xtrack names pieces if it ends up slicing a quad
    during a later insert() call (e.g. 'QFA_1R1' and any 'QFA_1R1'-prefixed
    slice fragments all roll up into the same 'QFA_1R1' total).

    quad_prefixes: optional list of base quad names to track (e.g. from
    ring.get_table().rows[...].name at the time of the snapshot). If None,
    every current Quadrupole element is tracked under its own full name.
    """
    tab = ring.get_table()
    quad_rows = tab.rows[tab.element_type == 'Quadrupole']

    if quad_prefixes is None:
        quad_prefixes = list(quad_rows.name)

    totals = {p: 0.0 for p in quad_prefixes}
    for name in quad_rows.name:
        for p in quad_prefixes:
            if name == p or name.startswith(p + '.') or name.startswith(p + '_'):
                elem = ring[name]
                k1 = getattr(elem, 'k1', 0.0)
                length = getattr(elem, 'length', 0.0)
                totals[p] += k1 * length
                break
    return totals


def check_quad_strength_conserved(ring, snapshot_before, tol=1e-9, verbose=True):
    """
    Compare each quad's integrated strength against a snapshot taken with
    snapshot_quad_strengths() before an insertion step. Flags any quad
    whose total k1*length changed by more than `tol` -- the signature of
    an insertion having sliced through (and possibly removed part of) a
    quad rather than landing cleanly in an adjacent drift, since
    Line.insert() does this silently with no exception raised.

    Returns a list of (name, before, after, delta) tuples for anything
    flagged; empty list means nothing detected.
    """
    after = snapshot_quad_strengths(ring, quad_prefixes=list(snapshot_before.keys()))

    flagged = []
    for name, before_val in snapshot_before.items():
        after_val = after.get(name, 0.0)
        delta = after_val - before_val
        if abs(delta) > tol:
            flagged.append((name, before_val, after_val, delta))

    if verbose:
        if flagged:
            print(f"WARNING: {len(flagged)} quad(s) show a changed integrated "
                  f"strength after insertion -- likely silently sliced/altered:")
            for name, before_val, after_val, delta in flagged:
                print(f"  {name}: before={before_val:.6e}  after={after_val:.6e}  "
                      f"delta={delta:+.3e}")
        else:
            print(f"OK: all {len(snapshot_before)} tracked quads have unchanged "
                  f"integrated strength after insertion.")

    return flagged