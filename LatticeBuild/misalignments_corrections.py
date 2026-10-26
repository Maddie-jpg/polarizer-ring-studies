"""
Magnet errors, BPMs, correctors and closed-orbit correction.

Typical use from an analysis script:

    import LatticeBuild.misalignments_corrections as mc
    mc.insert_bpms_and_correctors(pdr, design, config)
    line = mc.prepare_seed_line(pdr.lines['ring'], seed=42, correct=True)

Contents
--------
Instrumentation  insert_BPMs*, insert_correctors*, insert_bpms_and_correctors
Errors           misalignments (magnets), misalignments_correctors, bpm_reading_errors
                 insert_multipole_errors / apply_multipole_errors - field-quality
                 errors, applied only if multipole_errors.json exists
Correction       orbit_correction, report_eigenvector_counts
Workflow         prepare_seed_line  - misalign + (optionally) correct one seed
                 setup_spin_reference - set g-2 and energy for spin tracking

All error sigmas are in m (shifts) and rad (rolls). Defaults live in config.py.
"""

import os
import re
import sys
import json
from math import factorial
from pathlib import Path

import numpy as np
import xtrack as xt
import xobjects as xo

# Make the repository root importable (for config.py) however this file is reached.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)
import config as cfg

_DRIFT_CLASSES = ('Drift', 'DriftSlice')


def _element_length(line, name):
    """
    True length of an element.

    DriftSlice stores `weight` and `parent_name`, NOT `length`, so a plain
    getattr(el, 'length', 0.0) silently returns 0.0 for every drift that a
    sextupole (or any other insertion) has split. That is what made
    get_safe_insertion treat sextupole-region drifts as unusable and hug the
    quad edge instead of centring in the drift.
    """
    el = line.element_dict[name]
    L = getattr(el, 'length', None)
    if L is not None:
        return float(L)
    w = getattr(el, 'weight', None)
    parent = getattr(el, 'parent_name', None)
    if w is not None and parent is not None and parent in line.element_dict:
        return float(w) * float(getattr(line.element_dict[parent], 'length', 0.))
    return 0.0


def _drift_span(line, idx, direction):
    """
    Walk away from element `idx` in `direction` (+1 downstream, -1 upstream)
    over contiguous drift space, and return

        (drift_length, bounding_index)

    Drift and DriftSlice both count towards the length; zero-length elements
    (Marker, BPM) are transparent and stepped over, because a BPM marker
    inserted earlier splits the drift without consuming any of it. The walk
    stops at the first element with real length -- the magnet bounding the
    drift. bounding_index is None if the line end is reached first.
    """
    names = line.element_names
    j = idx + direction
    total = 0.0
    while 0 <= j < len(names):
        cls = line.element_dict[names[j]].__class__.__name__
        L = _element_length(line, names[j])
        if cls in _DRIFT_CLASSES:
            total += L
            j += direction
            continue
        if L == 0.0:                       # marker / BPM: transparent
            j += direction
            continue
        return total, j                    # real magnet bounds the drift
    return total, None


def _bounded_by_sextupole(line, idx, direction):
    """True if the drift beside element idx (direction +1 downstream, -1
    upstream) ends at a sextupole."""
    _, bound = _drift_span(line, idx, direction)
    if bound is None:
        return False
    return 'Sextupole' in line.element_dict[line.element_names[bound]].__class__.__name__


def sextupole_free_side(line, quad_name, min_drift=None):
    """Side of quad_name for its BPM and corrector: '+' (downstream) or '-'
    (upstream), whichever neighbouring drift has no sextupole and can hold a
    corrector (min_drift, default l_kick or 0.1 m). Downstream when both
    qualify; if neither fits a corrector, the sextupole-free side; downstream
    when nothing else decides. Read from the lattice, so it is the same in R
    and L sextants, where the beam meets the elements in opposite order."""
    if min_drift is None:
        try:
            min_drift = line.vars['l_kick']._value
        except Exception:
            min_drift = 0.1
    idx = list(line.element_names).index(quad_name)
    free = {d: not _bounded_by_sextupole(line, idx, d) for d in (+1, -1)}
    room = {d: _drift_span(line, idx, d)[0] >= min_drift for d in (+1, -1)}
    for d in (+1, -1):
        if free[d] and room[d]:
            return '+' if d > 0 else '-'
    for d in (+1, -1):
        if free[d]:
            return '+' if d > 0 else '-'
    return '+'


def get_safe_insertion(ring, prefix, elem, sign, default_offset):
    """
    Return (anchor_element, from_anchor, offset) placing a corrector at the
    CENTRE of the drift space next to prefix+elem.

    The anchor is always the END of the magnet that bounds the drift on its
    upstream side, and the offset is +drift/2, so the corrector centre lands
    exactly mid-drift:

        sign '-'   [upstream magnet]===|<-- drift/2 -->K<-- -->|===[the quad]
        sign '+'   [the quad]=========|<-- drift/2 -->K<-- -->|===[next magnet]

    Previously this anchored on the quad's own start/end with a fixed
    +-l_kick/2 whenever the neighbouring drift was not a plain `Drift` long
    enough to hold the kicker. In a sextupole region the neighbour is a
    DriftSlice (the sextupole split the drift) whose length read as 0.0, so
    that fallback fired every time and the corrector was jammed hard against
    the quad face.
    """
    ref_element = prefix + elem
    names = ring.element_names

    if ref_element not in names:
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
    try:
        l_kick = ring.vars['l_kick']._value
    except Exception:
        l_kick = 0.1

    want = -1 if sign == '-' else +1
    span, bound = _drift_span(ring, idx, want)
    other, other_bound = _drift_span(ring, idx, -want)

    # Preferred side first, then the other side, then whichever is larger.
    if span < l_kick and other >= l_kick:
        want, span, bound = -want, other, other_bound

    if span < l_kick:
        raise RuntimeError(
            f"get_safe_insertion: no drift beside '{ref_element}' can hold a "
            f"{l_kick} m corrector (upstream {span if want<0 else other:.4f} m, "
            f"downstream {other if want<0 else span:.4f} m). Shorten l_kick or "
            f"open up the layout.")

    if want < 0:
        # drift sits between the bounding magnet and the quad: anchor on that
        # magnet's end. If nothing bounds it, fall back to the quad's start.
        if bound is None:
            return ref_element, 'start', -span / 2.0
        return names[bound], 'end', span / 2.0
    # drift sits after the quad: the quad itself is the upstream magnet
    return ref_element, 'end', span / 2.0
        
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
        sign = sextupole_free_side(ring, prefix + suffix)

        if plane == 'h' and prefix == 'QFA_' and is_center:
            dynamic_offset = '(((l_drift/2)+l_quad)/2)-l_sext'
        else:
            current_drift = drift_map.get(prefix, 'l_drift')
            dynamic_offset = f'({current_drift}+l_quad) / 2'
        return sign, dynamic_offset

    def safe_insert(new_element, prefix, elem, sign, default_offset):
        """
        Places new_element using get_safe_insertion()'s real-neighbor-length
        computation rather than the caller's generic drift_map guess.

        get_safe_insertion returns (ref_element, anchor, offset) -- the
        element to anchor against, which point ON IT to measure from, and
        the (string, deferred-expression) offset to place new_element's
        centre at. The anchor is therefore ring.insert's from_anchor.
        (It used to be passed as `anchor`, i.e. the kicker's own anchor,
        which measured the edge offsets from the quad centre and put 62
        correctors inside their quads.)
        """
        ref_element, anchor, insertion_offset = get_safe_insertion(
            ring, prefix, elem, sign, default_offset)

        ring.insert(new_element, at=insertion_offset, from_=ref_element,
                    from_anchor=anchor)

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
                      rcond_x=cfg.RCOND_X, rcond_y=cfg.RCOND_Y,
                      n_eig_x=cfg.N_EIG_X, n_eig_y=cfg.N_EIG_Y,
                      eig_fraction=None,
                      bpm_error=True, sigma=cfg.BPM_SIGMA, seed=1234):
    """SVD closed-orbit correction with BPM reading errors.

    How many singular values (eigenvectors) are kept in each plane, in order
    of precedence:
      n_eig_x / n_eig_y   count for that plane (default config.N_EIG_X /
                          N_EIG_Y = 15); capped at the number available
      eig_fraction        if the count is None: that fraction of the
                          available singular values, rounded down
      rcond_x / rcond_y   only if both of the above are None: drop singular
                          values below rcond * largest
    When a count is used the rcond cut is switched off, so the count alone
    decides. The numbers used are printed for each correction.
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

    monitor_align = None
    if bpm_error:
        BPMs = [name for name in tt.name if 'BPM' in name]
        rng = np.random.default_rng(seed)
        monitor_align = bpm_reading_errors(BPMs, sigma, rng)

    def run_correction(corr_handler):
        for plane, n_eig, rcond in (('x', n_eig_x, rcond_x), ('y', n_eig_y, rcond_y)):
            corr = getattr(corr_handler, f'{plane}_correction')
            n_avail = len(corr.singular_values)
            if n_eig is None and eig_fraction is not None:
                n_eig = max(1, int(np.floor(eig_fraction * n_avail)))
            if n_eig is not None:
                n_eig = min(int(n_eig), n_avail)
                corr.n_singular_values = n_eig
                corr.rcond = None
                print(f"  orbit correction {plane}: {n_eig} of {n_avail} singular values")
            else:
                corr.n_singular_values = None
                corr.rcond = rcond
                kept = int(np.sum(corr.singular_values >= rcond * corr.singular_values[0]))
                print(f"  orbit correction {plane}: rcond={rcond:g} keeps {kept} of {n_avail}")
        corr_handler.correct()

    if threading is False:
        corr_handler = ring.correct_trajectory(twiss_table=twiss, run=False,
                                                monitor_alignment=monitor_align)
        run_correction(corr_handler)
    else:
        tw0 = twiss
        corr_handler = ring.correct_trajectory(twiss_table=tw0, run=False,
                                                monitor_alignment=monitor_align)
        corr_handler.thread(ds_thread=10., rcond_long=1e-2)

        tw1 = ring.twiss(method='6d')
        corr_handler_final = ring.correct_trajectory(twiss_table=tw1, run=False,
                                                       monitor_alignment=monitor_align)
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

    # ------------------------------------------------------------------
    # Build corrector target lists by DISCOVERING the quads actually in the
    # ring, instead of hardcoding cell numbers. Hardcoded lists broke across
    # designs (e.g. this one has QFA_1R1..1R6 + 1RC and NO 1R0, so the old
    # '1R0' entry referenced a nonexistent element). Discovery adapts to
    # whatever cell numbering / sector count a given lattice actually uses.
    #
    # side: sextupole_free_side() -- the drift next to the quad that has no
    # sextupole, the same side as the quad's BPM.
    # ------------------------------------------------------------------
    all_names = list(ring.element_names)

    # longest-prefix match so 'QDA_' doesn't swallow 'QDA_M...', etc.
    known_prefixes = ['QDA_M', 'QFA_M', 'QDDS_', 'QFDS_', 'QDDoub_',
                      'QFDoub_', 'QDTrip_', 'QFTripC_', 'QDA_', 'QFA_']

    def suffixes_for(prefix):
        """All element-name suffixes for `prefix`, excluding names that
        belong to a longer known prefix. Skips already-sliced pieces."""
        longer = [p for p in known_prefixes if p != prefix and p.startswith(prefix)]
        out = []
        for n in all_names:
            if not n.startswith(prefix):
                continue
            if any(n.startswith(lp) for lp in longer):
                continue
            suf = n[len(prefix):]
            # skip slice artefacts / sub-elements (e.g. '..0', '_entry')
            if '..' in suf or suf.endswith('_entry') or suf.endswith('_exit') \
               or 'entry_map' in suf or 'exit_map' in suf:
                continue
            out.append(suf)
        return sorted(set(out))

    def build_list(prefixes):
        out, skipped = [], []
        for prefix in prefixes:
            for suf in suffixes_for(prefix):
                out.append([suf, sextupole_free_side(ring, prefix + suf), prefix])
        return out, skipped

    # vertical correctors go on the DEFOCUSING (QD*) quads
    qy_prefixes = ['QDA_', 'QDA_M', 'QDDoub_', 'QDDS_', 'QDTrip_']
    # horizontal correctors go on the FOCUSING (QF*) quads
    qx_prefixes = ['QFA_', 'QFA_M', 'QFDoub_', 'QFDS_', 'QFTripC_']

    qy_ring_list, qy_skipped = build_list(qy_prefixes)
    qx_ring_list, qx_skipped = build_list(qx_prefixes)
    if qy_skipped or qx_skipped:
        print(f"insert_correctors_var2: could not infer sign for "
              f"{qy_skipped + qx_skipped} -- skipped.")

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


    # horizontal correctors (qx_ring_list built dynamically above)
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
    """BPM marker at the face of every quadrupole of the ring and the period
    (BPMx_ on focusing, BPMy_ on defocusing quads), on the side whose drift
    has no sextupole (sextupole_free_side), so the BPM and the corrector of
    each quad share the sextupole-free drift."""
    bpm = xt.Marker()
    for line in (pdr.lines['ring'], pdr.lines['period']):
        tab = line.get_table()
        for elem_name in tab.rows[tab.element_type == 'Quadrupole'].name:
            k1 = line[elem_name].k1
            if k1 == 0:
                continue
            downstream = sextupole_free_side(line, elem_name) == '+'
            line.insert(('BPMx_' if k1 > 0 else 'BPMy_') + elem_name, bpm,
                        at=0.0, from_=elem_name,
                        from_anchor='end' if downstream else 'start')
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

    import bend_tools as bt
    tab = line.get_table()
    etype = dict(zip(tab.name, tab.element_type))
    quads = list(tab.rows[tab.element_type == 'Quadrupole'].name)
    sexts = list(tab.rows[tab.element_type == 'Sextupole'].name)
    # {magnet: [elements]}: a hard-edge bend is [itself], a field-expansion
    # bend all its pieces, so the whole magnet gets ONE set of errors.
    bends = bt.bend_groups(line, tab)

    # Same draw order as before (quads, sexts, bends; 7 numbers per magnet),
    # so hard-edge lattices get exactly the same errors for a given seed.
    for magnet, elems in [(q, [q]) for q in quads + sexts] + list(bends.items()):
        err = dict(shift_x=sigma * truncnorm(cut, rng),
                   shift_y=sigma * truncnorm(cut, rng),
                   shift_s=sigma * truncnorm(cut, rng),
                   rot_s_rad=sigma * truncnorm(cut, rng),
                   rot_x_rad=sigma * truncnorm(cut, rng),
                   rot_y_rad=sigma * truncnorm(cut, rng))
        eps = truncnorm(cut, rng) * 1e-3

        for name in elems:
            ref = line.element_refs[name]
            for k, v in err.items():
                setattr(ref, k, v)
            t = etype[name]
            if t == 'Quadrupole':
                ref.k1 = ref.k1 * (1 + eps)
            elif t == 'Sextupole':
                ref.k2 = ref.k2 * (1 + eps)
            elif t in ('Bend', 'RBend'):
                line[name].k0_from_h = False      # keep geometry (h), perturb the field
                ref.k0 = ref.h * (1 + eps)
            elif t == 'BFieldExpansion':
                ref.kscale = ref.kscale * (1 + eps)   # whole field profile
            elif t == 'DipoleEdge':
                ref.k = ref.k * (1 + eps)             # edge focusing follows the field

    # Field-quality errors on top, if a tolerance scan has set a level.
    # No file -> nothing happens and this is the alignment-only model.
    apply_multipole_errors(line, seed, cut=cut)

    return line

def misalignments_correctors(line, sigma, seed=None, cut=2.5):
    rng = np.random.default_rng(seed)

    actual_seed = rng.bit_generator.seed_seq.entropy
    print(f"Applying misalignments with seed: {actual_seed}")

    def truncnorm(cut, rgen):
        var = rgen.normal()
        if abs(var) > cut:
            var = truncnorm(cut, rgen)
        return var

    tab = line.get_table()
    correctors = list(tab.rows[tab.element_type == 'Multipole'].name)

    for name in correctors:
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

def bpm_reading_errors(bpm_names, sigma, rng, cut=2.5):
    def truncnorm(cut, rgen):
        var = rgen.normal()
        if abs(var) > cut:
            var = truncnorm(cut, rgen)
        return var

    return {
        name: {
            'shift_x':   sigma * truncnorm(cut, rng),
            'shift_y':   sigma * truncnorm(cut, rng),
            'rot_s_rad': sigma * truncnorm(cut, rng),
        }
        for name in bpm_names
    }


# =============================================================================
# Shared workflow helpers - use these instead of copying the steps into scripts
# =============================================================================

# =============================================================================
# Multipole field errors (section 2.5.2)
# =============================================================================
# tolerance_scan.py writes the adopted field-quality level to a JSON file.
# If that file exists, the error multipoles it describes are added to every
# lattice built here; if it does not, nothing happens and the lattice carries
# alignment and main-field errors only, exactly as before.
#
# The kicks have to be their own thin Multipole elements. Setting knl/ksl on a
# thick Bend/Quadrupole/Sextupole has no effect in this version of Xtrack
# (checked 26/10: no change in px/py even with the values set at construction
# and num_multipole_kicks raised), which is why they are inserted rather than
# written onto the magnets.

MULTIPOLE_ERROR_FILE = getattr(
    cfg, 'MULTIPOLE_ERROR_FILE',
    str(Path(_REPO_ROOT) / 'multipole_errors.json'))

_MPE_PREFIX = 'MPE'          # MPEs_<magnet> at the start, MPEe_<magnet> at the end
_MPE_N_KNL = 14              # room for harmonics up to b14 / a14
_MAIN_ORDER = {'Bend': 1, 'Quadrupole': 2, 'Sextupole': 3}


def load_multipole_errors(path=None):
    """Return the adopted multipole error settings, or None if there are none.

    None is the normal case while the tolerance scan has not been run yet.
    """
    path = path or MULTIPOLE_ERROR_FILE
    if not os.path.exists(path):
        return None
    with open(path) as f:
        errors = json.load(f)
    missing = {'adopted_scale', 'harmonics', 'R_ref'} - set(errors)
    if missing:
        print(f'WARNING: {path} is missing {sorted(missing)}; ignoring it')
        return None
    return errors


def insert_multipole_errors(pdr, line=None, path=None):
    """Put empty thin multipoles at the ends of every magnet, ready to be filled.

    Called for you by insert_bpms_and_correctors. Does nothing if there is no
    error file, so a lattice built without one is unchanged.
    """
    errors = load_multipole_errors(path)
    if errors is None:
        return pdr

    line = line if line is not None else pdr.lines['ring']
    tt = line.get_table()
    if any(t in ('BFieldExpansion', 'ThickSliceBFieldExpansion') for t in tt.element_type):
        print('WARNING: multipole errors are only set up for hard-edge magnets; '
              'skipping them for this lattice.')
        return pdr

    places = []
    for magnet_type in _MAIN_ORDER:
        for m in tt.rows[tt.element_type == magnet_type].name:
            for end in ('start', 'end'):
                name = f'{_MPE_PREFIX}{end[0]}_{m}'
                pdr.new(name, xt.Multipole,
                        knl=np.zeros(_MPE_N_KNL), ksl=np.zeros(_MPE_N_KNL))
                places.append(pdr.place(name, at=f'{m}@{end}'))
    line.insert(places)
    print(f'Multipole errors: inserted {len(places)} thin kicks '
          f'(scale {errors["adopted_scale"]}x the {errors.get("table_name", "?")} table)')
    return pdr


def apply_multipole_errors(line, seed, errors=None, cut=2.5):
    """Fill the thin kicks with the adopted harmonic content.

    Each harmonic is a fixed fraction b_n of its magnet's own integrated
    strength, scaled by the reference radius:

        knl[n-1] = K_(m-1)L * b_n * (n-1)!/(m-1)! * R_ref^(m-n)

    Half goes at the magnet entry and half at the exit. The kicks inherit the
    magnet's shift and roll, so feed-down from the misalignments is automatic.

    Does nothing if there is no error file or the kicks were never inserted.
    """
    errors = errors if errors is not None else load_multipole_errors()
    if errors is None:
        return line
    if not any(n.startswith(_MPE_PREFIX) for n in line.get_table().name):
        return line                      # insert_multipole_errors was never run

    scale = errors['adopted_scale']
    # its own random stream, so the alignment draws above are untouched
    rng = np.random.default_rng([seed if seed is not None else 0, 1])
    tt = line.get_table()
    refs = line.element_refs

    for magnet_type, m_order in _MAIN_ORDER.items():
        harmonics = errors['harmonics'].get(magnet_type, [])
        R_ref = errors['R_ref'].get(magnet_type)
        if not harmonics or R_ref is None:
            continue
        for m in tt.rows[tt.element_type == magnet_type].name:
            # drawn for every harmonic whether or not it is used, so the
            # stream does not depend on which harmonics are switched on
            draws = [_truncnorm(cut, rng) for _ in harmonics]
            if magnet_type == 'Quadrupole':
                KL = refs[m].k1 * refs[m].length
            elif magnet_type == 'Sextupole':
                KL = refs[m].k2 * refs[m].length
            else:
                KL = refs[m].k0 * refs[m].length

            for (component, n, systematic, random_rms), z in zip(harmonics, draws):
                bn = scale * (systematic + random_rms * z)
                strength = (0.5 * bn * factorial(n - 1) / factorial(m_order - 1)
                            * R_ref ** (m_order - n))
                for end in 'se':
                    kick = refs[f'{_MPE_PREFIX}{end}_{m}']
                    if component == 'b':
                        kick.knl[n - 1] = KL * strength
                    else:
                        kick.ksl[n - 1] = KL * strength
            # follow the host magnet so feed-down happens by itself
            for end in 'se':
                kick = line[f'{_MPE_PREFIX}{end}_{m}']
                kick.shift_x = line[m].shift_x
                kick.shift_y = line[m].shift_y
                kick.rot_s_rad = line[m].rot_s_rad
    return line


def _truncnorm(cut, rgen):
    """Standard normal with anything beyond +-cut redrawn (not clipped)."""
    v = rgen.normal()
    while abs(v) > cut:
        v = rgen.normal()
    return v


def insert_bpms_and_correctors(pdr, design, config):
    """Insert BPM markers and orbit correctors into pdr.lines['ring'] in place.

    D1 C1 uses the original corrector layout (insert_correctors_var2); every
    other lattice uses insert_correctors. This is the rule the analysis,
    macroparticle, spin and tolerance scripts all used individually.
    """
    insert_BPMs_all_as_markers(pdr)
    if design == 1 and config == 1:
        insert_correctors_var2(pdr)
    else:
        insert_correctors(pdr)
    insert_multipole_errors(pdr)      # no-op unless multipole_errors.json exists
    return pdr


def setup_spin_reference(line, kinetic_energy=cfg.E0):
    """Set the anomalous magnetic moment and energy needed for spin tracking."""
    line.particle_ref.anomalous_magnetic_moment = cfg.ANOMALOUS_MAGNETIC_MOMENT
    line.particle_ref.kinetic_energy0 = kinetic_energy
    return line


def correct_orbit_with_fallback(line, twiss, seed, **kwargs):
    """Run orbit_correction; if the closed orbit cannot be found, retry with
    trajectory threading. Extra kwargs (rcond_x, rcond_y, ...) are passed on.

    `twiss` provides the response matrix. Pass the twiss of the PERFECT
    machine (with the same BPMs/correctors), not of the misaligned one: the
    misaligned machine may have no closed orbit until it has been threaded.
    """
    try:
        orbit_correction(line, twiss, threading=False, seed=seed, **kwargs)
    except Exception as e:
        print(f"  [seed {seed}] orbit_correction(threading=False) raised "
              f"{type(e).__name__}: {e} -- retrying with threading=True")
        orbit_correction(line, twiss, threading=True, seed=seed, **kwargs)
    return line


def prepare_seed_line(base_line, seed, correct, *,
                      sigma=cfg.MISALIGN_SIGMA,
                      corrector_sigma=cfg.CORRECTOR_SIGMA,
                      radiation='mean',
                      context=None,
                      copy=True,
                      report_orbit=False):
    """Return a line with one random error seed applied, optionally corrected.

    Steps:
      1. copy base_line (unless copy=False), set radiation model, build tracker
      2. misalignments(sigma, seed)                  - magnet shifts, rolls, field errors
      3. if correct:
           twiss 6d of the misaligned machine
           misalignments_correctors(corrector_sigma, seed + 1)
           orbit correction, falling back to threading if needed

    Corrector errors use seed + 1 so they are independent of the magnet errors
    but still reproducible. base_line must already contain BPMs and correctors
    if correct=True (see insert_bpms_and_correctors).

    Parameters
    ----------
    base_line : xt.Line       The perfect ring.
    seed : int                Random seed for this machine.
    correct : bool            Apply orbit correction after misaligning.
    sigma, corrector_sigma    RMS errors [m, rad]; defaults from config.py.
    radiation : str or None   Radiation model while preparing ('mean' by default).
    context                   xobjects context; defaults to CPU with OpenMP.
    copy : bool               Work on a copy (True) or modify base_line in place.
    report_orbit : bool       Print RMS closed orbit before/after correction
                              (costs one extra twiss).
    """
    line = base_line.copy() if copy else base_line
    line.configure_radiation(model=radiation)
    line.build_tracker(_context=context or xo.ContextCpu(omp_num_threads=0))

    # Response matrix from the PERFECT machine, computed before the errors go
    # in -- as in a real machine. The misaligned ring often has no closed
    # orbit before correction (e.g. the F-D-F lattice: 8/10 seeds at 0.25 mm),
    # so a twiss of it here would fail before threading could ever run.
    tw_ref = line.twiss(method='6d' if radiation else '4d') if correct else None

    misalignments(line, sigma, seed=seed)

    if correct:
        misalignments_correctors(line, corrector_sigma, seed + 1)
        correct_orbit_with_fallback(line, tw_ref, seed)

        if report_orbit:
            tw_after = line.twiss(method='6d' if radiation else '4d')
            print(f"  [seed {seed}] corrected orbit RMS x: {np.std(tw_after.x):.3e}, "
                  f"y: {np.std(tw_after.y):.3e}")

    return line
