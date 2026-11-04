"""
lattice_sketches.py
===================

Design sketches of the polarizer-ring lattice: every distinct cell type,
one full arc (sextant), and the whole ring.

Each sketch is dimensioned -- magnet names, magnet lengths and drift
lengths are all written on the figure -- so it can be used as a layout
drawing rather than just a cartoon.

Typical use in analysis.py
--------------------------
    import lattice_sketches as ls

    pdr = lo.three_fold_periodicity_long(matched=True, ...)

    ls.sketch_all(pdr, sextant='1R', outdir='Results/sketches')

or one at a time:

    ls.sketch_cell_types(pdr, sextant='1R')
    ls.sketch_arc(pdr, sextant='1R')
    ls.sketch_ring(pdr)
    ls.sketch_period_beam_size(pdr, beam)   # beam from corrections.ring_beam_sizes

The three entry points return matplotlib Figures (a dict of them for
sketch_cell_types), so you can restyle or save them yourself.

Nothing here modifies the lattice -- it only reads element names, s
positions and lengths.
"""

import os
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D


# =============================================================================
# Element classification
# =============================================================================
#
# Everything is driven off the element NAME, because that is the one thing
# that is stable across the different lattice functions in linear_optics.py.
# Patterns are tried in order, first match wins.
#
# height is the half-height of the box, in the magnet band (axis units):
# focusing quads sit above the axis, defocusing below, bends straddle it.
# That is the usual convention on a lattice layout drawing and makes the
# FODO rhythm readable at a glance.

#   Warm colours  = horizontally focusing, drawn ABOVE the beam axis
#   Cool colours  = horizontally defocusing, drawn BELOW the beam axis
#   Hatch         = which section the magnet belongs to, so the families are
#                   still distinguishable in greyscale or on a printout:
#                     plain  arc      //  matching      \\  disp. suppressor
#                     ..     doublet  xx  triplet
_STYLES = [
    # (regex,                    kind,          colour,    half-h, y-off, hatch)
    (r'^RFCav',                  'cavity',      '#9467bd', 0.55,  0.0,  ''),
    (r'^Bend1|^Bend2|^B1_|^B2_', 'bend',        '#5b9bd5', 0.45,  0.0,  ''),
    (r'^BendDS|^BDS_',           'bend_ds',     '#2e75b6', 0.45,  0.0,  '\\\\'),
    (r'^WigPos|^WigNeg',         'wiggler',     '#00b0a0', 0.35,  0.0,  ''),
    (r'^XF|^SF',                 'sext_f',      '#2ca02c', 0.40,  0.55, ''),
    (r'^XD|^SD',                 'sext_d',      '#8fbf3f', 0.40, -0.55, ''),
    (r'^QFA_M|^QFarcM',          'quad_f_m',    '#d1495b', 0.70,  0.75, '//'),
    (r'^QDA_M|^QDarcM',          'quad_d_m',    '#3d5a80', 0.70, -0.75, '//'),
    (r'^QFDS',                   'quad_f_ds',   '#e8743b', 0.70,  0.75, '\\\\'),
    (r'^QDDS',                   'quad_d_ds',   '#4a7ba7', 0.70, -0.75, '\\\\'),
    (r'^QFDoub',                 'quad_f_doub', '#f4a259', 0.70,  0.75, '..'),
    (r'^QDDoub',                 'quad_d_doub', '#8ecae6', 0.70, -0.75, '..'),
    (r'^QFTrip|^QFtr',           'quad_f_tr',   '#bf5700', 0.70,  0.75, 'xx'),
    (r'^QDTrip|^QDtr',           'quad_d_tr',   '#48718a', 0.70, -0.75, 'xx'),
    (r'^QFA|^QFarc|^QFH|^QF_',   'quad_f',      '#c1272d', 0.70,  0.75, ''),
    (r'^QDA|^QDarc|^QD_',        'quad_d',      '#1f4e79', 0.70, -0.75, ''),
    (r'^Ctr|^Mkr|^Marker',       'marker',      '#7f7f7f', 0.25,  0.0,  ''),
]

# ---------------------------------------------------------------------------
# Correction, diagnostics and injection hardware.
#
# These are matched BEFORE _STYLES (see _classify).
#
# Correctors/kickers are MAGNETS and are drawn in line on the beam axis with
# everything else. They straddle the axis like dipoles, since that is what
# they are -- small steering dipoles -- and are given a shorter box than the
# main bends so they still read as correction elements. Up/down in this
# sketch means focusing/defocusing, which does not apply to a steering
# magnet, so H and V are told apart by colour instead.
#
# Only BPMs float: they measure rather than act, are usually zero-length,
# and there are enough of them that putting them in line would swamp the
# FODO rhythm the sketch exists to show. On the ring footprint they are
# pushed outside the ring as indicators, matching the existing survey_plot
# convention.
#
# Colours follow that same convention: horizontal = pink, vertical = cyan.
# Pure cyan (#00FFFF) is very light on white, so a slightly darker cyan is
# used for print legibility -- set COL_V = '#00FFFF' here to match exactly.
COL_H = '#FF69B4'      # hotpink, horizontal plane
COL_V = '#00A8B5'      # darkened cyan, vertical plane
COL_INJ = '#d62728'

# Naming conventions differ between groups, so several are accepted. The
# Mx/My and BPMx/BPMy forms are the ones used in this lattice; the rest are
# common alternatives kept as fallbacks. Add your own prefixes here rather
# than renaming lattice elements.
_DIAG_STYLES = [
    # (regex,                        kind,     colour,   label,   half-h, y-off)
    (r'^INJ|^SEPT|^Sept|^Inj',       'inj',    COL_INJ, 'Injection',  0.45, 0.0),
    # BPMs before correctors: BPMx must not be caught by a bare ^B rule.
    (r'^BPMx|^BPMX',                 'bpm_h',  COL_H,   'BPM (H)',    0.30, 0.0),
    (r'^BPMy|^BPMY',                 'bpm_v',  COL_V,   'BPM (V)',    0.30, 0.0),
    (r'^BPM|^MON[_0-9]|^PU[_0-9]|'
     r'^Mon_',                       'bpm',   '#111111','BPM',        0.30, 0.0),
    # Kickers / correctors -- in-line magnets. Mx / My are this lattice's
    # names; the rest are common alternatives kept as fallbacks.
    (r'^Mx|^MCBH|^CH[_0-9]|^CORH|'
     r'^COR_H|^HKICK|^XCOR|^HCOR',   'corr_h', COL_H,   'Corrector (H)', 0.32, 0.0),
    (r'^My|^MCBV|^CV[_0-9]|^CORV|'
     r'^COR_V|^VKICK|^YCOR|^VCOR',   'corr_v', COL_V,   'Corrector (V)', 0.32, 0.0),
    (r'^MCB|^COR|^KICK',             'corr',  '#9467bd','Corrector (H+V)',0.32, 0.0),
]

# Kinds NOT drawn as ordinary in-line magnets.
#   BPMs  -> floated in their own band / outside the ring
#   inj   -> drawn separately as a full-height flag plus an on-axis box
# Correctors are deliberately absent: they are magnets and are drawn in line.
_DIAG_KINDS = {'inj', 'bpm', 'bpm_h', 'bpm_v'}
# Kinds drawn as floating indicators rather than in-line hardware.
_BPM_KINDS = {'bpm', 'bpm_h', 'bpm_v'}
# Corrector kinds: thin kicks in the lattice, drawn at their physical length.
_CORR_KINDS = {'corr', 'corr_h', 'corr_v'}
CORR_MIN_LENGTH = 0.1   # [m] drawn length if a corrector carries no length

# Human-readable names for the legend, keyed by kind.
_KIND_LABEL = {
    'bend':        'Arc dipole',
    'bend_ds':     'DS dipole',
    'wiggler':     'Wiggler pole',
    'sext_f':      'Sextupole (F)',
    'sext_d':      'Sextupole (D)',
    'quad_f':      'QF  arc',
    'quad_d':      'QD  arc',
    'quad_f_m':    'QF  matching',
    'quad_d_m':    'QD  matching',
    'quad_f_ds':   'QF  disp. suppressor',
    'quad_d_ds':   'QD  disp. suppressor',
    'quad_f_doub': 'QF  doublet',
    'quad_d_doub': 'QD  doublet',
    'quad_f_tr':   'QF  triplet',
    'quad_d_tr':   'QD  triplet',
    'cavity':      'RF cavity',
    'marker':      'Marker',
    'other':       'Other',
    'inj':         'Injection',
    'corr_h':      'Corrector (H)',
    'corr_v':      'Corrector (V)',
    'corr':        'Corrector (H+V)',
    'bpm':         'BPM',
    'bpm_h':       'BPM (H)',
    'bpm_v':       'BPM (V)',
}


def _classify(name):
    """Return (kind, colour, half_height, y_offset, hatch) for an element."""
    # Correction / diagnostics / injection first, matched on their own
    # prefixes. Correctors carry real geometry here because they are drawn
    # in line with the magnets; for BPMs and injection the geometry is
    # unused, since draw_beamline gives those their own treatment.
    for pat, kind, colour, _lab, h, y0 in _DIAG_STYLES:
        if re.match(pat, name):
            return kind, colour, h, y0, ''
    for pat, kind, colour, h, y0, hatch in _STYLES:
        if re.match(pat, name):
            return kind, colour, h, y0, hatch
    return 'other', '#bbbbbb', 0.30, 0.0, ''


# ---------------------------------------------------------------------------
# Sliced lines
#
# Once a line is sliced -- which xtrack does on build_tracker(), on
# slice_thick_elements(), and often as a side effect of inserting elements
# into a line of thick magnets -- a single magnet stops existing under its
# own name. QFA_1R1 becomes:
#
#   QFA_1R1_entry  QFA_1R1..entry_map  drift_QFA_1R1..0  QFA_1R1..0
#   drift_QFA_1R1..1  QFA_1R1..1  ...  QFA_1R1..exit_map  QFA_1R1_exit
#
# so an anchored pattern like ^QFA_1R\d+$ matches nothing and the sketch
# silently finds no lattice. _base_name maps every slice back to its parent
# and _merge_slices glues them into one row with the parent's full length,
# so the rest of this module sees the unsliced lattice either way.
#
# ::N is deliberately NOT stripped: that suffix marks repeated placements of
# the same prototype (Drarc::0, Drarc::1), which are genuinely distinct
# elements at different s, not slices of one.
_SLICE_RE = re.compile(
    r'^(?:drift_)?(?P<base>.+?)'
    r'(?:\.\.(?:\d+|entry_map|exit_map)|_entry|_exit)$')


_FE_PIECE_RE = re.compile(r'^.+?_(?:fe\d+|edge_in|edge_out)\.(?P<magnet>.+)$')


def _base_name(name):
    """Map a slice name back to the element it was cut from, and a piece of
    a field-expansion dipole ('Bend_fe2.Bend1_1R8') to its magnet ('Bend1_1R8')."""
    m = _SLICE_RE.match(name)
    name = m.group('base') if m else name
    m = _FE_PIECE_RE.match(name)
    return m.group('magnet') if m else name


def _merge_slices(rows):
    """
    Collapse rows belonging to one sliced element into a single row carrying
    the parent name, its start s and its total length.

    A thin element placed inside a magnet (e.g. a corrector kick or marker
    inserted part-way along a quad) splits it into pieces with the same base
    name. Those pieces are merged too, and the thin element is kept as its
    own row after the magnet, so the magnet is drawn once at full length.
    """
    out, i, n = [], 0, len(rows)
    while i < n:
        base = _base_name(rows[i]['name'])
        j = i
        inside = []                                  # thin elements within the magnet
        while j + 1 < n:
            if _base_name(rows[j + 1]['name']) == base:
                j += 1
                continue
            # look past zero-length elements for another piece of `base`
            k = j + 1
            while k < n and rows[k].get('thin', rows[k]['length'] == 0) \
                    and _base_name(rows[k]['name']) != base:
                k += 1
            if k < n and k > j + 1 and _base_name(rows[k]['name']) == base \
                    and not rows[i]['is_drift']:
                inside.extend(rows[j + 1:k])
                j = k
                continue
            break
        grp = [r for r in rows[i:j + 1] if r not in inside]
        if len(grp) == 1 and grp[0]['name'] == base:
            out.append(grp[0])                       # untouched element
        else:
            kind, colour, h, y0, hatch = _classify(base)
            out.append(dict(name=base,
                            s=min(r['s'] for r in grp),
                            length=sum(r['length'] for r in grp),
                            kind=kind, colour=colour, half_h=h, y0=y0,
                            hatch=hatch, is_drift=_is_drift(base, None)))
        out.extend(inside)
        i = j + 1
    return out


def _is_drift(name, el):
    """True for drift space (a real Drift element, or an auto-inserted one)."""
    if el is not None:
        for tname in (type(el).__name__, el.__class__.__name__):
            if 'Drift' in tname:
                return True
    # Named drifts in linear_optics.py: Drarc, DrarcS, DrDSL, DrTrans,
    # DrDoub, DrTripl, DrTrips, plus xtrack's auto-inserted drift_N.
    return bool(re.match(r'^[Dd]r', name))


# =============================================================================
# Reading the lattice
# =============================================================================

def _get_element(line, name):
    """Fetch an element from a line/environment, tolerating API differences."""
    for getter in (
        lambda: line.element_dict[name],
        lambda: line[name],
        lambda: line.get(name),
    ):
        try:
            return getter()
        except Exception:
            continue
    return None


def _element_rows(line, merge_slices=True):
    """
    Flatten a line into a list of dicts:
        {name, s, length, kind, colour, half_h, y0, hatch, is_drift}

    s is the start of the element. Zero-length elements (markers) are kept
    so they can be drawn as ticks.

    merge_slices=True (default) glues sliced elements back together, so the
    sketch works on a line that has been sliced -- which happens as soon as
    a tracker is built or elements are inserted. Pass False to see the raw
    sliced names, which is occasionally useful when debugging.
    """
    tab = line.get_table()
    names = list(tab.name)
    svals = np.asarray(tab.s, dtype=float)
    n_tot = len(names)

    rows = []
    for i, nm in enumerate(names):
        if nm == '_end_point':
            continue
        el = _get_element(line, nm)

        # Length from consecutive s, not from el.length. Slice elements
        # (ThinSliceQuadrupole, DriftSliceQuadrupole, ...) report
        # length=None, so reading the attribute silently gives zero-length
        # magnets on any sliced line. The s difference is exact and works
        # for thick, thin and sliced elements alike; el.length is used only
        # as a fallback at the very last element, which has no successor.
        if i + 1 < n_tot:
            L = float(svals[i + 1] - svals[i])
            if L < 0:
                L = 0.0
        else:
            L = float(getattr(el, 'length', 0.0) or 0.0)

        drift = _is_drift(nm, el)
        kind, colour, h, y0, hatch = _classify(nm)
        s_start = float(svals[i])
        thin = (L == 0)
        if kind in _CORR_KINDS and L == 0:
            # Correctors are thin kicks (zero s-length), but the element
            # carries the physical corrector length (l_kick). Draw it at that
            # length, centred on the kick, so it is visible and dimensioned.
            L = float(getattr(el, 'length', 0.0) or 0.0) or CORR_MIN_LENGTH
            s_start -= L / 2.0
        rows.append(dict(name=nm, s=s_start, length=L, thin=thin,
                         kind=kind, colour=colour, half_h=h, y0=y0,
                         hatch=hatch, is_drift=drift))
    if merge_slices:
        rows = _merge_slices(rows)
    return rows


def _find_names(line, pattern):
    """All element names in `line` matching `pattern`, in s order."""
    tab = line.get_table()
    return [n for n in tab.name if n != '_end_point' and re.search(pattern, n)]


def _no_match_help(rows, sextant):
    """
    Build an explanatory message when no QFA_<sextant>N elements are found,
    listing the sextant labels that DO exist. A bare 'check the label' is
    not much help when the real cause is usually a different sextant naming
    or a line that has been sliced.
    """
    labels = sorted({m.group(1) for m in
                     (re.match(r'^QFA_([0-9A-Za-z]+?)\d+$', r['name'])
                      for r in rows) if m})
    sample = [r['name'] for r in rows if not r['is_drift']][:8]
    msg = [f"no QFA_{sextant}* elements in this line."]
    if labels:
        msg.append(f"Sextant labels present: {', '.join(labels)}. "
                   f"Pass one of these as sextant=.")
    else:
        msg.append("No QFA_<label><n> elements at all. If this line was "
                   "sliced, slice merging should have handled it -- check "
                   "that element names still start with QFA_/QDA_.")
        msg.append(f"First few non-drift names: {sample}")
    return ' '.join(msg)


def _index_of(rows, name):
    for i, r in enumerate(rows):
        if r['name'] == name:
            return i
    raise KeyError(f"element '{name}' not found in this line")


def _slice_rows(rows, start_name, end_name):
    """Inclusive slice of rows between two element names."""
    i0 = _index_of(rows, start_name)
    i1 = _index_of(rows, end_name)
    if i1 < i0:
        i0, i1 = i1, i0
    return rows[i0:i1 + 1]


# =============================================================================
# The drawing primitive
# =============================================================================

def _fmt_len(L):
    """Length label. mm precision is what a magnet drawing needs."""
    return f'{L:.3f}'.rstrip('0').rstrip('.') + ' m'


def draw_beamline(ax, rows, title=None, show_drifts=True,
                  label_every=1, min_label_gap=None, s_offset=None,
                  dim_row_frac=0.55, ref_span=None):
    """
    Draw one span of beamline onto `ax`, fully dimensioned.

    Layout, bottom to top:
        drift dimension row   (grey, italic)
        magnet dimension row  (black arrows + length)
        magnet band           (boxes, F above / D below the axis)
        BPM band              (floated indicators -- see below)
        name band             (staggered labels with leader lines)

    Correctors are magnets and are drawn in line on the beam axis with the
    quads and dipoles, straddling the axis as small steering dipoles and
    dimensioned like any other magnet.

    Only BPMs are floated, into their own band above the magnets, joined to
    the axis by a hairline showing where each one measures. They are often
    zero-length, so they are given a minimum drawn width. Injection is
    treated separately again, as a full-height flag.

    Parameters
    ----------
    rows : list of dict
        As returned by _element_rows / _slice_rows.
    show_drifts : bool
        Annotate the gaps between magnets with their length.
    label_every : int
        Label every n-th magnet (use >1 only for very dense spans).
    min_label_gap : float or None
        Minimum s separation between name labels, in metres. Labels closer
        than this are dropped. None = derive from the span width.
    s_offset : float or None
        Value subtracted from s so the span starts at 0. None = auto.
    ref_span : float or None
        Span [m] that label rotation, symbol widths, label spacing and the
        side margins are scaled to. None = the span drawn. Pass a cell length
        to draw a long span (e.g. the whole ring) at cell-plot scale.
    """
    diags = [r for r in rows if r['kind'] in _DIAG_KINDS]
    mags = [r for r in rows if not r['is_drift'] and r['length'] > 0
            and r['kind'] not in _DIAG_KINDS]
    marks = [r for r in rows if r['length'] == 0 and r['kind'] == 'marker']
    # Everything that physically occupies space, magnets and hardware alike.
    # Dimensions and free-drift gaps are computed from this, not from the
    # magnets alone -- otherwise a 0.15 m corrector sitting inside a 0.5 m
    # drift would still be reported as 0.5 m of free space, which is exactly
    # the number someone reads off a layout drawing to check clearances.
    occ = sorted([r for r in rows if not r['is_drift'] and r['length'] > 0],
                 key=lambda r: r['s'])

    if not rows:
        ax.text(0.5, 0.5, 'no elements in span', ha='center', va='center',
                transform=ax.transAxes)
        ax.axis('off')
        return

    s0 = rows[0]['s'] if s_offset is None else s_offset
    s1 = rows[-1]['s'] + rows[-1]['length']
    span = s1 - s0
    if span <= 0:
        span = 1.0
    ref = ref_span or span

    # ---- bands -------------------------------------------------------
    Y_AX = 0.0
    has_diag = bool(diags)
    Y_DIAG = 1.95                # centre of the diagnostics band
    DIAG_H = 0.34                # half-height of a diagnostics symbol
    # The name band sits above whichever is topmost.
    Y_NAME_BASE = (Y_DIAG + DIAG_H + 0.55) if has_diag else 1.75
    Y_NAME_STEP = 0.52           # stagger step between name levels
    N_NAME_LEVELS = 3
    Y_DIM = -2.10                # magnet dimension row (clears the QD boxes,
                                 # whose bottom edge sits at -1.45)
    Y_DRIFT = -3.30              # drift dimension row (clears rotated labels)

    # ---- beam axis ---------------------------------------------------
    ax.plot([0, span], [Y_AX, Y_AX], color='0.35', lw=1.0, zorder=1)

    # ---- magnet boxes ------------------------------------------------
    for r in mags:
        x = r['s'] - s0
        rect = Rectangle((x, r['y0'] - r['half_h']), r['length'],
                         2 * r['half_h'],
                         facecolor=r['colour'], edgecolor='black',
                         hatch=r.get('hatch', ''),
                         linewidth=0.8,
                         # correctors on top, so one that overlaps a magnet
                         # (older lattices) is still visible
                         zorder=4 if r['kind'] in _CORR_KINDS else 3)
        ax.add_patch(rect)

    # ---- diagnostics band: BPMs, correctors, injection ----------------
    if has_diag:
        ax.plot([0, span], [Y_DIAG, Y_DIAG], color='0.80', lw=0.8,
                ls=(0, (4, 3)), zorder=1)
        ax.text(-ref * 0.012, Y_DIAG, 'BPMs',
                ha='right', va='center', fontsize=6.5, color='0.45')

        w_min = ref * 0.006           # so a thin/zero-length element is seen
        for r in diags:
            x = r['s'] - s0
            w = max(r['length'], w_min)
            xc = x + r['length'] / 2.0

            if r['kind'] in _BPM_KINDS:
                # BPM: open square straddling the band line, plus a stem
                # down to the beam axis showing where it measures.
                ax.plot([xc, xc], [Y_AX, Y_DIAG - DIAG_H], color='0.6',
                        lw=0.6, zorder=2)
                ax.add_patch(Rectangle((xc - w_min, Y_DIAG - DIAG_H * 0.75),
                                       2 * w_min, DIAG_H * 1.5,
                                       facecolor='white',
                                       edgecolor=r['colour'],
                                       linewidth=1.3, zorder=4))
            elif r['kind'] == 'inj':
                continue              # drawn separately, full height
            else:
                ax.plot([xc, xc], [Y_AX, Y_DIAG - DIAG_H], color='0.6',
                        lw=0.6, zorder=2)
                ax.add_patch(Rectangle((x, Y_DIAG - DIAG_H), w, 2 * DIAG_H,
                                       facecolor=r['colour'],
                                       edgecolor='black',
                                       linewidth=0.7, zorder=4))

            ax.text(xc, Y_DIAG + DIAG_H + 0.08, r['name'], rotation=90,
                    ha='center', va='bottom', fontsize=6.0,
                    color=r['colour'], zorder=5)

    # ---- injection point: a full-height flag --------------------------
    for r in [d for d in diags if d['kind'] == 'inj']:
        xc = r['s'] - s0 + r['length'] / 2.0
        # Sit the label above every name level so it cannot collide with
        # the staggered magnet labels.
        y_flag = Y_NAME_BASE + N_NAME_LEVELS * Y_NAME_STEP + 0.05
        ax.axvline(xc, color=r['colour'], lw=1.6, ls=(0, (5, 2)),
                   zorder=5, alpha=0.9)
        if r['length'] > 0:
            ax.add_patch(Rectangle((r['s'] - s0, -0.30), r['length'], 0.60,
                                   facecolor=r['colour'], edgecolor='black',
                                   linewidth=0.8, zorder=5))
        ax.annotate(f'  {r["name"]}', xy=(xc, Y_AX), xytext=(xc, y_flag),
                    fontsize=9, fontweight='bold', color=r['colour'],
                    ha='left', va='bottom', zorder=6,
                    arrowprops=dict(arrowstyle='-|>', color=r['colour'],
                                    lw=1.2))

    # ---- markers (zero length) ---------------------------------------
    for r in marks:
        x = r['s'] - s0
        ax.plot([x, x], [-1.05, 1.05], color=r['colour'], lw=1.2,
                ls=(0, (3, 2)), zorder=2)
        ax.text(x, 1.12, r['name'], rotation=90, ha='center', va='bottom',
                fontsize=6.5, color=r['colour'], zorder=4)

    # ---- magnet names, staggered -------------------------------------
    if min_label_gap is None:
        min_label_gap = ref / 60.0

    last_lab_x = -1e9
    lvl = 0
    for i, r in enumerate(mags):
        if i % label_every:
            continue
        xc = r['s'] - s0 + r['length'] / 2.0
        if xc - last_lab_x < min_label_gap:
            continue
        last_lab_x = xc

        y_txt = Y_NAME_BASE + (lvl % N_NAME_LEVELS) * Y_NAME_STEP
        lvl += 1

        y_box_top = r['y0'] + r['half_h']
        ax.plot([xc, xc], [y_box_top, y_txt - 0.08],
                color='0.55', lw=0.6, zorder=2)
        ax.text(xc, y_txt, r['name'], ha='center', va='bottom',
                fontsize=7.5, zorder=4,
                bbox=dict(boxstyle='round,pad=0.18', facecolor='white',
                          edgecolor='0.8', linewidth=0.5))

    # ---- length dimensions (magnets and hardware) ---------------------
    for r in occ:
        x = r['s'] - s0
        L = r['length']
        ax.annotate('', xy=(x, Y_DIM), xytext=(x + L, Y_DIM),
                    arrowprops=dict(arrowstyle='<->', color='black',
                                    lw=0.8, shrinkA=0, shrinkB=0),
                    zorder=3)
        # tick marks tying the dimension to the magnet edges
        for xe in (x, x + L):
            ax.plot([xe, xe], [Y_DIM - 0.14, Y_DIM + 0.14],
                    color='black', lw=0.7, zorder=3)
        # Horizontal labels go above the arrow; when the magnet is too narrow
        # the label is rotated, and must then go BELOW the arrow or it runs
        # up into the magnet boxes that hang under the axis.
        if L / ref > 0.035:
            ax.text(x + L / 2.0, Y_DIM + 0.22, _fmt_len(L),
                    ha='center', va='bottom', fontsize=7, zorder=4)
        else:
            ax.text(x + L / 2.0, Y_DIM - 0.22, _fmt_len(L),
                    ha='center', va='top', fontsize=7, rotation=90, zorder=4)

    # ---- drift dimensions --------------------------------------------
    if show_drifts and len(occ) > 1:
        for a, b in zip(occ[:-1], occ[1:]):
            xa = a['s'] - s0 + a['length']
            xb = b['s'] - s0
            gap = xb - xa
            if gap <= 1e-9:
                continue
            ax.annotate('', xy=(xa, Y_DRIFT), xytext=(xb, Y_DRIFT),
                        arrowprops=dict(arrowstyle='<->', color='0.45',
                                        lw=0.7, shrinkA=0, shrinkB=0),
                        zorder=3)
            for xe in (xa, xb):
                ax.plot([xe, xe], [Y_DRIFT - 0.12, Y_DRIFT + 0.12],
                        color='0.45', lw=0.6, zorder=3)
            rot = 0 if gap / ref > 0.030 else 90
            ax.text((xa + xb) / 2.0, Y_DRIFT - 0.20, _fmt_len(gap),
                    ha='center', va='top', fontsize=6.5, color='0.35',
                    style='italic', rotation=rot, zorder=4)

    # ---- overall length ----------------------------------------------
    Y_TOT = Y_DRIFT - 1.10
    ax.annotate('', xy=(0, Y_TOT), xytext=(span, Y_TOT),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.1,
                                shrinkA=0, shrinkB=0), zorder=3)
    for xe in (0, span):
        ax.plot([xe, xe], [Y_TOT - 0.16, Y_TOT + 0.16], color='black',
                lw=0.9, zorder=3)
    ax.text(span / 2.0, Y_TOT - 0.22,
            f'total length = {_fmt_len(span)}',
            ha='center', va='top', fontsize=8.5, fontweight='bold')

    # ---- axes cosmetics ----------------------------------------------
    y_top = Y_NAME_BASE + N_NAME_LEVELS * Y_NAME_STEP + 0.35
    if any(d['kind'] == 'inj' for d in diags):
        y_top += 0.45           # room for the injection flag label
    ax.set_xlim(-ref * 0.035, span + ref * 0.035)
    ax.set_ylim(Y_TOT - 1.0, y_top)
    ax.set_yticks([])
    ax.set_xlabel('s along the span  [m]', fontsize=9)
    ax.tick_params(axis='x', labelsize=8)
    for side in ('left', 'right', 'top'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_position(('outward', 6))
    ax.grid(axis='x', color='0.9', lw=0.6, zorder=0)
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', pad=14)


def _legend_for(rows, ax, ncol=5):
    """Legend built only from the kinds actually present in `rows`."""
    seen = {}
    for r in rows:
        # Diagnostics are often zero-length, so they are kept regardless.
        if r['kind'] not in _DIAG_KINDS and (r['is_drift'] or r['length'] == 0):
            continue
        if r['is_drift']:
            continue
        seen.setdefault(r['kind'], (r['colour'], r.get('hatch', '')))
    # BPMs are drawn as open squares on the figure, so their legend key must
    # be open too -- otherwise a BPM and a kicker of the same plane show as
    # two identical filled boxes.
    handles = [Rectangle((0, 0), 1, 1,
                         facecolor=('white' if k in _BPM_KINDS else c),
                         edgecolor=(c if k in _BPM_KINDS else 'black'),
                         hatch=h,
                         linewidth=(1.4 if k in _BPM_KINDS else 0.7),
                         label=_KIND_LABEL.get(k, k))
               for k, (c, h) in seen.items()]
    if handles:
        ax.legend(handles=handles, loc='upper center',
                  bbox_to_anchor=(0.5, -0.30), ncol=ncol, frameon=False,
                  fontsize=8.5)


# =============================================================================
# Cell-type discovery
# =============================================================================

def discover_cells(pdr, sextant='1R', line_name='ring'):
    """
    Work out, from the element names actually present, where each distinct
    cell type starts and ends in one sextant.

    Returns an ordered dict  {label: (start_name, end_name)}.

    The naming convention assumed is the one used throughout
    linear_optics.py:
        QFA_<sext><i>, QDA_<sext><i>, Bend1_<sext><i>, Bend2_<sext><i>
        QFA_M<sext><n>, QDA_M<sext><n>
        QFDS_<sext>, QDDS_<sext>, BendDS_<sext>
        QFDoub_<sext>, QDDoub_<sext>, QFTrip_<sext>1 / QDTrip_<sext>1

    Anything not found is skipped with a warning rather than raising, so
    this still works on the variants (F-D-F straight, sextupoles added,
    wigglers added).
    """
    line = pdr.lines[line_name]
    rows = _element_rows(line)
    present = {r['name'] for r in rows}

    def have(n):
        return n in present

    spans = {}

    # --- regular arc FODO cell -------------------------------------
    qfa = sorted(
        [n for n in present if re.match(rf'^QFA_{sextant}\d+$', n)],
        key=lambda n: int(re.search(r'(\d+)$', n).group(1)))
    if len(qfa) >= 3:
        # take a mid-arc cell, away from both boundaries
        k = len(qfa) // 2
        spans['Regular arc cell (FODO)'] = (qfa[k], qfa[k + 1]) \
            if k + 1 < len(qfa) else (qfa[k - 1], qfa[k])
    elif len(qfa) == 2:
        spans['Regular arc cell (FODO)'] = (qfa[0], qfa[1])

    # --- arc cell carrying sextupoles -------------------------------
    sext_names = [n for n in present
                  if re.match(rf'^(XF|XD|SF|SD)\w*_{sextant}\d+$', n)]
    if sext_names and len(qfa) >= 2:
        # find the first QFA cell that contains a sextupole
        for a, b in zip(qfa[:-1], qfa[1:]):
            ia, ib = _index_of(rows, a), _index_of(rows, b)
            if any(r['kind'] in ('sext_f', 'sext_d') for r in rows[ia:ib + 1]):
                spans['Arc cell with sextupoles'] = (a, b)
                break

    # --- matching cell ----------------------------------------------
    qfam = [n for n in present if re.match(rf'^QFA_M{sextant}\d+$', n)]
    if qfam and have(f'QDDS_{sextant}'):
        spans['Matching cell'] = (qfam[0], f'QDDS_{sextant}')
    elif qfam and have(f'QFDS_{sextant}'):
        spans['Matching cell'] = (qfam[0], f'QFDS_{sextant}')

    # --- dispersion suppressor --------------------------------------
    if have(f'QFDS_{sextant}') and have(f'BendDS_{sextant}'):
        spans['Dispersion suppressor'] = (f'QFDS_{sextant}',
                                          f'BendDS_{sextant}')

    # --- straight / insertion ---------------------------------------
    tail = None
    for cand in (f'QFTrip_{sextant}1', f'QDTrip_{sextant}1',
                 f'QFTripC_{sextant}2H', f'QDTripC_{sextant}2H',
                 f'QFTripC_{sextant}2', f'QDTripC_{sextant}2'):
        if have(cand):
            tail = cand
    head = None
    for cand in (f'BendDS_{sextant}', f'QFDoub_{sextant}',
                 f'QDDoub_{sextant}'):
        if have(cand):
            head = cand
            break
    if head and tail:
        spans['Straight section (insertion)'] = (head, tail)

    # --- wiggler period, if wigglers were inserted -------------------
    wig = sorted([n for n in present if re.match(r'^WigPos_.*_p0$', n)])
    if wig:
        base = wig[0].rsplit('_p0', 1)[0]
        last = f'{base}_p1'
        if have(last):
            spans['Wiggler period'] = (wig[0], last)

    # --- injection region, if an injection element exists ------------
    inj = [r for r in rows if r['kind'] == 'inj']
    if inj:
        i = _index_of(rows, inj[0]['name'])
        lo = max(0, i - 14)
        hi = min(len(rows) - 1, i + 14)
        spans['Injection region'] = (rows[lo]['name'], rows[hi]['name'])

    if not spans:
        print(f'discover_cells: nothing matched for sextant {sextant!r}. '
              f'Check the sextant label against ring element names.')
    return spans


# =============================================================================
# Public: the three sketch families
# =============================================================================

def draw_beam_size(axb, beam, s_a, s_b, n_sigma=None, legend_loc='lower left'):
    """
    Plot the rms beam sizes of `beam` between s_a and s_b on `axb`, against
    s - s_a (the convention draw_beamline uses with s_offset=s_a), so it lines
    up with a layout drawn underneath on a shared x axis.

    beam : dict with 's', 'sig_x', 'sig_y' (arrays over the ring, metres), as
        returned by corrections.ring_beam_sizes.
    n_sigma : also draw the n_sigma envelope (dashed), or None for rms only.
    Returns the top of the y axis [mm].
    """
    s = np.asarray(beam['s'], dtype=float)
    keep = (s >= s_a - 1e-9) & (s <= s_b + 1e-9)
    s = s[keep] - s_a
    sx = np.asarray(beam['sig_x'], dtype=float)[keep] * 1e3
    sy = np.asarray(beam['sig_y'], dtype=float)[keep] * 1e3
    if not len(s):
        raise ValueError(f'draw_beam_size: no beam-size points between s = {s_a:.3f} '
                         f'and {s_b:.3f} m')

    axb.plot(s, sx, color=COL_H, lw=1.4, label=r'$\sigma_x$ (rms)')
    axb.plot(s, sy, color=COL_V, lw=1.4, label=r'$\sigma_y$ (rms)')
    if n_sigma:
        axb.plot(s, n_sigma * sx, color=COL_H, lw=0.9, ls='--',
                 label=rf'{n_sigma:g}$\sigma_x$')
        axb.plot(s, n_sigma * sy, color=COL_V, lw=0.9, ls='--',
                 label=rf'{n_sigma:g}$\sigma_y$')
    y_max = (n_sigma or 1) * max(sx.max(), sy.max()) * 1.15
    axb.set_ylim(0, y_max)
    axb.set_ylabel('beam size  [mm]', fontsize=10)
    axb.tick_params(axis='y', labelright=True, right=True)
    axb.tick_params(axis='x', labelbottom=False)
    axb.grid(color='0.92', lw=0.6)
    axb.legend(loc=legend_loc, fontsize=9, framealpha=0.95)
    for sig, col, lab in ((sx, COL_H, 'x'), (sy, COL_V, 'y')):
        i = int(np.argmax(sig))
        axb.plot(s[i], sig[i], 'o', color=col, mec='black', ms=6, zorder=5)
        near_end = s[i] > 0.8 * (s_b - s_a)
        axb.annotate(rf'max $\sigma_{lab}$ = {sig[i]:.2f} mm', xy=(s[i], sig[i]),
                     xytext=(-8 if near_end else 8, 6), textcoords='offset points',
                     ha='right' if near_end else 'left', fontsize=8.5,
                     zorder=6, bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                         edgecolor=col, linewidth=0.8))
    return y_max


def sketch_cell_types(pdr, sextant='1R', line_name='ring', spans=None,
                      outdir=None, dpi=200, width=15.0, beam=None, n_sigma=None):
    """
    One dimensioned figure per distinct cell type.

    spans : dict {label: (start_name, end_name)} to override auto-discovery.
    beam : None (default) draws the layout only. A beam-size dict (as from
        corrections.ring_beam_sizes) adds the rms beam sizes above each
        layout on the same s axis, saved as cell_<label>_<sextant>_beam.png.
    n_sigma : with beam, also draw the n_sigma envelope.
    Returns {label: Figure}.
    """
    line = pdr.lines[line_name]
    rows_all = _element_rows(line)
    spans = spans or discover_cells(pdr, sextant, line_name)

    figs = {}
    for label, (a, b) in spans.items():
        try:
            rows = _slice_rows(rows_all, a, b)
        except KeyError as e:
            print(f'sketch_cell_types: skipping {label!r} -- {e}')
            continue

        n_mag = len([r for r in rows if not r['is_drift'] and r['length'] > 0])
        h = 5.4 if n_mag <= 12 else 6.0
        title = f'{label}   —   sextant {sextant}'
        if beam is None:
            fig, ax = plt.subplots(figsize=(width, h))
            draw_beamline(ax, rows, title=title)
        else:
            fig = plt.figure(figsize=(width, h + 2.8))
            gs = fig.add_gridspec(2, 1, height_ratios=[2.6, h], hspace=0.05)
            axb = fig.add_subplot(gs[0])
            ax = fig.add_subplot(gs[1], sharex=axb)
            draw_beamline(ax, rows)
            draw_beam_size(axb, beam, rows[0]['s'],
                           rows[-1]['s'] + rows[-1]['length'], n_sigma=n_sigma)
            axb.set_title(title, fontsize=12, fontweight='bold', pad=10)
        _legend_for(rows, ax)
        fig.tight_layout()

        if outdir:
            os.makedirs(outdir, exist_ok=True)
            safe = re.sub(r'[^A-Za-z0-9]+', '_', label).strip('_').lower()
            tag = '_beam' if beam is not None else ''
            fig.savefig(os.path.join(outdir, f'cell_{safe}_{sextant}{tag}.png'),
                        dpi=dpi, bbox_inches='tight')
        figs[label] = fig

    return figs


def sketch_arc(pdr, sextant='1R', line_name='ring', outdir=None, dpi=200,
               n_cells_shown=2, width=17.0, beam=None, n_sigma=None):
    """
    One sextant, drawn as a block diagram.

    A full sextant is far too dense to label element by element (12 cells
    is ~100 elements), so the repeating arc cells are collapsed into a
    single '<n> x regular arc cell' block while the matching cell, the
    dispersion suppressor and the straight are drawn individually. The
    first `n_cells_shown` cells are drawn in detail so the rhythm is
    visible.

    beam : None (default) for the layout only. A beam-size dict (as from
        corrections.ring_beam_sizes) adds the rms beam sizes above the
        detailed cells, saved as arc_<sextant>_beam.png.
    n_sigma : with beam, also draw the n_sigma envelope.
    """
    line = pdr.lines[line_name]
    rows_all = _element_rows(line)
    present = {r['name'] for r in rows_all}

    qfa = sorted([n for n in present if re.match(rf'^QFA_{sextant}\d+$', n)],
                 key=lambda n: int(re.search(r'(\d+)$', n).group(1)))
    if not qfa:
        raise KeyError(_no_match_help(rows_all, sextant))

    # span of the whole sextant: first arc quad -> last triplet quad
    first = qfa[0]
    last = None
    # Candidates are listed most-downstream first, so take the first hit --
    # without the break this picks an upstream quad and the straight section
    # gets cut short.
    for cand in (f'QFTripC_{sextant}2H', f'QDTripC_{sextant}2H',
                 f'QFTripC_{sextant}2', f'QDTripC_{sextant}2',
                 f'QFTrip_{sextant}1', f'QDTrip_{sextant}1',
                 f'QDDoub_{sextant}'):
        if cand in present:
            last = cand
            break
    if last is None:
        last = rows_all[-1]['name']

    rows = _slice_rows(rows_all, first, last)
    s0 = rows[0]['s']
    s_end = rows[-1]['s'] + rows[-1]['length']

    # ---- where the detailed cells end -------------------------------
    k = min(n_cells_shown, len(qfa) - 1)
    s_detail_end = next(r['s'] for r in rows_all if r['name'] == qfa[k])

    # ---- where the arc cells stop (start of matching cell) ----------
    qfam = [n for n in present if re.match(rf'^QFA_M{sextant}\d+$', n)]
    s_arc_end = (next(r['s'] for r in rows_all if r['name'] == qfam[0])
                 if qfam else s_detail_end)

    title_top = f'Arc (sextant {sextant}) — first {k} cells in detail'
    detail_rows = [r for r in rows if r['s'] < s_detail_end]
    if beam is None:
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(width, 10.5),
            gridspec_kw=dict(height_ratios=[1.0, 1.25], hspace=0.45))
    else:
        fig = plt.figure(figsize=(width, 14.0))
        outer = fig.add_gridspec(2, 1, height_ratios=[1.6, 1.25], hspace=0.30)
        inner = outer[0].subgridspec(2, 1, height_ratios=[0.6, 1.0], hspace=0.05)
        ax_beam = fig.add_subplot(inner[0])
        ax_top = fig.add_subplot(inner[1], sharex=ax_beam)
        ax_bot = fig.add_subplot(outer[1])

    # ================= top: schematic of the whole sextant ===========
    draw_beamline(ax_top, detail_rows,
                  title=title_top if beam is None else None, s_offset=s0)
    if beam is not None:
        draw_beam_size(ax_beam, beam, s0,
                       detail_rows[-1]['s'] + detail_rows[-1]['length'],
                       n_sigma=n_sigma)
        ax_beam.set_title(title_top, fontsize=12, fontweight='bold', pad=10)

    # ================= bottom: block diagram =========================
    ax = ax_bot
    total = s_end - s0
    n_arc_cells = len(qfa)

    blocks = []
    blocks.append((0.0, s_arc_end - s0,
                   f'{n_arc_cells} × regular arc cell', '#5b9bd5'))
    if qfam:
        s_ds = next((r['s'] for r in rows_all
                     if r['name'] == f'QFDS_{sextant}'), None)
        if s_ds is not None:
            blocks.append((s_arc_end - s0, s_ds - s0,
                           'matching cell', '#d1495b'))
            s_str = next((r['s'] for r in rows_all
                          if r['name'] == f'BendDS_{sextant}'), None)
            if s_str is not None:
                s_str_end = s_str + next(
                    r['length'] for r in rows_all
                    if r['name'] == f'BendDS_{sextant}')
                blocks.append((s_ds - s0, s_str_end - s0,
                               'dispersion suppressor', '#e8743b'))
                blocks.append((s_str_end - s0, total,
                               'straight (insertion)', '#f2a65a'))

    for x0, x1, lab, col in blocks:
        w = x1 - x0
        if w <= 0:
            continue
        ax.add_patch(Rectangle((x0, -0.5), w, 1.0, facecolor=col,
                               edgecolor='black', lw=1.0, alpha=0.85))
        # A short block cannot hold its own label, so float it above with a
        # leader line rather than letting the text spill over its neighbours.
        if w / total >= 0.16:
            ax.text(x0 + w / 2, 0.0, lab, ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='0.25',
                              alpha=0.35, edgecolor='none'))
        else:
            ax.plot([x0 + w / 2, x0 + w / 2], [0.5, 0.72],
                    color='0.5', lw=0.7)
            ax.text(x0 + w / 2, 0.76, lab, ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color=col,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor='0.8', linewidth=0.5))
        # dimension under the block
        ax.annotate('', xy=(x0, -0.95), xytext=(x1, -0.95),
                    arrowprops=dict(arrowstyle='<->', color='black', lw=0.9,
                                    shrinkA=0, shrinkB=0))
        for xe in (x0, x1):
            ax.plot([xe, xe], [-1.05, -0.85], color='black', lw=0.8)
        ax.text(x0 + w / 2, -1.12, _fmt_len(w), ha='center', va='top',
                fontsize=8.5)

    ax.annotate('', xy=(0, -1.75), xytext=(total, -1.75),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.2,
                                shrinkA=0, shrinkB=0))
    ax.text(total / 2, -1.90,
            f'sextant length = {_fmt_len(total)}',
            ha='center', va='top', fontsize=10, fontweight='bold')

    ax.set_xlim(-total * 0.02, total * 1.02)
    ax.set_ylim(-2.4, 1.45)
    ax.set_yticks([])
    ax.set_xlabel('s along the sextant  [m]', fontsize=9)
    for side in ('left', 'right', 'top'):
        ax.spines[side].set_visible(False)
    ax.grid(axis='x', color='0.9', lw=0.6)
    ax.set_title(f'Sextant {sextant} — block layout', fontsize=12,
                 fontweight='bold', pad=12)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        tag = '_beam' if beam is not None else ''
        fig.savefig(os.path.join(outdir, f'arc_{sextant}{tag}.png'),
                    dpi=dpi, bbox_inches='tight')
    return fig


def sketch_ring(pdr, line_name='ring', outdir=None, dpi=200, width=13.0,
                annotate_sextants=True, bpm_offset=3.0):
    """
    The whole ring: a survey footprint with the magnets drawn on it, plus a
    summary panel of the global parameters.

    BPMs are floated `bpm_offset` metres transversely outside the ring and
    joined to their true position by a hairline, so they read as position
    indicators rather than as elements in the beamline -- the same
    convention as the existing survey_plot. Kickers are drawn on the ring
    itself, with a heavier line so they stay visible despite being short.

    Falls back to a linear strip if survey() is unavailable.
    """
    line = pdr.lines[line_name]
    rows = _element_rows(line)

    try:
        sv = line.survey()
        X = np.asarray(sv.X, dtype=float)
        Z = np.asarray(sv.Z, dtype=float)
        sv_names = list(sv.name)
        have_survey = True
    except Exception as e:
        print(f'sketch_ring: survey() failed ({e}); drawing a linear strip')
        have_survey = False

    fig = plt.figure(figsize=(width, width * 0.80))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.22)
    ax = fig.add_subplot(gs[0])
    axt = fig.add_subplot(gs[1])

    if have_survey:
        ax.plot(Z, X, color='0.75', lw=1.0, zorder=1)

        # Index survey points by PARENT name, keeping every slice index, so
        # a sliced magnet is drawn over its whole footprint rather than just
        # its first slice (or missed entirely).
        spans_idx = {}
        for i, n in enumerate(sv_names):
            spans_idx.setdefault(_base_name(n), []).append(i)
        idx = {k: v[0] for k, v in spans_idx.items()}

        # theta is used to push BPMs transversely off the ring, matching the
        # existing survey_plot convention where they are indicators rather
        # than in-line elements.
        try:
            TH = np.asarray(sv.theta, dtype=float)
        except Exception:
            TH = None

        # theta + pi/2 is the transverse normal, but whether it points out
        # of the ring or into it depends on the sign convention of the
        # bends. Decide once, by majority vote over the BPMs, so they all
        # float the same way and always end up OUTSIDE the ring.
        Zc0, Xc0 = 0.5*(Z.min()+Z.max()), 0.5*(X.min()+X.max())
        bpm_sign = 1.0
        if TH is not None:
            votes = 0
            for r in rows:
                if r['kind'] not in _BPM_KINDS:
                    continue
                ii0 = spans_idx.get(r['name'])
                if not ii0:
                    continue
                i0 = min(ii0)
                d0 = TH[i0] + np.pi / 2.0
                r_in = np.hypot(Z[i0] - Zc0, X[i0] - Xc0)
                r_out = np.hypot(Z[i0] + np.cos(d0) - Zc0,
                                 X[i0] + np.sin(d0) - Xc0)
                votes += 1 if r_out > r_in else -1
            if votes < 0:
                bpm_sign = -1.0

        drawn = {}
        bpm_drawn = {}
        for r in rows:
            if r['is_drift']:
                continue
            ii = spans_idx.get(r['name'])
            if not ii:
                continue

            if r['kind'] in _BPM_KINDS:
                i = min(ii)
                if TH is not None:
                    d = TH[i] + np.pi / 2.0
                    oz = Z[i] + bpm_sign * bpm_offset * np.cos(d)
                    ox = X[i] + bpm_sign * bpm_offset * np.sin(d)
                else:                       # no theta: push radially outward
                    Zc, Xc = 0.5*(Z.min()+Z.max()), 0.5*(X.min()+X.max())
                    dz, dx = Z[i]-Zc, X[i]-Xc
                    nn = np.hypot(dz, dx) or 1.0
                    oz = Z[i] + bpm_offset*dz/nn
                    ox = X[i] + bpm_offset*dx/nn
                ax.plot([Z[i], oz], [X[i], ox], color=r['colour'],
                        lw=0.5, alpha=0.5, zorder=4)
                ax.scatter(oz, ox, color=r['colour'], s=16, zorder=5)
                bpm_drawn.setdefault(r['kind'], r['colour'])
                continue

            if r['length'] == 0:
                continue
            lo, hi = min(ii), min(max(ii) + 1, len(Z) - 1)
            if hi <= lo:
                hi = min(lo + 1, len(Z) - 1)
            # Kickers are short; draw them heavier so they stay visible.
            lw = 7.0 if r['kind'] in ('corr_h', 'corr_v', 'corr') else 4.0
            ax.plot(Z[lo:hi + 1], X[lo:hi + 1], color=r['colour'],
                    lw=lw, solid_capstyle='butt',
                    zorder=4 if lw > 4 else 3)
            drawn.setdefault(r['kind'], r['colour'])
        drawn.update(bpm_drawn)

        if annotate_sextants:
            # Push each label radially OUTWARD from the centre of the ring
            # footprint. Scaling the coordinates about the origin instead
            # would drop labels inside the ring, or on top of the title,
            # because the footprint is not centred on (0, 0).
            Zc, Xc = 0.5 * (Z.min() + Z.max()), 0.5 * (X.min() + X.max())
            Rc = max(Z.max() - Z.min(), X.max() - X.min()) / 2.0
            for sx in ('1R', '2L', '2R', '3L', '3R', '1L'):
                nm = next((n for n in sv_names
                           if re.match(rf'^QFA_{sx}1$', n)), None)
                if nm is None:
                    continue
                i = idx[nm]
                dz, dx = Z[i] - Zc, X[i] - Xc
                norm = np.hypot(dz, dx) or 1.0
                # Clear the floated BPMs, which sit bpm_offset outside.
                f = (0.13 * Rc + 1.6 * bpm_offset) / norm
                ax.annotate(sx, xy=(Z[i], X[i]),
                            xytext=(Z[i] + dz * f, X[i] + dx * f),
                            fontsize=11, fontweight='bold',
                            ha='center', va='center',
                            arrowprops=dict(arrowstyle='-', color='0.5',
                                            lw=0.8))
            ax.margins(0.20)

        # ---- injection point on the footprint ------------------------
        inj = [r for r in rows if r['kind'] == 'inj']
        for r in inj:
            i = idx.get(r['name'])
            if i is None:
                continue
            Zc, Xc = 0.5 * (Z.min() + Z.max()), 0.5 * (X.min() + X.max())
            dz, dx = Z[i] - Zc, X[i] - Xc
            norm = np.hypot(dz, dx) or 1.0
            Rc = max(Z.max() - Z.min(), X.max() - X.min()) / 2.0
            f = 0.28 * Rc / norm
            ax.plot(Z[i], X[i], marker='*', ms=20, color='#d62728',
                    markeredgecolor='black', markeredgewidth=0.8, zorder=6)
            ax.annotate(r['name'], xy=(Z[i], X[i]),
                        xytext=(Z[i] + dz * f, X[i] + dx * f),
                        fontsize=11, fontweight='bold', color='#d62728',
                        ha='center', va='center', zorder=6,
                        arrowprops=dict(arrowstyle='-|>', color='#d62728',
                                        lw=1.4),
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='white', edgecolor='#d62728',
                                  linewidth=1.0))
            ax.margins(0.22)

        ax.set_aspect('equal')
        ax.set_xlabel('Z  [m]', fontsize=10)
        ax.set_ylabel('X  [m]', fontsize=10)
        ax.grid(color='0.92', lw=0.6)

        handles = [Line2D([0], [0], color=c, lw=4,
                          label=_KIND_LABEL.get(k, k))
                   for k, c in drawn.items()]
        ax.legend(handles=handles, loc='center', fontsize=8.5,
                  frameon=False, ncol=2)
    else:
        rows_m = [r for r in rows if not r['is_drift'] and r['length'] > 0]
        drawn = {}
        for r in rows_m:
            ax.add_patch(Rectangle((r['s'], r['y0'] - r['half_h']),
                                   max(r['length'], 0.05), 2 * r['half_h'],
                                   facecolor=r['colour'], edgecolor='none'))
            drawn.setdefault(r['kind'], r['colour'])
        ax.axhline(0, color='0.4', lw=0.8)
        ax.set_xlim(0, line.get_length())
        ax.set_ylim(-2.2, 2.2)
        ax.set_yticks([])
        ax.set_xlabel('s  [m]', fontsize=10)
        for side in ('left', 'right', 'top'):
            ax.spines[side].set_visible(False)
        handles = [Rectangle((0, 0), 1, 1, facecolor=c,
                             label=_KIND_LABEL.get(k, k))
                   for k, c in drawn.items()]
        ax.legend(handles=handles, loc='upper center',
                  bbox_to_anchor=(0.5, -0.16), ncol=6, frameon=False,
                  fontsize=8)

    ax.set_title('Full ring layout', fontsize=13, fontweight='bold', pad=22)

    # ---- parameter table --------------------------------------------
    axt.axis('off')
    C = line.get_length()

    def _v(key, fmt='{:.4f}'):
        try:
            return fmt.format(pdr[key])
        except Exception:
            return '—'

    counts = {}
    for r in rows:
        if r['is_drift'] or r['length'] == 0:
            continue
        counts[r['kind']] = counts.get(r['kind'], 0) + 1

    n_bend = counts.get('bend', 0) + counts.get('bend_ds', 0)
    n_quad = sum(v for k, v in counts.items() if k.startswith('quad'))
    n_sext = counts.get('sext_f', 0) + counts.get('sext_d', 0)
    # Diagnostics are counted over all rows, since BPMs are usually
    # zero-length and would be missed by the magnet-only count above.
    n_bpm = sum(1 for r in rows if r['kind'] in _BPM_KINDS)
    n_corr = sum(1 for r in rows
                 if r['kind'] in ('corr', 'corr_h', 'corr_v'))

    left = [
        ('Circumference',        f'{C:.3f} m'),
        ('Cells per sextant',    _v('N_cells_S', '{:.0f}')),
        ('Arc cell length',      _v('l_cell') + ' m'),
        ('Arc dipole length',    _v('l_bend') + ' m'),
        ('DS dipole length',     _v('l_bendDS') + ' m'),
        ('Quadrupole length',    _v('l_quad') + ' m'),
    ]
    right = [
        ('Arc drift',            _v('l_drift') + ' m'),
        ('Straight drift',       _v('l_tripl') + ' m'),
        ('Bending angle / dip.', _v('hBarc', '{:.6f}') + ' rad/m'),
        ('Dipoles',              f'{n_bend}'),
        ('Quadrupoles',          f'{n_quad}'),
        ('Sextupoles',           f'{n_sext}'),
    ]
    if n_bpm or n_corr:
        right += [('BPMs', f'{n_bpm}'), ('Correctors', f'{n_corr}')]

    for col, items in ((0.02, left), (0.52, right)):
        for j, (k, v) in enumerate(items):
            y = 0.92 - j * 0.155
            axt.text(col, y, k, fontsize=9.5, va='top', color='0.3')
            axt.text(col + 0.30, y, v, fontsize=9.5, va='top',
                     fontweight='bold', family='monospace')

    axt.text(0.02, 1.02, 'Ring parameters', fontsize=11, fontweight='bold',
             va='bottom')

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, 'ring.png'), dpi=dpi,
                    bbox_inches='tight')
    return fig


CELL_PLOT_IN_PER_M = 15.0 / 3.7   # sketch_cell_types: 15 in for a ~3.7 m cell


def sketch_period_beam_size(pdr, beam, sextants=('3R', '1L'), line_name='ring',
                            inject_s=None, n_sigma=None, outdir=None,
                            in_per_m=0.5, ref_span=None, dpi=200):
    """
    One period of the ring (two neighbouring sextants: arc centre -> straight
    -> arc centre) as a layout drawing at the scale of the cell plots
    (draw_beamline: names, BPMs, magnet and drift lengths), with the injected
    beam's rms sizes plotted above it on the same s axis. The optics repeat
    every period, so this covers the whole ring.

    beam : dict with 's', 'sig_x', 'sig_y' (arrays over the ring, metres) and
        optionally 'eps_x', 'eps_y' (m rad) and 'sigma_delta' for the title --
        as returned by corrections.ring_beam_sizes from the macroparticle
        emittances.
    sextants : the sextants to draw, in ring order. The default ('3R', '1L')
        is the period holding the injection straight; ('2L', '2R') is centred
        on a whole arc instead.
    inject_s : s of the injection point [m], drawn if it falls in the span.
    n_sigma : also draw the n_sigma envelope (dashed), or None for rms only.
    in_per_m : horizontal scale [in per m]. 0.5 gives a ~38 in wide figure;
        CELL_PLOT_IN_PER_M (~4) draws it at the scale of the cell plots.
    ref_span : the span [m] that label rotation, label spacing and symbol
        sizes are scaled to. None = the span that 15 in covers at in_per_m,
        i.e. the width a cell plot gives one cell, so the style matches the
        cell plots at any scale.
    outdir : saves period_beam_size_<sextants>.png. A PNG cannot be wider than
        65536 px, so dpi is lowered to fit if needed.
    """
    line = pdr.lines[line_name]
    rows_all = _element_rows(line)

    # ---- span of the chosen sextants --------------------------------------
    spans = {}
    for r in rows_all:
        for tag in sextants:
            if re.search(rf'_{tag}(?:C|CH|\d+H?)?$', r['name']):
                lo, hi = spans.get(tag, (np.inf, -np.inf))
                spans[tag] = (min(lo, r['s']), max(hi, r['s'] + r['length']))
    missing = [t for t in sextants if t not in spans]
    if missing:
        raise KeyError(f'sketch_period_beam_size: no elements found for sextant(s) '
                       f'{missing}')

    def centre_quad(tag):
        return next((r for r in rows_all if re.match(
            rf'^(?:Q[FD]TripC_{tag}\d*|QFA_{tag}C)$', r['name'])), None)

    # From the arc-centre quad that opens the first R sextant (QFA_<R>C) to
    # the one that opens the next, so the span is exactly one period; fall
    # back to the sextant extents.
    s_a = min(lo for lo, _ in spans.values())
    s_b = max(hi for _, hi in spans.values())
    first, last = sextants[0], sextants[-1]
    q = centre_quad(first)
    if q is not None and first.endswith('R'):
        s_a = q['s']
    if last.endswith('L'):
        q = centre_quad('R'.join(last.rsplit('L', 1)))     # 1L -> 1R
        if q is not None:
            s_b = q['s'] if q['s'] > s_a else line.get_length()
    rows = [r for r in rows_all
            if r['s'] >= s_a - 1e-9 and r['s'] + r['length'] <= s_b + 1e-9]

    L = s_b - s_a
    if ref_span is None:
        ref_span = 15.0 / in_per_m

    width = in_per_m * (L + 2 * 0.035 * ref_span) + 2.0
    fig = plt.figure(figsize=(width, 11.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[4.2, 6.0], hspace=0.04,
                          left=1.0 / width, right=1.0 - 1.0 / width,
                          top=0.94, bottom=0.12)
    axb = fig.add_subplot(gs[0])
    axl = fig.add_subplot(gs[1], sharex=axb)

    # ---- layout, exactly as in the cell plots --------------------------
    draw_beamline(axl, rows, s_offset=s_a, ref_span=ref_span)
    ticks = np.arange(np.ceil(s_a), s_b + 1e-9, 1.0)
    axl.xaxis.set_major_locator(plt.FixedLocator(ticks - s_a))
    axl.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v + s_a:.0f}'))
    axl.set_xlabel('s along the ring  [m]', fontsize=9)

    # ---- beam sizes --------------------------------------------------
    y_max = draw_beam_size(axb, beam, s_a, s_b, n_sigma=n_sigma)
    title = (f'One period (sextants {" + ".join(sextants)}) with the injected beam size'
             f'  —  the optics repeat every period')
    if 'eps_x' in beam and 'eps_y' in beam:
        title += (rf';  $\epsilon_x$ = {beam["eps_x"]*1e6:.3f} $\mu$m, '
                  rf'$\epsilon_y$ = {beam["eps_y"]*1e6:.3f} $\mu$m')
    if 'sigma_delta' in beam:
        title += rf', $\sigma_\delta$ = {beam["sigma_delta"]*1e3:.2f}‰'
    axb.text(0.0, 1.03, title, transform=axb.transAxes, ha='left', va='bottom',
             fontsize=12, fontweight='bold')

    # ---- sextant labels, straight / arc centres ------------------------
    for k, tag in enumerate(sextants):
        x0 = max(spans[tag][0], s_a) - s_a
        axb.text(x0 + 0.3, y_max * 0.97, f'sextant {tag}', ha='left', va='top',
                 fontsize=11, fontweight='bold', color='0.3')
        if k + 1 < len(sextants):
            nxt = sextants[k + 1]
            straight = tag.endswith('R') and nxt.endswith('L')
            q = centre_quad(nxt)
            x = (q['s'] + q['length'] / 2 if q is not None else spans[nxt][0]) - s_a
            for a_ in (axb, axl):
                a_.axvline(x, color='0.55', lw=0.9, ls='--' if straight else ':',
                           zorder=0)
            axb.text(x, y_max * 0.90, ' straight centre' if straight else ' arc centre',
                     ha='left', va='top', fontsize=9, color='0.4')

    if inject_s is not None and s_a <= inject_s <= s_b:
        for a_ in (axb, axl):
            a_.axvline(inject_s - s_a, color=COL_INJ, lw=1.6, ls=(0, (5, 2)), zorder=6)
        axb.text(inject_s - s_a, y_max * 0.82, ' injection', ha='left', va='top',
                 fontsize=11, fontweight='bold', color=COL_INJ)

    seen = {}
    for r in rows:
        if not r['is_drift'] and (r['length'] > 0 or r['kind'] in _DIAG_KINDS):
            seen.setdefault(r['kind'], (r['colour'], r.get('hatch', '')))
    handles = [Rectangle((0, 0), 1, 1,
                         facecolor=('white' if k in _BPM_KINDS else c),
                         edgecolor=(c if k in _BPM_KINDS else 'black'),
                         hatch=h, linewidth=0.7, label=_KIND_LABEL.get(k, k))
               for k, (c, h) in seen.items()]
    if handles:
        axl.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.0, -0.14),
                   ncol=len(handles), frameon=False, fontsize=8.5)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        dpi_png = min(dpi, int(65000 / (width + 1.0)))
        fig.savefig(os.path.join(outdir, f'period_beam_size_{"_".join(sextants)}.png'),
                    dpi=dpi_png, bbox_inches='tight')
    return fig


def sketch_all(pdr, sextant='1R', line_name='ring', outdir='sketches',
               dpi=200, show=False, beam=None, n_sigma=None):
    """
    Produce the complete set: every cell type, the arc, and the ring.
    beam=None draws the layouts only; a beam-size dict (as from
    corrections.ring_beam_sizes) adds the beam sizes to the cell and arc
    plots. Saves to `outdir` and returns {name: Figure}.
    """
    out = {}
    out.update(sketch_cell_types(pdr, sextant, line_name, outdir=outdir,
                                 dpi=dpi, beam=beam, n_sigma=n_sigma))
    out['arc'] = sketch_arc(pdr, sextant, line_name, outdir=outdir, dpi=dpi,
                            beam=beam, n_sigma=n_sigma)
    out['ring'] = sketch_ring(pdr, line_name, outdir=outdir, dpi=dpi)

    print(f'\nsketch_all: {len(out)} figures written to {outdir}/')
    for k in out:
        print(f'   {k}')
    if show:
        plt.show()
    return out