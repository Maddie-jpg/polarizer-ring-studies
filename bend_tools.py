"""
Treat field-expansion dipoles as bends everywhere in the analysis.

A hard-edge dipole is one xt.Bend element (element_type 'Bend'). A
field-expansion dipole built by linear_optics._field_expansion_bend is a
group of elements cloned per magnet, e.g. for Bend1_1R8:

    Bend_fe0.Bend1_1R8  Bend_edge_in.Bend1_1R8  Bend_fe1.Bend1_1R8
    Bend_fe2.Bend1_1R8  Bend_fe3.Bend1_1R8  Bend_edge_out.Bend1_1R8
    Bend_fe4.Bend1_1R8

(element types BFieldExpansion / DipoleEdge). xtrack's element_type is the
Python class, so it can't be made to say 'Bend'; use these helpers instead
of `element_type == 'Bend'`:

    magnet_of(name)      'Bend_fe2.Bend1_1R8' -> 'Bend1_1R8' (else unchanged)
    is_bend(line, name)  True for xt.Bend / RBend and every field-expansion piece
    bend_groups(line)    {magnet: [element names]} in ring order
    magnets(line, types) one record per magnet (bends grouped), with
                         s_start, s_end, length, angle, h (peak k0), k1, k2
"""
import re

import numpy as np

_FE_PIECE = re.compile(r'^(?P<proto>.+?)_(?:fe\d+|edge_in|edge_out)\.(?P<magnet>.+?)(?:\.\..*)?$')
_HARD_BEND_TYPES = ('Bend', 'RBend')
_FE_TYPES = ('BFieldExpansion', 'ThickSliceBFieldExpansion', 'DipoleEdge')


def magnet_of(name):
    """Name of the magnet an element belongs to ('Bend_fe2.Bend1_1R8' -> 'Bend1_1R8')."""
    m = _FE_PIECE.match(name)
    return m.group('magnet') if m else name


def is_fe_piece(name):
    return _FE_PIECE.match(name) is not None


def is_bend(line, name, element_type=None):
    """True for a hard-edge bend or any piece of a field-expansion bend."""
    if is_fe_piece(name):
        return True
    if name == '_end_point':
        return False
    et = element_type if element_type is not None else type(line[name]).__name__
    return et in _HARD_BEND_TYPES


def peak_k0(el, element_type=None):
    """Peak dipole strength k0 = 1/rho [1/m] of one element (0 if none).
    Pass element_type from the table: line[name] can return a view object
    whose Python type is not the element class."""
    cls = element_type or type(el).__name__
    if cls in _HARD_BEND_TYPES:
        L = el.length
        return float(el.angle / L) if L else 0.0
    if cls in ('BFieldExpansion', 'ThickSliceBFieldExpansion'):
        k0 = float(getattr(el, 'k0', 0.0))
        knc = np.asarray(getattr(el, 'knc', np.zeros((0, 0))), dtype=float)
        if knc.size and knc.shape[0] > 0:
            s = el.s_start + np.linspace(0.0, el.length, 41)
            prof = np.polynomial.polynomial.polyval(s, knc[0])
            k0 = k0 + float(np.max(np.abs(prof)))
        return abs(k0) * float(getattr(el, 'kscale', 1.0))
    return 0.0


def bend_groups(line, table=None):
    """{magnet name: [element names]} for every dipole, in ring order.
    A hard-edge bend maps to [itself]; a field-expansion bend to its pieces
    (including its thin edge elements)."""
    tt = table if table is not None else line.get_table()
    out = {}
    for name, et in zip(tt.name, tt.element_type):
        if is_bend(line, name, et):
            out.setdefault(magnet_of(name), []).append(name)
    return out


def magnets(line, types=('Bend', 'Quadrupole', 'Sextupole'), table=None):
    """One record per magnet of the given types, in ring order. Bends are
    grouped (field-expansion pieces -> one magnet). Keys: name, type,
    s_start, s_end, length, angle, h (peak k0, = angle/length for xt.Bend),
    k1, k2, elements."""
    tt = table if table is not None else line.get_table()
    out = {}
    for name, et, s in zip(tt.name, tt.element_type, tt.s):
        kind = 'Bend' if is_bend(line, name, et) else et
        if kind not in types:
            continue
        el = line[name]
        L = float(getattr(el, 'length', 0.0) or 0.0)
        key = magnet_of(name)
        rec = out.setdefault(key, dict(name=key, type=kind, s_start=float(s),
                                       s_end=float(s), length=0.0, angle=0.0,
                                       h=0.0, k1=0.0, k2=0.0, elements=[]))
        rec['elements'].append(name)
        rec['s_end'] = max(rec['s_end'], float(s) + L)
        rec['length'] += L
        rec['angle'] += float(getattr(el, 'angle', 0.0) or 0.0)
        if kind == 'Bend':
            rec['h'] = max(rec['h'], peak_k0(el, et))
        else:
            rec['k1'] = float(getattr(el, 'k1', 0.0))
            rec['k2'] = float(getattr(el, 'k2', 0.0))
    return list(out.values())
