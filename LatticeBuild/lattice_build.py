#%%

import sys
import os
import xtrack as xt

# Adds the parent directory to the search path
parent_dir = os.path.abspath('..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import linear_optics as lo
import sextupole_configs as sc
import misalignments_corrections as mc

from pathlib import Path

#%%

design=1
config=1
mode='perfect' # 'perfect', 'misaligned', or 'corrected'

#%%


if mode == 'perfect':
    #Linear optics - uncomment desired optics
    pdr=lo.three_fold_periodicity_120_deg_many_sext(fringe_fields=True)

    #sextupole configuration

#Misalignments
elif mode == 'misaligned':
    pdr=xt.Environment.from_json(f'../JSON Files/D{design}/C{config}/pdr_perfect.json')
    ring=pdr.lines['ring']
    mc.misalignments(ring, 0.25e-3)

elif mode=='corrected':
    pdr=xt.Environment.from_json(f'../JSON Files/D{design}/C{config}/pdr_perfect.json')
    ring=pdr.lines['ring']

    mc.insert_BPMs_all_as_markers(pdr)
    mc.insert_correctors(pdr)

    twiss=ring.twiss(method='6d', radiation_integrals=True, eneloss_and_damping=True,
                   spin=True, polarization=True )

    mc.misalignments(ring,0.25e-3)

    try:
        ring.discard_tracker()
        mc.orbit_correction(ring, twiss, threading=False, rcond_x=1e-4, rcond_y=1e-2)
        
    except:
        mc.orbit_correction(ring, twiss, threading=True, rcond_x=1e-4, rcond_y=1e-2)
        



#%%
#Save json file

file_path_str = f'../JSON Files/D{design}/C{config}/pdr_{mode}_120.json'

output_file = Path(file_path_str)

output_file.parent.mkdir(parents=True, exist_ok=True)

pdr.to_json(file_path_str)