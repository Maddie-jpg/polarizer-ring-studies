#%%

import sys
import os

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
config=6
mode='misaligned' # 'perfect', 'misaligned', or 'corrected'

#%%

#Linear optics - uncomment desired optics

pdr=lo.three_fold_periodicity_90_deg()

#pdr=lo.two_fold_periodicity_90_deg()

#sextupole configuration

sc.config_D1_C6(pdr)

#Misalignments
if mode == 'misaligned':
    ring=pdr.lines['ring']
    mc.misalignments(ring)

elif mode=='corrected':
    ring=pdr.lines['ring']

    mc.insert_BPMs_all_as_markers(pdr)
    mc.insert_correctors(pdr)

    twiss=ring.twiss(method='6d', radiation_integrals=True, eneloss_and_damping=True,
                   spin=True, polarization=True )

    mc.misalignments(ring)

    try:
        ring.discard_tracker()
        mc.orbit_correction(pdr, twiss, threading=False, rcond_x=1e-4, rcond_y=1e-2)
        
    except:
        mc.orbit_correction(pdr, twiss, threading=False, rcond_x=1e-4, rcond_y=1e-2)
        


#%%


new_json_folder=f'./JSON Files/D{design}/C{config}'
if not os.path.exists(new_json_folder):
    os.makedirs(new_json_folder)
#%%
#Save json file

file_path_str = f'../JSON Files/D{design}/C{config}/pdr_{mode}.json'

output_file = Path(file_path_str)

output_file.parent.mkdir(parents=True, exist_ok=True)

pdr.to_json(file_path_str)