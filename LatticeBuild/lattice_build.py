#%%

import sys
import os
import xtrack as xt

# Adds the parent directory to the search path
parent_dir = os.path.abspath('..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import LatticeBuild.linear_optics as lo
import sextupole_configs as sc
import misalignments_corrections as mc

from pathlib import Path

#%%

design=3
config=0
mode='perfect' # 'perfect', 'misaligned', or 'corrected'
phase=90
changes=None

#%%
SEED=123456789

if mode == 'perfect':
    #Linear optics - uncomment desired optics
    pdr=lo.two_fold_racetrack_3straight(fringe_fields=True, matched=True,WP=(13.65, 13.23))
    #sc.config_D2_C3(pdr)
    


#Misalignments
elif mode == 'misaligned':
    if changes is not None:
        pdr=xt.Environment.from_json(f'../JSON Files/D{design}/C{config}/pdr_perfect_{phase}_{changes}.json')
    else:
        pdr=xt.Environment.from_json(f'../JSON Files/D{design}/C{config}/pdr_perfect_{phase}.json')
    ring=pdr.lines['ring']
    mc.misalignments(ring, 0.25e-3,SEED)

elif mode=='corrected':
    if changes is not None:
        pdr=xt.Environment.from_json(f'../JSON Files/D{design}/C{config}/pdr_misaligned_{phase}_{changes}.json')
    else:
        pdr=xt.Environment.from_json(f'../JSON Files/D{design}/C{config}/pdr_misaligned_{phase}.json')
    ring=pdr.lines['ring']

    mc.insert_BPMs_all_as_markers(pdr)
    mc.insert_correctors(pdr)
    mc.misalignments_correctors(ring,0.25e-3,SEED)

    twiss=ring.twiss(method='6d', radiation_integrals=True, eneloss_and_damping=True,
                   spin=True, polarization=True )



    try:
        ring.discard_tracker()
        mc.orbit_correction(ring, twiss, threading=False, rcond_x=1e-4, rcond_y=1e-2)
        
    except:
        mc.orbit_correction(ring, twiss, threading=True, rcond_x=1e-4, rcond_y=1e-2)
        



#%%
#Save json file
if changes is not None:
    file_path_str = f'../JSON Files/D{design}/C{config}/pdr_{mode}_{phase}_{changes}.json'
else:
    file_path_str = f'../JSON Files/D{design}/C{config}/pdr_{mode}_{phase}.json'

output_file = Path(file_path_str)

output_file.parent.mkdir(parents=True, exist_ok=True)

pdr.to_json(file_path_str)