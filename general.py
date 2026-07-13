#%%
import xtrack as xt
import pandas as pd

pdr= xt.Environment.from_json(f"/Users/maddiewatson/Library/CloudStorage/OneDrive-Personal/University/Research year/polarizer-ring-studies/JSON Files/D1/C1/pdr_misaligned.json")
ring=pdr.lines['ring']

twiss=ring.twiss()
print(twiss)

df=pd.DataFrame(twiss)

twiss.to_tfs('twiss.tfs')
# %%

import xtrack as xt
pdr= xt.Environment.from_json(f"/Users/maddiewatson/Library/CloudStorage/OneDrive-Personal/University/Research year/polarizer-ring-studies/JSON Files/D1/C1/pdr_misaligned.json")
line=pdr.lines['ring']

tt = line.get_table(attr=True)


print([m for m in dir(line) if 'spin' in m.lower()])
help(line.twiss)   # look for a `spin` kwarg in the signature

line.configure_spin('auto')   # exact kwargs TBD from your version's docstring
tw = line.twiss(spin=True)  # or spin may just follow from configure_spin — check the signature
tw.cols['name s spin_x spin_y spin_z'].show()

# %%
