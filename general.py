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
