import xtrack as xt
import matplotlib.pyplot as plt

pdr= xt.Line.from_json("clic_pdr.json")

pr= xt.Line.from_json("/Users/maddiewatson/Library/CloudStorage/OneDrive-Personal/University/Research year/polarizer-ring-studies/Junk Draw/polarizer_ring_current_config.json")

pdr.survey().plot()
plt.savefig('damping_ring.png')