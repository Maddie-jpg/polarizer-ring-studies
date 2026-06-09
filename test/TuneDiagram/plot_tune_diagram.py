import numpy as np
import pylab as plt

from lib.TuneDiagram.tune_diagram import resonance_lines

resonance_orders = (1,2,3,4)
Qx_range = (14.95, 15.55)
Qy_range = (14.95, 15.55)
resonances = resonance_lines(Qx_range,Qy_range,resonance_orders,3)
fig, ax = plt.subplots(1, figsize=(5,5))
resonances.plot_resonance(fig)

resonance_orders = (1,2,3,4)
Qx_range = (13.95, 14.55)
Qy_range = (14.95, 15.55)
resonances = resonance_lines(Qx_range,Qy_range,resonance_orders,3)
fig, ax = plt.subplots(1, figsize=(5,5))
resonances.plot_resonance(fig)

plt.show()