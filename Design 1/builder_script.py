import sys
import os

# Adds the parent directory to the search path
parent_dir = os.path.abspath('..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import xtrack as xt
import numpy as np
import matplotlib.pyplot  as plt
import matplotlib.patches as patches
import xpart as xp
import xobjects as xo
import math
from TuneDiagram.lib.TuneDiagram.tune_diagram import resonance_lines
from prettytable import PrettyTable
import xutil_DA_CC.xsuite_plot_functions as my_xpf
import xutil_DA_CC.xsuite_utilities as xutil
import Essentials.constants
import sextupole_configs as sc

#%%

pdr = xt.Environment.from_json("json_files/pdr_var1.json")

#%%
#Sextupole configuration implementation
sc.config_D1(pdr)
