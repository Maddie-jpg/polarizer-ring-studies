# =========================
# Standard library
# =========================
import os
import re
import copy
import json
import yaml
import time
import glob
import shutil
import logging
from pathlib import Path
from multiprocessing import Pool
from contextlib import redirect_stdout, redirect_stderr, contextmanager

# =========================
# Core scientific stack
# =========================
import numpy as np
import pandas as pd
import math

import scipy
import scipy.constants as sp_co
import scipy.optimize as opt
from scipy import integrate
from scipy.interpolate import interp1d, griddata
from scipy.linalg import cholesky
from scipy.stats import uniform, truncnorm, gamma
from scipy.special import hyp2f1  # add explicitly as needed
# from scipy.special import *

# =========================
# High-precision math (explicit use only)
# =========================
import mpmath as mp

# =========================
# Accelerator / beam physics
# =========================
import xtrack as xt
import xfields as xf
import xpart as xp
import xcoll as xc
import nafflib as pnf

# =========================
# I/O and data formats
# =========================
import h5py

# =========================
# Parallel / batch submission
# =========================
# from pylhc_submitter.job_submitter import main as htcondor_submit

# =========================
# Plotting (consider moving into functions)
# =========================
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# =========================
# Progress / utilities
# =========================
from tqdm import tqdm

# =========================
# Logging
# =========================
logger = logging.getLogger(__name__)



def save_dict_to_h5(filename, data_dict):
    """
    Saves a dictionary to an HDF5 file. Supports both flat and nested dictionaries.

    :param filename: Name of the HDF5 file (e.g., 'data.h5')
    :param data_dict: Dictionary containing NumPy arrays (can be nested)
    """
    def recursively_save(group, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):  # If value is a nested dictionary, create a subgroup
                subgroup = group.create_group(key)
                recursively_save(subgroup, value)  # Recursive call
            else:
                group.create_dataset(key, data=value)  # Save NumPy arrays

    with h5py.File(filename, "w") as f:
        recursively_save(f, data_dict)  # Call recursive function

    print(f"Dictionary saved to {filename}")

def load_dict_from_h5(filename):
    """
    Loads an HDF5 file into a dictionary, automatically handling nested structures.

    :param filename: Name of the HDF5 file to load
    :return: Dictionary with the same structure as the original data
    """
    def recursively_load(group):
        result = {}
        for key, item in group.items():
            if isinstance(item, h5py.Group):  # If it's a group, recurse
                result[key] = recursively_load(item)
            else:  # If it's a dataset, load it as a NumPy array
                result[key] = item[()]
        return result

    with h5py.File(filename, "r") as f:
        return recursively_load(f)

def ensure_list(value):
    # Check if the value is a single string
    if isinstance(value, (str, int, float)):
        return [value]
    # Return as-is if it's already a list or another iterable
    return value

def madx_add_fccee_wigglers (madx, sequence_name='fccee_p_ring'):

    # ! --------------------------------------------------------------------------------------------------
    # ! Wiggler installation script
    # ! Creates MAD-X Wiggler elements follwing the parameters presented in https://arxiv.org/pdf/1909.12245.pdf ,
    # ! and install them in the MAD-X sequence.
    # ! --------------------------------------------------------------------------------------------------

    brhobeam = madx.sequence[sequence_name].beam.brho
    madx.input(f'''brhobeam={brhobeam};''')
    madx.input(f'''sequence_name={sequence_name};''')
    madx.input(f'''

    r_asymmetry = 6;
    B_plus = 0.7;
    L_plus = 0.430;
    L_minus = L_plus*r_asymmetry/2.;
    L_dis = 0.250;
    L_qa = 2.9;
    brhobeam = {madx.sequence[sequence_name].beam.brho};

    on_wiggler_h = 0;
    wiggler_angle_h:= on_wiggler_h * B_plus * L_plus / brhobeam;
    on_wiggler_v = 0;
    wiggler_angle_v:= on_wiggler_v * B_plus * L_plus / brhobeam;

    Print, Text="The critical energy of the wiggler is:";
    VALUE, 2.21*10^(-6)*(BEAM->ENERGY)^3/(brhobeam/B_plus/1000);
    Print, Text="MeV.";
    ''')

    madx.input('''
    define_wigglers_as_multipoles() : macro {

        ! MWIM: MULTIPOLE, LRAD=L_minus, knl:={-0.5*wiggler_angle_h}, ksl:={-0.5*wiggler_angle_v};
        ! MWIP: MULTIPOLE, LRAD=L_plus,  knl:={ wiggler_angle_h},     ksl:={wiggler_angle_v};

        ! I put horizontal sbends (will tilt the multipoles in xsuite)
        MWIM: SBEND, L=L_minus, angle:=-0.5*wiggler_angle_v; !, tilt=pi/2;
        MWIP: SBEND, L=L_plus,  angle:=wiggler_angle_v; !, tilt=pi/2;

        MWI.A4RA: MWIM;
        MWI.B4RA: MWIP;
        MWI.C4RA: MWIM;
        MWI.D4RA: MWIM;
        MWI.E4RA: MWIP;
        MWI.F4RA: MWIM;
        MWI.G4RA: MWIM;
        MWI.H4RA: MWIP;
        MWI.I4RA: MWIM;

        MWI.A5RA: MWIM;
        MWI.B5RA: MWIP;
        MWI.C5RA: MWIM;
        MWI.D5RA: MWIM;
        MWI.E5RA: MWIP;
        MWI.F5RA: MWIM;
        MWI.G5RA: MWIM;
        MWI.H5RA: MWIP;
        MWI.I5RA: MWIM;

        MWI.A4RD: MWIM;
        MWI.B4RD: MWIP;
        MWI.C4RD: MWIM;
        MWI.D4RD: MWIM;
        MWI.E4RD: MWIP;
        MWI.F4RD: MWIM;
        MWI.G4RD: MWIM;
        MWI.H4RD: MWIP;
        MWI.I4RD: MWIM;

        MWI.A5RD: MWIM;
        MWI.B5RD: MWIP;
        MWI.C5RD: MWIM;
        MWI.D5RD: MWIM;
        MWI.E5RD: MWIP;
        MWI.F5RD: MWIM;
        MWI.G5RD: MWIM;
        MWI.H5RD: MWIP;
        MWI.I5RD: MWIM;

        MWI.A4RG: MWIM;
        MWI.B4RG: MWIP;
        MWI.C4RG: MWIM;
        MWI.D4RG: MWIM;
        MWI.E4RG: MWIP;
        MWI.F4RG: MWIM;
        MWI.G4RG: MWIM;
        MWI.H4RG: MWIP;
        MWI.I4RG: MWIM;

        MWI.A5RG: MWIM;
        MWI.B5RG: MWIP;
        MWI.C5RG: MWIM;
        MWI.D5RG: MWIM;
        MWI.E5RG: MWIP;
        MWI.F5RG: MWIM;
        MWI.G5RG: MWIM;
        MWI.H5RG: MWIP;
        MWI.I5RG: MWIM;

        MWI.A4RJ: MWIM;
        MWI.B4RJ: MWIP;
        MWI.C4RJ: MWIM;
        MWI.D4RJ: MWIM;
        MWI.E4RJ: MWIP;
        MWI.F4RJ: MWIM;
        MWI.G4RJ: MWIM;
        MWI.H4RJ: MWIP;
        MWI.I4RJ: MWIM;

        MWI.A5RJ: MWIM;
        MWI.B5RJ: MWIP;
        MWI.C5RJ: MWIM;
        MWI.D5RJ: MWIM;
        MWI.E5RJ: MWIP;
        MWI.F5RJ: MWIM;
        MWI.G5RJ: MWIM;
        MWI.H5RJ: MWIP;
        MWI.I5RJ: MWIM;

    }
    ''')

    madx.input('''
    install_wigglers(sequence_name) : macro {
        SEQEDIT, SEQUENCE=sequence_name;
        FLATTEN;
        
            REMOVE, ELEMENT = fwig.1;
            INSTALL, ELEMENT=MWI.A4RA, AT= L_qa/2. + 1 * L_dis + 0.5 * L_minus + 0.0 * L_plus, FROM=QA4.1;
            INSTALL, ELEMENT=MWI.B4RA, AT= L_qa/2. + 2 * L_dis + 1.0 * L_minus + 0.5 * L_plus, FROM=QA4.1;
            INSTALL, ELEMENT=MWI.C4RA, AT= L_qa/2. + 3 * L_dis + 1.5 * L_minus + 1.0 * L_plus, FROM=QA4.1;
            INSTALL, ELEMENT=MWI.D4RA, AT= L_qa/2. + 4 * L_dis + 2.5 * L_minus + 1.0 * L_plus, FROM=QA4.1;
            INSTALL, ELEMENT=MWI.E4RA, AT= L_qa/2. + 5 * L_dis + 3.0 * L_minus + 1.5 * L_plus, FROM=QA4.1;
            INSTALL, ELEMENT=MWI.F4RA, AT= L_qa/2. + 6 * L_dis + 3.5 * L_minus + 2.0 * L_plus, FROM=QA4.1;
            INSTALL, ELEMENT=MWI.G4RA, AT= L_qa/2. + 7 * L_dis + 4.5 * L_minus + 2.0 * L_plus, FROM=QA4.1;
            INSTALL, ELEMENT=MWI.H4RA, AT= L_qa/2. + 8 * L_dis + 5.0 * L_minus + 2.5 * L_plus, FROM=QA4.1;
            INSTALL, ELEMENT=MWI.I4RA, AT= L_qa/2. + 9 * L_dis + 5.5 * L_minus + 3.0 * L_plus, FROM=QA4.1;
            
            REMOVE, ELEMENT = fwig.2;
            INSTALL, ELEMENT=MWI.A5RA, AT= L_qa/2. + 1 * L_dis + 0.5 * L_minus + 0.0 * L_plus, FROM=QA5.1;
            INSTALL, ELEMENT=MWI.B5RA, AT= L_qa/2. + 2 * L_dis + 1.0 * L_minus + 0.5 * L_plus, FROM=QA5.1;
            INSTALL, ELEMENT=MWI.C5RA, AT= L_qa/2. + 3 * L_dis + 1.5 * L_minus + 1.0 * L_plus, FROM=QA5.1;
            INSTALL, ELEMENT=MWI.D5RA, AT= L_qa/2. + 4 * L_dis + 2.5 * L_minus + 1.0 * L_plus, FROM=QA5.1;
            INSTALL, ELEMENT=MWI.E5RA, AT= L_qa/2. + 5 * L_dis + 3.0 * L_minus + 1.5 * L_plus, FROM=QA5.1;
            INSTALL, ELEMENT=MWI.F5RA, AT= L_qa/2. + 6 * L_dis + 3.5 * L_minus + 2.0 * L_plus, FROM=QA5.1;
            INSTALL, ELEMENT=MWI.G5RA, AT= L_qa/2. + 7 * L_dis + 4.5 * L_minus + 2.0 * L_plus, FROM=QA5.1;
            INSTALL, ELEMENT=MWI.H5RA, AT= L_qa/2. + 8 * L_dis + 5.0 * L_minus + 2.5 * L_plus, FROM=QA5.1;
            INSTALL, ELEMENT=MWI.I5RA, AT= L_qa/2. + 9 * L_dis + 5.5 * L_minus + 3.0 * L_plus, FROM=QA5.1;

            REMOVE, ELEMENT = fwig.3;
            INSTALL, ELEMENT=MWI.A4RD, AT= L_qa/2. + 1 * L_dis + 0.5 * L_minus + 0.0 * L_plus, FROM=QA4.2;
            INSTALL, ELEMENT=MWI.B4RD, AT= L_qa/2. + 2 * L_dis + 1.0 * L_minus + 0.5 * L_plus, FROM=QA4.2;
            INSTALL, ELEMENT=MWI.C4RD, AT= L_qa/2. + 3 * L_dis + 1.5 * L_minus + 1.0 * L_plus, FROM=QA4.2;
            INSTALL, ELEMENT=MWI.D4RD, AT= L_qa/2. + 4 * L_dis + 2.5 * L_minus + 1.0 * L_plus, FROM=QA4.2;
            INSTALL, ELEMENT=MWI.E4RD, AT= L_qa/2. + 5 * L_dis + 3.0 * L_minus + 1.5 * L_plus, FROM=QA4.2;
            INSTALL, ELEMENT=MWI.F4RD, AT= L_qa/2. + 6 * L_dis + 3.5 * L_minus + 2.0 * L_plus, FROM=QA4.2;
            INSTALL, ELEMENT=MWI.G4RD, AT= L_qa/2. + 7 * L_dis + 4.5 * L_minus + 2.0 * L_plus, FROM=QA4.2;
            INSTALL, ELEMENT=MWI.H4RD, AT= L_qa/2. + 8 * L_dis + 5.0 * L_minus + 2.5 * L_plus, FROM=QA4.2;
            INSTALL, ELEMENT=MWI.I4RD, AT= L_qa/2. + 9 * L_dis + 5.5 * L_minus + 3.0 * L_plus, FROM=QA4.2;
            
            REMOVE, ELEMENT = fwig.4;
            INSTALL, ELEMENT=MWI.A5RD, AT= L_qa/2. + 1 * L_dis + 0.5 * L_minus + 0.0 * L_plus, FROM=QA5.2;
            INSTALL, ELEMENT=MWI.B5RD, AT= L_qa/2. + 2 * L_dis + 1.0 * L_minus + 0.5 * L_plus, FROM=QA5.2;
            INSTALL, ELEMENT=MWI.C5RD, AT= L_qa/2. + 3 * L_dis + 1.5 * L_minus + 1.0 * L_plus, FROM=QA5.2;
            INSTALL, ELEMENT=MWI.D5RD, AT= L_qa/2. + 4 * L_dis + 2.5 * L_minus + 1.0 * L_plus, FROM=QA5.2;
            INSTALL, ELEMENT=MWI.E5RD, AT= L_qa/2. + 5 * L_dis + 3.0 * L_minus + 1.5 * L_plus, FROM=QA5.2;
            INSTALL, ELEMENT=MWI.F5RD, AT= L_qa/2. + 6 * L_dis + 3.5 * L_minus + 2.0 * L_plus, FROM=QA5.2;
            INSTALL, ELEMENT=MWI.G5RD, AT= L_qa/2. + 7 * L_dis + 4.5 * L_minus + 2.0 * L_plus, FROM=QA5.2;
            INSTALL, ELEMENT=MWI.H5RD, AT= L_qa/2. + 8 * L_dis + 5.0 * L_minus + 2.5 * L_plus, FROM=QA5.2;
            INSTALL, ELEMENT=MWI.I5RD, AT= L_qa/2. + 9 * L_dis + 5.5 * L_minus + 3.0 * L_plus, FROM=QA5.2;

            REMOVE, ELEMENT = fwig.5;
            INSTALL, ELEMENT=MWI.A4RG, AT= L_qa/2. + 1 * L_dis + 0.5 * L_minus + 0.0 * L_plus, FROM=QA4.3;
            INSTALL, ELEMENT=MWI.B4RG, AT= L_qa/2. + 2 * L_dis + 1.0 * L_minus + 0.5 * L_plus, FROM=QA4.3;
            INSTALL, ELEMENT=MWI.C4RG, AT= L_qa/2. + 3 * L_dis + 1.5 * L_minus + 1.0 * L_plus, FROM=QA4.3;
            INSTALL, ELEMENT=MWI.D4RG, AT= L_qa/2. + 4 * L_dis + 2.5 * L_minus + 1.0 * L_plus, FROM=QA4.3;
            INSTALL, ELEMENT=MWI.E4RG, AT= L_qa/2. + 5 * L_dis + 3.0 * L_minus + 1.5 * L_plus, FROM=QA4.3;
            INSTALL, ELEMENT=MWI.F4RG, AT= L_qa/2. + 6 * L_dis + 3.5 * L_minus + 2.0 * L_plus, FROM=QA4.3;
            INSTALL, ELEMENT=MWI.G4RG, AT= L_qa/2. + 7 * L_dis + 4.5 * L_minus + 2.0 * L_plus, FROM=QA4.3;
            INSTALL, ELEMENT=MWI.H4RG, AT= L_qa/2. + 8 * L_dis + 5.0 * L_minus + 2.5 * L_plus, FROM=QA4.3;
            INSTALL, ELEMENT=MWI.I4RG, AT= L_qa/2. + 9 * L_dis + 5.5 * L_minus + 3.0 * L_plus, FROM=QA4.3;
            
            REMOVE, ELEMENT = fwig.6;
            INSTALL, ELEMENT=MWI.A5RG, AT= L_qa/2. + 1 * L_dis + 0.5 * L_minus + 0.0 * L_plus, FROM=QA5.3;
            INSTALL, ELEMENT=MWI.B5RG, AT= L_qa/2. + 2 * L_dis + 1.0 * L_minus + 0.5 * L_plus, FROM=QA5.3;
            INSTALL, ELEMENT=MWI.C5RG, AT= L_qa/2. + 3 * L_dis + 1.5 * L_minus + 1.0 * L_plus, FROM=QA5.3;
            INSTALL, ELEMENT=MWI.D5RG, AT= L_qa/2. + 4 * L_dis + 2.5 * L_minus + 1.0 * L_plus, FROM=QA5.3;
            INSTALL, ELEMENT=MWI.E5RG, AT= L_qa/2. + 5 * L_dis + 3.0 * L_minus + 1.5 * L_plus, FROM=QA5.3;
            INSTALL, ELEMENT=MWI.F5RG, AT= L_qa/2. + 6 * L_dis + 3.5 * L_minus + 2.0 * L_plus, FROM=QA5.3;
            INSTALL, ELEMENT=MWI.G5RG, AT= L_qa/2. + 7 * L_dis + 4.5 * L_minus + 2.0 * L_plus, FROM=QA5.3;
            INSTALL, ELEMENT=MWI.H5RG, AT= L_qa/2. + 8 * L_dis + 5.0 * L_minus + 2.5 * L_plus, FROM=QA5.3;
            INSTALL, ELEMENT=MWI.I5RG, AT= L_qa/2. + 9 * L_dis + 5.5 * L_minus + 3.0 * L_plus, FROM=QA5.3;

            REMOVE, ELEMENT = fwig.7;
            INSTALL, ELEMENT=MWI.A4RJ, AT= L_qa/2. + 1 * L_dis + 0.5 * L_minus + 0.0 * L_plus, FROM=QA4.4;
            INSTALL, ELEMENT=MWI.B4RJ, AT= L_qa/2. + 2 * L_dis + 1.0 * L_minus + 0.5 * L_plus, FROM=QA4.4;
            INSTALL, ELEMENT=MWI.C4RJ, AT= L_qa/2. + 3 * L_dis + 1.5 * L_minus + 1.0 * L_plus, FROM=QA4.4;
            INSTALL, ELEMENT=MWI.D4RJ, AT= L_qa/2. + 4 * L_dis + 2.5 * L_minus + 1.0 * L_plus, FROM=QA4.4;
            INSTALL, ELEMENT=MWI.E4RJ, AT= L_qa/2. + 5 * L_dis + 3.0 * L_minus + 1.5 * L_plus, FROM=QA4.4;
            INSTALL, ELEMENT=MWI.F4RJ, AT= L_qa/2. + 6 * L_dis + 3.5 * L_minus + 2.0 * L_plus, FROM=QA4.4;
            INSTALL, ELEMENT=MWI.G4RJ, AT= L_qa/2. + 7 * L_dis + 4.5 * L_minus + 2.0 * L_plus, FROM=QA4.4;
            INSTALL, ELEMENT=MWI.H4RJ, AT= L_qa/2. + 8 * L_dis + 5.0 * L_minus + 2.5 * L_plus, FROM=QA4.4;
            INSTALL, ELEMENT=MWI.I4RJ, AT= L_qa/2. + 9 * L_dis + 5.5 * L_minus + 3.0 * L_plus, FROM=QA4.4;
            
            REMOVE, ELEMENT = fwig.8;
            INSTALL, ELEMENT=MWI.A5RJ, AT= L_qa/2. + 1 * L_dis + 0.5 * L_minus + 0.0 * L_plus, FROM=QA5.4;
            INSTALL, ELEMENT=MWI.B5RJ, AT= L_qa/2. + 2 * L_dis + 1.0 * L_minus + 0.5 * L_plus, FROM=QA5.4;
            INSTALL, ELEMENT=MWI.C5RJ, AT= L_qa/2. + 3 * L_dis + 1.5 * L_minus + 1.0 * L_plus, FROM=QA5.4;
            INSTALL, ELEMENT=MWI.D5RJ, AT= L_qa/2. + 4 * L_dis + 2.5 * L_minus + 1.0 * L_plus, FROM=QA5.4;
            INSTALL, ELEMENT=MWI.E5RJ, AT= L_qa/2. + 5 * L_dis + 3.0 * L_minus + 1.5 * L_plus, FROM=QA5.4;
            INSTALL, ELEMENT=MWI.F5RJ, AT= L_qa/2. + 6 * L_dis + 3.5 * L_minus + 2.0 * L_plus, FROM=QA5.4;
            INSTALL, ELEMENT=MWI.G5RJ, AT= L_qa/2. + 7 * L_dis + 4.5 * L_minus + 2.0 * L_plus, FROM=QA5.4;
            INSTALL, ELEMENT=MWI.H5RJ, AT= L_qa/2. + 8 * L_dis + 5.0 * L_minus + 2.5 * L_plus, FROM=QA5.4;
            INSTALL, ELEMENT=MWI.I5RJ, AT= L_qa/2. + 9 * L_dis + 5.5 * L_minus + 3.0 * L_plus, FROM=QA5.4;
        FLATTEN;
        ENDEDIT;

    }
    ''')

    madx.input("exec, define_wigglers_as_multipoles()")
    madx.input(f'''exec, install_wigglers({sequence_name})''')
    madx.use(sequence=sequence_name)

    return

def xsuite_add_fccee_wigglers(line,optics_type='LCC',install_new_location_GHC=False):
    # ! --------------------------------------------------------------------------------------------------
    # ! Wiggler installation script
    # ! Creates MAD-X Wiggler elements follwing the parameters presented in https://arxiv.org/pdf/1909.12245.pdf ,
    # ! and install them in the MAD-X sequence.
    # ! --------------------------------------------------------------------------------------------------
    
    # get the environment from the line
    env = line.env

    # Define beam parameters
    brhobeam = line.particle_ref.p0c  / line.particle_ref.q0 / 0.299792458 * 1e-9 # brho [Tm] for relativistic protons/electrons

    # Wiggler geometry/field parameters
    r_asymmetry = 6
    B_plus = 0.7  # Tesla
    L_plus = 0.430  # m
    L_minus = L_plus * r_asymmetry / 2.0  # m
    L_dis = 0.250  # m
    L_qa = 2.9     # m

    # Wiggler angles
    # line.vars['on_wiggler_h'] = 0
    line.vars['on_wiggler_v'] = 0

    # Print out equivalent of VALUE
    critical_energy = 2.21e-6 * (line.particle_ref.energy*1e-9)**3 / (brhobeam / B_plus / 1000)  # MeV
    print("The critical energy of the wiggler is:")
    print(f"{critical_energy[0]:.6f} MeV")

    env.new(
            'MWIM',
            xt.Bend,
            k0_from_h=True,
            length=L_minus,
            angle=f'(-0.5 * (on_wiggler_v * {B_plus} * {L_plus} / {brhobeam[0]}))',
            # tilt=np.pi/2  # optional if needed
        )

    env.new(
            'MWIP',
            xt.Bend,
            k0_from_h=True,
            length=L_plus,
            angle=f'(on_wiggler_v * {B_plus} * {L_plus} / {brhobeam[0]})',
            # tilt=np.pi/2  # optional if needed
        )

    offsets1=[0.5,1.0,1.5,2.5,3.0,3.5,4.5,5.0,5.5,6.5,7.0,7.5]
    offsets2=[0.0,0.5,1.0,1.0,1.5,2.0,2.0,2.5,3.0,3.0,3.5,4.0]

    if optics_type=='GHC':
        elem_list = ["A", "B", "C", "D", "E", "F"] 
        elemp_list = ['B', 'E'] 
        for i in range(1,9):
            line.remove(f"fwig.{i}")
    if optics_type=='LCC':
        elem_list = ["A", "B", "C", "D", "E", "F"]#, "G", "H", "I", "J", "K", "L"]
        elemp_list = ['B', 'E']#, 'H', 'K']
        tt0 = line.get_table(attr=True)
        tt_rfc_name = tt0.rows['rfc::.*'].name
        tt_qfl2a_name = tt0.rows['qfl2a:.*'].name
        # tt_d__04_name = {'0':'dc_04','1':'di_04','2':'dd_04','3':'dl_04'}
        # tt_d__04_name = {'0':'qd10c','1':'qd10i','2':'qd10d','3':'qd10f'}
        if 'qf9m.1' in line.element_names:
                sep = '.'
        elif 'qf9m:1' in line.element_names:
            sep = ':'
        if 'qf9m' in tt0.rows['qf9m.*'].name:
            tt_qf9m_name = {'0':'qf9m','1':f'qf9m{sep}1','2':f'qf9m{sep}3','3':f'qf9m{sep}5'}
        else:
            tt_qf9m_name = {'0':'qf9m.0','1':'qf9m.2','2':'qf9m.4','3':'qf9m.6'}
        tt_qd8m_name = {'0':f'qd8m{sep}0','1':f'qd8m{sep}2','2':f'qd8m{sep}4','3':f'qd8m{sep}6'}

    part_list = ['4R','5R']
    if ((optics_type=='GHC') and (install_new_location_GHC==True)): 
        part_list = ['5R'] # will install 4 wiggler elements instead of 8
        Cfcc = line.get_length()
        s0 = 10380+8.1 # distance from IPs with long drift, ~zero beta-beat in y, Dx<0.001, but slightly negative beta-beat in x
        wig_len = 7.27 # total length of the wiggler
        s_start_dict = { # start locations of the wigglers w.r.t IPs
            0: (line.get_s_position(f"ipa.1") + (s0-wig_len/2)) % Cfcc, # 1st wiggler
            1: (line.get_s_position(f"ipd.1") + (s0-wig_len/2)) % Cfcc, # 2nd wiggler
            2: (line.get_s_position(f"ipg.1") + (s0-wig_len/2)) % Cfcc, # 3rd wiggler
            3: (line.get_s_position(f"ipj.1") + (s0-wig_len/2)) % Cfcc # 4th wiggler
        }
    inserts = []
    for ss, straight in enumerate(["A","D","G","J"]):
        for ee, element in enumerate(elem_list): #can't use B for naming convention of arc dipoles.
            for pp,part in enumerate(part_list):   #GHC naming convention
                # assign correct element typeS
                element_type =  'MWIP' if element in elemp_list else 'MWIM'

                # reconstruct element name
                element_name = f"MWI.{element}{part}{straight}"

                # Calculate the position of the first wiggler element
                if optics_type=='GHC':   
                    if pp == 0:
                        if install_new_location_GHC==True:
                            s_start = s_start_dict[ss]
                        else:
                            s_start = line.get_s_position(f"qb4.{ss+1}") + L_qa
                    elif pp == 1:
                        s_start = line.get_s_position(f"qa4.{ss+1}") + L_qa
                elif optics_type=='LCC':
                    if f"rfc::{ss}" in tt_rfc_name:
                        s_start = tt0.rows[f"rfc::{ss}"].s[0] + L_qa + pp * 17
                    elif f"qfl2a:{2*ss}" in tt_qfl2a_name:
                        s_start = line.get_s_position(f"qfl2a:{2*ss}") + L_qa / 2 + pp * 17
                    else:
                        # s_start = line.get_s_position(tt_d__04_name[str(ss)]) + L_qa + pp * 17 #L_qa / 2 + pp * 17
                        if pp == 0:
                            s_start = line.get_s_position(tt_qd8m_name[str(ss)]) + L_qa
                        elif pp == 1:
                            s_start = line.get_s_position(tt_qf9m_name[str(ss)]) + L_qa 


                # offset for each wiggler element
                offset = (ee+1) * L_dis + offsets1[ee] * L_minus + offsets2[ee] * L_plus

                # create the wiggler element in the environment
                env.new(name=element_name, parent=element_type)

                # insert the wiggler element
                inserts.append(env.place(
                    name=element_name,
                    at=s_start + offset
                    ))
                
    # Add the wiggler elements to the line
    print('''Adding wiggler elements to the line...''')
    line.insert(inserts)

    return


def log_parameters (file_name=None, 
                    operation_mode='z', 
                    particle_type=None, 
                    dict_to_save=None, 
                    modes={'z': 45.6e9, 'w': 80e9, 'h': 120e9, 'tt': 182.5e9}
                    )-> dict:

    if dict_to_save is None:

        if particle_type is not None:
            particle = particle_type
        else:
            particle = 'positron'
            print('No particle type is provided, thus the default value - positron - is used!')

        pdg_id = xp.get_pdg_id_from_name(particle)
        pdg_atrr = xp.reference_from_pdg_id(pdg_id)
        particle_mass = pdg_atrr.mass0 
        particle_charge = pdg_atrr.charge[0]
        particle_classical_radius0 = pdg_atrr.get_classical_particle_radius0()

        energy = modes[operation_mode]  # eV
        gamma_relativistic = energy / particle_mass 
        beta_relativistic = np.sqrt(1 - 1 / gamma_relativistic**2) 

        if file_name is not None: 
            ref_par_imp = json.load(open(file_name))
            params = ref_par_imp.get(operation_mode, {})

            energy_rpf = params.get('ENERGY')
            if energy_rpf is not None:
                print('Energy is taken from the referens paramiter file!')
                energy = energy_rpf*1e9  # eV
                gamma_relativistic = energy / particle_mass 
                beta_relativistic = np.sqrt(1 - 1 / gamma_relativistic**2) 

            energy_loss_per_turn = params.get('ENERGYLOSS_PER_TURN')
            if energy_loss_per_turn is not None:
                energy_loss_per_turn *= 1e9  # eV

            longitudinal_damping_time = params.get('LONGITUDINAL_DAMPING_TIME')
            emittance_x = params.get('EMITTANCE_X')
            emittance_y = params.get('EMITTANCE_Y')
            bunch_population = params.get('BUNCH_POPULATION')

            energy_spread_SR = params.get('ENERGYSPREAD_SR')
            if energy_spread_SR is not None:
                energy_spread_SR *= 1e-2

            energy_spread_BS = params.get('ENERGYSPREAD_BS')
            if energy_spread_BS is not None:
                energy_spread_BS *= 1e-2

            bunch_length_SR = params.get('BUNCHLENGTH_SR')
            if bunch_length_SR is not None:
                bunch_length_SR *= 1e-3  # m

            bunch_length_BS = params.get('BUNCHLENGTH_BS')
            if bunch_length_BS is not None:
                bunch_length_BS *= 1e-3  # m

            circumference = params.get('CIRCUMFERENCE')
            qx_fractional = params.get('Q1')
            qy_fractional = params.get('Q2')
            qs = params.get('QS')
            dqx = params.get('QPRIME1')
            dqy = params.get('QPRIME2')
            betastar_x = params.get('BETASTAR_X')
            betastar_y = params.get('BETASTAR_Y')
            beambeam_tuneshift_x = params.get('BEAMBEAM_TUNESHIFT_X')
            beambeam_tuneshift_y = params.get('BEAMBEAM_TUNESHIFT_Y')
            momentum_compaction = params.get('MOMENTUM_COMPACTION')

            
            reference_parameters = {
                'reference_parameters_file_name': file_name,
                'operation_mode': operation_mode,
                'particle': particle,
                'particle_charge': particle_charge,
                'particle_mass': particle_mass,  # eV/c2
                'particle_classical_radius0': particle_classical_radius0,  # m
                'energy': energy,  # eV
                'gamma_relativistic': gamma_relativistic,
                'beta_relativistic': beta_relativistic,
                'energy_loss_per_turn': energy_loss_per_turn,  # eV
                'longitudinal_damping_time': longitudinal_damping_time,  # turns
                'emittance_x': emittance_x,  # m
                'emittance_y': emittance_y,  # m
                'normalized_emittance_x': emittance_x * gamma_relativistic * beta_relativistic if emittance_x and gamma_relativistic and beta_relativistic else None,
                'normalized_emittance_y': emittance_y * gamma_relativistic * beta_relativistic if emittance_y and gamma_relativistic and beta_relativistic else None,
                'bunch_population': bunch_population,
                'energy_spread_SR': energy_spread_SR,
                'energy_spread_BS': energy_spread_BS,
                'bunch_length_SR': bunch_length_SR,  # m
                'bunch_length_BS': bunch_length_BS,  # m
                'energy_spread': energy_spread_SR,
                'bunch_length': bunch_length_SR,  # m
                'momentum_compaction': momentum_compaction,  
                'circumference': circumference,
                'qx_fractional': qx_fractional,
                'qy_fractional': qy_fractional,
                'qs': qs,
                'dqx': dqx,
                'dqy': dqy,
                'betastar_x': betastar_x,
                'betastar_y': betastar_y,
                'beambeam_tuneshift_x': beambeam_tuneshift_x,
                'beambeam_tuneshift_y': beambeam_tuneshift_y
            }

        elif file_name is None:

            reference_parameters = {
                'reference_parameters_file_name': None,
                'operation_mode': operation_mode,
                'particle': particle,
                'particle_charge': particle_charge,
                'particle_mass': particle_mass,  # eV/c2
                'particle_classical_radius0': particle_classical_radius0,  # m
                'energy': energy,  # eV
                'gamma_relativistic': gamma_relativistic,
                'beta_relativistic': beta_relativistic,
                'energy_loss_per_turn': None,  # eV
                'longitudinal_damping_time': None,  # turns
                'emittance_x': None,  # m
                'emittance_y': None,  # m
                'normalized_emittance_x': None,
                'normalized_emittance_y': None,
                'bunch_population': None,
                'energy_spread_SR': None,
                'energy_spread_BS': None,
                'bunch_length_SR': None,  # m
                'bunch_length_BS': None,  # m
                'energy_spread': None,
                'bunch_length': None,  # m
                'momentum_compaction': None,  
                'circumference': None,
                'qx_fractional': None,
                'qy_fractional': None,
                'qs': None,
                'dqx': None,
                'dqy': None,
                'betastar_x': None,
                'betastar_y': None,
                'beambeam_tuneshift_x': None,
                'beambeam_tuneshift_y': None
            }


        parameters = {'reference_parameters':reference_parameters}

        return parameters

    if file_name is not None and dict_to_save is not None :
        
        with open(file_name, "w") as file:
            # Save to a YAML file
            if file_name.split(".")[-1] == 'yaml':
                clean_dict_to_save = convert_types(dict_to_save)
                yaml.dump(clean_dict_to_save, file, default_flow_style=False, sort_keys=False, allow_unicode=True)
            # Save to a JSON file
            elif file_name.split(".")[-1] == 'json':
                    json.dump(dict_to_save, file, indent=4)

def get_q_s(voltage,energy_loss_per_turn,rf_frequency,circumference,momentum_compaction,gamma_relativistic,energy):
    phi_s = np.arcsin(energy_loss_per_turn/voltage)
    q_s = np.sqrt(voltage*2.0*np.pi*rf_frequency*circumference*np.abs(momentum_compaction - 1/gamma_relativistic**2)/(energy*sp_co.c)*np.cos(phi_s))/(2.0*np.pi)
    return q_s
    
def adjust_voltage_for_q_s(target_q_s,voltage,energy_loss_per_turn,rf_frequency,circumference,momentum_compaction,gamma_relativistic,energy):
    penalty = lambda voltage0: np.abs(target_q_s-get_q_s(voltage0[0],energy_loss_per_turn,rf_frequency,circumference,momentum_compaction,gamma_relativistic,energy))
    opt_result = opt.minimize(penalty,x0=[voltage],method='Nelder-mead')
    return opt_result.x[0]

def Vtot(phi, V1, r, n, phi2, U0):
    vtot = V1 * (np.sin(phi) + r * np.sin(n * phi + phi2)) - U0
    return vtot


def W_rf(phi, phis, voltage_ratio, n, phi2):
    return (np.cos(phi) - np.cos(phis) + \
            voltage_ratio / n * (np.cos(n * phi + phi2) - np.cos(n * phis + phi2)) + \
            (phi - phis) * (np.sin(phis) + voltage_ratio * np.sin(
                n * phis + phi2)))  # Eq. (2.41) PhD thesis T. Argyropoulos
                
def get_momentum_acceptance(voltage,energy_loss_per_turn,rf_frequency,circumference,momentum_compaction,gamma_relativistic,energy,voltage_harm,rf_frequency_harm,harmonic_number):
    if energy_loss_per_turn > voltage+voltage_harm:
        return 0.0
    phi2 = -np.pi / 2
    if voltage_harm != 0:
        n = rf_frequency / rf_frequency_harm
        r = voltage / voltage_harm
        phi = np.linspace(0, 2 * np.pi, 10000)
        vtots = Vtot(phi,voltage_harm, r, n, phi2, energy_loss_per_turn)
        imax = np.argmax(vtots)
        if vtots[imax] > 0:
            p_max = phi[imax]
            phi_s = opt.brentq(Vtot, a=p_max, b=p_max + 0.5, args=(voltage_harm, r, n, phi2, energy_loss_per_turn))
            phi_u = opt.brentq(Vtot, a=p_max - 0.5, b=p_max, args=(voltage_harm, r, n, phi2, energy_loss_per_turn))
        else:
            return 0.0
    else:
        phi_s = np.arcsin(energy_loss_per_turn/voltage)
        phi_u = np.pi - phi_s
        r = 0
        n = 1
    eta = np.abs(momentum_compaction - 1/gamma_relativistic**2)
    diff = W_rf(phi_u, phi_s, r, n, phi2) - W_rf(phi_s, phi_s, r, n, phi2)
    if diff > 0.0:
        dee_max = np.sqrt(diff) * np.sqrt(voltage_harm / harmonic_number / eta / energy / np.pi)
        return dee_max
    else:
        return 0.0

def adjust_voltage_for_momentum_acceptance(target_momentum_acceptance,voltage,energy_loss_per_turn,rf_frequency,circumference,momentum_compaction,gamma_relativistic,energy,voltage_harm,rf_frequency_harm,harmonic_number):
    penalty = lambda voltage0: np.abs(target_momentum_acceptance-get_momentum_acceptance(voltage0[0],energy_loss_per_turn,rf_frequency,circumference,momentum_compaction,gamma_relativistic,energy,voltage_harm,rf_frequency_harm,harmonic_number))
    opt_result = opt.minimize(penalty,x0=[voltage],method='Nelder-mead')
    return opt_result.x[0]

def _voltage_penalty(args,twiss,parameters, BS_scale_factor=1.0,update_type='all',voltage_lower_limit=90E6):
    voltage = args[0]
    phi_s = np.arcsin(parameters['reference_parameters']['energy_loss_per_turn']/voltage)
    RF_frequency = parameters['reference_parameters']['rf_frequency_400']
    if 'rf_frequency_800' in parameters['reference_parameters'].keys() and parameters['reference_parameters']['rf_frequency_800']is not None:
        RF_frequency = parameters['reference_parameters']['rf_frequency_800']
    q_s = np.sqrt(voltage*2.0*np.pi*RF_frequency*parameters['reference_parameters']['circumference']*np.abs(parameters['reference_parameters']['momentum_compaction'] - 1/parameters['reference_parameters']['gamma_relativistic']**2)/(parameters['reference_parameters']['energy']*sp_co.c)*np.cos(phi_s))/(2.0*np.pi)
    harmonic_number = RF_frequency/(parameters['reference_parameters']['beta_relativistic']*sp_co.c/parameters['reference_parameters']['circumference'])
    momentum_acceptance = 2*q_s/(harmonic_number*parameters['reference_parameters']['momentum_compaction'])*np.sqrt(1+(phi_s-np.pi/2)*np.tan(phi_s))
    
    longitudinal_sigmas = get_sigma_zeta_and_delta_from_ws_beambeam(twiss, parameters['reference_parameters'], parameters['beam_beam_parameters'], BS_scale_factor=BS_scale_factor)
    if update_type == 'all':
        parameters['reference_parameters']['energy_spread_BS']= longitudinal_sigmas['energy_spread_w_BS']
        parameters['reference_parameters']['bunch_length_BS']= longitudinal_sigmas['bunch_length_w_BS']
    elif update_type == 'empty':
        if parameters['reference_parameters']['energy_spread_BS'] is None:
            parameters['reference_parameters']['energy_spread_BS']= longitudinal_sigmas['energy_spread_w_BS']
        if parameters['reference_parameters']['bunch_length_BS'] is None:
            parameters['reference_parameters']['bunch_length_BS']= longitudinal_sigmas['bunch_length_w_BS']
    voltage_penalty = 0.0
    if voltage < voltage_lower_limit:
        voltage_penalty = voltage_lower_limit-voltage
    print(f'voltage {voltage} acceptance {momentum_acceptance}, energy spread',longitudinal_sigmas['energy_spread_w_BS'])
    xi_penalty = np.abs(0.1-longitudinal_sigmas['energy_spread_w_BS']/momentum_acceptance)
    print(f'penalties {xi_penalty} {voltage_penalty}')
    return xi_penalty+voltage_penalty

def _bunch_population_penalty(args,line,twiss,parameters, BS_scale_factor=1.0,max_bb_param=0.1,update_type='all'):
    bunch_population = args[0]
    parameters['reference_parameters']['bunch_population'] = bunch_population
    #opt_result = opt.minimize(_voltage_penalty,np.array([parameters['reference_parameters']['rf_voltage_400']]),args=(twiss,parameters, BS_scale_factor,update_type),method='Nelder-Mead')
    longitudinal_sigmas = get_sigma_zeta_and_delta_from_ws_beambeam(twiss, parameters['reference_parameters'], parameters['beam_beam_parameters'], BS_scale_factor=BS_scale_factor)
    if update_type == 'all':
        parameters['reference_parameters']['energy_spread_BS']= longitudinal_sigmas['energy_spread_w_BS']
        parameters['reference_parameters']['bunch_length_BS']= longitudinal_sigmas['bunch_length_w_BS']
    elif update_type == 'empty':
        if parameters['reference_parameters']['energy_spread_BS'] is None:
            parameters['reference_parameters']['energy_spread_BS']= longitudinal_sigmas['energy_spread_w_BS']
        if parameters['reference_parameters']['bunch_length_BS'] is None:
            parameters['reference_parameters']['bunch_length_BS']= longitudinal_sigmas['bunch_length_w_BS']    
    beambeam_xi_x, beambeam_xi_y, min_emit_pplane = get_beambeam_tune_shift (line, parameters['reference_parameters'], parameters['beam_beam_parameters'], max_bb_tune_shift=max_bb_param)
    return np.abs(beambeam_xi_y-max_bb_param)

def update_reference_parameters_from_line (line, parameters, eneloss=True, BS_scale_factor: float = 1.0, update_type='empty',max_bb_param=0.1):

    reference_parameters = parameters['reference_parameters']

    twiss = line.twiss(eneloss_and_damping=eneloss)
    ref_part = line.particle_ref
    ip_list = get_ip_list(line)

    # if reference_parameters['optics_type'] == 'GHC':
    #     ip_list = ['ipa.1','ipd.1','ipg.1','ipj.1']
    # elif reference_parameters['optics_type'] == 'LCC':
    #     ip_list = twiss.rows['ip.*'].name.tolist()[::2]
    # else:
    #     ip_list = None

    # Fill existing and empty parameters
    if update_type == 'all':
        reference_parameters['particle_charge']= ref_part.q0
        reference_parameters['particle_mass']= ref_part.mass0  # eV/c2
        reference_parameters['energy']= ref_part.p0c[0]  # eV
        reference_parameters['gamma_relativistic']= twiss.gamma0
        reference_parameters['beta_relativistic']= twiss.beta0
        if eneloss:
            eq_ge_z = twiss.eq_gemitt_zeta
            sig_z = np.sqrt(eq_ge_z*np.abs(twiss.slip_factor)*twiss.circumference/(2*np.pi*twiss.qs))
            sig_dd = eq_ge_z / sig_z
            reference_parameters['bunch_length_SR']= sig_z
            reference_parameters['energy_spread_SR']= sig_dd
            reference_parameters['emittance_x']= twiss.eq_gemitt_x
            reference_parameters['emittance_x']= twiss.eq_gemitt_x  # m
            reference_parameters['emittance_y']= twiss.eq_gemitt_y  # m
            reference_parameters['normalized_emittance_x']= twiss.eq_nemitt_x
            reference_parameters['normalized_emittance_y']= twiss.eq_nemitt_y
            reference_parameters['energy_loss_per_turn']= twiss.eneloss_turn  # eV
            reference_parameters['longitudinal_damping_time']= ref_part.p0c[0]/twiss.eneloss_turn  # turns
        reference_parameters['momentum_compaction']= twiss.momentum_compaction_factor  
        reference_parameters['circumference']= twiss.circumference
        reference_parameters['qx_fractional']= twiss.qx % 1
        reference_parameters['qy_fractional']= twiss.qy % 1
        reference_parameters['qs']= twiss.qs
        reference_parameters['dqx']= twiss.dqx
        reference_parameters['dqy']= twiss.dqy
        if ip_list is None:
            reference_parameters['betastar_x']= np.min(twiss.rows['ip.*'].betx)
            reference_parameters['betastar_y']= np.min(twiss.rows['ip.*'].bety)
        else:
            reference_parameters['betastar_x']= np.min(twiss.rows[ip_list].betx)
            reference_parameters['betastar_y']= np.min(twiss.rows[ip_list].bety)

    # Fill only empty parameters
    elif update_type == 'empty':
        if reference_parameters['particle_charge'] is None:
            reference_parameters['particle_charge']= ref_part.q0
        if reference_parameters['particle_mass'] is None:
            reference_parameters['particle_mass']= ref_part.mass0  # eV/c2
        if reference_parameters['energy'] is None:
            reference_parameters['energy']= ref_part.p0c[0] # eV
        if reference_parameters['gamma_relativistic'] is None:
            reference_parameters['gamma_relativistic']= twiss.gamma0
        if reference_parameters['beta_relativistic'] is None:
            reference_parameters['beta_relativistic']= twiss.beta0
        if eneloss:
            eq_ge_z = twiss.eq_gemitt_zeta
            sig_z = np.sqrt(eq_ge_z*np.abs(twiss.slip_factor)*twiss.circumference/(2*np.pi*twiss.qs))
            sig_dd = eq_ge_z / sig_z
            if reference_parameters['bunch_length_SR'] is None:
                reference_parameters['bunch_length_SR']= sig_z
            if reference_parameters['energy_spread_SR'] is None:
                reference_parameters['energy_spread_SR']= sig_dd
            if reference_parameters['emittance_x'] is None:
                reference_parameters['emittance_x']= twiss.eq_gemitt_x  # m
            if reference_parameters['emittance_y'] is None:
                reference_parameters['emittance_y']= twiss.eq_gemitt_y  # m
            if reference_parameters['normalized_emittance_x'] is None:
                reference_parameters['normalized_emittance_x']= twiss.eq_nemitt_x
            if reference_parameters['normalized_emittance_y'] is None:  
                reference_parameters['normalized_emittance_y']= twiss.eq_nemitt_y
            if reference_parameters['energy_loss_per_turn'] is None:
                reference_parameters['energy_loss_per_turn']= twiss.eneloss_turn # eV
            if reference_parameters['longitudinal_damping_time'] is None:
                reference_parameters['longitudinal_damping_time']= ref_part.p0c[0]/twiss.eneloss_turn # turns
        if reference_parameters['momentum_compaction'] is None:
            reference_parameters['momentum_compaction']= twiss.momentum_compaction_factor
        if reference_parameters['circumference'] is None:
            reference_parameters['circumference']= twiss.circumference
        if reference_parameters['qx_fractional'] is None:
            reference_parameters['qx_fractional']= twiss.qx % 1
        if reference_parameters['qy_fractional'] is None:
            reference_parameters['qy_fractional']= twiss.qy % 1
        if reference_parameters['qs'] is None:
            reference_parameters['qs']= twiss.qs
        if reference_parameters['dqx'] is None:
            reference_parameters['dqx']= twiss.dqx
        if reference_parameters['dqy'] is None:
            reference_parameters['dqy']= twiss.dqy
        if reference_parameters['betastar_x'] is None:
            if ip_list is None:
                reference_parameters['betastar_x']= np.min(twiss.rows['ip.*'].betx)
            else:
                reference_parameters['betastar_x']= np.min(twiss.rows[ip_list].betx)
        if reference_parameters['betastar_y'] is None:
            if ip_list is None:
                reference_parameters['betastar_y']= np.min(twiss.rows['ip.*'].bety)
            else:
                reference_parameters['betastar_y']= np.min(twiss.rows[ip_list].bety)
    
    if update_type == 'all' or (update_type == 'empty' and parameters['reference_parameters']['bunch_population'] is None):
        opt_result = opt.minimize(_bunch_population_penalty,np.array([1.5E11]),args=(line,twiss,parameters, BS_scale_factor,max_bb_param,update_type),method='Nelder-Mead')
    else:
        longitudinal_sigmas = get_sigma_zeta_and_delta_from_ws_beambeam(twiss, parameters['reference_parameters'], parameters['beam_beam_parameters'], BS_scale_factor=BS_scale_factor)
        if parameters['reference_parameters']['energy_spread_BS'] is None:
            parameters['reference_parameters']['energy_spread_BS']= longitudinal_sigmas['energy_spread_w_BS']
        if parameters['reference_parameters']['bunch_length_BS'] is None:
            parameters['reference_parameters']['bunch_length_BS']= longitudinal_sigmas['bunch_length_w_BS']
    correct_parameters_conflicts(parameters)

    return 

def load_configuration_parameters (file_name):

    with open(file_name, "r") as file:
        # Load YAML file into a dictionary
        if file_name.split(".")[-1] == 'yaml':
                config_dict = yaml.safe_load(file)
        # Load JSON file into a dictionary
        elif file_name.split(".")[-1] == 'json':
            config_dict = json.load(file)

    return config_dict


def convert_types(obj):
    """
    Recursively convert NumPy and other non-serializable types
    into native Python types that YAML can safely dump/load.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy arrays to lists
    elif isinstance(obj, (np.generic,)):  
        # Handles np.float64, np.int64, np.bool_, etc.
        return obj.item()
    elif isinstance(obj, (set, tuple)):
        return [convert_types(v) for v in obj]  # Convert sets/tuples to lists
    elif isinstance(obj, dict):
        return {convert_types(k): convert_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_types(v) for v in obj]
    elif isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", errors="ignore")  # Ensure strings
    elif obj is None:
        return None  # YAML 'null'
    else:
        return obj  # Leave native types untouched
    

def correct_parameters_conflicts(parameters, update_study_parameters_from_reference=False):

    # update energy dependent parameters
    if parameters['reference_parameters']['energy'] is not None:
        energy = parameters['reference_parameters']['energy']
        particle_mass = parameters['reference_parameters']['particle_mass']
        gamma_relativistic = energy / particle_mass 
        beta_relativistic = np.sqrt(1 - 1 / gamma_relativistic**2) 
        parameters['reference_parameters']['gamma_relativistic'] = gamma_relativistic
        parameters['reference_parameters']['beta_relativistic'] = beta_relativistic

    # if collisions is True set correctly the energy_spread and bunch_length
    if 'beam_beam_parameters' in parameters:
        if parameters['beam_beam_parameters']['collisions']:
            parameters['reference_parameters']['energy_spread'] = parameters['reference_parameters']['energy_spread_BS']
            parameters['reference_parameters']['bunch_length'] = parameters['reference_parameters']['bunch_length_BS']
            print ('The collisions is True thus the energy_spread and bunch_length are updated to BS ones')
        if not parameters['beam_beam_parameters']['collisions']:
            parameters['reference_parameters']['energy_spread'] = parameters['reference_parameters']['energy_spread_SR']
            parameters['reference_parameters']['bunch_length'] = parameters['reference_parameters']['bunch_length_SR']
            print ('The collisions is False thus the energy_spread and bunch_length are updated to SR ones')
        
    # if update_stady_parameters_from_reference is True then update the study parameters
    if update_study_parameters_from_reference:
        parameters['study_parameters']['ini_cond_nemittance_x'] = parameters['reference_parameters']['normalized_emittance_x']
        parameters['study_parameters']['ini_cond_nemittance_y'] = parameters['reference_parameters']['normalized_emittance_y']
        parameters['study_parameters']['ini_cond_bunch_length'] = parameters['reference_parameters']['bunch_length']
        parameters['study_parameters']['ini_cond_energy_spread'] = parameters['reference_parameters']['energy_spread']
        
    return 


def load_sequence_from_madx (madx, reference_parameters, collimators_on=False):

    ref_param = reference_parameters

    madx.call(f"{ref_param['acc_model_path']}/lattices/{ref_param['operation_mode']}/{ref_param['lattice_file_name']}")
    if len(madx.sequence.keys())!=0:
        sequence_name = list(madx.sequence.keys())[0]
    else:
        sequence_name = f"fccee_{ref_param['particle'][0]}_ring" 
    madx.input(f"""beam, particle={ref_param['particle']}, 
            sequence={sequence_name},  
            energy:={ref_param['energy']*1e-9}, 
            NPART:={ref_param['bunch_population']},  
            ex:={ref_param['emittance_x']}, 
            ey:={ref_param['emittance_y']},  
            sige:={ref_param['energy_spread']}, 
            sigt:={ref_param['bunch_length']}""")

    # use sequence
    madx.use(sequence=sequence_name)
    ref_param['brho'] = madx.sequence[sequence_name].beam.brho


    # add collimators and apertures if needed
    if collimators_on:
        madx_add_collimators(madx, ref_param['aperture_dir'],sequence_name=sequence_name)
        madx_add_apertures (madx, ref_param['aperture_dir'], sequence_name=sequence_name)

    return sequence_name


def element_selection_from_line(line, selection_criteria, get_thin_element_parent=False):

    tt = line.get_table(attr=True)
    sv = line.survey()

    # Extract regex patterns and element types from selection_criteria
    regex_patterns = [re.compile(item) for item in selection_criteria if ".*" in item]

    # Extract relevant elements dynamically
    filtered_elements = []
    for name in line.element_names:
        if ~line[name].isthick and '_parent' in line[name].get_expr().keys() and get_thin_element_parent:
            if name.startswith("drift_"):
                continue
            else:
                element = line[name]._parent
                element_type = element.__class__.__name__
                element_name = name.split('..')[0]
                elem_name_for_s = element_name+'_entry'
        else:
            element = line[name]
            element_type = element.__class__.__name__
            element_name = name
            elem_name_for_s = element_name
        if element_type in selection_criteria or any(pattern.match(name) for pattern in regex_patterns):
            filtered_elements.append(
                {
                    "name": element_name,
                    "type": element_type,  # Get the element type
                    "s": elem_name_for_s,  # Get the longitudinal position (s) 
                    "X": elem_name_for_s,  # Get the survey position (X) 
                    "Y": elem_name_for_s,  # Get the survey position (Y) 
                    "Z": elem_name_for_s,  # Get the survey position (Z) 
                    #"length": getattr(element, "length", 0),  # Length of element
                    #"k1": getattr(element, "k1", 0),  # Quadrupole strength
                    #"k1s": getattr(element, "k1s", 0),  # Skew Quadrupole strength
                    #"k2": getattr(element, "k2", 0),  # Sextupole strength
                    #"k2s": getattr(element, "k2s", 0),  # Skew Sextupole strength
                    #"k3": getattr(element, "k3", 0),  # Octupole strength
                    #"k3s": getattr(element, "k3s", 0),  # Skew Octupole strength
                    #"shift_x": getattr(element, "shift_x", 0),  # X shift of the element
                    #"shift_y": getattr(element, "shift_y", 0),  # Y shift of the element
                    #"shift_s": getattr(element, "shift_s", 0),  # S shift of the element
                })

    # Convert to Pandas DataFrame
    df = pd.DataFrame(filtered_elements).drop_duplicates(subset=["name"], keep="first").set_index('name', drop=False).rename_axis(None)
    df['s'] = tt.rows[df.s[:]].s
    df['X'] = sv.rows[df.X[:]].X
    df['Y'] = sv.rows[df.Y[:]].Y
    df['Z'] = sv.rows[df.Z[:]].Z

    return df

def add_misalignment_error (line, element_familys, error_class='systimatic', seeds=201,
                    shift_x=0,shift_y=0,shift_s=0,rot_s_rad=0, girder_misalignment=False, knob_name=None):
    
    '''
    loop over the element_familys entris and add erros
    for evry entry an error knob is generated 
    
    element_familys: 'Dipole', 'Quadrupole', ... or 'sf.*', 'sd.*', ...
    error_class: 'systimatic', 'random' (in 3sigma)
    seeds: seed number for each shift or rotation.
    '''
    #make a dictionary with the errors for BPM's, to be given to the orbit correction function
    if element_familys== 'bpm':

        tt = line.get_table(attr=True)

        misalignment_dict = {}
        BPMs=tt.rows[f"{element_familys}.*"].name
        n_bpm = len(BPMs)
        x1_array = np.zeros(n_bpm)
        y1_array = np.zeros(n_bpm)
        r1_array = np.zeros(n_bpm)
        for jj, BPM in enumerate(BPMs):
            match = re.search(rf"^{element_familys}_(.*)$", BPM)

            element_name = match.group(1)

            # Try element_name, or element_name+'a', or element_name+'b'
            try:
                _ = line[element_name]
            except KeyError:
                # If that fails, try element_name + 'a'
                try:
                    _ = line[element_name + 'a']
                    element_name += 'a'
                except KeyError:
                    # If that fails, try element_name + 'b'
                    try:
                        _ = line[element_name + 'b']
                        element_name += 'b'

                    except KeyError:
                        continue  # Skip if none exist
            misalignment_dict[BPM] = {}

            #find the misalignment of the element the BPM is attached to 
            misalignments=tt.rows[element_name].cols['shift_x','shift_y','rot_s_rad']
            mean = 0
            std_dev = 1
            lower_bound = -2.5 # in sigma
            upper_bound = 2.5 # in sigma
            size=len(BPMs)
            seeds2 =[seeds, seeds+1,seeds+2, seeds+3]
            for ii, name in enumerate(['sx','sy','rs']):
                if name == 'sx':
                    np.random.seed(seeds2[ii])
                    x1 = misalignments['shift_x'][0].item()
                    x1_array[jj]=x1
                    x2=shift_x*truncnorm.rvs(lower_bound, upper_bound, loc=mean, scale=std_dev,size=size)
                elif name == 'sy':
                    np.random.seed(seeds2[ii])
                    y1 = misalignments['shift_y'][0].item()
                    y1_array[jj]=y1
                    y2=shift_y*truncnorm.rvs(lower_bound, upper_bound, loc=mean, scale=std_dev, size=size)
                elif name == 'rs':
                    np.random.seed(seeds2[ii])
                    el = line[element_name]
                    r1 = float(el._rot_s_rad_no_frame)
                    r1_array[jj]=r1
                    r2=rot_s_rad*truncnorm.rvs(lower_bound, upper_bound, loc=mean, scale=std_dev, size=size)
        for jj, BPM in enumerate(BPMs):
                #add the element misalignment and the random BPM beam based alignment error to obtain the total BPM position error
                misalignment_dict[BPM]['shift_x']=float(x1_array[jj]+x2[jj])
                misalignment_dict[BPM]['shift_y']=float(y1_array[jj]+y2[jj])
                misalignment_dict[BPM]['rot_s_rad']=float(r1_array[jj]+r2[jj])
        return misalignment_dict


    else:
        #misalignments for dipoles, quadrupoles, sextupoles, girders
        tt = line.get_table(attr=True)
        tt_no_parent_elem = tt.rows[tt.parent_name==None]
        tt_no_marker_no_parent_elem = tt_no_parent_elem.rows[tt_no_parent_elem.element_type!='Marker']

        element_family_list = ensure_list(element_familys)
        elements_with_types = [(name, line.element_dict[name].__class__.__name__) for name in line.element_names]


        for ii,element_family in enumerate(element_family_list):
            #off switch to regulate the misalignments without calling the function
            if knob_name is None:
                mis_switch_name = 'mis_'+element_family+'_'+error_class[:3]+'_'+str(seeds)
                line[mis_switch_name] = 1
            elif np.size(knob_name)== np.size(element_familys):
                mis_switch_name = knob_name[ii]
                line[mis_switch_name] = 1
            else:
                mis_switch_name = knob_name[0]
                line[mis_switch_name] = 1


            ## in order to include the parent elements if the line has thin element
            parent_names = [item for item in tt.parent_name if item is not None]
            if len(parent_names)>0:
                parent_types = [line[parent_names[ii]].__class__.__name__ for ii in range(len(parent_names))]
            else:
                parent_types = []

            if element_family in np.unique(np.append(tt.element_type,parent_types)):
                parent_type_names = [name for ii, name in enumerate(parent_names) if parent_types[ii] == element_family]
                element_names = np.append(tt_no_marker_no_parent_elem.rows[tt_no_marker_no_parent_elem.element_type==element_family].name,parent_type_names)
            else:
                parent_type_names = [name for name in parent_names if re.match(element_family, name)] 
                element_names = np.append(tt_no_marker_no_parent_elem.rows[element_family].name,parent_type_names)

            size = len(element_names)

            if error_class == 'systimatic':
                shift_values_x = np.full(shift_x, size)
                shift_values_y = np.full(shift_y, size)
                shift_values_s = np.full(shift_s, size)
                rot_values_s = np.full(rot_s_rad, size)
            elif error_class == 'random':
                mean = 0
                std_dev = 1
                lower_bound = -2.5 # in sigma
                upper_bound = 2.5 # in sigma
                seeds2 =[seeds, seeds+1,seeds+2, seeds+3]

                for ii, name in enumerate(['sx','sy','ss','rs']):
                    if name == 'sx':
                        np.random.seed(seeds2[ii])
                        shift_values_x = shift_x*truncnorm.rvs(lower_bound, upper_bound, loc=mean, scale=std_dev, size=size)
                    elif name == 'sy':
                        np.random.seed(seeds2[ii])
                        shift_values_y = shift_y*truncnorm.rvs(lower_bound, upper_bound, loc=mean, scale=std_dev, size=size)
                    elif name == 'ss':
                        np.random.seed(seeds2[ii])
                        shift_values_s = shift_s*truncnorm.rvs(lower_bound, upper_bound, loc=mean, scale=std_dev, size=size)
                    elif name == 'rs':
                        np.random.seed(seeds2[ii])
                        rot_values_s = rot_s_rad*truncnorm.rvs(lower_bound, upper_bound, loc=mean, scale=std_dev, size=size)
            
            if girder_misalignment:

                #for girders the total element missalignment is the misalignment of the element plus the misalignment of the girder on which the element is placed
                for ii, name in enumerate(element_names):
                    idx = line.element_names.index(name)
                    i_left = idx
                    #a girder contains a quadrupole and the nearest sextupole, unless there is a dipole imnbetween, in which case it only contains a quadrupole
                    while i_left >= 0 and elements_with_types[i_left][1] != "RBend":
                        i_left -= 1

                    # Scan right
                    i_right = idx

                    while i_right < len(elements_with_types) and elements_with_types[i_right][1] != "RBend":
                        i_right += 1

                    # all_of_them = [elements_with_types[i][0] for i in range(i_left + 1, i_right)]

                    all_of_them = [elements_with_types[i][0] 
                            for i in range(i_left + 1, i_right) 
                            if elements_with_types[i][1] not in ("Drift", "Marker")]
                    for aa,el in enumerate(all_of_them):
                        line[el].shift_x = line.ref[mis_switch_name]*line[el].shift_x+line.ref[mis_switch_name]*shift_values_x[ii]
                        line[el].shift_y = line.ref[mis_switch_name]*line[el].shift_y+line.ref[mis_switch_name]*shift_values_y[ii]
                        line[el].shift_s = line.ref[mis_switch_name]*line[el].shift_s+line.ref[mis_switch_name]*shift_values_s[ii]
                        line[el]._rot_s_rad_no_frame =line.ref[mis_switch_name]*line[el].rot_s_rad_no_frame+ line.ref[mis_switch_name]*rot_values_s[ii]
            
            else:

                 #for all other element types the misalignment is simply the value obtained
                for ii, name in enumerate(element_names):
                    line[name].shift_x = line.ref[mis_switch_name]*shift_values_x[ii]
                    line[name].shift_y = line.ref[mis_switch_name]*shift_values_y[ii]
                    line[name].shift_s = line.ref[mis_switch_name]*shift_values_s[ii]
                    line[name]._rot_s_rad_no_frame = line.ref[mis_switch_name]*rot_values_s[ii]

    return


# def add_misalignment_error (line, element_familys, error_class='systimatic', seeds=[201,202,203,204],
#                     shift_x=0,shift_y=0,shift_s=0,rot_s_rad=0):
    
#     '''
#     loop over the element_familys entris and add erros
#     for evry entry an error knob is generated 
    
#     element_familys: 'Dipole', 'Quadrupole', ... or 'sf.*', 'sd.*', ...
#     error_class: 'systimatic', 'random' (in 3sigma)
#     seeds: seed number for each shift or rotation
#     '''
    
#     tt = line.get_table(attr=True)
#     tt_no_parent_elem = tt.rows[tt.parent_name==None]
#     tt_no_marker_no_parent_elem = tt_no_parent_elem.rows[tt_no_parent_elem.element_type!='Marker']

#     element_family_list = ensure_list(element_familys)

#     for element_family in element_family_list:

#         mis_switch_name = 'mis_'+element_family+'_'+error_class[:3]+'_'+str(seeds)
#         line[mis_switch_name] = 1

#         ## in order to includ the parent elements if the line has thin element
#         parent_names = [item for item in tt.parent_name if item is not None]
#         if len(parent_names)>0:
#             parent_types = [line[parent_names[ii]].__class__.__name__ for ii in range(len(parent_names))]
#         else:
#             parent_types = []

#         if element_family in np.unique(np.append(tt.element_type,parent_types)):
#             parent_type_names = [name for ii, name in enumerate(parent_names) if parent_types[ii] == element_family]
#             element_names = np.append(tt_no_marker_no_parent_elem.rows[tt_no_marker_no_parent_elem.element_type==element_family].name,parent_type_names)
#         else:
#             parent_type_names = [name for name in parent_names if re.match(element_family, name)] 
#             element_names = np.append(tt_no_marker_no_parent_elem.rows[element_family].name,parent_type_names)

#         size = len(element_names)

#         if error_class == 'systimatic':
#             shift_values_x = np.full(shift_x, size)
#             shift_values_y = np.full(shift_y, size)
#             shift_values_s = np.full(shift_s, size)
#             rot_values_s = np.full(rot_s_rad, size)
#         elif error_class == 'random':
#             mean = 0
#             std_dev = 1
#             lower_bound = -3 # in sigma
#             upper_bound = 3 # in sigma
#             for ii, name in enumerate(['sx','sy','ss','rs']):
#                 if name == 'sx':
#                     np.random.seed(seeds[ii])
#                     shift_values_x = shift_x*truncnorm.rvs(lower_bound, upper_bound, loc=mean, scale=std_dev, size=size)
#                 elif name == 'sy':
#                     np.random.seed(seeds[ii])
#                     shift_values_y = shift_y*truncnorm.rvs(lower_bound, upper_bound, loc=mean, scale=std_dev, size=size)
#                 elif name == 'ss':
#                     np.random.seed(seeds[ii])
#                     shift_values_s = shift_s*truncnorm.rvs(lower_bound, upper_bound, loc=mean, scale=std_dev, size=size)
#                 elif name == 'rs':
#                     np.random.seed(seeds[ii])
#                     rot_values_s = rot_s_rad*truncnorm.rvs(lower_bound, upper_bound, loc=mean, scale=std_dev, size=size)

#         for ii, name in enumerate(element_names):
#             line[name].shift_x = line.ref[mis_switch_name]*shift_values_x[ii]
#             line[name].shift_y = line.ref[mis_switch_name]*shift_values_y[ii]
#             line[name].shift_s = line.ref[mis_switch_name]*shift_values_s[ii]
#             line[name].rot_s_rad = line.ref[mis_switch_name]*rot_values_s[ii]
            
#     return


def add_optics_correctors(line, corector_names, corector_type=None):
    '''
    Similar to add_steering_correctors. This function adds any order correctors (dipole, quadrupole, sextupole ....) through the {knl,ksl} funcitonality of the elements.
    X and Y planes can have separate correctors.

    line: the line the correctors will be added to
    corector_names: the names of the elements the correctors will be added to
    corector_type:defines the order and plane of the correctors added. eg. for a corrector names input of ['qf2a:.*','sf1a:.*'] the corrector type input will be 
    [['ksl1','knl1', knl2], ['ksl3','knl4']] which adds quadrupole correctors in x, y plane and sextupole correctors in the x plane, all placed on the 'qf2a:.*' family.
    On the 'sf1a:.*' family octupole correctors are placed on the y plane, and decapole correctors on th x plane.
    '''
    #adds any order correctors (dipole, quadrupole, sextupole ....)
    tt = line.get_table(attr=True)
    element_groups1 = []
    corector_groups1 = []
    knl_elements = []
    ksl_elements = []

    for jj, name in enumerate(corector_names):
        if name in ['Bend','RBend','Quadrupole','Sextupole']:
            tt_name = tt.rows[tt.element_type==name]
        else:
            tt_name = tt.rows[name]
        if len(corector_names) != len(corector_type):
            corector_type = [corector_type[0] for _ in corector_names]
        expanded_names = tt_name.name.tolist()
        element_groups1.extend(expanded_names)
        corector_groups1.extend([corector_type[jj]] * len(expanded_names))
    
    for i, element in enumerate(element_groups1):
        types = corector_groups1[i]

        for t in types:
            # extract the number (indicative of the order of the corrector)
            num = int(re.search(r'\d+', t).group())  
            
            if "knl" in t.lower():
                line.vars[f'knl{num}' + str(element)] = 0
                line[element].knl[num] = line.vars[f'knl{num}' + str(element)]
                if element not in knl_elements:
                    knl_elements.append(element)
            
            elif "ksl" in t.lower():
                line.vars[f'ksl{num}' + str(element)] = 0
                line[element].ksl[num] = line.vars[f'ksl{num}' + str(element)]
                if element not in ksl_elements:
                    ksl_elements.append(element)

    return (ksl_elements, knl_elements)


def off_switch(element_list, switch_rate):
    ''' 
    Given the list of elements affected it will remove elements randomly. Used for correctors and BPM's to see how stable the solution is.

    element_list: list of elements to be treated. Accepts nested lists eg. [[family10, family11],[family20, family21]]
    switch_rate: the rate at which the element will be removed, maximum value 1, minimum 0.

    returns the lists in the order and format provided with the modifications.
    '''
    new_element_list = []
    removed_element_list = []
    np.random.seed()
    for sublists in element_list:
        cleaned_sublists = []
        trash = []
        for sub in sublists:
            kept = [e for e in sub if np.random.rand >= switch_rate]
            discarded = [e for e in sub if np.random.rand < switch_rate]
            cleaned_sublists.append(kept)
            trash.append(discarded)
        new_element_list.append(cleaned_sublists)
        removed_element_list.append(trash)
    return new_element_list, removed_element_list


def add_steering_correctors(line, corector_names, corrector_plane='hv'):
    '''
    Similar to add_optics_correctors. This function only returns dipole correctors, with an option to have them only in the x or y plane.
    Adds new elements to the line rather than using the {knl,ksl} funcitonality contrary to add_optics_correctors.

    line: the line the correctors will be added to
    corector_names: the names of the elements the correctors will be added to
    corrector_plane: the plane on which the correctors will act.
    '''
    tt = line.get_table(attr=True)
    element_groups1 = []
    knl_elements = []
    ksl_elements = []

    for jj, name in enumerate(corector_names):
        if name in ['Bend','RBend','Quadrupole','Sextupole', 'Octupole']:
            tt_name = tt.rows[tt.element_type==name]
        else:
            tt_name = tt.rows[name]
        expanded_names = tt_name.name.tolist()
        element_groups1.extend(expanded_names)
    
    for i, element in enumerate(element_groups1):
        line.vars['knl' + f'{element}'] = 0
        line.vars['ksl' + f'{element}'] = 0

        if corrector_plane == 'hv':
            line.insert_element("xsteering_corector_"+ f'{element}', xt.Multipole(knl=np.array([0])), at=f'{element}')
            line.insert_element("ysteering_corector_"+ f'{element}', xt.Multipole(ksl=np.array([0])), at=f'{element}')
            line["xsteering_corector_"+ f'{element}'].knl = line.vars['knl' + f'{element}']
            line["ysteering_corector_"+ f'{element}'].ksl = line.vars['ksl' + f'{element}']
            if element not in knl_elements:
                knl_elements.append(element)
            if element not in ksl_elements:
                ksl_elements.append(element)
            line.steering_correctors_x=knl_elements
            line.steering_correctors_y=knl_elements 

        elif corrector_plane == 'h':
            line.insert_element("xsteering_corector_"+ f'{element}', xt.Multipole(knl=np.array([0])), at=f'{element}')
            line["xsteering_corector_"+ f'{element}'].knl = line.vars['knl' + f'{element}']
            if element not in knl_elements:
                knl_elements.append(element)
            line.steering_correctors_x=knl_elements

        elif corrector_plane == 'v': 
            line.insert_element("ysteering_corector_"+ f'{element}', xt.Multipole(ksl=np.array([0])), at=f'{element}')
            line["ysteering_corector_"+ f'{element}'].ksl = line.vars['ksl' + f'{element}']
            if element not in ksl_elements:
                ksl_elements.append(element)
            line.steering_correctors_y=knl_elements 

    return


def add_markers(line, marker_placement, marker_name=None):
    '''
    Adds marker elements to the latticel.

    line: the line the markers will be added to
    marker_placement: the elements next to which the markers will be added
    marker_name: the distinguishing word attached to the start of the markers to distinguish them eg. {BPM}_{sf1a:1}, where marker_name=BPM and the element on 
    which the marker is attached is sf1a:1.
    '''
    tt = line.get_table(attr=True)
    # env = xt.Environment()

    added_markers = []
    element_groups1 = []
    for jj, name in enumerate(marker_placement):
        if name in ['Bend','RBend','Dipole','Quadrupole','Sextupole']:
            tt_name = tt.rows[tt.element_type==name]
        else:
            tt_name = tt.rows[name]
        expanded_names = tt_name.name.tolist()
        element_groups1.extend(expanded_names)
    # marker_places = []

    for i, element in enumerate(element_groups1): 
        if marker_name is None: 
            marker = f"{element}_marker" 
        else: 
            marker = f"{marker_name}_{element}" 
        line.insert_element(marker, xt.Marker() , at=f"{element}")
        # Insert all markers in one call
        # line.insert(marker_places)

        added_markers.append(marker)
    line.steering_monitors_x =added_markers
    line.steering_monitors_y =added_markers

    return added_markers


def add_field_error (line, element_familys, error_class='systimatic', seed=0,
                    error_type=None, error_category='relative', error_strength=1, reference_radius=1E-2, B_ref=None):
    
    '''
    loop over the element_familys entris and add erros
    for evry entry an error knob is generated 
    
    element_familys: 'Dipole', 'Quadrupole', ... or 'sf.*', 'sd.*', ...
    error_class: 'systimatic', 'random' (in 3sigma)
    error_type: 'b1', 'a1', 'b2', 'a2', ... k0, k1, k1s, ...
    error_category: 'relative', 'absolute' 
    B_ref: reference field for absolute errors, needed for b and a type errors, also for relative errors if the error not mesured at the nominal field
    '''
    
    tt = line.get_table(attr=True)
    tt_no_parent_elem = tt.rows[tt.parent_name==None]
    tt_no_marker_no_parent_elem = tt_no_parent_elem.rows[tt_no_parent_elem.element_type!='Marker']

    element_family_list = ensure_list(element_familys)

    for element_family in element_family_list:

        error_switch_name = 'err_'+element_family+'_'+error_type+'_'+error_category[:3]+'_'+error_class[:3]+'_'+str(seed)
        line[error_switch_name] = 1
        
        ## in order to includ the parent elements if the line has thin element
        parent_names = [item for item in tt.parent_name if item is not None]
        if len(parent_names)>0:
            parent_types = [line[parent_names[ii]].__class__.__name__ for ii in range(len(parent_names))]
        else:
            parent_types = []

        if element_family in np.unique(np.append(tt.element_type,parent_types)):
            parent_type_names = [name for ii, name in enumerate(parent_names) if parent_types[ii] == element_family]
            element_names = np.append(tt_no_marker_no_parent_elem.rows[tt_no_marker_no_parent_elem.element_type==element_family].name,parent_type_names)
        else:
            parent_type_names = [name for name in parent_names if re.match(element_family, name)] 
            element_names = np.append(tt_no_marker_no_parent_elem.rows[element_family].name,parent_type_names)

        size = len(element_names)

        if error_class == 'systimatic':
            error_values = np.ones(size)*error_strength
        elif error_class == 'random':
            np.random.seed(seed)
            std_dev = 1
            mean = 0
            lower_bound = -3 # in sigma
            upper_bound = 3 # in sigma
            error_values = error_strength*truncnorm.rvs(lower_bound, upper_bound, loc=mean, scale=std_dev, size=size)
        
        if error_type[0] == 'k':
            if error_type[-1]=='s':
                normal_skew = 's'
                order = int(error_type[1:-1])
            else:
                normal_skew = 'n'
                order = int(error_type[1:])
        
        if error_type[0] in ['b','a']:
            order = int(error_type[1:])
            if error_type[0] == 'b':
                normal_skew = 'n'
            else:
                normal_skew = 's'

        if normal_skew == 'n':
            sign = 1
        else:
            sign = 1 # some bibliografy use -1 for skew but in MAD-X and XSuite the implimentation is with +1, so I keep it like this for the moment, but it can be changed if needed

        for ii, name in enumerate(element_names):
            elem_tt = line[name].get_table()
            if elem_tt.name[0] == 'k0':
                main_k_name = 'k0'
                if elem_tt.rows[main_k_name].value == 'from_h':
                    main_k_value = elem_tt.rows['h'].value[0]
                else:
                    main_k_value = elem_tt.rows[main_k_name].value[0]
                main_order = int(main_k_name[1:])+1
            elif elem_tt.name[0][0] == 'k':
                main_k_name = elem_tt.name[np.argmax(np.abs(elem_tt.value[:2]))]
                main_k_value = elem_tt.rows[main_k_name].value[0]
                main_order = int(main_k_name[1:])+1
            elif 'order' in elem_tt.name:
                knl_max_index = np.where(np.max(np.abs(line[name].knl))==np.abs(line[name].knl))[0][0]
                ksl_max_index = np.where(np.max(np.abs(line[name].ksl))==np.abs(line[name].ksl))[0][0]
                if knl_max_index > ksl_max_index:
                    main_order = ksl_max_index + 1
                    main_k_value = line[name].ksl[main_order-1]
                else:
                    main_order = knl_max_index + 1
                    main_k_value = line[name].knl[main_order-1]

            Brho = (line.particle_ref.p0c/sp_co.c)/line.particle_ref.q0
            
            if error_category == 'relative':
                if B_ref is None or np.abs(main_k_value) == 0:
                    scale_factor = 1
                else:
                    scale_factor = math.factorial(main_order-1)*B_ref/(Brho*reference_radius**(main_order-1))/main_k_value
                error_values[ii] = sign*error_values[ii]*main_k_value*scale_factor
                
                if error_type[0] in ['b','a']:
                    factorial_ratio = math.factorial(order-1)/math.factorial(main_order-1)
                    rref_term = reference_radius**(main_order-order)
                    error_values[ii] = error_values[ii]*factorial_ratio*rref_term
            
            if error_category == 'absolute':
                if error_type[0] in ['b','a']:
                    if B_ref is None:
                        raise ValueError("The B_ref is needed!")
                    else:
                        rref_term = reference_radius**(order-1)
                        error_values[ii] = sign*error_values[ii]*B_ref * math.factorial(order-1)/(rref_term*Brho)
                       
            # Add multipolar components to elements
            if normal_skew == 'n':
                if len(line[name].knl) < order:
                    line.extend_knl_ksl(order=order-1, element_names=[name])
                line[name].knl[order-1] += line.ref[error_switch_name]*error_values[ii]*elem_tt.rows['length'].value[0]
            elif normal_skew == 's':
                if len(line[name].ksl) < order:
                    line.extend_knl_ksl(order=order-1, element_names=[name])
                line[name].ksl[order-1] += line.ref[error_switch_name]*error_values[ii]*elem_tt.rows['length'].value[0]

    return

def magnet_sign_switch (ismag=0, isrot = 0, isb4 = 0, order_N = 0, order_n = 0, isskew_N = 0, isskew_n = 0):
    tot_exp = ismag + isrot + isb4

    # ismag = 1 # 1 if magnetic ref frame else 0
    # isrot = 0 # 1 if beam see coil end first else 0
    # isb4 = 0 # 1 if beam4 else 0
    # order_N = 1 # order of the main multipole (1 for dipole, 2 for quadrupole, etc.)
    # order_n = 2 # order of the error multipole (1 for dipole, 2 for quadrupole, etc.)
    # isskew_N = 0 # 1 if main is skew else 0
    # isskew_n = 1 # 1 if error is skew else 0

    signs = {'sign_n': 1, 'sign_N': 1, 'sign_nN': 1}

    if tot_exp > 0:
        sign_n = (-1)**(tot_exp+order_n+isskew_n)
        sign_N = (-1)**(tot_exp+order_N+isskew_N)
        signs['sign_n'] = sign_n
        signs['sign_N'] = sign_N
        signs['sign_nN'] = sign_n/sign_N

    return signs

def set_integrator (line):

    tt = line.get_table()
    tt_bend = tt.rows[
        (tt.element_type=='Bend') |
        (tt.element_type=='RBend') |
        (tt.element_type=='ThickSliceBend') |
        (tt.element_type=='ThickSliceRBend')]
    tt_wigg = tt.rows['mw.*']
    tt_quad = tt.rows[(tt.element_type=='Quadrupole')|
        (tt.element_type=='ThickSliceQuadrupole')]
    tt_sext = tt.rows[(tt.element_type=='Sextupole')]

    line.set(tt_bend, integrator='uniform', num_multipole_kicks=3, model='mat-kick-mat') #'drift-kick-drift-exact')
    line.set(tt_wigg, integrator='teapot', num_multipole_kicks=11, model='mat-kick-mat')
    line.set(tt_quad, integrator='uniform', num_multipole_kicks=3, model='mat-kick-mat')
    line.set(tt_sext, integrator='yoshida4', num_multipole_kicks=1)

    # line.set(tt_bend, integrator='yoshida4', num_multipole_kicks=1)
    # line.set(tt_quad, integrator='yoshida4', num_multipole_kicks=1)
    # line.set(tt_sext, integrator='yoshida4', num_multipole_kicks=1)

    return


def add_chroma_knobs(line, optics_type=None):

    tt0 = line.get_table(attr=True)

    if optics_type is None:
        if len(tt0.rows['sy.*'].name) == 0:
            optics_type = 'LCC'
        else:
            optics_type = 'GHC'

    line.vars['sf.k2n.chroma.knob'] = 1
    line.vars['sd.k2n.chroma.knob'] = 1
    if optics_type == 'GHC':
        for ii in tt0.rows['sf.*'].name:
            if '_aper' not in ii:
                line.element_refs[ii].k2 = line.vars['sf.k2n.chroma.knob']*line.element_refs[ii].k2._expr

        for ii in tt0.rows['sd.*'].name:
            if '_aper' not in ii:
                line.element_refs[ii].k2 = line.vars['sd.k2n.chroma.knob']*line.element_refs[ii].k2._expr

    elif optics_type == 'LCC':
        for ii in tt0.rows['sf[12]a.*'].name:
            if '_aper' not in ii:
                line.element_refs[ii].k2 = line.vars['sf.k2n.chroma.knob']*line.element_refs[ii].k2._expr

        for ii in tt0.rows['sd[12]a.*'].name:
            if '_aper' not in ii:
                line.element_refs[ii].k2 = line.vars['sd.k2n.chroma.knob']*line.element_refs[ii].k2._expr

    return


def add_chroma_knobs_factor(line, optics_type=None):
    
    return (add_chroma_knobs(line, optics_type=None))


def add_chroma_knobs_step(line, optics_type=None):

    tt0 = line.get_table(attr=True)

    if optics_type is None:
        if len(tt0.rows['sy.*'].name) == 0:
            optics_type = 'LCC'
        else:
            optics_type = 'GHC'

    line.vars['sf.k2n.chroma.knob'] = 0
    line.vars['sd.k2n.chroma.knob'] = 0
    if optics_type == 'GHC':
        for ii in tt0.rows['sf.*'].name:
            if '_aper' not in ii:
                line.element_refs[ii].k2 = line.vars['sf.k2n.chroma.knob']+line.element_refs[ii].k2._expr

        for ii in tt0.rows['sd.*'].name:
            if '_aper' not in ii:
                line.element_refs[ii].k2 = line.vars['sd.k2n.chroma.knob']+line.element_refs[ii].k2._expr

    elif optics_type == 'LCC':
        for ii in tt0.rows['sf[12]a.*'].name:
            if '_aper' not in ii:
                line.element_refs[ii].k2 = line.vars['sf.k2n.chroma.knob']+line.element_refs[ii].k2._expr

        for ii in tt0.rows['sd[12]a.*'].name:
            if '_aper' not in ii:
                line.element_refs[ii].k2 = line.vars['sd.k2n.chroma.knob']+line.element_refs[ii].k2._expr

    return


def add_chroma_knobs_step_v(line, optics_type=None):

    tt0 = line.get_table(attr=True)

    if optics_type is None:
        if len(tt0.rows['sy.*'].name) == 0:
            optics_type = 'LCC'
        else:
            optics_type = 'GHC'

    line.vars['sf.k2n.chroma.knob'] = 0
    line.vars['sd.k2n.chroma.knob'] = 0
    if optics_type == 'GHC':
        for ii in tt0.rows['sf.*'].name:
            if '_aper' not in ii:
                line.element_refs[ii].k2 = line.vars['sf.k2n.chroma.knob']+line[ii].k2

        for ii in tt0.rows['sd.*'].name:
            if '_aper' not in ii:
                line.element_refs[ii].k2 = line.vars['sd.k2n.chroma.knob']+line[ii].k2

    elif optics_type == 'LCC':
        for ii in tt0.rows['sf[12]a.*'].name:
            if '_aper' not in ii:
                line.element_refs[ii].k2 = line.vars['sf.k2n.chroma.knob']+line[ii].k2

        for ii in tt0.rows['sd[12]a.*'].name:
            if '_aper' not in ii:
                line.element_refs[ii].k2 = line.vars['sd.k2n.chroma.knob']+line[ii].k2

    return


def install_phase_trombone(line, locations, delta_phases_x=None, delta_phases_y=None):
    if delta_phases_x is None:
        delta_phases_x = np.zeros(len(locations))
    if delta_phases_y is None:
        delta_phases_y = np.zeros(len(locations))
        
    ## Initialize phase advance trombone / functions
    phase_trombone = xt.FirstOrderTaylorMap(length=0) #radiation_flag=0,
    def optics_gamma(beta,alfa):
        return ((1+alfa**2)/beta)
    def R11(phi,alfa):
        return (np.cos(phi)+alfa*np.sin(phi))
    def R12(phi,beta):
        return (beta*np.sin(phi))
    def R21(phi,gamma):
        return (-gamma*np.sin(phi))
    def R22(phi,alfa):
        return (np.cos(phi)-alfa*np.sin(phi))

    ## Install phase trombone
    line.discard_tracker()
    
    ph_tr_names = []
    for location in locations:
        ph_tr_name = 'phase_tromb_'+location
        ph_tr_names.append(ph_tr_name)
        line.insert_element(ph_tr_name, phase_trombone, at=location)

    line.build_tracker()

    # Compute lattice functions
    tw_tromb = line.twiss(method='4d') #eneloss_and_damping=True)

    for ph_tr_name, delta_phase_x, delta_phase_y in zip(ph_tr_names, delta_phases_x, delta_phases_y):

        gamma_x = optics_gamma(tw_tromb.rows[ph_tr_name].betx[0],tw_tromb.rows[ph_tr_name].alfx[0])
        line[ph_tr_name].m1[0,0] = R11(delta_phase_x,tw_tromb.rows[ph_tr_name].alfx[0])
        line[ph_tr_name].m1[0,1] = R12(delta_phase_x,tw_tromb.rows[ph_tr_name].betx[0])
        line[ph_tr_name].m1[1,0] = R21(delta_phase_x,gamma_x)
        line[ph_tr_name].m1[1,1] = R22(delta_phase_x,tw_tromb.rows[ph_tr_name].alfx[0])

        gamma_y = optics_gamma(tw_tromb.rows[ph_tr_name].bety[0],tw_tromb.rows[ph_tr_name].alfy[0])
        line[ph_tr_name].m1[2,2] = R11(delta_phase_y,tw_tromb.rows[ph_tr_name].alfy[0])
        line[ph_tr_name].m1[2,3] = R12(delta_phase_y,tw_tromb.rows[ph_tr_name].bety[0])
        line[ph_tr_name].m1[3,2] = R21(delta_phase_y,gamma_y)
        line[ph_tr_name].m1[3,3] = R22(delta_phase_y,tw_tromb.rows[ph_tr_name].alfy[0])

    return


# def sextupoles_strength_edit (line, family_name=all, error_strength=1, optics_type=None):

#     tt = line.get_table(attr=True)

#     if optics_type is None:
#         if len(tt.rows['sy.*'].name) == 0:
#             optics_type = 'LCC'
#         else:
#             optics_type = 'GHC'

#     line.vars['k2n.weight'] = error_strength
#     tt_sext = tt.rows[tt.element_type=='Sextupole']


#     if family_name == 'all':
#         for ii in tt_sext.name:
#             line.element_refs[ii].k2 = line.vars['k2n.weight']*line.element_refs[ii].k2._expr
    
#     elif family_name == 'ir':
#         if optics_type == 'LCC':
#             for ii in tt_sext.rows['scrab.*|sdm.*|sdy.*|sfm.*|sfx.*'].name:
#                 line.element_refs[ii].k2 = line.vars['k2n.weight']*line.element_refs[ii].k2._expr
#         elif optics_type == 'GHC':
#             for ii in tt_sext.rows['sy.*'].name:
#                 line.element_refs[ii].k2 = line.vars['k2n.weight']*line.element_refs[ii].k2._expr
    
#     elif family_name == 'arc':
#         if optics_type == 'LCC':
#             for ii in tt_sext.rows['sf[12]a.*|sd[12]a.*'].name:
#                 line.element_refs[ii].k2 = line.vars['k2n.weight']*line.element_refs[ii].k2._expr
#         elif optics_type == 'GHC':
#             for ii in tt_sext.rows['sf.*|sd.*'].name:
#                 line.element_refs[ii].k2 = line.vars['k2n.weight']*line.element_refs[ii].k2._expr

#     return


def sextupoles_strength_edit (line, family_name=all, error_strength=1, optics_type=None):

    tt = line.get_table(attr=True)

    if optics_type is None:
        if len(tt.rows['sy.*'].name) == 0:
            optics_type = 'LCC'
        else:
            optics_type = 'GHC'
    if family_name == 'all':
        line.vars['k2n.weight_ir'] = error_strength
        line.vars['k2n.weight_arc'] = error_strength

    elif family_name == 'ir':
        line.vars['k2n.weight_ir'] = error_strength

    elif family_name == 'arc':
        line.vars['k2n.weight_arc'] = error_strength


    line.vars['k2n.weight'] = error_strength
    tt_sext = tt.rows[tt.element_type=='Sextupole']


    if family_name == 'all':
        for ii in tt_sext.rows['scrab.*|sdm.*|sdy.*|sfm.*|sfx.*'].name:
            line.element_refs[ii].k2 = line.vars['k2n.weight_ir']*line.element_refs[ii].k2._expr
        for ii in tt_sext.rows['sf[12]a.*|sd[12]a.*'].name:
            line.element_refs[ii].k2 = line.vars['k2n.weight_arc']*line.element_refs[ii].k2._expr
    elif family_name == 'ir':
        if optics_type == 'LCC':
            for ii in tt_sext.rows['scrab.*|sdm.*|sdy.*|sfm.*|sfx.*'].name:
                line.element_refs[ii].k2 = line.vars['k2n.weight_ir']*line.element_refs[ii].k2._expr
        elif optics_type == 'GHC':
            for ii in tt_sext.rows['sy.*'].name:
                line.element_refs[ii].k2 = line.vars['k2n.weight_ir']*line.element_refs[ii].k2._expr
    
    elif family_name == 'arc':
        if optics_type == 'LCC':
            for ii in tt_sext.rows['sf[12]a.*|sd[12]a.*'].name:
                line.element_refs[ii].k2 = line.vars['k2n.weight_arc']*line.element_refs[ii].k2._expr
        elif optics_type == 'GHC':
            for ii in tt_sext.rows['sf.*|sd.*'].name:
                line.element_refs[ii].k2 = line.vars['k2n.weight_arc']*line.element_refs[ii].k2._expr

    return


def responce_matrix(line,dk, observables, obs_points, corr_elements, reference_twiss, bipolar=True):
    response = {oo: np.zeros((len(obs_points), len(corr_elements)))
                for oo in observables}


    for ii, cc in enumerate(corr_elements):
        nn = cc
        print(f'Processing {ii}/{len(corr_elements)}')
        line.vars[nn]+= dk
        twp = line.twiss4d()

        line.vars[nn] -= dk

#
        for observable in observables:
            response[observable][:, ii] = (
                twp.rows[obs_points][observable] - reference_twiss.rows[obs_points][observable])  / dk
    response_array1 = np.vstack([response[obs] for obs in observables])



    response2 = {oo: np.zeros((len(obs_points), len(corr_elements)))
                for oo in observables}

    dk =-1e-4
    for ii, cc in enumerate(corr_elements):
        nn = cc
        print(f'Processing {ii}/{len(corr_elements)}')
        line.vars[nn]+= dk
        twp = line.twiss4d()

        line.vars[nn] -= dk


        for observable in observables:
            response2[observable][:, ii] = (
                twp.rows[obs_points][observable] - reference_twiss.rows[obs_points][observable])  / dk
    response_array2 = np.vstack([response2[obs] for obs in observables])

    if bipolar == True:
        response_array=(response_array2+response_array1)/2
    if bipolar== False:
        response_array=response_array1

    return response_array


def apply_optics_correction(line, responce_matrix, reference_twiss, observables, observation_points, correctors, correction_weight=1):
    tol=1e-7  
    twiss=line.twiss4d()
  
    U, S, Vt = np.linalg.svd(responce_matrix, full_matrices=False)
    S_inv = np.diag([1/s if s > tol* np.max(S) else 0 for s in S])
    p_inverse=Vt.T @ S_inv @ U.T

    ideal_values = {o: [reference_twiss.rows[bpm][o] for bpm in observation_points] for o in observables}
    measured_values   = {o: [twiss.rows[bpm][o] for bpm in observation_points] for o in observables}

    #corrections?
    delta_y = {o: np.array(ideal_values[o]) - np.array(measured_values[o]) for o in observables}
    delta_y_vec = np.concatenate([delta_y[o] for o in observables]) 

    delta_p = p_inverse @ delta_y_vec

    knobs = correctors             
    dp = delta_p.flatten()             

    # if fails
    assert len(knobs) == len(dp), f"Length mismatch: {len(knobs)} knobs, {len(dp)} deltas"


    for name, magnet_shift in zip(knobs, dp):
        line.vars[name] += correction_weight*magnet_shift

    tw_corr = line.twiss4d()

    return tw_corr


def sextupoles_phase_error (line, family_name=all, error_class='random', seed=0, error_strength=1, optics_type=None):

    tt = line.get_table(attr=True)

    if optics_type is None:
        if len(tt.rows['sy.*'].name) == 0:
            optics_type = 'LCC'
        else:
            optics_type = 'GHC'

    if family_name == 'ir':
        if optics_type == 'GHC':
            tt_sext = tt.rows['sy1l.[1357]|sy2r.[1357]']
        elif optics_type == 'LCC':
            tt_sext = tt.rows['ipimag[24].*']

    size = len(tt_sext.name)
    if error_class == 'systimatic':
        error_values_x = np.ones(size)*error_strength
        error_values_y = np.ones(size)*error_strength
    elif error_class == 'random':
        np.random.seed(seed)
        std_dev = 1
        mean = 0
        lower_bound = -3 # in sigma
        upper_bound = 3 # in sigma
        error_values_x = error_strength*truncnorm.rvs(lower_bound, upper_bound, loc=mean, scale=std_dev, size=size)
        error_values_y = error_strength*truncnorm.rvs(lower_bound, upper_bound, loc=mean, scale=std_dev, size=size)


    ## Initialize phase advance trombone / functions
    phase_trombone = xt.FirstOrderTaylorMap(length=0) #radiation_flag=0,
    def optics_gamma(beta,alfa):
        return ((1+alfa**2)/beta)
    def R11(phi,alfa):
        return (np.cos(phi)+alfa*np.sin(phi))
    def R12(phi,beta):
        return (beta*np.sin(phi))
    def R21(phi,gamma):
        return (-gamma*np.sin(phi))
    def R22(phi,alfa):
        return (np.cos(phi)-alfa*np.sin(phi))

    ## Install phase trombone
    line.discard_tracker()
    
    for ii in tt_sext.name:
        line.insert_element('phase_tromb_'+ii, phase_trombone, at=ii)

    line.build_tracker()


    # Compute lattice functions
    tt_tromb = line.get_table(attr=True)
    tw_tromb = line.twiss(method='4d') #eneloss_and_damping=True)

    line.vars['phase_tromb.knob'] = 0
    for ii, name in enumerate(tt_tromb.rows['phase_tromb.*'].name):

        gamma_x = optics_gamma(tw_tromb.rows[name].betx[0],tw_tromb.rows[name].alfx[0])
        line[name].m1[0,0] = (1 - line.vars['phase_tromb.knob']) + line.vars['phase_tromb.knob'] * R11(error_values_x[ii],tw_tromb.rows[name].alfx[0])
        line[name].m1[0,1] = line.vars['phase_tromb.knob']*R12(error_values_x[ii],tw_tromb.rows[name].betx[0])
        line[name].m1[1,0] = line.vars['phase_tromb.knob']*R21(error_values_x[ii],gamma_x)
        line[name].m1[1,1] = (1 - line.vars['phase_tromb.knob']) + line.vars['phase_tromb.knob'] * R22(error_values_x[ii],tw_tromb.rows[name].alfx[0])

        gamma_y = optics_gamma(tw_tromb.rows[name].bety[0],tw_tromb.rows[name].alfy[0])
        line[name].m1[2,2] = (1 - line.vars['phase_tromb.knob']) + line.vars['phase_tromb.knob'] * R11(error_values_y[ii],tw_tromb.rows[name].alfy[0])
        line[name].m1[2,3] = line.vars['phase_tromb.knob']*R12(error_values_y[ii],tw_tromb.rows[name].bety[0])
        line[name].m1[3,2] = line.vars['phase_tromb.knob']*R21(error_values_y[ii],gamma_y)
        line[name].m1[3,3] = (1 - line.vars['phase_tromb.knob']) + line.vars['phase_tromb.knob'] * R22(error_values_y[ii],tw_tromb.rows[name].alfy[0])

    return

def get_ip_list(line):
    if 'ipa.1' in line.element_names:
        ip_list = ['ipa.1','ipd.1','ipg.1','ipj.1']
    elif 'ip.0' in line.element_names:
        ip_list = ['ip.0','ip.2','ip.4','ip.6']
    elif 'ip:0' in line.element_names:
        ip_list = ['ip:0','ip:2','ip:4','ip:6']
    else:
        print(f'No IPs found in the line')
    return ip_list

def set_beam_beam_scale(line,optics_type,value, ip_list=None):
    if ip_list is None:
        ip_list = get_ip_list(line)
    for ii in ip_list:
        line[f'beambeam_{ii}'].scale_strength = value

def set_lumi_flag(line,optics_type,value, ip_list=None):
    if ip_list is None:
        ip_list = get_ip_list(line)
    for ii in ip_list:
        line[f'beambeam_{ii}'].flag_luminosity = value

def make_thin_and_rematch (line_to_thin, target_twiss, machine='FCCee'):

    Strategy = xt.slicing.Strategy
    Teapot = xt.slicing.Teapot
    if machine == 'FCCee':
        slicing_strategies = [
            Strategy(slicing=Teapot(1)),  # Default catch-all as in MAD-X
            Strategy(slicing=Teapot(3), element_type=xt.Bend),
            Strategy(slicing=Teapot(3), element_type=xt.CombinedFunctionMagnet),
            #Strategy(slicing=Teapot(50), element_type=xt.Quadrupole), # Starting point
            Strategy(slicing=Teapot(30), name=r'^mwi\..*'),
            Strategy(slicing=Teapot(10), name=r'^qf.*'),
            Strategy(slicing=Teapot(10), name=r'^qd.*'),
            Strategy(slicing=Teapot(5), name=r'^qfg.*'),
            Strategy(slicing=Teapot(5), name=r'^qdg.*'),
            Strategy(slicing=Teapot(5), name=r'^ql.*'),
            Strategy(slicing=Teapot(5), name=r'^qs.*'),
            Strategy(slicing=Teapot(10), name=r'^qb.*'),
            Strategy(slicing=Teapot(10), name=r'^qg.*'),
            Strategy(slicing=Teapot(10), name=r'^qh.*'),
            Strategy(slicing=Teapot(10), name=r'^qi.*'),
            Strategy(slicing=Teapot(10), name=r'^qr.*'),
            Strategy(slicing=Teapot(10), name=r'^qu.*'),
            Strategy(slicing=Teapot(10), name=r'^qx.*'),
            Strategy(slicing=Teapot(10), name=r'^qy.*'),
            Strategy(slicing=Teapot(50), name=r'^qa.*'),
            Strategy(slicing=Teapot(50), name=r'^qc.*'),
            Strategy(slicing=Teapot(130), name=r'^qf1b.*'),
            Strategy(slicing=Teapot(130), name=r'^qf1a.*'),
            Strategy(slicing=Teapot(130), name=r'^qd0b.*'),
            Strategy(slicing=Teapot(130), name=r'^qd0a.*'),
            Strategy(slicing=Teapot(20), name=r'^sy.*'),
            Strategy(slicing=Teapot(20), name=r'^sc.*'),
            Strategy(slicing=Teapot(10), name=r'^sf.*'),
            Strategy(slicing=Teapot(10), name=r'^sd.*')
        ]
    elif machine == 'LHC':
        slicing_strategies = [
            Strategy(slicing=Teapot(1)),  # Default catch-all as in MAD-X
            Strategy(slicing=Teapot(2), element_type=xt.Bend),
            Strategy(slicing=None, element_type=xt.Solenoid),
            Strategy(slicing=Teapot(2), element_type=xt.Quadrupole),
            Strategy(slicing=Teapot(2), element_type=xt.Sextupole),
            Strategy(slicing=Teapot(2), element_type=xt.Octupole),
            Strategy(slicing=Teapot(4), name=r'^mbx.*'),
            Strategy(slicing=Teapot(4), name=r'^mbrb.*'),
            Strategy(slicing=Teapot(4), name=r'^mbrc.*'),
            Strategy(slicing=Teapot(4), name=r'^mbrs.*'),
            Strategy(slicing=Teapot(4), name=r'^mbh.*'),
            Strategy(slicing=Teapot(2), name=r'^mq.*'),
            Strategy(slicing=Teapot(16), name=r'^mqxa.*'),
            Strategy(slicing=Teapot(16), name=r'^mqxb.*'),
            Strategy(slicing=Teapot(4), name=r'^mqwa.*'),
            Strategy(slicing=Teapot(4), name=r'^mqwb.*'),
            Strategy(slicing=Teapot(4), name=r'^mqy.*'),
            Strategy(slicing=Teapot(4), name=r'^mqm.*'),
            Strategy(slicing=Teapot(4), name=r'^mqmc.*'),
            Strategy(slicing=Teapot(4), name=r'^mqml.*'),
            Strategy(slicing=Teapot(2), name=r'^mqtlh.*'),
            Strategy(slicing=Teapot(2), name=r'^mqtli.*'),
            Strategy(slicing=Teapot(2), name=r'^mqt.*')
        ]
    else:
        raise Exception('The slicing of asked machine is not suported yet!')

    line_to_thin.slice_thick_elements(slicing_strategies=slicing_strategies)


    ## Transfer lattice on context and compile tracking code
    line_to_thin.build_tracker()


    ## Compute thin lattice functions withoud rf and radiation before rematching
    tw_thin_4d_no_rad_before = line_to_thin.twiss(method='4d') 
                            #line_to_thin.twiss(start=line_to_thin.element_names[0], end=line_to_thin.element_names[-1],
                            #method='4d', init=target_twiss.get_twiss_init(0))

    # Compare tunes
    print('Before rematching:')

    print('Tunes thick model:')
    print(target_twiss.qx, target_twiss.qy)
    print('Tunes thin model:')
    print(tw_thin_4d_no_rad_before.mux[-1], tw_thin_4d_no_rad_before.muy[-1])

    print('Beta beating at ips:')
    print('H:', np.max(np.abs(
        tw_thin_4d_no_rad_before.rows['ip.*'].betx / target_twiss.rows['ip.*'].betx -1)))
    print('V:', np.max(np.abs(
        tw_thin_4d_no_rad_before.rows['ip.*'].bety / target_twiss.rows['ip.*'].bety -1)))

    #print('Number of elements: ', len(line))
    print('\n')

    print('tune match names',line_to_thin.vars.get_table().name)

    if 'k1qf2' in line_to_thin.vars.get_table().name:
        opt_tune = line_to_thin.match(
            method='4d',
            #start=line_to_thin.element_names[0], end=line_to_thin.element_names[-1],
            #init=target_twiss.get_twiss_init(0),
            vary=[xt.VaryList(['k1qf4', 'k1qf2', 'k1qd3', 'k1qd1',], step=1e-8, tag='quad'),
                ],
            targets=[
                    #xt.TargetSet(at=xt.END, mux=target_twiss.qx, muy=target_twiss.qy, tol=1e-5),
                    xt.TargetSet(qx=target_twiss.qx, qy=target_twiss.qy, tol=1e-6, tag='tune'),
                    #xt.TargetSet(betx=target_twiss['betx','ip.8'], bety=target_twiss['bety','ip.8'], at='ip.8', tol=1e-5),
                    ])
    
    elif 'kqf2' in line_to_thin.vars.get_table().name:
        opt_tune = line_to_thin.match(
            method='4d',
            #start=line_to_thin.element_names[0], end=line_to_thin.element_names[-1],
            #init=target_twiss.get_twiss_init(0),
            vary=[xt.VaryList(['kqf2', 'kqf4', 'kqf6', 'kqd1', 'kqd3', 'kqd5',], step=1e-8, tag='quad'),
                ],
            targets=[
                    #xt.TargetSet(at=xt.END, mux=target_twiss.qx, muy=target_twiss.qy, tol=1e-5),
                    xt.TargetSet(qx=target_twiss.qx, qy=target_twiss.qy, tol=1e-6, tag='tune'),
                    #xt.TargetSet(betx=target_twiss['betx','ip.8'], bety=target_twiss['bety','ip.8'], at='ip.8', tol=1e-5),
                    ])
    

    opt_chroma = line_to_thin.match(
        method='4d',
        vary=[xt.VaryList(['sf.k2n.chroma.knob', 'sd.k2n.chroma.knob',], step=1e-3, tag='sext'),
            ],
        targets=[xt.TargetSet(dqx=target_twiss.dqx, dqy=target_twiss.dqy, tol=1e-2, tag='chrom'),
                ])

    # Inspect optimization outcome
    opt_tune.target_status()
    opt_tune.vary_status()
    opt_chroma.target_status()
    opt_chroma.vary_status()

    tw_thin_4d_no_rad = line_to_thin.twiss(method='4d')

    print('After rematching:')
    print('Tunes thick model:')
    print(target_twiss.qx, target_twiss.qy)
    print('Tunes thin model:')
    print(tw_thin_4d_no_rad.qx, tw_thin_4d_no_rad.qy)

    print('Beta beating at ips:')
    print('H:', np.max(np.abs(
        tw_thin_4d_no_rad.rows['ip.*'].betx / target_twiss.rows['ip.*'].betx -1)))
    print('V:', np.max(np.abs(
        tw_thin_4d_no_rad.rows['ip.*'].bety / target_twiss.rows['ip.*'].bety -1)))

    #print('Number of elements: ', len(line))

    print('Change on arc quadrupoles:')
    print(opt_tune.log().vary[-1]/opt_tune.log().vary[0] - 1)

    print('\n Beta at the IPs:')
    tw_thin_4d_no_rad.rows['ip.*'].cols['betx bety'].show()

    return 

def make_thin (line_to_thin, machine):

    Strategy = xt.slicing.Strategy
    Teapot = xt.slicing.Teapot
    if machine == 'FCCee':
        slicing_strategies = [
            Strategy(slicing=Teapot(1)),  # Default catch-all as in MAD-X
            Strategy(slicing=Teapot(3), element_type=xt.Bend),
            Strategy(slicing=Teapot(3), element_type=xt.CombinedFunctionMagnet),
            #Strategy(slicing=Teapot(50), element_type=xt.Quadrupole), # Starting point
            Strategy(slicing=Teapot(10), name=r'^qf.*'),
            Strategy(slicing=Teapot(10), name=r'^qd.*'),
            Strategy(slicing=Teapot(5), name=r'^qfg.*'),
            Strategy(slicing=Teapot(5), name=r'^qdg.*'),
            Strategy(slicing=Teapot(5), name=r'^ql.*'),
            Strategy(slicing=Teapot(5), name=r'^qs.*'),
            Strategy(slicing=Teapot(10), name=r'^qb.*'),
            Strategy(slicing=Teapot(10), name=r'^qg.*'),
            Strategy(slicing=Teapot(10), name=r'^qh.*'),
            Strategy(slicing=Teapot(10), name=r'^qi.*'),
            Strategy(slicing=Teapot(10), name=r'^qr.*'),
            Strategy(slicing=Teapot(10), name=r'^qu.*'),
            Strategy(slicing=Teapot(10), name=r'^qx.*'),
            Strategy(slicing=Teapot(10), name=r'^qy.*'),
            Strategy(slicing=Teapot(50), name=r'^qa.*'),
            Strategy(slicing=Teapot(50), name=r'^qc.*'),
            Strategy(slicing=Teapot(130), name=r'^qf1b.*'),
            Strategy(slicing=Teapot(130), name=r'^qf1a.*'),
            Strategy(slicing=Teapot(130), name=r'^qd0b.*'),
            Strategy(slicing=Teapot(130), name=r'^qd0a.*'),
            Strategy(slicing=Teapot(20), name=r'^sy.*'),
            Strategy(slicing=Teapot(20), name=r'^sc.*'),
            Strategy(slicing=Teapot(10), name=r'^sf.*'),
            Strategy(slicing=Teapot(10), name=r'^sd.*')
        ]
    elif machine == 'LHC':
        slicing_strategies = [
            Strategy(slicing=Teapot(1)),  # Default catch-all as in MAD-X
            Strategy(slicing=Teapot(2), element_type=xt.Bend),
            Strategy(slicing=None, element_type=xt.Solenoid),
            Strategy(slicing=Teapot(2), element_type=xt.Quadrupole),
            Strategy(slicing=Teapot(2), element_type=xt.Sextupole),
            Strategy(slicing=Teapot(2), element_type=xt.Octupole),
            Strategy(slicing=Teapot(4), name=r'^mbx.*'),
            Strategy(slicing=Teapot(4), name=r'^mbrb.*'),
            Strategy(slicing=Teapot(4), name=r'^mbrc.*'),
            Strategy(slicing=Teapot(4), name=r'^mbrs.*'),
            Strategy(slicing=Teapot(4), name=r'^mbh.*'),
            Strategy(slicing=Teapot(2), name=r'^mq.*'),
            Strategy(slicing=Teapot(16), name=r'^mqxa.*'),
            Strategy(slicing=Teapot(16), name=r'^mqxb.*'),
            Strategy(slicing=Teapot(4), name=r'^mqwa.*'),
            Strategy(slicing=Teapot(4), name=r'^mqwb.*'),
            Strategy(slicing=Teapot(4), name=r'^mqy.*'),
            Strategy(slicing=Teapot(4), name=r'^mqm.*'),
            Strategy(slicing=Teapot(4), name=r'^mqmc.*'),
            Strategy(slicing=Teapot(4), name=r'^mqml.*'),
            Strategy(slicing=Teapot(2), name=r'^mqtlh.*'),
            Strategy(slicing=Teapot(2), name=r'^mqtli.*'),
            Strategy(slicing=Teapot(2), name=r'^mqt.*')
        ]
    else:
        raise Exception('The slicing of asked machine is not suported yet!')

    line_to_thin.slice_thick_elements(slicing_strategies=slicing_strategies)

    return 


def get_ip_optics (twiss):

    if len(twiss.rows['ipa.1'].name) != 0:
        optics_type = 'GHC'
    elif len(twiss.rows['ip[.:]0'].name) != 0:
        optics_type = 'LCC'

    if optics_type == 'GHC':
        ip_optics = twiss.rows['ip[adgj].1'].cols['betx bety alfx alfy dx dpx']
    elif optics_type == 'LCC':
        ip_optics = twiss.rows['ip[.:][0246]'].cols['betx bety alfx alfy dx dpx']

    ip_optics_dict = {}
    for ii in ip_optics.name:
        ip_optics_dict[ii] = {
            'betx': ip_optics.rows[ii].betx[0],
            'bety': ip_optics.rows[ii].bety[0],
            'alfx': ip_optics.rows[ii].alfx[0],
            'alfy': ip_optics.rows[ii].alfy[0],
            'dx': ip_optics.rows[ii].dx[0],
            'dpx': ip_optics.rows[ii].dpx[0],
        }
    
    return ip_optics_dict


def match_tune_chroma (line, target_twiss, match_quantities='tune_chroma', method='6d', machine='FCCee'):

    ip_marker = 'ipa.1' if 'ipa.1' in line.element_names else 'ip.0'
    if ip_marker == 'ip.0':
        ip_marker = 'ip.0' if 'ip.0' in line.element_names else 'ip:0'
        
    if isinstance(target_twiss, dict):
        if 'tune' in match_quantities:
            target_qx = target_twiss['qx']
            target_qy = target_twiss['qy']
            if 'name' in target_twiss.keys():
                ip_index = np.where(target_twiss['name']==ip_marker)[0]
                target_betx_ip = target_twiss['betx'][ip_index]
                target_bety_ip = target_twiss['bety'][ip_index]
                target_alfx_ip = target_twiss['alfx'][ip_index]
                target_alfy_ip = target_twiss['alfy'][ip_index]
                target_dx_ip = target_twiss['dx'][ip_index]
                target_dpx_ip = target_twiss['dpx'][ip_index]
            else:
                target_betx_ip = target_twiss[ip_marker]['betx']
                target_bety_ip = target_twiss[ip_marker]['bety']
                target_alfx_ip = target_twiss[ip_marker]['alfx']
                target_alfy_ip = target_twiss[ip_marker]['alfy']
                target_dx_ip = target_twiss[ip_marker]['dx']
                target_dpx_ip = target_twiss[ip_marker]['dpx']

        if 'chroma' in match_quantities:
            target_dqx = target_twiss['dqx']
            target_dqy = target_twiss['dqy']
    else:
        if 'tune' in match_quantities:
            target_qx = target_twiss.qx
            target_qy = target_twiss.qy
            target_betx_ip = target_twiss.rows[ip_marker].betx[0]
            target_bety_ip = target_twiss.rows[ip_marker].bety[0]
            target_alfx_ip = target_twiss.rows[ip_marker].alfx[0]
            target_alfy_ip = target_twiss.rows[ip_marker].alfy[0]
            target_dx_ip = target_twiss.rows[ip_marker].dx[0]
            target_dpx_ip = target_twiss.rows[ip_marker].dpx[0]
        if 'chroma' in match_quantities:
            target_dqx = target_twiss.dqx
            target_dqy = target_twiss.dqy

    if machine == 'FCCee':
        if 'tune' in match_quantities:
            if 'k1qf2' in line.vars.get_table().name:
                opt_tune = line.match(
                    method=method,
                    vary=[xt.VaryList(['k1qrfr1', 'k1qrdr1', 'k1qrfr2', 'k1qrdr2',
                                       'k1qrfr3', 'k1qrdr3', 'k1qrfr4', 'k1qrdr4',
                                       'k1qrfr5', 'k1qrdr5', 'k1qrfr6', 'k1qrdr6',

                                       'k1qi1', 'k1qi2', 'k1qi3',
                                       'k1qi4', 'k1qi5', 'k1qi6', 
                                       'k1qi7', 'k1qi8', 'k1qi9', 
                                        ], step=1e-8, tag='quad'),
                        ],
                    targets=[xt.TargetSet(qx=target_qx, qy=target_qy, tol=1e-6, tag='tune'),
                             xt.TargetSet(betx=target_betx_ip, alfx=target_alfx_ip, 
                                          bety=target_bety_ip, alfy=target_alfy_ip, 
                                          dx=target_dx_ip, dpx=target_dpx_ip,
                                          at='ipa.1', tol=1e-6, tag='bet_alf'),
                        ])
            
            elif 'kqf6' in line.vars.get_table().name:
                opt_tune = line.match(
                    method=method,
                    vary=[xt.VaryList(['kqf2', 'kqf4', 'kqf6', 'kqd1', 'kqd3', 'kqd5',], step=1e-8, tag='quad'),
                        ],
                    targets=[xt.TargetSet(qx=target_qx, qy=target_qy, tol=1e-6, tag='tune'),
                        ])
            
            elif 'kqf2' in line.vars.get_table().name:
                opt_tune = line.match(
                    method=method,
                    vary=[xt.VaryList(['kqf2', 'kqd1', ], step=1e-8, tag='quad'),
                        ],
                    targets=[xt.TargetSet(qx=target_qx, qy=target_qy, tol=1e-6, tag='tune'),
                        ])
            
            opt_tune.target_status()
            opt_tune.vary_status()

        if 'chroma' in match_quantities:
            chroma_knob_refs = line.vars['sf.k2n.chroma.knob']._find_dependant_targets()[1:]
            if len(chroma_knob_refs) == 0:
                print('WARNING: Chroma knobs are not defined, could not match chroma')
            else:
                opt_chroma = line.match(
                    method=method,
                    vary=[xt.VaryList(['sf.k2n.chroma.knob', 'sd.k2n.chroma.knob',], step=1e-3, tag='sext'),
                        ],
                    targets=[xt.TargetSet(dqx=target_dqx, dqy=target_dqy, tol=1e-2, tag='chrom'),
                        ])
                opt_chroma.target_status()
                opt_chroma.vary_status()


    elif machine == 'LHC':
        #if 'dqx.b1' in line.vars.get_table().name:
        if 'tune' in match_quantities:
            opt_tune = line.match(
                method=method,
                vary=[xt.VaryList(['dqx.b1','dqy.b1',], step=1e-8, tag='quad'),
                    ],
                targets=[xt.TargetSet(qx=target_qx, qy=target_qy, tol=1e-5, tag='tune'),
                    ])
            opt_tune.target_status()
            opt_tune.vary_status()
        
        if 'chroma' in match_quantities:
            opt_chroma = line.match(
                method=method,
                vary=[xt.VaryList(['dqpx.b1', 'dqpy.b1',], step=1e-3, tag='sext'),
                    ],
                targets=[xt.TargetSet(dqx=target_dqx, dqy=target_dqy, tol=1e-2, tag='chrom'),
                    ])
            opt_chroma.target_status()
            opt_chroma.vary_status()

    else:
        raise Exception('The matching of asked machine is not suported yet!')

    return


def match_tune_chroma_emit_y (line, target_values):

    ip_marker = 'ipa.1' if 'ipa.1' in line.element_names else 'ip.0'
    if ip_marker == 'ip.0':
        ip_marker = 'ip.0' if 'ip.0' in line.element_names else 'ip:0'

    if isinstance(target_values, dict):
        target_qx = target_values['qx']
        target_qy = target_values['qy']
        target_dqx = target_values['dqx']
        target_dqy = target_values['dqy']
        target_betx_ip = target_values[ip_marker]['betx']
        target_bety_ip = target_values[ip_marker]['bety']
        target_alfx_ip = target_values[ip_marker]['alfx']
        target_alfy_ip = target_values[ip_marker]['alfy']
        target_dx_ip = target_values[ip_marker]['dx']
        target_dpx_ip = target_values[ip_marker]['dpx']
        target_eq_emit_y = target_values['eq_emit_y']
    else:
        target_qx = target_values.qx
        target_qy = target_values.qy
        target_dqx = target_values.dqx
        target_dqy = target_values.dqy
        target_betx_ip = target_values.rows[ip_marker].betx[0]
        target_bety_ip = target_values.rows[ip_marker].bety[0]
        target_alfx_ip = target_values.rows[ip_marker].alfx[0]
        target_alfy_ip = target_values.rows[ip_marker].alfy[0]
        target_dx_ip = target_values[ip_marker].dx[0]
        target_dpx_ip = target_values[ip_marker].dpx[0]
        target_eq_emit_y = target_values.eq_gemitt_y

    if 'k1qf2' in line.vars.get_table().name:
        opt_tune = line.match(
            method='6d',
            vary=[xt.VaryList(['k1qrfr1', 'k1qrdr1', 'k1qrfr2', 'k1qrdr2',
                                'k1qrfr3', 'k1qrdr3', 'k1qrfr4', 'k1qrdr4',
                                'k1qrfr5', 'k1qrdr5', 'k1qrfr6', 'k1qrdr6',

                                'k1qi1', 'k1qi2', 'k1qi3',
                                'k1qi4', 'k1qi5', 'k1qi6', 
                                'k1qi7', 'k1qi8', 'k1qi9',], step=1e-8, tag='quad'),
                  xt.VaryList(['sf.k2n.chroma.knob', 'sd.k2n.chroma.knob',], step=1e-3, tag='sext'),
                  xt.VaryList(['on_wiggler_v'], step=1e-3, tag='wigg_y'),
                ],
            targets=[xt.TargetSet(qx=target_qx, qy=target_qy, tol=1e-6, tag='tune'),
                     xt.TargetSet(betx=target_betx_ip, alfx=target_alfx_ip, 
                                  bety=target_bety_ip, alfy=target_alfy_ip, 
                                  dx=target_dx_ip, dpx=target_dpx_ip,
                                  at='ipa.1', tol=1e-6, tag='bet_alf'),
                     xt.TargetSet(dqx=target_dqx, dqy=target_dqy, tol=1e-2, tag='chrom'),
                     xt.TargetSet(eq_gemitt_y=target_eq_emit_y, tol=1e-7, tag='eq_emit_y'),
                ])
    
    elif 'kqf6' in line.vars.get_table().name:
        opt = line.match(
            method='6d',
            eneloss_and_damping=True,
            compensate_radiation_energy_loss=True,
            vary=[xt.VaryList(['kqf2', 'kqf4', 'kqf6', 'kqd1', 'kqd3', 'kqd5',], step=1e-8, tag='quad'),
                  xt.VaryList(['sf.k2n.chroma.knob', 'sd.k2n.chroma.knob',], step=1e-3, tag='sext'),
                  xt.VaryList(['on_wiggler_v'], step=1e-3, tag='wigg_y'),
                ],
            targets=[xt.TargetSet(qx=target_qx, qy=target_qy, tol=1e-6, tag='tune'),
                     xt.TargetSet(dqx=target_dqx, dqy=target_dqy, tol=1e-2, tag='chrom'),
                     xt.TargetSet(eq_gemitt_y=target_eq_emit_y, tol=1e-7, tag='eq_emit_y'),
                ])
    
    elif 'kqf2' in line.vars.get_table().name:
        opt = line.match(
            method='6d',
            eneloss_and_damping=True,
            compensate_radiation_energy_loss=True,
            vary=[xt.VaryList(['kqf2', 'kqd1', ], step=1e-8, tag='quad'),
                  xt.VaryList(['sf.k2n.chroma.knob', 'sd.k2n.chroma.knob',], step=1e-3, tag='sext'),
                  xt.VaryList(['on_wiggler_v'], step=1e-3, tag='wigg_y'),
                ],
            targets=[xt.TargetSet(qx=target_qx, qy=target_qy, tol=1e-6, tag='tune'),
                     xt.TargetSet(dqx=target_dqx, dqy=target_dqy, tol=1e-2, tag='chrom'),
                     xt.TargetSet(eq_gemitt_y=target_eq_emit_y, tol=1e-7, tag='eq_emit_y'),
                ])
    opt.target_status()
    opt.vary_status()

    return


def match_wiggler_for_eq_emitt (line, target_eq_em_x=None, target_eq_em_y=None):

    if target_eq_em_x is not None and target_eq_em_y is not None:
        line.vars['on_wiggler_h']=0.1
        line.vars['on_wiggler_v']=0.1
        opt_eq_em = line.match(
            method='6d',
            eneloss_and_damping=True,
            vary=[xt.VaryList(['on_wiggler_h','on_wiggler_v'], step=1e-3, tag='wigg'),
                ],
            targets=[xt.TargetSet(eq_gemitt_x=target_eq_em_x, tol=1e-15, tag='eq_emit_x'),
                     xt.TargetSet(eq_gemitt_y=target_eq_em_y, tol=1e-15, tag='eq_emit_y'),
                ])

    elif  target_eq_em_x is not None and target_eq_em_y is None:
        line.vars['on_wiggler_h']=0.1
        opt_eq_em = line.match(
            method='6d',
            eneloss_and_damping=True,
            vary=[xt.VaryList(['on_wiggler_h'], step=1e-3, tag='wigg_x'),
                ],
            targets=[xt.TargetSet(eq_gemitt_x=target_eq_em_x, tol=1e-15, tag='eq_emit_x'),
                ])

    elif  target_eq_em_x is None and target_eq_em_y is not None:
        line.vars['on_wiggler_v']=0.1
        opt_eq_em = line.match(
            method='6d',
            eneloss_and_damping=True,
            vary=[xt.VaryList(['on_wiggler_v'], step=1e-3, tag='wigg_y'),
                ],
            targets=[xt.TargetSet(eq_gemitt_y=target_eq_em_y, tol=1e-15, tag='eq_emit_y'), #tol=10**(np.floor(np.log10(abs(target_eq_em_y)))-2)
                ])

    else:

        raise Exception('The wigglers target values not set correctly!')

    opt_eq_em.target_status()
    opt_eq_em.vary_status()

    return

def initial_conditions_grid (study, energy_spread=None, cartesian_polar=None, min_r_y=None, max_r_y=None, num_r_y_points=None, min_theta_x=None, max_theta_x=None, num_theta_x_points=None, r_range_x=(5,10), r_range_y=(5,10), theta_range_x=(0,2*np.pi), theta_range_y=(0,2*np.pi), delta_initial_values=None, num_particles=None, rnd_seed=105):

    """
    Generate initial conditions grid for a given study.

    Parameters
    ----------
    study : str
        Study type. Can be 'DA', 'MA' or 'halo'.
    energy_spread : float
        Energy spread of the beam.
    cartesian_polar : str
        Type of initial condition for 'DA' and 'MA' studies. Can be 'cartesian' or 'polar'.
    min_r_y : float
        Minimum y for cartesian initial conditions or minimum radius for polar.
    max_r_y : float
        Maximum y for cartesian initial conditions or maximum radius for polar.
    num_r_y_points : int
        Number of points in y or radial plane.
    min_theta_x : float
        Minimum x for cartesian initial conditions or minimum theta for polar.
    max_theta_x : float
        Maximum x for cartesian initial conditions or minimum theta for polar.
    num_theta_x_points : int
        Number of points in x or theta plane.
    r_range_x : tuple
        Range of radial distances in phase space for horizontal coordinates for halo.
    r_range_y : tuple
        Range of radial distances in phase space for vertical coordinates for halo.
    theta_range_x : tuple
        Range of thata angles in phase space for horizontal coordinates for halo.
    theta_range_y : tuple
        Range of thata angles in phase space for vertical coordinates for halo.
    delta_initial_values : array_like
        Initial values of delta.
    num_particles : int
        Number of particles needed only for halo distribution.
    rnd_seed : int
        Random number seed.

    Returns for 'MA' and 'DA'
    -------
    x_normalized : array_like
        Normalized x coordinates of the particles.
    y_normalized : array_like
        Normalized y coordinates of the particles.
    delta_init : array_like
        Initial values of delta.
    num_theta_x_points : int
        Number of points in x plane.
    num_r_y_points : int
        Number of points in y plane.
    num_delta : int
        Number of initial values of delta.
    num_particles : int
        Number of particles.

    Returns for 'halo'
    -------
    x_normalized : array_like
        Normalized x coordinates of the particles.
    y_normalized : array_like
        Normalized y coordinates of the particles.
    px_normalized : array_like
        Normalized px coordinates of the particles.
    py_normalized : array_like
        Normalized py coordinates of the particles.
    """

    np.random.seed(rnd_seed)

    if min_r_y is None:
        if study in ['DA','MA']:
            min_r_y = 0
    
    if max_r_y is None:
        if study=='DA':
            max_r_y = 50
        elif study=='MA':
            max_r_y = 30

    if num_r_y_points is None:
        if study=='DA':
            num_r_y_points = 51
        elif study=='MA':
            num_r_y_points = 31

    if min_theta_x is None:
        if study=='DA':
            min_theta_x = -20
        elif study=='MA':
            min_theta_x = np.pi/4

    if max_theta_x is None:
        if study=='DA':
            max_theta_x= 20
        elif study=='MA':
            max_theta_x= np.pi/4

    if num_theta_x_points is None:
        if study=='DA':
            num_theta_x_points = 41
        elif study=='MA':
            num_theta_x_points = 1

    if delta_initial_values is None:
        if study=='DA':
            delta_initial_values = 0
        elif study=='MA':
            delta_initial_values = np.linspace(-25*energy_spread, 25*energy_spread, 51) 

    
    if study=='DA':
        if cartesian_polar is None or cartesian_polar=='cartesian':
            x_norm_points = np.linspace(min_theta_x, max_theta_x, num_theta_x_points)
            y_norm_points = np.linspace(min_r_y, max_r_y, num_r_y_points)
            x_norm_grid, y_norm_grid = np.meshgrid(x_norm_points, y_norm_points)
            x_normalized = x_norm_grid.flatten()
            y_normalized = y_norm_grid.flatten()

        elif cartesian_polar=='polar':
            x_normalized, y_normalized, r_xy, theta_xy = xp.generate_2D_polar_grid(
                r_range=(min_r_y, max_r_y), # beam sigmas
                theta_range=(min_theta_x, max_theta_x),
                nr=num_r_y_points, ntheta=num_theta_x_points)
            
    if study=='MA':
        if cartesian_polar is None or cartesian_polar=='polar':
            x_normalized, y_normalized, r_xy, theta_xy = xp.generate_2D_polar_grid(
                r_range=(min_r_y, max_r_y), # beam sigmas
                theta_range=(min_theta_x, max_theta_x),
                nr=num_r_y_points, ntheta=num_theta_x_points)
            
        elif cartesian_polar=='cartesian':
            x_norm_points = np.linspace(min_theta_x, max_theta_x, num_theta_x_points)
            y_norm_points = np.linspace(min_r_y, max_r_y, num_r_y_points)
            x_norm_grid, y_norm_grid = np.meshgrid(x_norm_points, y_norm_points)
            x_normalized = x_norm_grid.flatten()
            y_normalized = y_norm_grid.flatten()

    if study in ['DA','MA']:
        num_delta = np.size(delta_initial_values)
        num_particles = num_delta*num_theta_x_points*num_r_y_points
        if num_delta != 1:
            x_normalized = np.tile(x_normalized, num_delta)
            y_normalized = np.tile(y_normalized, num_delta)
            delta_init = np.repeat(delta_initial_values, np.size(x_normalized)/num_delta)
        else:
            delta_init = delta_initial_values

    if study == 'halo':
        (x_normalized, px_normalized, r_points, theta_points)= xp.generate_2D_uniform_circular_sector(
                                            num_particles=num_particles,
                                            r_range=r_range_x, # beam sigmas
                                            theta_range=theta_range_x
                                            )   

        (y_normalized, py_normalized, r_points, theta_points)= xp.generate_2D_uniform_circular_sector(
                                            num_particles=num_particles,
                                            r_range=r_range_y, # beam sigmas
                                            theta_range=theta_range_y
                                            )
        return (x_normalized, y_normalized, px_normalized, py_normalized)
    
    return (x_normalized, y_normalized, delta_init, num_theta_x_points, num_r_y_points, num_delta, num_particles)

def univariate_q_gaussian(size, q, beta=1):
    """
    Generate univariate q-Gaussian samples with an arbitrary β.
    """

    if q == 1:
    # Standard Gaussian with variance 1/(2β)
        return np.random.normal(scale=np.sqrt(1 / (2 * beta)), size=size)

    elif q > 1:
        # # Heavy-tailed case: Variance rescaling
        # nu = 2 / (q - 1) - 1 # Degrees of freedom
        # W = np.random.gamma(shape=nu/2, scale=2/nu, size=size)
        # Z = np.random.normal(size=size)
        # return Z / np.sqrt(W * beta)
    
        # Heavy-tailed case: Variance rescaling
        nu = 2 / (q - 1) - 1 # Degrees of freedom
        Z = np.random.standard_t(df=nu, size=size)
        scale = np.sqrt(nu / ((3 - q) * beta))
        return scale * Z

    elif q < 1:
        # Compact support: Rejection sampling
        
        # samples = []
        # while len(samples) < size:
        #     Z = np.random.uniform(-1, 1, size=1) # Proposal
        #     pdf_ratio = (1 - (1 - q) * beta * Z**2) ** (1 / (1 - q))
        #     if uniform.rvs() < pdf_ratio:
        #         samples.append(Z[0])
        # return np.array(samples) * np.sqrt(1 / beta) # Adjust scaling

        R = np.random.beta(a=0.5, b=1/(1-q) - 0.5, size=size)
        sign = np.random.choice([-1, 1], size=size)
        return sign * np.sqrt(R / ((1 - q) * beta))

    

# def multivariate_q_gaussian(mean, cov, q, beta=1, size=1):
#     """
#     Generate multivariate q-Gaussian samples for any beta value.
#     """
#     mean = np.asarray(mean)
#     d = len(mean) # Dimensionality
#     # Decompose covariance matrix: Σ = L L^T (Cholesky)
#     L = cholesky(cov, lower=True)
#     # Generate independent q-Gaussian samples
#     Z = np.array([univariate_q_gaussian(size, q, beta) for _ in range(d)]).T # Shape (size, d)
#     # Transform into correlated samples
#     samples = Z @ L.T + mean # Apply covariance structure

#     return samples


def multivariate_q_gaussian(mean, cov, q, beta=1, size=1):
    """
    Generate multivariate q-Gaussian samples with mean and covariance.

    Parameters
    ----------
    mean : array_like
        Mean vector (d,)
    cov : array_like
        Covariance matrix (d, d)
    q : float
        q parameter
    beta : float
        Inverse scale parameter
    size : int
        Number of samples

    Returns
    -------
    samples : ndarray
        Samples with shape (size, d)
    """
    mean = np.asarray(mean)
    d = mean.size

    # Cholesky decomposition of covariance
    L = cholesky(cov, lower=True)

    # Gaussian case (exact)
    if q == 1:
        Z = np.random.normal(size=(size, d))
        return mean + Z @ L.T

    # Heavy-tailed case (q > 1)
    if q > 1:
        # Degrees of freedom
        nu = 2 / (q - 1) - d

        if nu <= 0:
            raise ValueError(
                f"q={q} too large for dimension d={d}: nu={nu} <= 0"
            )

        # Shared radial scaling (elliptical symmetry)
        chi2 = np.random.chisquare(nu, size=size)
        S = nu / chi2

        # Gaussian core
        Z = np.random.normal(size=(size, d))

        # Correct beta scaling
        scale = np.sqrt(1 / ((3 - q) * beta))

        return mean + scale * np.sqrt(S)[:, None] * (Z @ L.T)

    # Compact-support case (q < 1)
    else:
        # Maximum radius
        r_max = np.sqrt(1 / ((1 - q) * beta))

        # Generate isotropic directions
        Z = np.random.normal(size=(size, d))
        Z /= np.linalg.norm(Z, axis=1)[:, None]

        # Radial distribution
        a = d / 2
        b = 1 / (1 - q) - d / 2
        R = np.random.beta(a, b, size=size)

        radii = r_max * np.sqrt(R)

        return mean + radii[:, None] * (Z @ L.T)
    

def generate_q_gaussian_halo(
    mean,
    cov,
    q,
    beta,
    n_particles,
    r_min,
    r_max,
    rng=None,
):
    """
    Generate halo particles from a multivariate q-Gaussian distribution,
    constrained to lie between inner and outer radii in each phase-space direction.

    Parameters
    ----------
    mean : array_like, shape (d,)
        Mean (closed orbit)
    cov : array_like, shape (d, d)
        Covariance matrix
    q : float
        q parameter (q=1 Gaussian, q>1 heavy tails)
    beta : float
        Beta parameter of the q-Gaussian
    n_particles : int
        Number of halo particles to generate
    r_min : array_like, shape (d,)
        Minimum radius (in sigma units) per coordinate
    r_max : array_like, shape (d,)
        Maximum radius (in sigma units) per coordinate
    rng : np.random.Generator, optional
        Random number generator

    Returns
    -------
    halo : ndarray, shape (n_particles, d)
        Halo particle coordinates
    """
    if rng is None:
        rng = np.random.default_rng()

    mean = np.asarray(mean)
    r_min = np.asarray(r_min)
    r_max = np.asarray(r_max)

    d = mean.size

    if r_min.shape != (d,) or r_max.shape != (d,):
        raise ValueError("r_min and r_max must have shape (d,)")

    if np.any(r_min < 0) or np.any(r_max <= r_min):
        raise ValueError("Require 0 <= r_min < r_max for all dimensions")

    # Cholesky factor of covariance
    L = cholesky(cov, lower=True)

    halo = []
    batch = max(4 * n_particles, 1000)

    while len(halo) < n_particles:
        # --- Sample from full multivariate q-Gaussian ---
        samples = multivariate_q_gaussian(
            mean=np.zeros(d),
            cov=np.eye(d),
            q=q,
            beta=beta,
            size=batch,
        )

        # --- Apply covariance and mean ---
        samples = samples @ L.T + mean

        # --- Compute normalized amplitudes ---
        # Convert to sigma units
        sigma = np.sqrt(np.diag(cov))
        normed = (samples - mean) / sigma

        # --- Halo condition: outside core, inside max ---
        mask = np.ones(batch, dtype=bool)
        for i in range(d):
            mask &= (np.abs(normed[:, i]) >= r_min[i])
            mask &= (np.abs(normed[:, i]) <= r_max[i])

        accepted = samples[mask]
        halo.extend(accepted.tolist())

    return np.array(halo[:n_particles])


def initial_conditions_distribution (twiss, normalized_emittance_x, normalized_emittance_y, emittance_zeta, number_of_particles=500, new_closed_orbit=None, covariance_dispertion_free=False, qq=1, bb=1, rnd_seed=101):
    """
    Generate initial conditions of particles in phase space.

    Parameters
    ----------
    twiss : xtrack.Twiss
        Twiss data of the machine.
    normalized_emittance_x : float
        Normalized emittance in x.
    normalized_emittance_y : float
        Normalized emittance in y.
    emittance_zeta : float
        Longitudinal emittance.
    number_of_particles : int
        Number of particles.
    qq : float
        q parameter of the q-Gaussian distribution.
    bb : float
        beta parameter of the q-Gaussian distribution.
    rnd_seed : int
        Random seed.

    Returns
    -------
    x, px, y, py, zeta, delta : tuple of arrays
        Initial conditions of particles in phase space.
    """
    np.random.seed(rnd_seed)

    closed_orbit_dic = {'x':twiss.x[0], 
                        'px':twiss.px[0], 
                        'y':twiss.y[0], 
                        'py':twiss.py[0], 
                        'zeta':twiss.zeta[0], 
                        'delta':twiss.delta[0]
                        }
    if new_closed_orbit is not None:
        for ii in new_closed_orbit.keys():
            if new_closed_orbit[ii] is not None:
                closed_orbit_dic[ii] = new_closed_orbit[ii]
    
    closed_orbit = [closed_orbit_dic['x'], 
                    closed_orbit_dic['px'], 
                    closed_orbit_dic['y'], 
                    closed_orbit_dic['py'], 
                    closed_orbit_dic['zeta'], 
                    closed_orbit_dic['delta']
                    ]
    # closed_orbit = [twiss.x[0], twiss.px[0], twiss.y[0], twiss.py[0], twiss.zeta[0], twiss.delta[0]]

    covariance_f=twiss.get_beam_covariance(nemitt_x=normalized_emittance_x, nemitt_y=normalized_emittance_y, gemitt_zeta=emittance_zeta)
    covariance = covariance_f.Sigma[0]
    if covariance_dispertion_free:

        disp_vector = np.array([twiss.dx[0], twiss.dpx[0], twiss.dy[0], twiss.dpy[0], 0.0, 1.0])
        sigma_delta2 = covariance[-1,-1]
        covariance = covariance - sigma_delta2 * np.outer(disp_vector, disp_vector)
        covariance[-1,-1] = sigma_delta2  

    if qq == 1:
        rng = np.random.default_rng()
        part_inj = rng.multivariate_normal(mean=closed_orbit, cov=covariance, size=number_of_particles)
    else:
        part_inj = multivariate_q_gaussian(closed_orbit, covariance, qq, beta=bb, size=number_of_particles)

    x = part_inj[:, 0]
    px = part_inj[:, 1]
    y = part_inj[:, 2]
    py = part_inj[:, 3]
    zeta = part_inj[:, 4]
    delta = part_inj[:, 5]

    return (x, px, y, py, zeta, delta)

# def get_off_momentum_twiss(line, delta_offset, method='4d'):
#     """
#     Get the twiss parameters for an off-momentum closed orbit.
    
#     If the closed orbit can be found directly, this function simply calls
#     `line.twiss` with the given `delta_offset`. Otherwise, it performs a
#     search for the closed orbit by starting from the on-momentum orbit and
#     gradually increasing the delta offset.

#     For a null delta offset this function returns the on-momentum twiss.

#     Parameters
#     ----------
#     line : Line
#         The lattice to be used.
#     delta_offset : float
#         The off-momentum value.

#     Returns
#     -------
#     tw : Twiss
#         The twiss parameters for the off-momentum closed orbit.
#     """
#     try:
#         # Try to find the closed orbit directly
#         return line.twiss(method = method, delta0=delta_offset) #, steps_r_matrix=steps_r_matrix)
#     except Exception:
#         # If the closed orbit can't be found directly, perform a search
#         in_div = 4
#         ini_delta = delta_offset / in_div
#         step_delta = delta_offset / 10 * (in_div - 1)/in_div
#         fin_delta = delta_offset + step_delta
#         co_guess = line.find_closed_orbit(delta0=ini_delta - step_delta)
#         for delta in np.arange(ini_delta, fin_delta, step_delta):
#             co_guess = line.find_closed_orbit(co_guess=co_guess, delta0=delta)
#         # Now find the twiss parameters for the off-momentum closed orbit
#         return line.twiss(method = method,  co_guess=co_guess, delta0=delta_offset)

def get_off_momentum_twiss(line_onmom, delta_offset, BB_off=True, method='4d'):
    """
    Compute off-momentum twiss by trimming RF frequency and adding a ZetaShift.

    Raises:
        ValueError: for invalid method/BB_off combinations or missing RF cavity.
        RuntimeError: for failures in twiss computation (keeps original exception as __cause__).
    """
    # ---- Validate inputs early (raise ValueError, not mysterious crashes)
    if method not in ('4d', '6d'):
        raise ValueError(f"method must be '4d' or '6d', got {method!r}")

    if not BB_off and method != '6d':
        raise ValueError("When BB_off=False (BB on), method must be '6d'")

    # ---- Snapshot state we are about to mutate, so we can restore it
    tab = line_onmom.get_table()
    bb_elem_name = tab.rows[tab.element_type == 'BeamBeamBiGaussian3D'].name

    # Save current BB strengths
    prev_bb_strength = {jj: line_onmom[jj].scale_strength for jj in bb_elem_name}

    # If you also toggle radiation, we should restore it too.
    # If your xtrack version doesn't expose a clean getter, at least restore deterministically.
    # Here we just remember whether we set it to None in this function.
   
    SR_model = line_onmom._radiation_model
    BS_model = line_onmom._beamstrahlung_model
    radiation_was_forced_off = False
    
    try:
        # ---- Apply BB scaling on the input line
        for jj in bb_elem_name:
            line_onmom[jj].scale_strength = 0 if BB_off else 1

        # ---- Radiation / method logic
        if method == '4d' and BB_off:
            # You explicitly want SR off in this mode
            line_onmom.configure_radiation(model=None)
            radiation_was_forced_off = True
        elif method == '6d' and BB_off:
            # BB off, SR on, 6d: do nothing special (as per your intent)
            pass
        # (case BB_on handled by validation above)

        # ---- On-momentum twiss
        try:
            tw_onmom = line_onmom.twiss(method=method)
        except Exception as e:
            raise RuntimeError("Failed to compute on-momentum twiss") from e

        tw_onmom = line_onmom.twiss(method=method)
        
        # ---- Work on a copy for off-momentum manipulation 
        line = line_onmom.copy(shallow=any(tab.element_type == 'Geant4Collimator'))

        # ---- Trim RF frequency
        tt = line.get_table()
        cav_names = tt.rows[tt.element_type == "Cavity"].name
        if len(cav_names) == 0:
            raise ValueError("No Cavity element found in line; cannot compute RF frequency trim")

        eta = tw_onmom.slip_factor
        if eta == 0:
            raise ValueError("slip_factor (eta) is 0; cannot compute frequency trim")

        f_rf = line[cav_names[0]].frequency

        # Frequency trim to have the desired delta
        df_hz = -eta * delta_offset * f_rf

        # Zeta shift to add
        dzeta = tw_onmom.circumference * df_hz / f_rf

        # Append delay element to the line
        line.append('zeta_shift', xt.ZetaShift(dzeta=dzeta))

        # ---- Off-momentum twiss on modified copy
        try:
            tw1 = line.twiss()
            if any(tab.element_type == 'Geant4Collimator'):
                line.remove('zeta_shift')
        except Exception as e:
            raise RuntimeError("Failed to compute off-momentum twiss on modified line") from e


        # Optional diagnostics (won't affect exceptions)
        f_rev = 1 / tw_onmom.T_rev0
        h_rf = f_rf / f_rev
        df_rev = df_hz / h_rf
        delta_expected = -df_rev / f_rev / eta

        print(f'Obtained delta: {tw1.delta[0]}')
        print(f'Difference between expected and measured delta: {tw1.delta[0] - delta_expected}')
        return tw1

    finally:
        # ---- ALWAYS restore mutated state on the original line_onmom
        for jj, val in prev_bb_strength.items():
            try:
                line_onmom[jj].scale_strength = val
            except Exception:
                # Avoid masking the original exception in finally
                pass

        # If you forced radiation off, you might want to restore it to "default/on".
        # If your workflow expects radiation to remain off, remove this.
        if radiation_was_forced_off:
            try:
                line.configure_radiation(model=SR_model, model_beamstrahlung=BS_model)
                pass
            except Exception:
                pass

def compute_injection_twiss_with_fallbacks(line, parameters):
    delta = parameters['injection_parameters']['energy_offset']

    attempts = [
        # label, kwargs passed to get_off_momentum_twiss
        ("BB on + SR on (6d, 0.9*delta)",
         dict(delta_offset=0.9 * delta, BB_off=False, method='6d')),

        ("BB off + SR on (6d, delta)",
         dict(delta_offset=0.98 * delta, BB_off=True, method='6d')),

        ("BB off + SR off (4d, delta)",
         dict(delta_offset=0.98 * delta, BB_off=True, method='4d')),
    ]

    errors = []
    for label, kwargs in attempts:
        try:
            tw = get_off_momentum_twiss(line, **kwargs)
            return tw
        except Exception as e:
            print(f"WARNING: {label} failed: {type(e).__name__}: {e}")
            errors.append((label, e))

    # If we get here, all attempts failed
    last_label, last_exc = errors[-1]
    summary = " ; ".join(lbl for lbl, _ in errors)
    raise RuntimeError(
        f"Could not calculate off-momentum twiss with any fallback. Attempts: {summary}"
    ) from last_exc


def generate_particle_distribution (line, study_param, method='6d', particle_capacity=None, 
                                    q_factor=1, b_factor=1, beambeam_strength_used=None, radiation_off=False, rnd_seed=107):

    if radiation_off:
        wig_v = line.vv['on_wiggler_v']
        SR_model = line._radiation_model
        BS_model = line._beamstrahlung_model
        line.vars['on_wiggler_v']=0
        line.configure_radiation(model=None, model_beamstrahlung=None)
    if beambeam_strength_used is not None:
        tt = line.get_table(attr=True)
        bb_elem_name = tt.rows[tt.element_type=='BeamBeamBiGaussian3D'].name
        dd_bb = {}
        for jj in bb_elem_name:
            dd_bb[jj] = line[jj].scale_strength
            line[jj].scale_strength = beambeam_strength_used
    
    if particle_capacity is None:
        capacity = study_param['number_of_particles']
    else:
        capacity = particle_capacity

    if 'distribution' in study_param['ini_cond_type']:
        if 'matched' in study_param['ini_cond_type']: # relies on twiss used tw_matched
            delta_offset = study_param['ini_cond_energy_offset']
            if delta_offset is None:
                tw_matched = line.twiss(method=method)
            else:   
                tw_matched = get_off_momentum_twiss(line, delta_offset) #, method=method) 4d is currently forced by Xsuite for off-momentum twiss  
            
            x_in, px_in, y_in, py_in, zeta_in, delta_in = initial_conditions_distribution (tw_matched, study_param['ini_cond_nemittance_x'], 
                                                                                                study_param['ini_cond_nemittance_y'], 
                                                                                                study_param['ini_cond_bunch_length']*study_param['ini_cond_energy_spread'], 
                                                                                                number_of_particles=study_param['number_of_particles'], 
                                                                                                new_closed_orbit=study_param['new_closed_orbit'],
                                                                                                covariance_dispertion_free=study_param['covariance_dispertion_free'],
                                                                                                qq=q_factor, bb=b_factor, rnd_seed=rnd_seed)
            
            if delta_offset is not None: # with delta0 the twiss has to be 4d thus the longitudinal distribution should be set separately
                long_closed_orbit = [np.mean(zeta_in), np.mean(delta_in)] # [0, inject_param['energy_offset']] # 
                long_covariance = [[study_param['ini_cond_bunch_length']**2, 0], [0, study_param['ini_cond_energy_spread']**2]]
                np.random.seed(rnd_seed+1)
                long_part_inj = np.random.multivariate_normal(mean=long_closed_orbit, cov=long_covariance, size=study_param['number_of_particles'])
                zeta_in = long_part_inj[:,0]
                delta_in = long_part_inj[:,1]

        else:
            raise ValueError(f"Unknown initial condition!")
       
        particles = line.build_particles(_capacity=capacity,
                                        x = x_in, 
                                        px = px_in,
                                        y = y_in, 
                                        py = py_in,
                                        zeta = zeta_in,
                                        delta = delta_in)
        
    if radiation_off:
        line.vars['on_wiggler_v']=wig_v
        line.configure_radiation(model=SR_model, model_beamstrahlung=BS_model)
    if beambeam_strength_used is not None:
        for jj in bb_elem_name:
            line[jj].scale_strength = dd_bb[jj]

    return (particles)


def generate_particle_grid (line, study_param, particle_capacity=None, cartesian_polar=None, 
                            min_r_y=None, max_r_y=None, num_r_y_points=None, min_theta_x=None, max_theta_x=None, 
                            num_theta_x_points=None, r_range_x=(5,10), r_range_y=(5,10), 
                            theta_range_x=(0,2*np.pi), theta_range_y=(0,2*np.pi), delta_initial_values=None, 
                            beambeam_strength_used=None, radiation_off=False, rnd_seed=101):

    if radiation_off:
        wig_v = line.vv['on_wiggler_v']
        SR_model = line._radiation_model
        BS_model = line._beamstrahlung_model
        line.vars['on_wiggler_v']=0
        line.configure_radiation(model=None, model_beamstrahlung=None)
    if beambeam_strength_used is not None:
        tt = line.get_table(attr=True)
        bb_elem_name = tt.rows[tt.element_type=='BeamBeamBiGaussian3D'].name
        dd_bb = {}
        for jj in bb_elem_name:
            dd_bb[jj] = line[jj].scale_strength
            line[jj].scale_strength = beambeam_strength_used

    tw = line.twiss(eneloss_and_damping=True)
    ref_part = line.particle_ref
    if 'grid' in study_param['ini_cond_type']:
        # The longitudinal closed orbit needs to be manually supplied for now
        zeta_co = tw.zeta[0] 
        delta_co = tw.delta[0]
        if 'halo' in study_param['ini_cond_type']:
            # Circulating halo beam physical coordinates
 
            (x_normalized, y_normalized, px_normalized, py_normalized) = initial_conditions_grid ('halo', 
                                                                                                 r_range_x=r_range_x, r_range_y=r_range_y, 
                                                                                                 theta_range_x=theta_range_x, theta_range_y=theta_range_y, 
                                                                                                 num_particles=study_param['number_of_particles'], rnd_seed=rnd_seed) 

            print(f'Paramter sigma_z > 0, preparing a longitudinal distribution matched to the RF bucket')
            zeta, delta = xp.generate_longitudinal_coordinates(
                            line=line,
                            num_particles=study_param['number_of_particles'], distribution='gaussian',
                            sigma_z=study_param['ini_cond_bunch_length'], particle_ref=ref_part)
            zeta = zeta + zeta_co
            delta = delta + delta_co
            
            grid_details = {}

        elif any(ii in study_param['ini_cond_type'] for ii in ['DA', 'MA']):
            found = next(ii for ii in ['DA', 'MA'] if ii in study_param['ini_cond_type'])
            (x_normalized, y_normalized, delta_init, num_theta_x_points, num_r_y_points, num_delta, num_particles) = initial_conditions_grid (found, 
                                                                                                                                             study_param['ini_cond_energy_spread'],
                                                                                                                                             cartesian_polar=cartesian_polar, 
                                                                                                                                             min_r_y=min_r_y, max_r_y=max_r_y, num_r_y_points=num_r_y_points, 
                                                                                                                                             min_theta_x=min_theta_x, max_theta_x=max_theta_x, num_theta_x_points=num_theta_x_points, 
                                                                                                                                             delta_initial_values=delta_initial_values, rnd_seed=rnd_seed)
            px_normalized = 0 
            py_normalized = 0
            zeta = zeta_co
            delta = delta_init + delta_co

            grid_details = {
            'num_theta_x_points':num_theta_x_points,
            'num_r_y_points':num_r_y_points,
            'num_delta':num_delta,
            'num_particles':num_particles,
            'x_normalized':x_normalized,
            'y_normalized':y_normalized,
            'delta_init':delta_init
            }
        
        else:
            raise ValueError(f"Unknown initial condition!")
        
        if particle_capacity is None:
            capacity = len(x_normalized)
        else:
            capacity = particle_capacity
        
        particles = line.build_particles(_capacity=capacity,
                x_norm=x_normalized, 
                y_norm=y_normalized,  
                px_norm=px_normalized,
                py_norm=py_normalized,
                nemitt_x=study_param['ini_cond_nemittance_x'], 
                nemitt_y=study_param['ini_cond_nemittance_y'], 
                zeta= zeta, 
                delta= delta) 
        
    if radiation_off:
        line.vars['on_wiggler_v']=wig_v
        line.configure_radiation(model=SR_model, model_beamstrahlung=BS_model)
    if beambeam_strength_used is not None:
        for jj in bb_elem_name:
            line[jj].scale_strength = dd_bb[jj]

    return (particles, grid_details)


# Thanks to beam-beam team, Khoi Le Nguyen Nguyen, Xavier Buffat, Peter Kicsiny, Tatiana Pieloni and Roxana Soos
# https://arxiv.org/abs/2404.09012
# https://gitlab.cern.ch/xbuffat/khoi/-/blob/master/bs_1D_flipflop_hgcw_correction.py?ref_type=heads
def get_sigma_zeta_and_delta_from_ws_beambeam(twiss, reference_parameters, beam_beam_parameters, BS_scale_factor=1):

    r_e = sp_co.physical_constants["classical electron radius"][0]                             # Classical electron radius in m
    m_e = sp_co.physical_constants["electron mass energy equivalent in MeV"][0]                # Electron mass in MeV

    ### Intermediate parameters
    ###
    def sigma_delta(tw, sigma_zeta):
        return (2*np.pi*tw.qs*sigma_zeta/tw.circumference/np.abs(tw.slip_factor))
    
    def B(x):
        B = theta_c/x**2
                # B_w/s: x = s_xw/s
        return(B)

    def D(n,a,b):
        D = n/(2*a**2) + 1/(2*b**2)
                # D_w/s: a = s_zs/w, b = s_zw/s
        return (D)
        
    def G(n,a,b,c):
        G = theta_c**2/(2*a**2)+(2*n)/(n*b**2+c**2)
                # G_nw/s: a = s_xw/s, b = s_zw/s, c = s_zs/w
        return (G)

    def L(n,a,b,c,d):
        L = a/np.sqrt(theta_c**2*(n*b**2+c**2)+4*n*d**2)
                # L_nw/s: a = s_xs/w, b = s_zw/s, c = s_zs/w, d = s_xw/s
        return (L)

    def M(n,a,b,c,d):
        M = n/2+2*n*L(n,a,b,c,d)**2
                # M_nw/s: a = s_xs/w, b = s_zw/s, c = s_zs/w, d = s_xw/s
        return (M)

    def y_ratio(a,b):
        y_ratio = a/b
                # y_ratio_w/s: a = s_ys/w, b = s_yw/s
        return(y_ratio)

    def P(n,a,b,c,d,e,f):
        P = 1 - y_ratio(e,f)**2/(2*M(n,a,b,c,d))
                # P_nw/s: a = s_xs/w, b = s_zw/s, c = s_zs/w, d = s_xw/s, e = s_ys/w, f = s_yw/s
        return (P)

    def Q(n,a,b,c):
        Q = (2*np.pi)**(-1/2)*a**(1-n)*((2*r_e*b)/(gamma_factor*c)*np.sqrt(2/np.pi))**n
                # Q_nw/s: a = s_zs/w, b = N_s/w, c = s_xs/w
        return (Q)


    def R(n,a,b,c,d,e,f,g):
        R = np.pi*mp.gamma(n/2+1)*L(n,a,b,c,d)*y_ratio(e,f)*Q(n,c,g,a)
                # R_nw/s: a = s_xs/w, b = s_zw/s, c = s_zs/w, d = s_xw/s, e = s_ys/w, f = s_yw/s, g = N_s/w
        return (R)

    """
    Integrals I with/without hourglass and/or crabwaist
    """
    def I_0(n,a,b,c,d,e,f,g):
        I_0 = R(n,a,b,c,d,e,f,g)*M(n,a,b,c,d)**(-(n/2+1))*hyp2f1(n/2+1,1/2,1,P(n,a,b,c,d,e,f))
                # I_0_nw/s: a = s_xs/w, b = s_zw/s, c = s_zs/w, d = s_xw/s, e = s_ys/w, f = s_yw/s_ys, g = N_s/w
        return (I_0)
    def I_hg(n,a,b,c,d,e,f,g):
        I_hg = I_0(n,a,b,c,d,e,f,g)-(R(n,a,b,c,d,e,f,g))/(2*G(n,a,b,c)*beta_y**2*M(n,a,b,c,d)**(n/2+1))*(mp.appellf1(1/2,-1,n/2+1,1,y_ratio(e,f)**2-n,P(n,a,b,c,d,e,f))+(B(d)**2*a**2)/(4*G(n,d,b,c)*M(n,a,b,c,d))*(n/2+1)*mp.appellf1(1/2,-1,n/2+2,2,y_ratio(e,f)**2-n,P(n,a,b,c,d,e,f)))
                # I_hg_nw/s: a = s_xs/w, b = s_zw/s, c = s_zs/w, d = s_xw/s, e = s_ys/w, f = s_yw/s_ys, g = N_s/w
        return(I_hg)
    def I_cw(n,a,b,c,d,e,f,g):
        I_cw = I_0(n,a,b,c,d,e,f,g)-(R(n,a,b,c,d,e,f,g))/(16*D(n,c,b)*beta_y**2*M(n,a,b,c,d)**(n/2+1))*((1+(4*d**2*c**2)/(a**2*b**2)*L(n,a,b,c,d)**2)*mp.appellf1(1/2,-1,n/2+1,1,y_ratio(e,f)**2-n,P(n,a,b,c,d,e,f))+(4*D(n,c,b)*a**2)/(theta_c**2*M(n,a,b,c,d))*(1-(theta_c**2*c**2)/(a**2)*L(n,a,b,c,d)**2)**2*(n/2+1)*mp.appellf1(1/2,-1,n/2+2,2,y_ratio(e,f)**2-n,P(n,a,b,c,d,e,f)))
                # I_cw_nw/s: a = s_xs/w, b = s_zw/s, c = s_zs/w, d = s_xw/s, e = s_ys/w, f = s_yw/s, g = N_s/w
        return(I_cw)
        
    def I_hgcw(n,a,b,c,d,e,f,g):
        """
        Function to solve the two coupled equations for the equilibrium bunch lengths with weak hg, weak cw
        """
        I_hgcw = I_0(n,a,b,c,d,e,f,g)-(R(n,a,b,c,d,e,f,g))/(16*D(n,c,b)*beta_y**2*M(n,a,b,c,d)**(n/2+1))*((n/2+1)*(a**2*D(n,c,b))/M(n,a,b,c,d)*(2/theta_c+(2*n*theta_c*b**2)/a**2*L(n,a,b,c,d)**2)**2*mp.appellf1(1/2,-1,n/2+2,2,y_ratio(e,f)**2,P(n,a,b,c,d,e,f))+(1+4*n**2*(b**2*d**2)/(c**2*a**2)*L(n,a,b,c,d)**2)*mp.appellf1(1/2,-1,n/2+1,1,y_ratio(e,f)**2,P(n,a,b,c,d,e,f)))
        -(n*R(n,a,b,c,d,e,f,g))/(16*G(n,d,b,c)*beta_y**2*M(n,a,b,c,d)**(n/2+1))*((2*G(n,d,b,c)*a**2)/M(n,a,b,c,d)*(1/theta_c-theta_c/a**2*(n*b**2+c**2)*L(n,a,b,c,d)**2)**2*(n/2+1)*mp.hyp2f1(n/2+2,3/2,3,P(n,a,b,c,d,e,f))+2*mp.hyp2f1(n/2+1,3/2,2,P(n,a,b,c,d,e,f)))
                # I_hgcw_nw/s: a = s_xs/w, b = s_zw/s, c = s_zs/w, d = s_xw/s, e = s_ys/w, f = s_yw/s, g = N_s/w
        return(I_hgcw)


    """
    Function to solve the two coupled equations for the equilibrium bunch lengths
    """
    def eqm_bunch_lengths_no_hg_no_cw(p):
                    # p[0] = s_zw_eq, p[1] = s_zs_eq
        return (
            2/tau_z_SR*(s_zw_SR**2+A_2*I_0(3,s_xs,p[0],p[1],s_xw,s_ys,s_yw,N_s))-(2/tau_z_SR+n_IP*4/3*r_e*gamma_factor**3*I_0(2,s_xs,p[0],p[1],s_xw,s_ys,s_yw,N_s))*p[0]**2,
            2/tau_z_SR*(s_zs_SR**2+A_2*I_0(3,s_xw,p[1],p[0],s_xs,s_yw,s_ys,N_w))-(2/tau_z_SR+n_IP*4/3*r_e*gamma_factor**3*I_0(2,s_xw,p[1],p[0],s_xs,s_yw,s_ys,N_w))*p[1]**2
        )
        
    def eqm_bunch_lengths_weak_hg_no_cw(p):
                    # p[0] = s_zw_eq, p[1] = s_zs_eq
        return (
            2/tau_z_SR*(s_zw_SR**2+A_2*I_hg(3,s_xs,p[0],p[1],s_xw,s_ys,s_yw,N_s))-(2/tau_z_SR+n_IP*4/3*r_e*gamma_factor**3*I_hg(2,s_xs,p[0],p[1],s_xw,s_ys,s_yw,N_s))*p[0]**2,
            2/tau_z_SR*(s_zs_SR**2+A_2*I_hg(3,s_xw,p[1],p[0],s_xs,s_yw,s_ys,N_w))-(2/tau_z_SR+n_IP*4/3*r_e*gamma_factor**3*I_hg(2,s_xw,p[1],p[0],s_xs,s_yw,s_ys,N_w))*p[1]**2
        )

    def eqm_bunch_lengths_no_hg_weak_cw(p):
                    # p[0] = s_zw_eq, p[1] = s_zs_eq
        return (
            2/tau_z_SR*(s_zw_SR**2+A_2*I_cw(3,s_xs,p[0],p[1],s_xw,s_ys,s_yw,N_s))-(2/tau_z_SR+n_IP*4/3*r_e*gamma_factor**3*I_cw(2,s_xs,p[0],p[1],s_xw,s_ys,s_yw,N_s))*p[0]**2,
            2/tau_z_SR*(s_zs_SR**2+A_2*I_cw(3,s_xw,p[1],p[0],s_xs,s_yw,s_ys,N_w))-(2/tau_z_SR+n_IP*4/3*r_e*gamma_factor**3*I_cw(2,s_xw,p[1],p[0],s_xs,s_yw,s_ys,N_w))*p[1]**2
        )
        
    def eqm_bunch_lengths_weak_hg_weak_cw(p):
                    # p[0] = s_zw_eq, p[1] = s_zs_eq
        return (
            2/tau_z_SR*(s_zw_SR**2+A_2*I_hgcw(3,s_xs,p[0],p[1],s_xw,s_ys,s_yw,N_s))-(2/tau_z_SR+n_IP*4/3*r_e*gamma_factor**3*I_hgcw(2,s_xs,p[0],p[1],s_xw,s_ys,s_yw,N_s))*p[0]**2,
            2/tau_z_SR*(s_zs_SR**2+A_2*I_hgcw(3,s_xw,p[1],p[0],s_xs,s_yw,s_ys,N_w))-(2/tau_z_SR+n_IP*4/3*r_e*gamma_factor**3*I_hgcw(2,s_xw,p[1],p[0],s_xs,s_yw,s_ys,N_w))*p[1]**2
        )
                    
    ##### COMPOSITE JACOBIAN AND LUMINOSITY CODE

    #### Beam Parameters for tt2 resonance at FCC-ee
    C = reference_parameters['circumference']                         # Circumference in m
    theta_c = 2*np.abs(beam_beam_parameters['half_xing_angle'])       # Full crossing angle
    BE = reference_parameters['energy']*1e-9                   # Beam energy in GeV
    N_s = reference_parameters['bunch_population']*BS_scale_factor                    # Target bunch population
    N_w = reference_parameters['bunch_population']*BS_scale_factor                    # Source bunch population
    a_p = reference_parameters['momentum_compaction']                 # Momentum compaction
    beta_x = reference_parameters['betastar_x']                       # Twiss beta_x parameter in m
    beta_y = reference_parameters['betastar_y']                       # Twiss beta_y parameter in m
    s_xw = np.sqrt(reference_parameters['betastar_x']*reference_parameters['emittance_x'])              # Source x-size at IP in m
    s_xs = np.sqrt(reference_parameters['betastar_x']*reference_parameters['emittance_x'])              # Target x-size at IP in m
    s_yw = np.sqrt(reference_parameters['betastar_y']*reference_parameters['emittance_y'])              # Weak y-size at IP in m
    s_ys = np.sqrt(reference_parameters['betastar_y']*reference_parameters['emittance_y'])              # Strong y-size at IP in m
    s_dw_SR = reference_parameters['energy_spread_SR']           # Energy spread in source bunch due to orbit SR
    s_ds_SR = reference_parameters['energy_spread_SR']           # Energy spread in target bunch due to orbit SR
    Q_sync = reference_parameters['qs']                          # Synchrotron tune
    tau_z_SR = reference_parameters['longitudinal_damping_time'] # Longitudinal damping time in turns
    n_IP = beam_beam_parameters['num_IPs']                       # Number of interaction points

    ####### Derived Parameters
    gamma_factor = BE*10**3/m_e
    A_1 = ((a_p*C)/(2*np.pi*Q_sync))**2
    A_2 = 55/(24*np.sqrt(3))*A_1*(n_IP*tau_z_SR)/4*r_e**2*gamma_factor**5*137

    s_zw_SR = np.sqrt(A_1)*s_dw_SR
    s_zs_SR = np.sqrt(A_1)*s_ds_SR


    # for beta_y in beta_y_list:
    eq_sol_0=opt.root(eqm_bunch_lengths_no_hg_no_cw, [1e-2,1e-2])
    if eq_sol_0.success == True:
        #print("Initial longitudinal weak bunch length in mm", s_zw_SR*10**3, "Initial longitudinal strong bunch length in mm", s_zs_SR*10**3)
        print(f"Beamstrahlung bunch length: {eq_sol_0.x[0]*10**3:.3f} [mm]")
        #print("s_zw/s_zw_SR: ", eq_sol_0.x[0]/s_zw_SR)
        # beta_y_ratio_nohgnocw[counter] = beta_y/eq_sol_0.x[0]
        # length_nohgnocw[counter] = eq_sol_0.x[0]*10**3  
    else: 
        print("Failed to determine beamstrahlung bunch length")
    
    #eq_sol_1=opt.root(eqm_bunch_lengths_weak_hg_no_cw, [1e-2,1e-2])
    #if eq_sol_1.success == True:
    #    #print("Total longitudinal weak bunch length after interaction (weak hg no cw) in mm: ", eq_sol_1.x[0]*10**3, "; Total longitudinal strong bunch length after interaction in mm: ", eq_sol_1.x[1]*10**3)
    #    #print("s_zw/s_zw_SR: ", eq_sol_1.x[0]/s_zw_SR)
    #    # beta_y_ratio_weakhgnocw[counter] = beta_y/eq_sol_1.x[0]
    #    # length_weakhgnocw[counter] = eq_sol_1.x[0]*10**3  
    #else: 
    #    print("No result with given starting guesses (weak_hg_no_cw)")
    
    #eq_sol_2=opt.root(eqm_bunch_lengths_weak_hg_weak_cw, [1e-2,1e-2])
    #if eq_sol_2.success == True:
    #    #print("Total longitudinal weak bunch length after interaction (weak hg weak cw) in mm: ", eq_sol_2.x[0]*10**3, "; Total longitudinal strong bunch length after interaction in mm: ", eq_sol_2.x[1]*10**3)
    #    #print("s_zw/s_zw_SR: ", eq_sol_2.x[0]/s_zw_SR)
    #    # beta_y_ratio_weakhgweakcw[counter] = beta_y/eq_sol_2.x[0]
    #    # length_weakhgweakcw[counter] = eq_sol_2.x[0]*10**3  
    #else: 
    #    print("No result with given starting guesses (weak_hg_weak_cw)")
    
    # eq_sol_3=opt.root(eqm_bunch_lengths_no_hg_weak_cw, [1e-2,1e-2])
    # if eq_sol_3.success == True:
    # print("Total longitudinal weak bunch length after interaction (no hg weak cw) in mm: ", eq_sol_3.x[0]*10**3, "; Total longitudinal strong bunch length after interaction in mm: ", eq_sol_3.x[1]*10**3)
    # print("s_zw/s_zw_SR: ", eq_sol_3.x[0]/s_zw_SR)
    # else: 
    # print("No result with given starting guesses")
    # counter+=1

    # df = pd.DataFrame({"beta_y/sigma_z no hg no cw": beta_y_ratio_nohgnocw,
                    # "beta_y/sigma_z weak hg no cw": beta_y_ratio_weakhgnocw,
                    # "beta_y/sigma_z weak hg weak cw": beta_y_ratio_weakhgweakcw,
                    # "No hg no cw": length_nohgnocw,
                    # "Weak hg no cw": length_weakhgnocw,
                    # "Weak hg weak cw": length_weakhgweakcw})
    # df.to_csv("1D_bunch_lengths_variable_betay_WW_02Sep2022.csv", index = False, float_format = '%.15f')

    longitudinal_sigmas = {
        'bunch_length_w_SR': s_zw_SR,
        'energy_spread_w_SR': sigma_delta(twiss, s_zw_SR),
        'bunch_length_s_SR': s_zs_SR,
        'energy_spread_s_SR': sigma_delta(twiss, s_zs_SR),
        'bunch_length_w_BS': eq_sol_0.x[0],
        'energy_spread_w_BS': sigma_delta(twiss, eq_sol_0.x[0]),
        'bunch_length_s_BS': eq_sol_0.x[1],
        'energy_spread_s_BS': sigma_delta(twiss, eq_sol_0.x[1]),
        #'s_z_w_SR': s_zw_SR,
        #'s_delta_w_SR': sigma_delta(twiss, s_zw_SR),
        #'s_z_s_SR': s_zs_SR,
        #'s_delta_s_SR': sigma_delta(twiss, s_zs_SR),
        #'s_z_w_eq_no_hg_no_cw': eq_sol_0.x[0],
        #'s_delta_w_eq_no_hg_no_cw': sigma_delta(twiss, eq_sol_0.x[0]),
        #'s_z_s_eq_no_hg_no_cw': eq_sol_0.x[1],
        #'s_delta_s_eq_no_hg_no_cw': sigma_delta(twiss, eq_sol_0.x[1]),
        #'s_z_w_eq_weak_hg_no_cw': eq_sol_1.x[0],
        #'s_delta_w_eq_weak_hg_no_cw': sigma_delta(twiss, eq_sol_1.x[0]),
        #'s_z_s_eq_weak_hg_no_cw': eq_sol_1.x[1],
        #'s_delta_s_eq_weak_hg_no_cw': sigma_delta(twiss, eq_sol_1.x[1]),
        #'s_z_w_eq_weak_hg_weak_cw': eq_sol_2.x[0],
        #'s_delta_w_eq_weak_hg_weak_cw': sigma_delta(twiss, eq_sol_2.x[0]),
        #'s_z_s_eq_weak_hg_weak_cw': eq_sol_2.x[1],
        #'s_delta_s_eq_weak_hg_weak_cw': sigma_delta(twiss, eq_sol_2.x[1]),
        # 's_z_w_eq_no_hg_weak_cw': eq_sol_3.x[0],
        # 's_delta_w_eq_no_hg_weak_cw': sigma_delta(twiss, eq_sol_3.x[0]),
        # 's_z_s_eq_no_hg_weak_cw': eq_sol_3.x[1],
        # 's_delta_s_eq_no_hg_weak_cw': sigma_delta(twiss, eq_sol_3.x[1])
    }
    return (longitudinal_sigmas)


def get_beambeam_tune_shift(line, reference_parameters, beam_beam_parameters, max_bb_tune_shift=0.1):

    nemittance_x = reference_parameters['normalized_emittance_x']         # Geometric emittance in m
    nemittance_y = reference_parameters['normalized_emittance_y']         # Geometric emittance in m
    energy_spread = reference_parameters['energy_spread_BS']              # Energy spread with BS
    bunch_length = reference_parameters['bunch_length_BS']                # Bunch length in m with BS
    emittance_zeta = bunch_length*energy_spread                           # Longitudinal emittance
    tw = line.twiss()
    ip_list = get_ip_list(line)
    beta_x = tw.rows[ip_list[0]].rows[0].betx[0]                          # Beta star in m
    beta_y = tw.rows[ip_list[0]].rows[0].bety[0]                          # Beta star in m
    covariance_f=tw.get_beam_covariance(nemitt_x=nemittance_x, nemitt_y=nemittance_y, gemitt_zeta=emittance_zeta)
    sigma_x = np.sqrt(covariance_f.rows[ip_list[0]].rows[0].Sigma11[0])
    sigma_y = np.sqrt(covariance_f.rows[ip_list[0]].rows[0].Sigma33[0])
    sigma_z = bunch_length

    bunch_population = reference_parameters['bunch_population']        # Bunch population
    gamma_relativistic = reference_parameters['gamma_relativistic']     # Relativistic gamma factor
    r0 = reference_parameters['particle_classical_radius0'] # Classical radius  in m    

    half_xangle = beam_beam_parameters['half_xing_angle']  # Half crossing angle in rad
    if beam_beam_parameters['xing_plane'] == 0: # 0:Horizontal crossing
        piwinski = sigma_z*np.tan(half_xangle)/sigma_x
    elif beam_beam_parameters['xing_plane'] == 1: # 1:Vertical crossing
        piwinski = sigma_z*np.tan(half_xangle)/sigma_y

    if math.isclose(sigma_x, sigma_y, rel_tol=1e-9): 
        xi = lambda sigma: r0*bunch_population/(2*np.pi*gamma_relativistic*sigma**2)/np.sqrt(1+piwinski**2)
        bb_xi_x = xi(sigma_x)
        bb_xi_y = xi(sigma_y)
        if beam_beam_parameters['xing_plane'] == 0: # 0:Horizontal crossing
            min_emit_pplane = r0*bunch_population/(2*np.pi*gamma_relativistic*max_bb_tune_shift)/np.sqrt(1+piwinski**2)/beta_y
        elif beam_beam_parameters['xing_plane'] == 1: # 1:Vertical
            min_emit_pplane = r0*bunch_population/(2*np.pi*gamma_relativistic*max_bb_tune_shift)/np.sqrt(1+piwinski**2)/beta_x  
    else:
        xi = lambda beta, sigma: r0*bunch_population*beta/(2*np.pi*gamma_relativistic*sigma*(sigma_x+sigma_y))
        min_emittance_parallel_plane = lambda beta, sigma: ((-sigma+np.sqrt(sigma**2 + (2*r0*bunch_population*beta)/(np.pi*gamma_relativistic*max_bb_tune_shift*np.sqrt(1+piwinski**2))))/2)**2 / beta
        if beam_beam_parameters['xing_plane'] == 0: # 0:Horizontal crossing
            bb_xi_x = xi(beta_x, sigma_x)/(1+piwinski**2) 
            bb_xi_y = xi(beta_y, sigma_y)/np.sqrt(1+piwinski**2)
            min_emit_pplane = min_emittance_parallel_plane(beta_y, sigma_x)
        elif beam_beam_parameters['xing_plane'] == 1: # 1:Vertical crossing
            bb_xi_x = xi(beta_x, sigma_x)/np.sqrt(1+piwinski**2)
            bb_xi_y = xi(beta_y, sigma_y)/(1+piwinski**2)
            min_emit_pplane = min_emittance_parallel_plane(beta_x, sigma_y)


    return (bb_xi_x, bb_xi_y, min_emit_pplane)


def install_beam_beam_elements (line, reference_parameters, beam_beam_parameters, ip_list=None):

    normalized_emittance_x= reference_parameters['normalized_emittance_x']
    normalized_emittance_y = reference_parameters['normalized_emittance_y']
    bunch_length = reference_parameters['bunch_length']
    bunch_population = reference_parameters['bunch_population']
    half_xing_angle = beam_beam_parameters['half_xing_angle']
    xing_plane = beam_beam_parameters['xing_plane']
    beamstrahlung_on = beam_beam_parameters['beamstrahlung_on']
    num_slices = beam_beam_parameters['num_slices']
    binning_mode = beam_beam_parameters['binning_mode'] 

    # set the oposit sign for the other reference particle
    particle_ref_rev = line.particle_ref.copy()
    particle_ref_rev.q0 = -particle_ref_rev.q0

    # use the revers method due to the lack of the other beam lattice (for now)
    tw_rev = line.twiss(method='6d', particle_ref=particle_ref_rev, reverse=True)

    if ip_list is None:
        ip_list = get_ip_list(line)

        #ip_list = tw_rev.rows['ip.*'].name.tolist()[::-2]
    
    # definition and installation of beam beam
    line.discard_tracker() # needed to modify the line structure
    bb_elem_name = []
    sigmas = tw_rev.get_betatron_sigmas(nemitt_x=normalized_emittance_x, nemitt_y=normalized_emittance_y)
    #z_centroid, z_cuts, n_part_slice = constant_charge_slicing_gaussian(bunch_population, bunch_length, 70)
    slicer = xf.TempSlicer(n_slices = num_slices, sigma_z = bunch_length, mode = binning_mode)
    for ii in ip_list:
        bb_elem_name.append('beambeam_'+ii)

        if beamstrahlung_on:
            slices_other_beam_zeta_bin_width_star_beamstrahlung = slicer.bin_widths_beamstrahlung / np.cos(half_xing_angle)
            slices_other_beam_sqrtSigma_11_beamstrahlung = np.zeros(num_slices)+np.sqrt(sigmas['Sigma11',ii])
            slices_other_beam_sqrtSigma_33_beamstrahlung = np.zeros(num_slices)+np.sqrt(sigmas['Sigma33',ii])
            slices_other_beam_sqrtSigma_55_beamstrahlung = 2.0*slicer.bin_widths_beamstrahlung # -> Shitty implementation which doesn not take into account Xing angle -> factor 2 is empirical.
        else:
            slices_other_beam_zeta_bin_width_star_beamstrahlung = None 
            slices_other_beam_sqrtSigma_11_beamstrahlung = None
            slices_other_beam_sqrtSigma_33_beamstrahlung = None
            slices_other_beam_sqrtSigma_55_beamstrahlung = None     

        bb_element = xf.BeamBeamBiGaussian3D(

            phi = half_xing_angle,
            alpha = xing_plane,
            other_beam_q0 = particle_ref_rev.q0,

            slices_other_beam_num_particles = slicer.bin_weights * bunch_population,   #n_part_slice[::-1],
            slices_other_beam_zeta_center = slicer.bin_centers,   #z_centroid[::-1],

            slices_other_beam_Sigma_11 = sigmas['Sigma11',ii],
            slices_other_beam_Sigma_12 = sigmas['Sigma12',ii],
            slices_other_beam_Sigma_13 = sigmas['Sigma13',ii],
            slices_other_beam_Sigma_14 = sigmas['Sigma14',ii],
            slices_other_beam_Sigma_22 = sigmas['Sigma22',ii],
            slices_other_beam_Sigma_23 = sigmas['Sigma23',ii],
            slices_other_beam_Sigma_24 = sigmas['Sigma24',ii],
            slices_other_beam_Sigma_33 = sigmas['Sigma33',ii],
            slices_other_beam_Sigma_34 = sigmas['Sigma34',ii],
            slices_other_beam_Sigma_44 = sigmas['Sigma44',ii],
            slices_other_beam_zeta_bin_width_star_beamstrahlung = slices_other_beam_zeta_bin_width_star_beamstrahlung, 
            slices_other_beam_sqrtSigma_11_beamstrahlung = slices_other_beam_sqrtSigma_11_beamstrahlung,
            slices_other_beam_sqrtSigma_33_beamstrahlung = slices_other_beam_sqrtSigma_33_beamstrahlung,
            slices_other_beam_sqrtSigma_55_beamstrahlung = slices_other_beam_sqrtSigma_55_beamstrahlung              
        )

        line.insert_element(bb_elem_name[-1], bb_element, at=ii)
        
        # switch off beambeam
        line[bb_elem_name[-1]].scale_strength = 0

    line.build_tracker()

    return (bb_elem_name)


def save_track_to_h5(monitor, monitor_name='0', output_dir="plots"):
    """Saves multi-particle tracking data to an HDF5 file with gzip compression."""

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f'merged_data_{monitor_name}.h5')

    # Define attributes and expected NumPy types
    attributes = {
        'x': np.float32, 'px': np.float32, 'y': np.float32, 'py': np.float32,
        'zeta': np.float32, 'delta': np.float32, 's': np.float32, 'at_turn': np.int32
    }

    try:
        with h5py.File(file_path, 'w') as h5f:
            for attr, dtype in attributes.items():
                data = getattr(monitor, attr, None)
                
                if data is not None:
                    data = np.asarray(data, dtype=dtype)  # Ensure NumPy array
                    
                    # Handle empty arrays gracefully
                    if data.size > 0:
                        h5f.create_dataset(attr, data=data, compression='gzip')

        print(f"Data successfully saved to {file_path}")

    except Exception as e:
        raise RuntimeError(f"Failed to save data to {file_path}: {e}")

def install_collimators(line, collimation_param):
    
    line.discard_tracker()

    # Load collimator database and install collimators
    colldb = xc.CollimatorDatabase.from_yaml(
        collimation_param['collimator_file']
    )
    colldb.install_geant4_collimators(line=line, apertures=xt.LimitEllipse(a=0.03, b=0.03))
    # Update optics for collimators and build the tracker
    line.build_tracker()
    line.collimators.assign_optics()

    # Discard and rebuild tracker with updated global aperture limit
    line.discard_tracker()
    line.config.global_xy_limit = 1e3


def _save_particles_hdf(particles=None, lossmap_data=None, filename='part', reduce_particles_size=False):
    if not filename.endswith('.hdf'):
        filename += '.hdf'

    fpath = Path(filename)
    # Remove a potential old file as the file is open in append mode
    if fpath.exists():
        fpath.unlink()

    if particles is not None:
        df = particles.to_pandas(compact=True)
        if reduce_particles_size:
            for dtype in ('float64', 'int64'):
                thistype_columns = df.select_dtypes(include=[dtype]).columns
                df[thistype_columns] = df[thistype_columns].astype(dtype.replace('64', '32'))

        df.to_hdf(fpath, key='particles', format='table', mode='a',
                  complevel=9, complib='blosc')

    if lossmap_data is not None:
        for key, lm_df in lossmap_data.items():
            lm_df.to_hdf(fpath, key=key, mode='a', format='table',
                         complevel=9, complib='blosc')
            
def generate_lossmap(line, particles, ref_part, config_dict, output_dir="plots"):

    """
    Generate a loss map of particles in a lattice with scattering.

    Parameters
    ----------
    line : Line
        The lattice with scattering.
    particles : Particles
        The particles to track.
    ref_part : Particles
        The reference particles.
    collimation_param : dict
        The collimation parameters.
    study_param : dict
        The study parameters.
    output_dir : str, optional
        The directory to save the loss map. Defaults to "plots".
    impact : bool, optional
        Whether to save the impacts of the Geant4 engine. Defaults to True.

    Returns
    -------
    None
    """
    num_turns = config_dict['study_parameters']['number_of_turns']
    impacts = xc.InteractionRecord.start(line=line)

    xc.geant4.engine.particle_ref = ref_part
    xc.geant4.engine.seed = config_dict['collimation_parameters']['inv3']
    xc.geant4.engine.start(line=line)

    line.scattering.enable()
    line.enable_time_dependent_vars = True
    for turn in range(num_turns):
        print(f"Turn {turn}, surviving particles: {particles._num_active_particles}")

        line.track(particles, num_turns=1)

        if particles._num_active_particles == 0:
            print(f'All particles lost by turn {turn}, teminating.')
            break
    particles.remove_unused_space()
    
    line.scattering.disable()
    line.enable_time_dependent_vars = False

    xc.geant4.engine.stop()
    #print(f'Tracking {num_particles} turns done in: {time.time()-t0} s')

    impacts.stop()
    # Saving impacts table
    df = impacts.to_pandas()
    df.to_csv(os.path.join(output_dir,'impacts_line.csv'), index=False)
    
    aper_interp = config_dict['collimation_parameters']['aperture_interp']
    # Make loss map
    weights = config_dict['collimation_parameters'].get('weights', 'none')

    line.cycle(name_first_element='beambeam_ipa.1', inplace=True)

    df_line = line.to_pandas()
    beam_start_index= df_line[df_line['name'] == config_dict['study_parameters']['start_element']].index[0]
    max_index = df_line.index.max()
    beam_start_s = df_line[df_line['name'] ==  config_dict['study_parameters']['start_element']]['s'].values[0]
    s_max = df_line['s'].max()

    particles = particles.remove_unused_space()  

    particles.s = (particles.s + beam_start_s) % s_max 
    particles.at_element = ((particles.at_element + beam_start_index) % max_index).astype(int)

    file_path = os.path.join(output_dir, f'lossmap_full.json')

    if weights == 'energy':
        part_mass_ratio = particles.charge_ratio / particles.chi
        part_mass = part_mass_ratio * particles.mass0
        p0c = particles.p0c
        f_x = lambda x: 1  # No modification for x
        f_px = lambda px: 1  # No modification for px
        f_y = lambda y: 1  # No modification for y
        f_py = lambda py: 1  # No modification for py
        f_zeta = lambda zeta: 1 
        f_delta = lambda delta: np.sqrt(((delta + 1) * p0c * part_mass_ratio)**2 + part_mass**2)
        weight_function = [f_x, f_px, f_y, f_py, f_zeta, f_delta]
    else:
        weight_function = None

    LossMap_full = xc.LossMap(line=line,
                            part=particles,
                            line_is_reversed=False,
                            interpolation=aper_interp,
                            weights=None,
                            weight_function=weight_function
                            )
    LossMap_full.to_json(file_path)
    print(f'Lossmap for all turns saved to {file_path}')

    output_file = os.path.join(output_dir, f"part.hdf")
    _save_particles_hdf(
        particles, lossmap_data=None, filename=output_file)
    
def madx_add_apertures (madx, aperture_dir='fcc-ee-lattice-V25_GHC/aperture', sequence_name='fccee_p_ring'):

    # ! --------------------------------------------------------------------------------------------------
    # ! Aperture installation scriptand install them in the MAD-X sequence.
    # ! Assuming the aperture definition file is in the lattice repository and kept up to date by the collimation team
    # ! --------------------------------------------------------------------------------------------------
    
    aperture_file_path = os.path.join(aperture_dir, 'FCCee_aper_definitions.madx')
    
    madx.input('''option,update_from_parent=true;''')
    madx.input(f'''sequence_name={sequence_name};''')
    madx.call(f"{aperture_file_path}")
    madx.use(sequence=sequence_name)

    return

def madx_add_collimators (madx, aperture_dir='fcc-ee-lattice-V25_GHC/aperture', sequence_name='fccee_p_ring'):

    # INSTALL COLLIMATORS IN PREDEFINED LOCATION (HALO/MOM/SYNCR)
    # NOTE WE EXPECT TO HAVE UPLOADED IN THE LATTICE DIRECTORY THE install_coll_*.madx to the aperture directory 
    # NOTE: FOR COLLIMATION STUDY THEY SHOULD BE IMPLEMENTED AS COLLIMATOR, OTHERWISE MARKERS ARE SUFFICIENT
    # ! --------------------------------------------------------------------------------------------------
    # ! Collimators(HALO+MOM+SR) installation script and install them in the MAD-X sequence.
    # ! Install as COLLIMATOR to have the correct apertures when translate into xtrack line
    # ! --------------------------------------------------------------------------------------------------
    
    madx_files = [
        'install_primary_secondary_collimators.madx',
        'install_tertiary_collimators.madx',
        'install_synchrotron_radiation_collimators.madx',
        'install_synchrotron_radiation_mask.madx'
    ]

    for madx_file in madx_files:
        file_path = os.path.join(aperture_dir, madx_file)

        # Read and replace {SEQUENCE_NAME} with the actual sequence name
        with open(file_path, 'r') as file:
            madx_script = file.read().replace("{SEQUENCE_NAME}", sequence_name)

        # Pass the modified script to MAD-X
        madx.input(madx_script)

    madx.use(sequence=sequence_name)

    return

def check_and_insert_aperture(line_thick, line_thin):
    """
    Check if there are any elements in the thin lattice without apertures and
    insert apertures from the thick lattice if needed.

    Parameters
    ----------
    line_thick: Line object for thick lattice
    line_thin: Line object for thin lattice
    """
    # Check which elements in the thin lattice have no aperture
    aper = line_thin.check_aperture()

    if aper.has_aperture_problem.any():
        # Get the elements with no aperture
        aper_miss = aper[aper['has_aperture_problem']].copy()
        new_s_values = aper_miss['s'].values
        tab = line_thick.get_table()

        # Extract aperture values from thick lattice
        apers = tab.rows[tab.element_type == 'LimitEllipse'].s
        aper_x = [line_thick[nn].a for nn, ee in zip(tab.name, tab.element_type) if ee.startswith('LimitEllipse')]
        aper_y = [line_thick[nn].b for nn, ee in zip(tab.name, tab.element_type) if ee.startswith('LimitEllipse')]

        # Interpolate apertures values along the ring
        sorted_indices = np.argsort(apers)
        aper_miss.loc[:, 'a'] = np.interp(new_s_values, apers[sorted_indices], np.array(aper_x)[sorted_indices])
        aper_miss.loc[:, 'b'] = np.interp(new_s_values, apers[sorted_indices], np.array(aper_y)[sorted_indices])

        # Insert missing aperture elements
        for idx, row in aper_miss.iterrows():
            line_thin.insert_element(
                name=str(row['name']) + '_aper',  # Ensure string
                element=xt.LimitEllipse(a=row['a'], b=row['b']),
                at_s=row['s'],
                s_tol=1e-3
            )

        # Check aperture again
        new_aper = line_thin.check_aperture()
        if new_aper.has_aperture_problem.any():
            print("Aperture missing even after adding missing ones, please check the lattice...")
        else:
            print("Aperture missing have been added successfully")


def install_two_kick_injection_bump (line, injection_marker_name, injection_bump_height, phase_adv_left_of_inj_marker=0.25, method='6d'):
    """
    Installs a two-kick injection bump in the specified accelerator line.

    injection marker and inserts markers for these kicks. It then matches the
    kick angles to achieve the desired injection bump height at the marker.

    Parameters:
    - line: Line object where the injection bump is installed.
    - injection_marker_name: Name of the injection point marker where the injection bump should be centered.
    - injection_bump_height: Desired height of the injection bump at the marker.
    - phase_adv_left_of_inj_marker: Phase advance to the left of the injection marker (default is 0.25).

    The function iteratively searches for drifts suitable for placing the kicks
    by adjusting phase advances. It then calculates the required kick angles and
    inserts the kick elements at the appropriate positions.
    """

    tw0 = line.twiss(method=method)
    tt0 = line.get_table(attr=True)
    tt0_use = tt0.rows[
        (tt0.element_type == 'Drift') | ((tt0.element_type == 'Marker') & (tt0.name == injection_marker_name))
        ]
    index = tt0_use.name.tolist().index(injection_marker_name)
    index_left = index - 1
    index_right = index + 1
    phase_advance_left = phase_adv_left_of_inj_marker
    phase_advance_right = 0.5-phase_advance_left
    s_shift_left = None
    s_shift_right = None

    def drift_phase_advance_x(index, length=None): 
        if length is not None:
            return np.arctan2(1, tw0.rows[tt0_use.rows[index].name].betx/length - tw0.rows[tt0_use.rows[index].name].alfx)/(2*np.pi)
        else:
            if tt0_use.rows[index].length[0] == 0:
                return 0
            else:
                return np.arctan2(1, tw0.rows[tt0_use.rows[index].name].betx/tt0_use.rows[index].length[0] - tw0.rows[tt0_use.rows[index].name].alfx)/(2*np.pi)
    length_shift = lambda index, phase_advance: tw0.rows[tt0_use.rows[index].name].betx/(1/np.tan(phase_advance*2*np.pi)+tw0.rows[tt0_use.rows[index].name].alfx)
    delta_mux_func = lambda l_index, r_index : tw0.rows[tt0_use.rows[r_index].name].mux-tw0.rows[tt0_use.rows[l_index].name].mux

    jj=0
    exit=False
    repeat=False
    while jj<2:

        while s_shift_left is None:

            delta_mux = delta_mux_func(index_left, index)
            drift_phase_adv = drift_phase_advance_x(index_left)
            if delta_mux >= phase_advance_left: 
                if delta_mux-drift_phase_adv <= phase_advance_left:
                    s_shift_left = length_shift(index_left,delta_mux-phase_advance_left)
                    jj+=1

                    while s_shift_right is None:

                        delta_mux = delta_mux_func(index, index_right)
                        drift_phase_adv = drift_phase_advance_x(index_right)
                        if delta_mux+drift_phase_adv >= phase_advance_right:
                            if delta_mux <= phase_advance_right: 
                                s_shift_right = length_shift(index_right,phase_advance_right-delta_mux)
                                jj+=1
                            else:
                                index_left = index - 1
                                index_right = index + 1
                                if phase_advance_left + 0.01 <= 0.45:
                                    phase_advance_left = phase_advance_left + 0.00001
                                else:
                                    repeat = True
                                    exit = True
                                    print('Not possible to find requited possition for left and right kicks, search stoped!')
                                    break
                                phase_advance_right = 0.5-phase_advance_left 
                                s_shift_left = None
                                s_shift_right = None
                                repeat = True
                                jj-=1
                                break
                        else:
                            index_right += 1

                    if repeat:
                        break

                else:
                    index_left = index - 1
                    index_right = index + 1
                    if phase_advance_left + 0.01 <= 0.45:
                        phase_advance_left = phase_advance_left + 0.00001
                    else:
                        exit = True
                        print('Not possible to find requited possition for left and right kicks, search stoped!')
                        break
                    phase_advance_right = 0.5-phase_advance_left
                    s_shift_left = None
                    s_shift_right = None
                    break
            else:
                index_left -= 1

        if exit:
            break

        if jj==2:
            print(f'phase_advance_left conclude to {phase_advance_left}')
            print(f'phase_advance_right conclude to {phase_advance_right}')

        
    drift_name_left = tt0_use.rows[index_left].name
    drift_name_right = tt0_use.rows[index_right].name

    if s_shift_left is not None and s_shift_right is not None:
        line.discard_tracker()
        line.insert_element('kick_marker_left', xt.Marker(), at_s=s_shift_left[0]+tt0_use.rows[drift_name_left].s[0])
        line.insert_element('kick_marker_right', xt.Marker(), at_s=s_shift_right[0]+tt0_use.rows[drift_name_right].s[0])
    
        line.build_tracker()
        tw1 = line.twiss(method=method)
        line.discard_tracker()

        line.vars['injection_bump.knob'] = 1
        
        line.vars['kick_angle_left'] = 1e-7
        line.vars['kick_angle_right'] = line.vars['kick_angle_left'] * np.sqrt(tw1.rows['kick_marker_left'].betx/tw1.rows['kick_marker_right'].betx)[0]

        line.insert_element('injection_bump_kick_left', xt.Multipole(knl = [0,]) , at='kick_marker_left')
        line['injection_bump_kick_left'].knl = 'injection_bump.knob*kick_angle_left'
        line.insert_element('injection_bump_kick_right', xt.Multipole(knl = [0,]) , at='kick_marker_right')
        line['injection_bump_kick_right'].knl = 'injection_bump.knob*kick_angle_right'
        line.build_tracker()
        tw2 = line.twiss(method=method)

        opt_kick = line.match(
                method=method,
                start='injection_bump_kick_left', end='kick_marker_right',
                init=tw2, init_at='injection_bump_kick_left',
                vary=[xt.VaryList(['kick_angle_left'], step=1e-8, tag='kick_left'),
                    ],
                targets=[xt.TargetSet(x=injection_bump_height, at=injection_marker_name, tol=1e-8, tag='bump_hight'),
                    ])

        line.vars['injection_bump.knob'] = 0
        line.discard_tracker()
        line.remove('kick_marker_left')
        line.remove('kick_marker_right')
        line.build_tracker()

    return


def activate_dynamic_injection_bump(line, inject_param, twiss, peak_turn=10, dt=0):
    """
    Control the bumpers with a modulated piecewise function of time.

    The pulse is defined by the following parameters:
    - the risetime and falltime of the pulse
    - the turn at which the peak of the pulse has to be reached
    - the duration of the flat top

    The pulse is defined by a piecewise linear function of time.
    The function has the following points:
    - t=0: knob=0
    - t=peak_time - left_kick_dt - risefall_t/2: knob=0.5
    - t=peak_time - left_kick_dt: knob=1
    - t=peak_time - left_kick_dt + flattop_t: knob=1
    - t=peak_time - left_kick_dt + flattop_t + risefall_t/2: knob=0.5
    - t=peak_time - left_kick_dt + flattop_t + risefall_t: knob=0

    Parameters
    ----------
    line: line object
    inject_param: injection parameters dictionary
    twiss: twiss parameters
    peak_turn: the turn number of the peak of the pulse
    dt: time shift
    """

    # Constants
    rev_t_s = twiss.T_rev0  # revolution time in seconds
    bump_risefall_dt_s = inject_param['bump_risefall_time']  # risetime and falltime of the pulse in seconds
    bump_flattop_dt_s = inject_param['bump_flattop_time']  # duration of the flat top in seconds

    # Calculate the time difference between needed 
    start_to_inj_point_dt_s = (twiss.rows[inject_param['injection_marker_name']].s*rev_t_s/twiss.circumference)[0]
    left_kick_dt_s = ((twiss.rows[inject_param['injection_marker_name']].s - twiss.rows['injection_bump_kick_left'].s)*rev_t_s/twiss.circumference)[0]
    right_kick_dt_s = ((twiss.rows['injection_bump_kick_right'].s - twiss.rows[inject_param['injection_marker_name']].s)*rev_t_s/twiss.circumference)[0]

    # Calculate the time of the peak of the pulse from twiss start
    peak_time_s = peak_turn * rev_t_s + start_to_inj_point_dt_s + dt

    # Create the piecewise linear function
    line.functions['bump_pulse'] = xt.FunctionPieceWiseLinear(
        x=np.array([peak_time_s - left_kick_dt_s - bump_risefall_dt_s, # 0  
                    peak_time_s - left_kick_dt_s - bump_risefall_dt_s / 2, # 0.5
                    peak_time_s - left_kick_dt_s , #1
                    peak_time_s, #1
                    peak_time_s + right_kick_dt_s , #1
                    peak_time_s + bump_flattop_dt_s, #1
                    peak_time_s + bump_flattop_dt_s + bump_risefall_dt_s / 2, # 0.5
                    peak_time_s + bump_flattop_dt_s + bump_risefall_dt_s # 0
                    ]), # s
        y=np.array([0,     
                    0.5,              
                    1,                    
                    1,
                    1,
                    1,
                    0.5,
                    0])         # knob value
    )

    # Set the knob value of the injection bump to the value of the pulse
    line.vars['injection_bump.knob'] = line.functions['bump_pulse'](line.vars['t_turn_s'])

def find_apertures(line):
    i_apertures = []
    apertures = []
    for ii, ee in enumerate(line.elements):
        if ee.__class__.__name__.startswith('Limit'):
            i_apertures.append(ii)
            apertures.append(ee)
    return np.array(i_apertures), np.array(apertures)

def find_bb_lenses(line):
    i_apertures = []
    apertures = []
    for ii, ee in enumerate(line.elements):
        if ee.__class__.__name__.startswith('BeamBeamBiGaussian3D'):
            i_apertures.append(ii)
            apertures.append(ee)
    return np.array(i_apertures), np.array(apertures)

def insert_bb_lens_bounding_apertures(line):
    # Place aperture defintions around all beam-beam elements in order to ensure
    # the correct functioning of the aperture loss interpolation
    # the aperture definitions are taken from the nearest neighbour aperture in the line
    s_pos = line.get_s_elements(mode='upstream')
    apert_idx, apertures = find_apertures(line)
    apert_s = np.take(s_pos, apert_idx)

    bblens_idx, bblenses = find_bb_lenses(line)
    bblens_names = np.take(line.element_names, bblens_idx)
    bblens_s_start = np.take(s_pos, bblens_idx)
    bblens_s_end = np.take(s_pos, bblens_idx + 1)

    # Find the nearest neighbour aperture in the line
    bblens_apert_idx_start = np.searchsorted(apert_s, bblens_s_start, side='left')
    bblens_apert_idx_end = bblens_apert_idx_start + 1

    aper_start = apertures[bblens_apert_idx_start]
    aper_end = apertures[bblens_apert_idx_end]

    idx_offset = 0
    for ii in range(len(bblenses)):
        line.insert_element(name=bblens_names[ii] + '_aper_start',
                            element=aper_start[ii].copy(),
                            at=bblens_idx[ii] + idx_offset)
        idx_offset += 1

        line.insert_element(name=bblens_names[ii] + '_aper_end',
                            element=aper_end[ii].copy(),
                            at=bblens_idx[ii] + 1 + idx_offset)
        idx_offset += 1

def tracking_data_process (tracking_data, monitor_twiss=None, norm_emit_x=None, norm_emit_y=None, sigma_z=None, sigma_delta=None, particle_id_to_use='all'):
    
    # Create set of particles to use
    if particle_id_to_use == 'all':
        part_mask = np.any(tracking_data.state == 1, axis=1)
    elif particle_id_to_use == 'survived':
        part_mask = np.all(tracking_data.state == 1, axis=1)
    elif particle_id_to_use == 'lost':
        part_mask = np.any(tracking_data.state == 0, axis=1)
    elif isinstance(particle_id_to_use,(list, np.ndarray)):
        part_mask = np.all(np.isin(tracking_data.particle_id, particle_id_to_use), axis=1)

    # After a particle is lost, a zero value is used therefor,
    # a mask with these zero locations is constracted so to set them to nan
    turns = tracking_data.at_turn[part_mask]
    turns = turns.astype(float)
    turns_mask = (tracking_data.state[part_mask] == 0)
    turns[turns_mask] = np.nan

    # Used coordinates
    x_cor = tracking_data.x[part_mask]
    x_cor[turns_mask] = np.nan
    px_cor = tracking_data.px[part_mask]
    px_cor[turns_mask] = np.nan
    y_cor = tracking_data.y[part_mask]
    y_cor[turns_mask] = np.nan
    py_cor = tracking_data.py[part_mask]
    py_cor[turns_mask] = np.nan
    z_cor = tracking_data.zeta[part_mask]
    z_cor[turns_mask] = np.nan
    delta_cor = tracking_data.delta[part_mask]
    delta_cor[turns_mask] = np.nan

    if monitor_twiss is not None:

        sigmas = monitor_twiss.get_betatron_sigmas(nemitt_x=norm_emit_x, nemitt_y=norm_emit_y)

        geom_emit_x = norm_emit_x/(monitor_twiss.beta0*monitor_twiss.gamma0)
        x_s = x_cor/sigmas['sigma_x',0]
        x_n = x_cor/np.sqrt(monitor_twiss.betx[0])
        px_s = px_cor/sigmas['sigma_px',0]
        px_n = x_n*monitor_twiss.alfx[0] + px_cor*np.sqrt(monitor_twiss.betx[0])
        act_x = (x_n**2 + px_n**2)/2
        phi_x = np.arctan2(-px_n, x_n)

        geom_emit_y = norm_emit_y/(monitor_twiss.beta0*monitor_twiss.gamma0)
        y_s = y_cor/sigmas['sigma_y',0]
        y_n = y_cor/np.sqrt(monitor_twiss.bety[0])
        py_s = py_cor/sigmas['sigma_py',0]
        py_n = y_n*monitor_twiss.alfy[0] + py_cor*np.sqrt(monitor_twiss.bety[0])
        act_y = (y_n**2 + py_n**2)/2
        phi_y = np.arctan2(-py_n, y_n)

        z_s = z_cor/sigma_z
        delta_s = delta_cor/sigma_delta
    
    else:
        x_s = np.zeros(np.shape(x_cor))*np.nan
        x_n = np.zeros(np.shape(x_cor))*np.nan
        px_s = np.zeros(np.shape(x_cor))*np.nan
        px_n = np.zeros(np.shape(x_cor))*np.nan
        act_x = np.zeros(np.shape(x_cor))*np.nan
        phi_x = np.zeros(np.shape(x_cor))*np.nan
        y_s = np.zeros(np.shape(x_cor))*np.nan
        y_n = np.zeros(np.shape(x_cor))*np.nan
        py_s = np.zeros(np.shape(x_cor))*np.nan
        py_n = np.zeros(np.shape(x_cor))*np.nan
        act_y = np.zeros(np.shape(x_cor))*np.nan
        phi_y = np.zeros(np.shape(x_cor))*np.nan
        z_s = np.zeros(np.shape(x_cor))*np.nan
        delta_s = np.zeros(np.shape(x_cor))*np.nan


    sigma_matrix, emittances = sigma_matrix_and_emittances_from_tracking ({'x':x_cor,
                                                    'px':px_cor,
                                                    'y':y_cor,
                                                    'py':py_cor,
                                                    'zeta':z_cor,
                                                    'delta':delta_cor})

    x_s_dist_evol = x_cor/np.sqrt(sigma_matrix[0,0,:])
    x_n_dist_evol = x_s_dist_evol*np.sqrt(emittances[0,:])
    px_s_dist_evol = px_cor/np.sqrt(sigma_matrix[1,1,:])
    px_n_dist_evol = -x_n_dist_evol*sigma_matrix[0,1,:]/emittances[0,:] + px_cor*np.sqrt(sigma_matrix[0,0,:]/emittances[0,:])
    act_x_dist_evol = (x_n_dist_evol**2 + px_n_dist_evol**2)/2
    phi_x_dist_evol = np.arctan2(-px_n_dist_evol, x_n_dist_evol)

    y_s_dist_evol = y_cor/np.sqrt(sigma_matrix[2,2,:])
    y_n_dist_evol = y_s_dist_evol*np.sqrt(emittances[2,:])
    py_s_dist_evol = py_cor/np.sqrt(sigma_matrix[3,3,:])
    py_n_dist_evol = -y_n_dist_evol*sigma_matrix[2,3,:]/emittances[2,:] + py_cor*np.sqrt(sigma_matrix[2,2,:]/emittances[2,:])
    act_y_dist_evol = (y_n_dist_evol**2 + py_n_dist_evol**2)/2
    phi_y_dist_evol = np.arctan2(-py_n_dist_evol, y_n_dist_evol)

    z_s_dist_evol = z_cor/np.sqrt(sigma_matrix[4,4,:])
    delta_s_dist_evol = delta_cor/np.sqrt(sigma_matrix[5,5,:])

    particle_id_used = tracking_data.particle_id[part_mask][:,0]

    state_org = tracking_data.state

    trackind_data_postprocessed = {
                'x':x_cor,
                'x_sigma':x_s,
                'nx':x_n,
                'px':px_cor,
                'px_sigma':px_s,
                'npx':px_n,
                'action_x':act_x,
                'phi_x':phi_x,
                'y':y_cor,
                'y_sigma':y_s,
                'ny':y_n,
                'py':py_cor,
                'py_sigma':py_s,
                'npy':py_n,
                'action_y':act_y,
                'phi_y':phi_y,
                'zeta':z_cor,
                'zeta_sigma':z_s,
                'delta':delta_cor,
                'delta_sigma':delta_s,
                'at_turn':turns,
                'state':state_org,
                'particle_id_list':particle_id_used,
                
                'x_sigma_using_distribution_evolution':x_s_dist_evol,
                'nx_using_distribution_evolution':x_n_dist_evol,
                'px_sigma_using_distribution_evolution':px_s_dist_evol,
                'npx_using_distribution_evolution':px_n_dist_evol,
                'action_x_using_distribution_evolution':act_x_dist_evol,
                'phi_x_using_distribution_evolution':phi_x_dist_evol,
                'y_sigma_using_distribution_evolution':y_s_dist_evol,
                'ny_using_distribution_evolution':y_n_dist_evol,
                'py_sigma_using_distribution_evolution':py_s_dist_evol,
                'npy_using_distribution_evolution':py_n_dist_evol,
                'action_y_using_distribution_evolution':act_y_dist_evol,
                'phi_y_using_distribution_evolution':phi_y_dist_evol,
                'zeta_sigma_using_distribution_evolution':z_s_dist_evol,
                'delta_sigma_using_distribution_evolution':delta_s_dist_evol,
                'emit_x':emittances[0,:],
                'emit_px':emittances[1,:],
                'emit_y':emittances[2,:],
                'emit_py':emittances[3,:],
                'emit_zeta':emittances[4,:],
                'emit_delta':emittances[5,:],
                'sigma_matrix':sigma_matrix}
    
    return(trackind_data_postprocessed)

# # Sigma matrix from data
def sigma_matrix_and_emittances_from_tracking (dic_coordinates):
    antisimmetric_matrix = np.zeros((6,6))
    antisimmetric_matrix[0,1] = 1
    antisimmetric_matrix[1,0] = -1
    antisimmetric_matrix[2,3] = 1
    antisimmetric_matrix[3,2] = -1
    antisimmetric_matrix[4,5] = 1
    antisimmetric_matrix[5,4] = -1

    x = dic_coordinates['x']
    px = dic_coordinates['px']
    y = dic_coordinates['y']
    py = dic_coordinates['py']
    z = dic_coordinates['zeta']
    dd = dic_coordinates['delta']

    turns = len(x[0,:])
    n_part = len(x[:,0])

    # Store covariance matrices for each turn
    cov_matrices = []

    # Loop over turns
    for t in range(turns):
        # Extract data for this turn (shape: n_part × 6)
        data_turn = np.column_stack((x[:, t], px[:, t], y[:, t], py[:, t], z[:, t], dd[:, t]))

        # Remove rows with NaN values
        data_turn = data_turn[~np.isnan(data_turn).any(axis=1)]

        # Compute covariance matrix only if data remains
        if data_turn.shape[0] > 1:
            cov_matrix = np.cov(data_turn, rowvar=False)
            # Adjust the covariance matrix by multiplying by (n-1)/n to replace Bessel's correction
            cov_matrix *= (data_turn.shape[0] - 1) / data_turn.shape[0]
        else:
            cov_matrix = np.full((6, 6), np.nan)  # If no valid data, fill with NaNs

        # Store covariance matrix
        cov_matrices.append(cov_matrix)

    # Convert to numpy array with shape (6, 6, n_turn)
    cov_matrices = np.array(cov_matrices).transpose(1, 2, 0)
    
    sigma = cov_matrices

    # Initialize empty arrays to store the result
    
    sigma_dot_antisimmetric = np.zeros((6, 6, turns))*np.nan
    emittances = np.zeros((6, turns))*np.nan

    # Perform matrix manipulations for each slice
    for ii in range(turns):
        if np.isnan(sigma[:, :, ii]).any() or len(x[:,ii][~np.isnan(x[:,ii])])<119:
            print('Not enough particles (poor statistics) for acurate calculation of the sigma matrix and emittances therefore, are set to Nan !')
            break
        
        sigma_dot_antisimmetric[:, :, ii] = np.dot(sigma[:, :, ii], antisimmetric_matrix)
        eigenvalues, eigenvectors = np.linalg.eig(sigma_dot_antisimmetric[:, :, ii])
        
        max_indices = np.argmax(np.abs(eigenvectors), axis=0)
        
        # restor of the corrext order (x,y,z) of the emittances 
        for jj in range(3):
            
            if 2*jj in max_indices:
                indice = np.where(max_indices == 2*jj)[0][0]
                emittances[2*jj, ii] = np.abs(1j*eigenvalues[indice])
            elif 2*jj+1 in max_indices:
                indice = np.where(max_indices == 2*jj+1)[0][0]
                emittances[2*jj, ii] = np.abs(1j*eigenvalues[indice])

            if 2*jj+1 in max_indices:
                indice = np.where(max_indices == 2*jj+1)[0][0]
                emittances[2*jj+1, ii] = np.abs(1j*eigenvalues[indice])
            elif 2*jj in max_indices:
                indice = np.where(max_indices == 2*jj)[0][0]
                emittances[2*jj+1, ii] = np.abs(1j*eigenvalues[indice])
    
    return (cov_matrices, emittances)


def nafflib_tune_calculation (q_coordinates, pq_coordinates=None, particle_id_list=None, number_harmonics=1, window_order=1, data_skipped_from_start=0, remove_nan_from_data='no'): 

    # Dataframe input
    results = np.zeros((np.shape(q_coordinates)[0],number_harmonics*5+1))*np.nan
    columns = ['particle_id']
    for ii in range(number_harmonics):
        columns = columns + [f'Q{ii+1}', f'A{ii+1}', f'Phase{ii+1}', f'Re{ii+1}', f'Im{ii+1}']
    
    jj=0
    for ii in range(np.shape(q_coordinates)[0]):
        
        if particle_id_list is not None:
            results[jj,0] = particle_id_list[ii]
        else:
            results[jj,0] = ii

        if remove_nan_from_data == 'yes':
            q_data = q_coordinates[ii,:][~np.isnan(q_coordinates[ii,:])]
            if pq_coordinates is not None:
                pq_data = pq_coordinates[ii,:][~np.isnan(pq_coordinates[ii,:])]
                q_ef_data = q_data - 1j*pq_data - np.mean(q_data - 1j*pq_data)
            else:
                q_ef_data = q_data - np.mean(q_data)
        
        elif remove_nan_from_data == 'no':
            if pq_coordinates is not None:
                q_ef_data = q_coordinates[ii,:] - 1j*pq_coordinates[ii,:] - np.mean(q_coordinates[ii,:] - 1j*pq_coordinates[ii,:])
            else:
                q_ef_data = q_coordinates[ii,:] - np.mean(q_coordinates[ii,:])
        
        A, Q = pnf.harmonics(q_ef_data[data_skipped_from_start:], num_harmonics=number_harmonics, window_order=window_order, window_type="hann")

        for kk in range(number_harmonics):
            if number_harmonics==1:
                results[jj,1+5*kk] = np.abs(Q[kk])
            else:
                results[jj,1+5*kk] = Q[kk]
            results[jj,2+5*kk] = np.abs(A[kk])
            results[jj,3+5*kk] = np.angle(A[kk])
            results[jj,4+5*kk] = np.real(A[kk])
            results[jj,5+5*kk] = np.imag(A[kk])

        jj+=1
        print(ii,'/',np.shape(q_coordinates)[0])
        
    df_table = pd.DataFrame(results,columns=columns)
    df_table['particle_id'] = df_table['particle_id'].astype(int)
    
    return(df_table)


def naffuv_tune_calculation (q_coordinates, pq_coordinates=None, particle_id_list=None, number_harmonics=1, remove_nan_from_data='no', data_type='c', data_reading_step=1, sampling_frequency=1.0, 
                             frequency_window='han_1', amplitude_window='hft', modified_GS=1):
        
    control_file_input = ['input_data.txt', data_type, 'full', '0', '500', str(int(data_reading_step)), 
                          str(float(data_reading_step)), str(int(number_harmonics)), '0', 
                          frequency_window, amplitude_window, str(int(modified_GS)), 'freqiencies_and_amplitudes.txt']

    with open("naff_uv_control_file.txt", "r") as control_file:
        control_file_lines = control_file.readlines()
    jj=0
    for ii in range(len(control_file_lines)):
        if control_file_lines[ii][0] == '#':
            continue
        else:
            control_file_lines[ii] = control_file_input[jj]+'\n'
            jj+=1
    control_file = open("naff_uv_control_file.txt", "w")
    control_file.writelines(control_file_lines)
    control_file.close()

    # Dataframe input
    results = np.zeros((np.shape(q_coordinates)[0],number_harmonics*4+1))*np.nan
    columns = ['particle_id']
    for ii in range(number_harmonics):
        columns = columns + [f'Q{ii+1}', f'A{ii+1}', f'Re[A{ii+1}]', f'Im[A{ii+1}]']
    
    jj=0
    for ii in range(np.shape(q_coordinates)[0]):
        
        if particle_id_list is not None:
            results[jj,0] = particle_id_list[ii]
        else:
            results[jj,0] = ii       

        if remove_nan_from_data == 'yes':
            q_data = q_coordinates[ii,:][~np.isnan(q_coordinates[ii,:])]
            if pq_coordinates is not None:
                pq_data = pq_coordinates[ii,:][~np.isnan(pq_coordinates[ii,:])]
        elif remove_nan_from_data == 'no':
            q_data = q_coordinates[ii,:]
            if pq_coordinates is not None:
                pq_data = pq_coordinates[ii,:]
        
        if data_type == 'c':
            q_ef_data = np.array([q_data,-pq_data]).T
        elif data_type == 'r':
            q_ef_data = q_data
        else:
            print('Not supported data_type value in naffuv_tune_calculation')
        
        np.savetxt('input_data.txt', q_ef_data, fmt='%1.16e')
        os.system('./abc')
        tunes_amps_x = np.loadtxt('freqiencies_and_amplitudes.txt')
        os.system('rm -rf input_data.txt freqiencies_and_amplitudes.txt')
        
        
        if number_harmonics == 1:
            tunes_amps = tunes_amps.reshape(number_harmonics,len(tunes_amps))
        else:
            tunes_amps = tunes_amps[tunes_amps[:,2].argsort()[::-1],:]
        
        for kk in range(number_harmonics):
            if kk+1 <= np.shape(tunes_amps)[0]:
                if tunes_amps[kk,1] > 0.5:
                    results[jj,1+4*kk] = np.abs(tunes_amps[kk,1] - 1)
                else:
                    results[jj,1+4*kk] = np.abs(tunes_amps[kk,1])
                results[jj,2+4*kk:5+4*kk] = tunes_amps[kk,2:]

        jj+=1
        print(ii,'/',np.shape(q_coordinates)[0])
        
    df_table = pd.DataFrame(results,columns=columns)
    df_table['particle_id'] = df_table['particle_id'].astype(int)
    
    return(df_table)

# Define the Gaussian function
def gaussian(dd, mu, sigma):
    return (1/np.sqrt(2*np.pi*sigma**2)) * np.exp(-0.5*((dd-mu)/sigma)**2)

# Define the q-Gaussian function
def q_gaussian_pdf(dd, mu, beta, q):
    pdf = np.sqrt(beta)*np.maximum((1+(q-1)*beta*(dd-mu)**2),0)**(1/(1-q))
    if q < 1:
        norm_fuctor = 2*np.sqrt(np.pi)*gamma(1/(1-q))/((3-q)*np.sqrt(1-q)*gamma((3-q)/(2-2*q)))
    elif q > 1:
        norm_fuctor = np.sqrt(np.pi)*gamma((3-q)/(2-2*q))/(np.sqrt(q-1)*gamma(1/(q-1)))
    return (pdf/norm_fuctor)


# # Fit Gaussian or q-Gaussian function to the histogram data
def fit_distribution(data, q):
    hist, bins = np.histogram(data, bins=100, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    #For Gaussian
    if q==1:
        def error_func(params, dd, tar):
            mu, sigma = params
            return (gaussian(dd, mu, sigma) - tar)
        result = opt.least_squares(error_func, x0=[0, 1], bounds=([-np.inf, 0], [np.inf, np.inf]), args=(bin_centers, hist), loss='soft_l1')
    
    #For q-Gaussian
    else:
        def error_func(params, dd, tar):
            mu, beta, q = params
            return (q_gaussian_pdf(dd, mu, beta, q) - tar)

        if q > 1:
            result = opt.least_squares(error_func, x0=[0, 1, q], bounds=([-np.inf, 0, 1], [np.inf, np.inf, 3]), args=(bin_centers, hist), loss='soft_l1')
        elif q < 1:
            result = opt.least_squares(error_func, x0=[0, 1, q], bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, 1]), args=(bin_centers, hist), loss='soft_l1')
    
    #popt_x_q = fit_distribution(x,1.1)
    #popt_y_q = fit_distribution(y,1.1)

    return (result.x)

def save_track_along_ring_to_h5(monitor, output_dir="plots"):
    """
    Saves monitor data (assumed shape: (num_part, 201358)) to an HDF5 file.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, f'merged_data_along_ring.h5')

    try:
        with h5py.File(file_path, 'w') as h5f:
            # Extract and store relevant properties
            for attr in ['x', 'px', 'y', 'py', 'zeta', 'delta', 's', 'at_turn']:
                data = getattr(monitor, attr, None)
                if data is not None:
                    dtype = np.float32 if data.dtype.kind in 'fc' else np.int32
                    h5f.create_dataset(attr, data=data.astype(dtype), compression="gzip", chunks=True)

        print(f"Data successfully saved to {file_path}")
    except Exception as e:
        print(f"Failed to save data to {file_path}: {e}")

@contextmanager
def set_directory(path: Path):
    """
    Taken from: https://dev.to/teckert/changing-directory-with-a-python-context-manager-2bj8
    """
    origin = Path().absolute()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)

def resolve_and_cache_paths(iterable_obj, resolved_iterable_obj, cache_destination):
    if isinstance(iterable_obj, (dict, list)):
        for k, v in (iterable_obj.items() if isinstance(iterable_obj, dict) else enumerate(iterable_obj)):
            possible_path = Path(str(v))
            if not isinstance(v, (dict, list)) and possible_path.exists() and possible_path.is_file():
                shutil.copy(possible_path, cache_destination)
                resolved_iterable_obj[k] = possible_path.name
            resolve_and_cache_paths(v, resolved_iterable_obj[k], cache_destination)

def dump_dict_to_yaml(dict_obj, file_path):
        with open(file_path, 'w') as yaml_file:
            clean_dict_to_save = convert_types(dict_obj)
            yaml.dump(clean_dict_to_save, yaml_file, default_flow_style=False, sort_keys=False, allow_unicode=True)

def submit_jobs(config_dict, config_file):
    # Relative path from the config file should be relative to
    # the file itself, not to where the script is executed from
    if config_file:
        conf_path = Path(config_file).resolve()
        conf_dir = conf_path.parent
        conf_fname = conf_path.name
    else:
        conf_dir = Path().resolve()
        conf_fname = 'configuration.yaml' #'config_collimation.yaml'
        conf_path = Path(conf_dir, conf_fname)
        
    with set_directory(conf_dir):
        #config_dict = CONF_SCHEMA.validate(config_dict)

        sub_dict = config_dict['jobsubmission']
        workdir = str(Path(sub_dict['working_directory']).resolve())
        num_jobs = sub_dict['num_jobs']
        replace_dict_in = sub_dict.get('replace_dict', {})
        executable = sub_dict.get('executable', 'bash')
        mask_abspath = str(Path(sub_dict['mask']).resolve())
        
        max_local_jobs = 10
        if sub_dict.get('run_local', False) and num_jobs > max_local_jobs:
            raise Exception(f'Cannot run more than {max_local_jobs} jobs locally,'
                            f' {num_jobs} requested.')
            
        # Make a directory to copy the files for the submission
        input_cache = Path(workdir, 'input_cache')
        os.makedirs(workdir)
        os.makedirs(input_cache)

        # Copy the files to the cache and replace the path in the config
        # Copy the configuration file
        if conf_path.exists():
            shutil.copy(conf_path, input_cache)
        else:
            # If the setup came from a dict a dictionary still dump it to archive
            dump_dict_to_yaml(config_dict, Path(input_cache, conf_path.name))
            
        exclude_keys = {'jobsubmission',} # The submission block is not needed for running
        # Preserve the key order
        reduced_config_dict = {k: config_dict[k] for k in 
                               config_dict.keys() if k not in exclude_keys}
        resolved_config_dict = copy.deepcopy(reduced_config_dict)
        resolve_and_cache_paths(reduced_config_dict, resolved_config_dict, input_cache)

        resolved_conf_file = f'for_jobs_{conf_fname}' # config file used to run each job
        dump_dict_to_yaml(resolved_config_dict, Path(input_cache, resolved_conf_file))

        # compress the input cache to reduce network traffic
        shutil.make_archive(input_cache, 'gztar', input_cache)
        # for fpath in input_cache.iterdir():
        #     fpath.unlink()
        # input_cache.rmdir()

        # Set up the jobs
        inv1 = config_dict['study_parameters']['inv1']
        inv2 = config_dict['study_parameters']['inv2']

        # Creating a dictionary with inv1 and inv2 values, instead of seed
        replace_dict_base = {'inv1': inv1,
                             'inv2': inv2,
                             'config_file': resolved_conf_file,
                             'input_cache_archive': str(input_cache) + '.tar.gz'}

        # Pass through additional replace dict option and other job_submitter flags
        if replace_dict_in:
            replace_dict = {**replace_dict_base, **replace_dict_in}
        else:
            replace_dict = replace_dict_base
        
        processed_opts = {'working_directory', 'num_jobs','executable', 'mask'}
        submitter_opts = list(set(sub_dict.keys()) - processed_opts)
        submitter_options_dict = { op: sub_dict[op] for op in submitter_opts }
        
        # Send/run the jobs via the job_submitter interface
        htcondor_submit(
            mask=mask_abspath,
            working_directory=workdir,
            executable=executable,
            replace_dict=replace_dict,
            **submitter_options_dict)

        print('Done!')

def _read_particles_hdf(filename):
    return pd.read_hdf(filename, key='particles')

def load_output(directory, job_output_dir, output_file, match_pattern='*part.hdf*',imax=None, load_lossmap=False, load_particles=False):

    t0 = time.time()

    job_dirs = glob.glob(os.path.join(directory, 'Job.*')
                         )  # find directories to loop over

    job_dirs_sorted = []
    for i in range(len(job_dirs)):
        # Very inefficient, but it sorts the directories by their numerical index
        job_dir_idx = job_dirs.index(
            os.path.join(directory, 'Job.{}'.format(i)))
        job_dirs_sorted.append(job_dirs[job_dir_idx])

    part_hdf_files = []
    part_dataframes = []
    lossmap_dicts = []
    
    tqdm_ncols=100
    tqdm_miniters=10
    print(f'Parsing directories...')
    dirs_visited = 0
    files_loaded = 0
    for i, d in tqdm(enumerate(job_dirs_sorted), total=len(job_dirs_sorted), 
                     ncols=tqdm_ncols, miniters=tqdm_miniters):
        if imax is not None and i > imax:
            break

        #print(f'Processing {d}')
        dirs_visited += 1
        output_dir = os.path.join(d, job_output_dir)
        output_files = glob.glob(os.path.join(output_dir, match_pattern))
        if output_files:
            of = output_files[0]
            part_hdf_files.append(of)
            files_loaded += 1
        else:
            print(f'No output found in {d}')

    part_merged = None
    if load_particles:
        print(f'Loading particles...')
        with Pool() as p:
            part_dataframes = list(tqdm(p.imap(_read_particles_hdf, part_hdf_files), total=len(part_hdf_files), 
                                        ncols=tqdm_ncols, miniters=tqdm_miniters))
        part_objects = [xp.Particles.from_pandas(pdf) for pdf in tqdm(part_dataframes, total=len(part_dataframes),
                                                                      ncols=tqdm_ncols, miniters=tqdm_miniters)]

        print('Particles load finished, merging...')
        part_merged = xp.Particles.merge(list(tqdm(part_objects, total=len(part_objects),
                                              ncols=tqdm_ncols, miniters=tqdm_miniters)))

    
    _save_particles_hdf(particles=part_merged,
                        lossmap_data=None, filename=output_file)

    print('Directories visited: {}, files loaded: {}'.format(
        dirs_visited, files_loaded))
    print(f'Processing done in {time.time() -t0} s')


def generate_bash_script(enviroment_source='/afs/cern.ch/work/k/kskoufar/public/Xutil_submission_tools/setup_env.sh', output_file="mask_jobsubmission.sh"):
    # Get the absolute path of this script's directory
    script_location = Path(__file__).resolve().parent

    # Find run_tracking.py in the same folder
    python_script_path = script_location / "run_tracking.py"
    if not python_script_path.exists():
        raise FileNotFoundError(f"{python_script_path} not found!")

    # Convert to absolute paths
    python_script_path = python_script_path.resolve()

    # Prepare environment source line (if provided)
    env_source_line = f"source {Path(enviroment_source).resolve()}\n" if enviroment_source else ""

    # Define the script content (FIX: Escaped Bash variables properly)
    script_content = f"""#!/usr/bin/env bash
    
{env_source_line}

PYTHON_SCRIPT={python_script_path}

# Mask variables - replaced by the script, should only be edited by experts
INPUT_ARCHIVE=%(input_cache_archive)s
INV1=%(inv1)s
INV2=%(inv2)s
CONFIGFILE=%(config_file)s

# Copy the input files to the node
cp $INPUT_ARCHIVE .
tar -xzvf *.tar.gz

#Convert inv1 and inv2 from lists to single values (fix YAML formatting)
sed -i -E "/^  inv1:/,/^  inv2:/ s/^  inv1:.*/  inv1: ${{INV1}}/" ${{CONFIGFILE}}
sed -i -E "/^  inv2:/,/^.+/ s/^  inv2:.*/  inv2: ${{INV2}}/" ${{CONFIGFILE}}

sed -i -E "/^  inv1:/,/^  inv2:/ {{/^  - /d}}" ${{CONFIGFILE}}
sed -i -E "/^  inv2:/,/^$/ {{/^  - /d}}" ${{CONFIGFILE}}

# Run the script
python $PYTHON_SCRIPT --config_file ${{CONFIGFILE}} > tracker_${{INV1}}_${{INV2}}.log
"""

    # Write the script safely
    temp_file = f"{output_file}.tmp"
    with open(temp_file, "w") as file:
        file.write(script_content)

    # Rename temp file to final output (atomic operation)
    Path(temp_file).rename(output_file)

    print(f"Bash script generated: {output_file}")


# Formulas used are from T. Argyropoulos PhD and his original code 
#TODO: use twiss insted of passining hard coded values
class LongParamsCalculator:

    def __init__(self, RFvoltage, harmonicRatio, RFphase, mom, Vs=0, machine='LHC', gammat=None, atomicMass=1, Z=1, x=None, Vtot=None, h=None, lepton=False):

        self.machine = machine
        protonMass = 1.67e-27
        electronMass = 9.1093837e-31

        self.atomicMass = atomicMass
        self.Z = Z
        if lepton==True:
            self.mass = atomicMass*electronMass
        else:
            self.mass = atomicMass*protonMass

        if x is None:
            # self.x = np.linspace(0 - 0.5, 2 * np.pi + 0.5, 2 ** 15)  # phase vector in RF rads (0,2pi) = 1 bucket
            self.x = np.linspace(np.pi/4, 2 * np.pi - np.pi/4, 2 ** 15)  # phase vector in RF rads (0,2pi) = 1 bucket
            self.Vtot = Vtot
        else:
            if len(x)<2**10:
                Vtot_new = interp1d(x,Vtot)
                x_new = np.linspace(x[0],x[-1],2**15)
                V_new = Vtot_new(x_new)
            self.x = x_new
            self.Vtot = V_new
        
            
            
        self.V = self.Z*np.ascontiguousarray(RFvoltage) # in MV
        self.h = np.ascontiguousarray(harmonicRatio) # for the main RF should be 1
        self.phi = np.ascontiguousarray(RFphase)
        self.p = self.Z*mom # in MeV/c
        self.Vs = self.Z*Vs
        self.machine_params(gammat,h)

        self.U = None
        self.x_bucketRight = None
        self.phis = None
        self.E_phaseSpace = None


    def machine_params(self, gammat=None, h=None):
        # get the parameters for the machine and the calculated values in correct units
        E0 = ((self.mass * sp_co.c ** 2) / 1.6e-10) * 1e3  # in MeV

        if self.machine == 'SPS':
            R = 1100.0093  # radius in m
            bending_radius = 741.254598153917  # in m
            if gammat is None:
               gammat = 18 #22.773  # 18 #22.93662195  #22.8 #18 # transision gamma 
            if h is None:
                h = 4620  # harmonic number
        elif self.machine == 'LHC':
            R = 4242.89  # radius in m
            bending_radius = 2804.0  # in m
            if gammat is None:
                gammat = 53.61  # 53.61 # transision gamma 55.744024 b1, 55.852493 b2@injection
            if h is None:
                h = 35640  # harmonic number
        elif self.machine == 'LEIR':
            R = 78.54370266 / 2 / np.pi  # radius in m
            bending_radius = 4.17  # not correct
            if gammat is None:
                gammat = 2.8383
                
            if h is None:
                h = 2  # harmonic number
        elif self.machine == 'FCC':
            R = 90658.499/ 2 / np.pi 
            bending_radius = 10.021
            if gammat is None:
                gammat = 210.58465757133067
                
            if h is None:
                h = 121200  # harmonic number
        else:
            print('I do not have the parameters for the machine you request')

        # calculated parameters
        E = np.sqrt(self.p ** 2 + E0 ** 2)  # in MeV
        EK = (E- E0)/self.atomicMass
        gamma = E / E0
        beta = np.sqrt(1 - (1 / gamma ** 2))
        v = beta * c
        Trev = 2 * np.pi * R / v  # in s
        wrev = 2 * np.pi / Trev  # in radHz
        wrf = h * wrev  # in rad Hz
        frf = wrf / 2 / np.pi  # in Hz
        eta_0 = (1. / gammat) ** 2 - (1. / gamma) ** 2
        if eta_0 < 0: # if  below transition energy change the phase axis by pi
            self.x = self.x - np.pi

        # normalization factor
        emitScalingFactor = np.sqrt((2 * beta ** 2 * E) / (h * np.abs(eta_0) * wrev**2 * 2 * np.pi)) / h
        # this factor has to be multiplied by the DE (MeV) of the particles
        HamiltScalingFactor = (h * eta_0 * 2 * np.pi) / (beta** 2 * E)
        
        fsScalingFactor = np.sqrt((h * np.abs(eta_0) * wrev**2)/(2 * np.pi * beta** 2 * E))/(2 * np.pi)

        self.params = {'E':E, 'gamma':gamma, 'beta':beta, 'R':R, 'E0':E0, 'EK':EK,
                       'bending_radius':bending_radius, 'eta_0':eta_0, 'gammat':gammat, 'Trev':Trev, 'wrev':wrev,
                       'wrf':wrf, 'frf':frf, 'harmonicNumber':h,
                       'emitScalingFactor':emitScalingFactor,'HamiltScalingFactor':HamiltScalingFactor, 'fsScalingFactor':fsScalingFactor}

        return self.params

    def totalvoltage(self):
        # calculate the RF voltage waveform
        # h=1 for main harmonic
        Vtot = 0

        for i, v in enumerate(self.V):
            Vtot += v * np.sin(self.h[i] * self.x + self.phi[i])

        self.Vtot = Vtot - self.Vs

        return self.Vtot
    #
    def potentialWell(self):
        # calculate the potential well
        dx = self.x[1] - self.x[0]
        if self.Vtot is None:
            self.Vtot = self.totalvoltage()

        U = np.cumsum(self.Vtot) * dx

        if self.params['eta_0'] > 0:
            U = -U
            phis_indexes = np.where(np.diff(np.sign(self.Vtot)) < 0)[0]
        else:
            phis_indexes = np.where(np.diff(np.sign(self.Vtot)) > 0)[0]

        self.U = U - U.min()
        self.phis = self.x[phis_indexes[0]]

        return self.U, self.phis


    def calculateBucket(self):

        if self.U is None:
            self.U, self.phis = self.potentialWell()

        left = np.where(self.x <= self.phis)
        middleIdex = left[0][-1]
        right = np.where(self.x > self.phis)
        maxLeft = max(self.U[left])
        maxLeftIndex = np.where(self.U[left] == maxLeft)
        maxLeftIndex = maxLeftIndex[0][-1]
        maxRight = max(self.U[right])
        maxRightIndex = np.where(self.U[right] == maxRight)
        maxRightIndex = maxRightIndex[0][0] + middleIdex

        Hbucket = min(maxLeft, maxRight)
        Unew = self.U[maxLeftIndex:maxRightIndex]
        xnew = self.x[maxLeftIndex:maxRightIndex]

        bucketIndexes = np.where(Unew <= Hbucket)

        x_bucket = xnew[bucketIndexes]
        E_bucket = np.sqrt(Hbucket - Unew[bucketIndexes]) * 1e6 * self.params['emitScalingFactor']  # in V
        E_bucket = E_bucket.real


        self.bucketArea = 2 * np.trapz(E_bucket, x_bucket)/self.atomicMass

        self.x_bucketLeft = x_bucket[0]
        self.x_bucketRight = x_bucket[-1]
        self.bucketLength = self.x_bucketRight - self.x_bucketLeft
        # potential well only of the bucket
        self.Ubucket = self.U[(self.x >= self.x_bucketLeft) & (self.x <= self.x_bucketRight)]
        self.xbucket = self.x[(self.x >= self.x_bucketLeft) & (self.x <= self.x_bucketRight)]

        self.x_phaseSpace = np.append(x_bucket, x_bucket[::-1])
        self.E_phaseSpace = np.append(E_bucket, -E_bucket[::-1])
        self.E_phaseSpace = self.params['wrf'] * self.E_phaseSpace # in eV
        self.DEmax_bucket = np.max(self.E_phaseSpace)
        self.dp_over_p_bucket =(1/self.params['beta']**2)*(self.DEmax_bucket/self.params['E']/1e6)

        return self.bucketArea

    def calculateEmittance(self, bunchLength=None, epsilon=0.001):
        self.bunchLength = bunchLength # should be in seconds
        # find first the bucket in case is not yet calculated. Inside the potential well will be also calculated
        if self.E_phaseSpace is None: # find first the bucket
            self.calculateBucket()

        Hbucket = max(self.Ubucket)
        Umin = min(self.Ubucket)
        Umax = Hbucket

        if self.bunchLength is None:
            print('ERROR: NO bunch length input')
        else:
            bunchLength = self.params['wrf'] * self.bunchLength  # change the bunchlength from s to rads
            Hbunch = (Umax + Umin) / 2
            indx = np.where(self.Ubucket <= Hbunch)
            x_bunch = self.xbucket[indx] #self.Ubucket <= Hbunch]
            U_bunch = self.Ubucket[indx] #self.Ubucket <= Hbunch]
            bunchLength1 = x_bunch[-1] - x_bunch[0]
                 
            if bunchLength < self.bucketLength:
                
                while np.abs(bunchLength1 - bunchLength) > epsilon:

                    if bunchLength1 > bunchLength:
                        Umax = Hbunch
                        Hbunch = (Umax + Umin) / 2
                    elif bunchLength1 < bunchLength:
                        Umin = Hbunch
                        Hbunch = (Umax + Umin) / 2
                    indx = np.where(self.Ubucket <= Hbunch)
                    x_bunch = self.xbucket[indx] #self.Ubucket <= Hbunch]
                    U_bunch = self.Ubucket[indx] #self.Ubucket <= Hbunch]    
                    # x_bunch = self.xbucket[self.Ubucket <= Hbunch]
                    # U_bunch = self.Ubucket[self.Ubucket <= Hbunch]
                    bunchLength1 = x_bunch[-1] - x_bunch[0]

            else:
                print('ERROR: The bunch length you ask cannot fit the bucket')


        E_bunch = np.sqrt(Hbunch - U_bunch) * 1e6 * self.params['emitScalingFactor']  # in eV
        E_bunch = E_bunch.real
    
        self.bunchArea = 2 * np.trapz(E_bunch, x_bunch)/self.atomicMass
        self.x_bunchLeft = x_bunch[0]
        self.x_bunchRight = x_bunch[-1]
        self.x_bunchPhaseSpace = np.append(x_bunch, x_bunch[::-1])
        self.E_bunchPhaseSpace = self.params['wrf'] * np.append(E_bunch, -E_bunch[::-1])  # in eV

        self.DEmax = np.max(self.E_bunchPhaseSpace)
        self.dp_over_p=(1/self.params['beta']**2)*(self.DEmax/self.params['E']/1e6)
        # print('Dp/p = ',str(self.dp_over_p))
        # print('tau = ',str(bunchLength1/self.params['wrf']))
        
        return self.bunchArea

        # return bucketArea, bunchArea, xbucket, Ubucket, x_bunch, U_bunch, , x_bunchPhaseSpace, E_bunchPhaseSpace

    def calculateBunchLength(self, emittance=None, epsilon = 0.001):
    #calculate the bunch length for an input emittance in eVs
        self.emittance = emittance # should be in seconds
        # find first the bucket in case is not yet calculated. Inside the potential well will be also calculated
        if self.E_phaseSpace is None: # find first the bucket
            self.calculateBucket()

        Hbucket = max(self.Ubucket)
        Umin = min(self.Ubucket)
        Umax = Hbucket

        if self.emittance is None:
            print('ERROR: NO emittance input')
        else:
            Hbunch = Hbucket/2
            x_bunch = self.xbucket[self.Ubucket<=Hbunch]
            U_bunch = self.Ubucket[self.Ubucket<=Hbunch]
            E_bunch = np.sqrt(Hbunch-U_bunch)*1e6*self.params['emitScalingFactor'] # in eV
            E_bunch = E_bunch.real
            bunchArea = 2*np.trapz(E_bunch,x_bunch)
        #    print('Bucket Area = {:2.3f}'.format(bucketArea))
        #    print('Bunch Area = {:2.3f}'.format(bunchArea))
            while np.abs(emittance-bunchArea)>epsilon:
                if emittance<bunchArea:
                    Umax = Hbunch
                    Hbunch = (Umax+Umin)/2
                elif emittance>bunchArea:
                    Umin = Hbunch
                    Hbunch = (Umax+Umin)/2
                    
                x_bunch = self.xbucket[self.Ubucket<=Hbunch]
                U_bunch = self.Ubucket[self.Ubucket<=Hbunch]
                E_bunch = np.sqrt(Hbunch-U_bunch)*1e6*self.params['emitScalingFactor'] # in eV
                E_bunch = E_bunch.real
                bunchArea = 2*np.trapz(E_bunch,x_bunch)
            
            self.x_bunchLeft = x_bunch[0]
            self.x_bunchRight = x_bunch[-1]
            bunchLength = x_bunch[-1]-x_bunch[0] #in rad
            self.bunchLength = bunchLength/self.params['wrf'] #in s
            self.x_bunchPhaseSpace = np.append(x_bunch,x_bunch[::-1])
            self.E_bunchPhaseSpace = self.params['wrf']*np.append(E_bunch,-E_bunch[::-1])# in eV
            
            return self.bunchLength
            
    def synchFreq(self,n_fs=200):


        def second_derivative(f, x, dx=1e-5):
                return (-f(x + 2*dx) + 16*f(x + dx)
                        - 30*f(x)
                        + 16*f(x - dx) - f(x - 2*dx)) / (12*dx**2)
        
        
        if self.x_bucketRight is None:
        # if self.U is None:
            self.calculateBucket()
        
        x_edge_right = self.x_bucketRight
        x_edge_left = self.x_bucketLeft
        
        Ul = interp1d(self.x,self.U,kind='cubic')
        
        Ulmax = np.min([Ul(x_edge_left),Ul(x_edge_right)])
        U = self.U[(self.x>=x_edge_left) * (self.x<=x_edge_right)]
        phase = self.x[(self.x>=x_edge_left) * (self.x<=x_edge_right)]
        
        ind_min = np.argmin(U) #14133
        
        
        phis = phase[ind_min]
        
        
        
        fs0 = np.sqrt(second_derivative(Ul, phis))
        self.params['fs0'] = fs0 *self.params['fsScalingFactor']* np.abs(np.cos(self.phis))
        
        phi_right = interp1d(U[ind_min:],phase[ind_min:],kind='slinear')
        phi_left = interp1d(U[:ind_min+1],phase[:ind_min+1],kind='slinear')
        
        
        plt.figure()
        plt.plot(phase,Ul(phase))
        plt.plot(phis,0,'.',markersize=17)
        # plt.plot(x_edge,Ulmax,'.',markersize=17)
        
        plt.figure(20)
        plt.plot(U[ind_min:],phase[ind_min:])
        plt.plot(U[:ind_min+1],phase[:ind_min+1])
        plt.plot(U[ind_min:],phi_right(U[ind_min:]))
        plt.plot(U[:ind_min+1],phi_left(U[:ind_min+1]))
        
        fs_arr = np.zeros(n_fs)
        bl_arr = np.zeros(n_fs)
        e_fs_arr = np.linspace(Ulmax/n_fs,Ulmax,n_fs,endpoint = True)
        for i in range(n_fs):
            # print(e_fs_arr[i],phi_left(e_fs_arr[i]), phi_right(e_fs_arr[i]))
            plt.figure(20)
            plt.plot(e_fs_arr[i],phi_left(e_fs_arr[i]),'.r',markersize=17)
            plt.plot(e_fs_arr[i],phi_right(e_fs_arr[i]),'.r',markersize=17) 
            res = integrate.quad(lambda x: np.sqrt(2.)/np.sqrt(e_fs_arr[i] - Ul(x)), phi_left(e_fs_arr[i]), phi_right(e_fs_arr[i]))[0]
            fs_arr[i] = 2.*np.pi/res
            bl_arr[i] = phi_right(e_fs_arr[i]) - phi_left(e_fs_arr[i])
            
        e_fs_arr = np.append(0,e_fs_arr)
        fs_arr = np.append(fs0,fs_arr)
        bl_arr = np.append(0,bl_arr)
        # print(phi_left(e_fs_arr[i]),phi_right(e_fs_arr[i]))
        
        self.e_fs_arr = e_fs_arr[np.logical_not(np.isnan(fs_arr))]
        self.bl_arr = bl_arr[np.logical_not(np.isnan(fs_arr))]
        self.fs_arr = fs_arr[np.logical_not(np.isnan(fs_arr))]
        self.fs0 = fs0
    #    fs_arr[np.isnan(fs_arr)]=0
        
        return self.e_fs_arr,self.bl_arr,self.fs_arr    
    
    
    def unnormalizefs(self):
        self.fs_arr = self.params['fsScalingFactor'] * self.fs_arr * np.abs(np.cos(self.phis))
        
        return self.fs_arr
    

def insert_apertures(env, scen_path, sequence_name):

    line = env.lines[sequence_name]
    tab = line.get_table()
    with open(scen_path, 'r') as fid:
        scen = yaml.safe_load(fid)

    ############################################################
    # Set appropriate models and integrators
    ############################################################

    tt_bend = tab.rows[(tab.element_type=='Bend') | (tab.element_type=='RBend')]
    tt_quad = tab.rows[(tab.element_type=='Quadrupole')]
    tt_sext = tab.rows[(tab.element_type=='Sextupole')]
    tt_doublet = tt_quad.rows[scen['lattice_names']['doublet']]

    env.set(tt_bend, integrator='uniform', num_multipole_kicks=3, model='mat-kick-mat')
    env.set(tt_quad, integrator='uniform', num_multipole_kicks=5, model='mat-kick-mat')
    env.set(tt_sext, integrator='yoshida4', num_multipole_kicks=1, model='drift-kick-drift-expanded')
    env.set(tt_doublet, integrator='yoshida4', num_multipole_kicks=200, model='drift-kick-drift-exact')

    ############################################################
    # Aperture assignment
    ############################################################
    NEEDS_APERTURE = [
        "Cavity",
        "Multipole",
        "Bend",
        "RBend",
        "Quadrupole",
        "Sextupole",
    ]

    ip_names = tab.rows[scen['lattice_names']['ip']].name
    for nn, ee in zip(tab.name, tab.element_type):
        if ee not in NEEDS_APERTURE and nn not in ip_names:
            continue

        is_ip = nn in ip_names
        is_inner_doublet = nn in tab.rows[scen['lattice_names']['inner_doublet']].name
        is_outer_doublet = nn in tab.rows[scen['lattice_names']['outer_doublet']].name
        is_near_qc1 = ('qc1' in nn) and ('cor' in nn)
        is_near_qc2 = ('qc2' in nn) and ('cor' in nn)

        if is_ip:
            radius = scen['apertures']['r_ip']
        elif is_inner_doublet or is_near_qc1:
            radius = scen['apertures']['r_inner_doublet']
        elif is_outer_doublet or is_near_qc2:
            radius = scen['apertures']['r_outer_doublet']
        else:
            radius = scen['apertures']['r']

        aper_name = f'{nn}_aper'
        env.new(aper_name, xt.LimitEllipse, a=radius, b=radius)

        if is_ip:
            continue

        # Replica apertures for multipoles and cavities
        if ee in ('Multipole', 'Cavity'):
            env.new(aper_name + '..0', aper_name, mode='replica')
            env.new(aper_name + '..1', aper_name, mode='replica')

        env[nn].name_associated_aperture = aper_name


    ############################################################
    # Aperture installation for IPs, cavities and multipoles
    ############################################################
    # IPs
    line.insert([env.place(f'{ip}_aper', at=f'{ip}@start') for ip in ip_names])

    # Multipoles and Cavities
    placements = []
    for nn in tab.rows[(tab.element_type == 'Multipole') | (tab.element_type == 'Cavity')].name:
        placements.append(env.place(f'{nn}_aper..0', at=f'{nn}@start'))
        placements.append(env.place(f'{nn}_aper..1', at=f'{nn}@end'))

    line.insert(placements)


    ############################################################
    # Install SR masks
    ############################################################
    srm_rect = scen['lattice_names']['sr_mask_rect_prefix']
    srm_circ = scen['lattice_names']['sr_mask_circ_prefix']
    for ip in [1, 2, 3, 4]:
        env.new(f'{srm_rect}.{ip}.b1', xt.LimitRect, min_x=-0.007, max_x=0.007, min_y=-0.013, max_y=0.013)
        env.new(f'{srm_circ}.{ip}.b1', xt.LimitEllipse, a=0.015, b=0.015)

    line.insert([env.place(f'{srm_rect}.{i+1}.b1', at=env[coll].length / 2 + 0.42, from_=f'{coll}@end')
                for i, coll in enumerate(scen['lattice_names']['sr_mask_rect'])]
            + [env.place(f'{srm_circ}.{i+1}.b1', at=-0.03, from_=f'{coll}@start')
                for i, coll in enumerate(scen['lattice_names']['sr_mask_circ'])])

def thick_slicing(line, scen_path):

    with open(scen_path, 'r') as fid:
        scen = yaml.safe_load(fid)

    line.slice_thick_elements(
    slicing_strategies=[
        xt.Strategy(slicing=xt.Uniform(1, mode='thick')), # Default applied to all elements
        xt.Strategy(slicing=xt.Uniform(3, mode='thick'), element_type=xt.RBend),
        xt.Strategy(slicing=xt.Uniform(5, mode='thick'), element_type=xt.Quadrupole),
        xt.Strategy(slicing=xt.Uniform(1, mode='thick'), element_type=xt.Sextupole),
        xt.Strategy(None, element_type=xt.Cavity),
        xt.Strategy(slicing=xt.Uniform(30, mode='thick'), name=scen['lattice_names']['doublet']),
    ])

    return line
