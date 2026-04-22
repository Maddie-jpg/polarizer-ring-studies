# Polarizer Ring Studies

## Project Overview

FCC-ee is a study on an electron-positron collider with a circumference of about 90 km and to be operated at four different energies. The experiments need to know the energy of the colliding particles with high precision. This can be achieved by colliding low intensity polarized bunches (spin of particles having a preferential direction), which have to be depolarized to determine the beam energy. The baseline is to generate such polarized bunches in the collider prior to intensity ramp-up of high intensity colliding bunches. A dedicated synchrotron operated at lower energy to generate polarized bunches has been proposed as an alternative to improve the efficiency of the exploitation of the FCC-ee collider. The task is to further refine and optimize the design of such a polarizer ring and to determine the basic characteristics and performance.

**Aims and objectives are further outlined in the [project plan](/Assignments/Project%20Plan/ProjectPlan.pdf).**

## Folder Structure
```
Polarizer-Ring-Studies/
├── Assignments/                           # University related work goes in the assignments folder
│   ├── Final Dissertation/
│   ├── Interim Dissertation/
│   ├── Journal Paper Review/
│   └── Project Plan/
├── CC files/                 
├── Design 1/                              # Compact ring 90° phase advance
│   ├── linear_optics_variation_1.py       # Initial ring design
│   ├── linear_optics_variation_1.py       # Initial ring design with matching QDs
│   ├── sextupole_configs.py               # Functions for sextupole insertion (+ BPMs and correctors)
│   ├── constants.py                       # Values such as VRF and WP are put in here so they can be changed across all codes
│   ├── build_and_analysis.ipynb           # Loads in lattice and performs analysis
│   ├── json_files/                        # Saved lattice json files
│   └── config_Dn/                         # Each sextupole config gets its own folder containing results (config_D0, config_D1, etc.)
├── Tune Diagram/                          # Function to plot resonance lines for WP plots (from Hannes Bartosik)
├── xutil_DA_CC/                           # Contains functions to plot DA and MA (from Kyriacos Skoufaris)
├── my_functions.py                        # Misc. functions that can be used across designs   
├── Junk Draw/                             # Things I should probably delete but might need 
├── .gitignore              
└── README.md                
```
## Useful Reference Material
* [Xsuite Documentation](https://xsuite.readthedocs.io/en/latest/index.html)

* [Low-emittance storage rings](https://github.com/user-attachments/files/26247579/Low-emittance.storage.rings.pdf)

* [Polarized Beam Dynamics and Instrumentation in Particle Accelerators](https://github.com/user-attachments/files/26247593/Polarized.Beam.Dynamics.and.Instrumentation.in.Particle.Accelerators._.pdf)

* [Radiative Polarization, Computer Algorithms and Spin Matching in Electron Storage Rings](https://github.com/user-attachments/files/26247598/Radiative.Polarization.Computer.Algorithms.and.Spin.Matching.in.Electron.Storage.Rings.pdf)

* [Spin-Orbit Maps and Electron Spin Dynamics for the Luminosity Upgrade Project at HERA](https://github.com/user-attachments/files/26247600/Spin-Orbit.Maps.and.Electron.Spin.Dynamics.for.the.Luminosity.Upgrade.Project.at.HERA.pdf)

* [The Physics of Electron Storage Rings: An Introduction](https://github.com/user-attachments/files/26247602/The.Physics.of.Electron.Storage.Rings_.An.Introduction.pdf)

* [Future Circular Collider Feasibility Study Report: Volume 2, Accelerators, Technical Infrastructure and Safety](https://arxiv.org/pdf/2505.00274)

* [Compilation of papers on RF electronics](https://cds.cern.ch/record/1407402/files/p223.pdf)

![]([[https://cdn.dribbble.com/userupload/25473001/file/original-a900f3853728a14c547a0d3182815f03.gif](https://media.tenor.com/mMkJeuyHkRYAAAAj/cat-cat-on-computer.gif)]
