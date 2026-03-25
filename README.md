# Polarizer Ring Studies

## Project Overview

FCC-ee is a study on an electron-positron collider with a circumference of bout 90 km and to be operated at four different energies. The experiments need to know the energy of the colliding particles with high precision. This can be achieved by colliding low intensity polarized bunches (spin of particles having a preferential direction), which have to be depolarized to determine the beam energy. The baseline is to generate such polarized bunches in the collider prior to intensity ramp-up of high intensity colliding bunches. A dedicated synchrotron operated at lower energy to generate polarized bunches has been proposed as an alternative to improve the efficiency of the exploitation of the FCC-ee collider. The task is to further refine and optimize the design of such a polarizer ring and to determine the basic characteristics and performance.

**Aims and objectives are further outlined in the [project plan](/Assignments/Project%20Plan/ProjectPlan.pdf).**

## Folder Structure
```
polarizer-ring-studies/
├── assignments/                           # University related work goes in the assignments folder
│   ├── final dissertation/
│   ├── interim dissertation/
│   ├── journal paper review/
│   └── project plan/
├── CC files/                 
├── essentials/                            # Lattice design, analysis, and associated functions
│   ├── design_n/                          # Each design/edit gets its own folder (design_0, design_1, etc.)
│   │   ├── lattice_building_n.ipynb       # Script to build the lattice - including matching routines
│   │   ├── json_files                     # JSON outputs from lattice_building_n.ipynb
│   │   ├── plots                          # Plot outputs from lattice_analysis.ipynb
│   │   └── A Design Description           # Small .txt file describing the lattice being built and any changes from previous iterations
│   ├── lattice_analysis.ipynb             # Analysis of each design (Twiss, working point, DA, MA etc.)
│   ├── Tune Diagram/                      # Function to plot resonance lines for WP plots (from Hannes Bartosik)
│   ├── xutil_DA_CC/                       # Function to plot DA and MA (from Kyriacos Skoufaris)
│   └── constants.py                       # values such as VRF and WP are put in here so they can be changed across all codes
├── misc. plots/   
├── notebooks/   
├── .gitignore              
└── README.md                
```
## Useful Reference Material
* [Low-emittance storage rings](https://github.com/user-attachments/files/26247579/Low-emittance.storage.rings.pdf)

* [Polarized Beam Dynamics and Instrumentation in Particle Accelerators](https://github.com/user-attachments/files/26247593/Polarized.Beam.Dynamics.and.Instrumentation.in.Particle.Accelerators._.pdf)

* [Radiative Polarization, Computer Algorithms and Spin Matching in Electron Storage Rings](https://github.com/user-attachments/files/26247598/Radiative.Polarization.Computer.Algorithms.and.Spin.Matching.in.Electron.Storage.Rings.pdf)

* [Spin-Orbit Maps and Electron Spin Dynamics for the Luminosity Upgrade Project at HERA](https://github.com/user-attachments/files/26247600/Spin-Orbit.Maps.and.Electron.Spin.Dynamics.for.the.Luminosity.Upgrade.Project.at.HERA.pdf)

* [The Physics of Electron Storage Rings_ An Introduction](https://github.com/user-attachments/files/26247602/The.Physics.of.Electron.Storage.Rings_.An.Introduction.pdf)

* [Future Circular Collider Feasibility Study Report: Volume 2, Accelerators, Technical Infrastructure and Safety](https://arxiv.org/pdf/2505.00274)

