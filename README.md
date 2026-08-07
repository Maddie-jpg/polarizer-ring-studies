# Polarizer Ring Studies

FCC-ee is a study on an electron-positron collider with a circumference of about 90 km and to be operated at four different energies. The experiments need to know the energy of the colliding particles with high precision. This can be achieved by colliding low intensity polarized bunches (spin of particles having a preferential direction), which have to be depolarized to determine the beam energy. The baseline is to generate such polarized bunches in the collider prior to intensity ramp-up of high intensity colliding bunches. A dedicated synchrotron operated at lower energy to generate polarized bunches has been proposed as an alternative to improve the efficiency of the exploitation of the FCC-ee collider. The task is to further refine and optimize the design of such a polarizer ring and to determine the basic characteristics and performance.

### Current literature on the project

- **[Project Plan](/Assignments/Project%20Plan/ProjectPlan.pdf)** : Outlines aims and objectives of the project.
- **[Literature Review](/Assignments/Interim%20Dissertion%20(Literature%20Review)/Literature_Review.pdf)** : A report on the ideas and concepts behind the project.
- **[A Low Energy Polarizer Ring for the FCC-ee](https://github.com/user-attachments/files/30833367/WEP5048.pdf)** : IPAC '26 paper on the progress of the polarizer ring.

## Design variations and naming conventions
Every stored lattice file and results folder is indexed by the four independent choices outlined below.

### Design -`{design}`
This defines the linear optics, cell type, and periodicity of the lattice.
|Design	|Geometry|	
|--------|----------|
|D1|Three-fold symmetric (6 sextants), FODO arcs, 8 cells/sextant.|	
|D2|Two-fold ring with a single triplet straight section, FODO arcs.| 
|D3|Two-fold racetrack (4 sextants) with long straights, FODO arcs, 10 cells/sextant.|	


### Sextupole configuration - `{config}`
This defines the chromaticity correction scheme used in the lattice. All configurations for each design are kept [here](LatticeBuild/sextupole_configs.py), with each function containing a small description of the scheme used.

### Alignment state - `{mode}`

|State|Meaning|
|---|----|
|`perfect`| An ideal lattice without errors.|
|`misaligned`| Random magnet errors applied: 0.25mm rms, Gaussian truncated at 2.5 $\sigma$, on all quadrupoles, sextupoles, and dipoles. Full misalignment scheme can be found [here](LatticeBuild/misalignments_corrections.py)|
|`perfect`| Misaligned lattice that has then been orbit-corrected, containing BPMs and corrector magnets. The full correction scheme can be found [here](LatticeBuild/misalignments_corrections.py)|

### Cell phase advance - `{phase}`
Phase advance is input into the file name in degrees. Each linear optics design can be altered for a specific phase advance.

### Lattice changes - `{changes}`
For small changes to the lattice, e.g. changing a drift length to see its effect, an identifier can be added to the file. This parameter can (and mostly should) be `None` if this doesn't apply.

## Repository structure

### Lattice storage
Lattices are stored as Xsuite `Environment` JSON files:
 
```
JSON Files/D{design}/C{config}/pdr_{mode}_{phase}.json
```
 
Each file contains a `ring` line (the full closed ring) and the periodic cell.

### Folder structure

What each top-level folder is for. Paths follow the
`D{design}/C{config}` indexing described above.
 
| Path | Contents |
|---|---------|
| `JSON Files/` | **The lattices.** Xsuite `Environment` JSON files, indexed `D{design}/C{config}/pdr_{mode}_{phase}.json`. This is what you load. |
| `LatticeBuild/` | Code that **builds** a lattice from scratch: linear optics (`linear_optics.py`), sextupole/chromaticity configs (`sextupole_configs.py`), misalignments + orbit correction (`misalignments_corrections.py`), and the `lattice_build.py` driver. |
| `Results/` | **Generated outputs** — plots, parameter dumps, `.dat`/`.csv` — written under `D{design}/C{config}/...`.Reproducible from the code. An example set-up of this folder is provided below.|
| `TuneDiagram/` | Helper library for plotting resonance lines / working points. (thanks to Hannes Bartosik)|
| `xutil_DA_CC/` | Dynamic-aperture / momentum-aperture tracking toolkit used by `analysis.py`. (thanks to Kyriacos Skoufaris)|
|`PositronBeam_2p86GeV/`,`PositronBeam_2p86GeV_PolarizedEbeam/`| Current macroparticle distributions from the positron linac. (thanks to Iryna Chaikovska)|
|`Damping Ring/`| Any files pertaining to current damping ring designs. | 
| `External Files/` | Third-party or supervisor-provided inputs and reference lattices. |
| `Assignments/`, `Slides and Notes/` | Write-ups, literature, presentations for my masters degree. |

#### Results folder set-up

```bash
Results/
└── D{design}/                            e.g. D1
    ├── Macroparticle Distribution/       distribution-level: phase-space plots,
    │                                     TwissResults.json (design-level — not per-config)
    └── C{config}/                        e.g. C1
        └── {phase}deg_PhaseAdvance/      e.g. 90deg_PhaseAdvance 
            ├── perfect/ 
            │   └── MPD-{energy}MeV/      injection/ECS results: tracking evolution, survival 
            │                             curves, CompressorParams.json for a certain energy
            ├── misaligned/
            ├── corrected/
            ├── Comparison/               spin tracking scan (misaligned vs corrected) from 
            │                             spin_tracking.py
            └── SingleSeed_{seed}/        single-seed spin tracking results from 
                                          spin_trackin_single_seed.py

```


### Top-level scripts

| File | Role |
|---|---|
| `analysis.py` | Beam optics, tunes, damping, polarization, DA/MA. |
| `spin_tracking.py`,`spin_tracking_single_seed.py` | Spin-tracking / depolarization studies (single seed or multiple cases). |
| `macroparticles.py`| Injection efficiency simulations using the macroparticle distributions from the positron linac.|
| `constants.py` | Shared beam energy, RF voltage, working points that are used across scripts. |
| `my_functions.py`| Shared helpers (plotting, lattice utilities, resonance scans, spin tracking functions). |
| `run_full_sims.py` | Batch driver across designs/configs. |

## Loading a lattice
A simple loading script that loads in the lattice JSON file and produces Twiss.

```
import xtrack as xt

pdr  = xt.Environment.from_json("JSON Files/D1/C1/pdr_perfect_90.json")
ring = pdr.lines["ring"]

# 1. Reference particle: energy + electron anomalous moment (needed for spin).
ring.particle_ref.kinetic_energy0            = 2.86e9          # eV
ring.particle_ref.anomalous_magnetic_moment  = 0.001159652181

# 2. Enable spin transport.
ring.configure_spin("auto")

# 3. Radiation mode: 'mean' for twiss, 'quantum' for tracking.
ring.configure_radiation("mean")
tw = ring.twiss(method="6d", radiation_integrals=True, eneloss_and_damping=True,
                spin=True, polarization=True)
```

![](https://media.tenor.com/mMkJeuyHkRYAAAAj/cat-cat-on-computer.gif)

README last updated: 07/08/2026
