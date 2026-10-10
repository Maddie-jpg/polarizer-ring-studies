# Polarizer Ring Studies

FCC-ee is a study on an electron-positron collider with a circumference of about 90 km and to be operated at four different energies. The experiments need to know the energy of the colliding particles with high precision. This can be achieved by colliding low intensity polarized bunches (spin of particles having a preferential direction), which have to be depolarized to determine the beam energy. The baseline is to generate such polarized bunches in the collider prior to intensity ramp-up of high intensity colliding bunches. A dedicated synchrotron operated at lower energy to generate polarized bunches has been proposed as an alternative to improve the efficiency of the exploitation of the FCC-ee collider. The task is to further refine and optimize the design of such a polarizer ring and to determine the basic characteristics and performance.

### Current literature on the project


- **[Project Plan](https://github.com/user-attachments/files/30835136/ProjectPlan.pdf)** : Outlines aims and objectives of the project.
- **[Literature Review](https://github.com/user-attachments/files/30835120/Literature_Review.pdf)** : A report on the ideas and concepts behind the project.
- **[A Low Energy Polarizer Ring for the FCC-ee](https://github.com/user-attachments/files/30833367/WEP5048.pdf)** : IPAC '26 paper on the progress of the polarizer ring.

## Design variations and naming conventions
Every stored lattice file and results folder is indexed by the four independent choices outlined below.

### Design -`{design}`
This defines the linear optics, cell type, and periodicity of the lattice.
|Design	|Geometry|	
|--------|----------|
|D1|Three-fold symmetric (6 sextants), FODO arcs, 8 cells/period.|	
|D2|Two-fold ring with a single triplet straight section, FODO arcs.| 
|D3|Two-fold racetrack (4 arcs) with long straights, FODO arcs, 10 cells/period.|	


### Sextupole configuration - `{config}`
This defines the chromaticity correction scheme used in the lattice. All configurations for each design are kept [here](LatticeBuild/sextupole_configs.py), with each function containing a small description of the scheme used.

### Alignment state - `{mode}`

|State|Meaning|
|---|----|
|`perfect`| An ideal lattice without errors.|
|`misaligned`| Random magnet errors applied: 0.25mm rms, Gaussian truncated at 2.5 $\sigma$, on all quadrupoles, sextupoles, and dipoles. Full misalignment scheme can be found [here](LatticeBuild/misalignments_corrections.py)|
|`corrected`| Misaligned lattice that has then been orbit-corrected, containing BPMs and corrector magnets. The full correction scheme can be found [here](LatticeBuild/misalignments_corrections.py)|

### Cell phase advance - `{phase}`
Phase advance is input into the file name in degrees. Each linear optics design can be altered for a specific phase advance.

### Lattice changes - `{changes}`
For small changes to the lattice, e.g. changing a drift length to see its effect, an identifier can be added to the file. This parameter can (and mostly should) be `None` if this doesn't apply.

### Lattice index
Which builder and sextupole config produced each stored lattice. Please keep this up to date when adding one.

| Lattice | Builder (`linear_optics.py`) | Sextupoles (`sextupole_configs.py`) | Working point | Notes |
|---|---|---|---|---|
| D1 C9 90deg FDF | `three_fold_periodicity` | `config_D1_C9` | (15.42, 15.38) | current default (`config.DEFAULT_STUDY`) |
| D3 C2 90deg | `two_fold_racetrack_3straight` | | (13.65, 13.23) | |
| ... | | | | |

## Repository structure

### Lattice storage
Lattices are stored as Xsuite `Environment` JSON files:
 
```
JSON_Files/D{design}/C{config}/pdr_{mode}_{phase}[_{changes}].json
```

Each file contains a `ring` line (the full closed ring) and the periodic cell. Never build these paths by hand: use `paths.lattice_json_path(...)` or `paths.load_lattice(...)`.

### Folder structure

What each top-level folder is for. Paths follow the
`D{design}/C{config}` indexing described above.
 
| Path | Contents |
|---|---------|
| `JSON_Files/` | **The lattices.** Xsuite `Environment` JSON files, indexed `D{design}/C{config}/pdr_{mode}_{phase}.json`. This is what you load. |
| `LatticeBuild/` | Code that **builds** a lattice from scratch: linear optics (`linear_optics.py`), sextupole/chromaticity configs (`sextupole_configs.py`), misalignments + orbit correction (`misalignments_corrections.py`), and the `lattice_build.py` driver. |
| `Results/` | **Generated outputs** — plots, parameter dumps, `.dat`/`.csv` — written under `D{design}/C{config}/...`.Reproducible from the code. An example set-up of this folder is provided below.|
| `TuneDiagram/` | Helper library for plotting resonance lines / working points. (thanks to Hannes Bartosik)|
| `xutil_DA_CC/` | Dynamic-aperture / momentum-aperture tracking toolkit used by `analysis.py`. (thanks to Kyriacos Skoufaris)|
|`PositronBeam_2p86GeV/`,`PositronBeam_2p86GeV_PolarizedEbeam/`| Current macroparticle distributions from the positron linac. (thanks to Iryna Chaikovska)|
|`Damping Ring/`| Any files pertaining to current damping ring designs. | 
| `External Files/` | Third-party or supervisor-provided inputs and reference lattices. |
| `Assignments/`, `Slides and Notes/` | Write-ups, literature, presentations for my masters degree. |

#### Results folder set-up

Output folders are created by `paths.results_dir(design, config, phase, changes, metric, sub)`:

```bash
Results/
└── D{design}/C{config}/
    └── {phase}deg[_{changes}]/           e.g. 90deg_FDF
        ├── LatticeOptics/                analysis.py: optics, tunes, resonances, beta-beating
        │   └── {mode}/
        ├── DA_MA/                        analysis.py: dynamic aperture, momentum acceptance,
        │                                 seed overlays
        ├── InjectionEfficiency/          macroparticles.py
        │   ├── BeamSource/               injected-beam characterisation, TwissResults.json
        │   ├── {mode}/{label}_{E}MeV/    survival curves at a given injection energy
        │   └── SeedStudy/                misaligned/corrected seed overlays
        ├── Spin/
        │   ├── Scan/                     spin_tracking.py (misaligned vs corrected)
        │   └── Seed_{seed}/              spin_tracking_single_seed.py
        ├── PhysicalAperture/             macroparticles.py: chamber sizing and pole-tip fields
        └── ToleranceScan/                tolerance_scan.py (BeamStayClear/ = aperture it tracks with)
```


### Top-level scripts

| File | Role |
|---|---|
| `analysis.py` | Beam optics, tunes, damping, polarization, DA/MA. |
| `spin_tracking.py`,`spin_tracking_single_seed.py` | Spin-tracking / depolarization studies (single seed or multiple cases). |
| `macroparticles.py`| Injection efficiency simulations using the macroparticle distributions from the positron linac.|
| `config.py` | **Single source** of beam energy, RF, default working points, error sigmas, seeds and the default study. (`constants.py` is a deprecated alias.) |
| `paths.py` | Where lattices and results live: `lattice_json_path`, `load_lattice`, `results_dir`, `study_from_env`. Works from any directory. |
| `tolerance_scan.py` | Alignment/field-error tolerance scan (`--help` for options). Tracks with its own beam-stay-clear aperture (99.9 % beam containment + closed-orbit allowance), which is separate from the chamber design in `macroparticles.py`. |
| `wp_optimisation.py` | Working-point scans: cell phase advance, RDTs, realistic tune scan with DA. |
| `my_functions.py`| Shared helpers (plotting, lattice utilities, resonance scans, spin tracking functions). |
| `run_full_sims.py` | Batch driver: runs the scripts above for each lattice/mode. |

## Running a study

1. **Build the lattices** (`LatticeBuild/lattice_build.py`): set `design`, `config`, `phase`, `changes` in the *Study settings* cell, then run it three times with `mode = 'perfect'`, `'misaligned'`, `'corrected'`. Each run saves one JSON.
2. **Analyse** with `python run_full_sims.py` (edit the list at the bottom), or run one script directly. Scripts choose the lattice from the `DESIGN`, `CONFIG`, `MODE`, `PHASE`, `CHANGES` environment variables and fall back to `config.DEFAULT_STUDY`:

```bash
DESIGN=1 CONFIG=9 MODE=perfect PHASE=90 CHANGES=FDF python analysis.py
CHANGES= python analysis.py        # empty = no variant tag in the file name
```

All scripts can be launched from any folder.

## Shared helpers — use these rather than copying code

| Need | Use |
|---|---|
| Load a lattice | `pdr = paths.load_lattice(design, config, mode, phase, changes)` |
| Folder for outputs | `folder = paths.results_dir(design, config, phase, changes, metric='DA_MA')` |
| Add BPMs and correctors | `mc.insert_bpms_and_correctors(pdr, design, config)` |
| One misaligned (and corrected) machine | `line = mc.prepare_seed_line(pdr.lines['ring'], seed, correct=True)` |
| Orbit correction with threading fallback | `mc.correct_orbit_with_fallback(line, twiss, seed)` |
| Spin-tracking reference particle | `mc.setup_spin_reference(line)` |
| Constants, sigmas, seeds | `import config as cfg` → `cfg.E0`, `cfg.MISALIGN_SIGMA`, ... |

(`mc` is `LatticeBuild.misalignments_corrections`.) Corrector errors always use `seed + 1`, so a seed number fully defines a machine.

## Requirements

Python 3.10+, `pip install -r requirements.txt`. The DA toolkit in `xutil_DA_CC/` also imports `cpymad`, `pymadng` and `xplt`.

## Loading a lattice
A simple loading script that loads in the lattice JSON file and produces Twiss.

```
import paths
import LatticeBuild.misalignments_corrections as mc

pdr  = paths.load_lattice(design=1, config=9, mode='perfect', phase=90, changes='FDF')
ring = pdr.lines["ring"]

# 1. Reference particle: energy + electron anomalous moment (needed for spin).
mc.setup_spin_reference(ring)       # values from config.py

# 2. Enable spin transport.
ring.configure_spin("auto")

# 3. Radiation mode: 'mean' for twiss, 'quantum' for tracking.
ring.configure_radiation("mean")
tw = ring.twiss(method="6d", radiation_integrals=True, eneloss_and_damping=True,
                spin=True, polarization=True)
```

![](https://media.tenor.com/mMkJeuyHkRYAAAAj/cat-cat-on-computer.gif)

README last updated: 10/10/2026
