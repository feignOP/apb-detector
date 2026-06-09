# APB Detector

A Python tool for detecting **Anti-Phase Boundaries (APBs)** in B2 intermetallic alloys from LAMMPS molecular-dynamics simulations. Works for any B2 alloy (CoAl, NiAl, FeAl, TiAl, …). This was created using Claude and tested by me. 

---

## What is an Anti-Phase Boundary?

In a B2 intermetallic (e.g. CoAl), atoms occupy two interpenetrating simple-cubic sublattices:

```
Sublattice A (corners):       Co — Co — Co
Sublattice B (body-centres):  Al — Al — Al
```

An **Anti-Phase Boundary** is a planar defect where the sublattice assignment inverts across a plane — Co atoms appear on Al sites and vice versa. APBs are carried by partial dislocations and are key microstructural features under plastic deformation.

```
Perfect B2:       Co · Al · Co · Al · Co · Al
                              ↕  APB
APB region:       Co · Al · Co · Co · Al · Co
                                   ↑
                              sublattice inverted
```

## Detection Method

1. **Structure identification** via OVITO Polyhedral Template Matching (PTM). Only atoms classified as BCC or B2 enter the analysis — FCC regions, grain boundaries, and surfaces are excluded automatically.

2. **8-nearest-neighbour chemistry check** using a strain-independent rank-based query (SciPy KD-tree). For each B2-region atom, count how many of its 8 nearest neighbours are:
   - the **same type** as the atom itself, AND
   - also classified as **B2-structured** by PTM.

   | Atom environment        | `same_b2_nn` | `apb_flag` |
   |-------------------------|:------------:|:----------:|
   | Perfect B2 (interior)   | 0            | 0          |
   | APB plane atom          | 4            | **1**      |
   | Phase-transformed bcc   | 8            | 2 (excluded) |
   | FCC / surface / GB      | —            | -1         |

3. **Threshold**: `apb_nn_min = 4` (lower bound, catches {110} APBs) and `apb_nn_max = 4` (upper bound, excludes bcc-phase atoms that would otherwise be false positives).

## Repository Structure

```
apb-detector/
├── apb_detector/
│   ├── __init__.py
│   └── core.py            # detection logic — importable as a library
├── tests/
│   ├── generate_b2_sample.py   # synthetic B2 + APB sample generator
│   └── test_detection.py       # ground-truth validation test
├── detect_apb.py          # main CLI script
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
# 1. Clone
git clone https://github.com/<your-username>/apb-detector.git
cd apb-detector

# 2. Install OVITO via conda (pip version conflicts with Qt)
conda install -c conda-forge ovito

# 3. Install remaining dependencies
pip install -r requirements.txt
```

> **Python ≥ 3.9** and **OVITO ≥ 3.x** are required.

## Usage

### Run on a LAMMPS simulation

```bash
# Single dump file
python detect_apb.py path/to/dump.lammpstrj

# Multiple frames (quote the glob)
python detect_apb.py "GB_AlCo_sig13_*.cfg"

# With stress/strain plot (LAMMPS def1 thermo file)
python detect_apb.py "GB_AlCo_sig13_*.cfg" \
    --def1 CoAlNi_tens_100.def1.txt

# Custom atom types (default: type-a=1, type-b=2)
python detect_apb.py dump.lammpstrj --type-a 2 --type-b 1

# Skip already-processed frames, re-plot only
python detect_apb.py "GB_AlCo_sig13_*.cfg" --skip-if-done
```

Full options:

```
usage: detect_apb.py [-h] [--output-dir OUTPUT_DIR] [--def1 DEF1]
                     [--type-a TYPE_A] [--type-b TYPE_B]
                     [--n-nn N_NN] [--apb-min APB_MIN] [--apb-max APB_MAX]
                     [--rmsd-cutoff RMSD_CUTOFF] [--skip-if-done] [--no-dumps]
                     pattern
```

| Flag | Default | Description |
|------|---------|-------------|
| `pattern` | — | File path or glob pattern for dump/cfg files |
| `--output-dir` | `apb_output` | Directory for outputs |
| `--def1` | — | LAMMPS thermo file for stress/strain axes |
| `--type-a` | `1` | Particle type on sublattice A (corner sites) |
| `--type-b` | `2` | Particle type on sublattice B (body-centre sites) |
| `--n-nn` | `8` | Number of nearest neighbours |
| `--apb-min` | `4` | Min same-type B2 NN to flag as APB |
| `--apb-max` | `4` | Max same-type B2 NN to flag as APB |
| `--rmsd-cutoff` | `0.12` | PTM RMSD cutoff |
| `--skip-if-done` | off | Skip processing if all outputs already exist |
| `--no-dumps` | off | Skip writing per-frame dump files |

### Use as a Python library

```python
from ovito.io import import_file
from apb_detector import detect_apb_frame, get_b2_types, build_ptm_modifier

pipeline = import_file("dump.lammpstrj")
pipeline.modifiers.append(build_ptm_modifier())
data = pipeline.compute()

import numpy as np
pos    = np.array(data.particles["Position"])
ptype  = np.array(data.particles["Particle Type"])
struct = np.array(data.particles["Structure Type"])

apb_flag, same_b2_nn = detect_apb_frame(
    pos, ptype, struct,
    b2_types=get_b2_types(),
)
n_apb = (apb_flag == 1).sum()
print(f"APB atoms: {n_apb}")
```

## Output Files

| File | Description |
|------|-------------|
| `apb_output/apb_<timestep>.dump` | LAMMPS dump with `apb_flag` and `same_b2_nn` columns |
| `apb_output/apb_summary.csv` | Per-frame: timestep, APB fraction, atom counts |
| `apb_output/apb_fraction.png` | APB fraction vs timestep (or vs stress/strain) |

### `apb_flag` column values

| Value | Meaning |
|:-----:|---------|
| `-1` | Not B2 — FCC region, grain boundary, surface, or OTHER |
| `0` | Perfect B2 |
| `1` | **APB atom** |
| `2` | High same-type NN count (e.g. bcc phase-transformed region) |

Open any `apb_*.dump` in OVITO and colour by `apb_flag` to visualise the APB plane.

## Validation Test

A ground-truth test is included. It builds a synthetic 14×14×14 B2 supercell (5 488 atoms) with a known APB at the midplane, runs the full detector, and asserts zero false positives:

```bash
conda activate base
python tests/test_detection.py
```

Expected output:
```
✓ PASS — APB plane correctly detected, zero false positives.
```

The test also writes `tests/output/apb_5k_detected.dump` which you can open in OVITO for visual confirmation.

## Notes

- **Non-periodic boundaries**: The KD-tree query does not wrap around periodic boundaries. This is correct for simulations using `ss ss ss` (shrink-wrapped) boundaries. For periodic simulations, consider using OVITO's neighbor finder instead.
- **{110} vs {100} APBs**: The default threshold (4) targets {110} APBs, which are the most common under [001] compression of B2 CoAl. For {100} APBs (2 same-type NN per atom), set `--apb-min 2 --apb-max 2`.
- **Thermal noise**: At 300 K, thermal displacements occasionally place one same-type atom among the 8 NN. The threshold ≥ 4 eliminates this noise.

## Citation

If you use this tool in published work, please cite the simulation package and structure identification method:

- Larsen, P. M. et al. (2016). Robust structural identification via polyhedral template matching. *Modelling Simul. Mater. Sci. Eng.* **24**, 055007.
- OVITO: Stukowski, A. (2010). *Modelling Simul. Mater. Sci. Eng.* **18**, 015012.

## License

MIT — see [LICENSE](LICENSE).
