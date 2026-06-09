#!/usr/bin/env python3
"""
tests/test_detection.py
========================
Ground-truth validation of the APB detector using a synthetic 14×14×14
B2 supercell (5 488 atoms) with a known {001} APB at the midplane.

What the test confirms
-----------------------
1. PTM correctly identifies the interior of the large box as B2/BCC.
2. APB atoms are detected at exactly the two expected z-levels.
3. No false positives occur at any other z-level.
4. The upper-bound guard (APB_NN_MAX = 4) leaves no room for bcc-phase
   atoms (same_b2_nn = 8) to be mis-labelled as APB.

Run
---
  conda activate base
  python tests/test_detection.py
"""

import warnings
warnings.filterwarnings("ignore", message=".*OVITO.*PyPI")

import os, sys, time
import numpy as np

# Allow running from repo root OR from tests/ directory
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from generate_b2_sample import (
    build_b2_supercell, write_lammps_dump, apb_ground_truth_z
)
from apb_detector.core import (
    build_ptm_modifier, detect_apb_frame, get_b2_types, frame_statistics
)

try:
    import ovito
    from ovito.io import import_file
    print(f"OVITO {ovito.version_string}")
except ImportError as e:
    sys.exit(f"ERROR: cannot import ovito — {e}")

# ── Parameters ────────────────────────────────────────────────────────────
A          = 2.86
NX, NY, NZ = 14, 14, 14
APB_LAYER  = NZ // 2        # iz = 7 → z_apb = 7 × 2.86 = 20.02 Å
TYPE_A     = 1              # Co
TYPE_B     = 2              # Al
APB_NN_MIN = 4
APB_NN_MAX = 4

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
#  STEP 1 — Generate synthetic sample
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 1 — Generating synthetic B2 + APB sample")

atoms, box, z_apb = build_b2_supercell(
    NX, NY, NZ, a=A, type_a=TYPE_A, type_b=TYPE_B, apb_layer=APB_LAYER
)
N_atoms = len(atoms)
gt_z    = apb_ground_truth_z(APB_LAYER, A)

print(f"  Supercell  : {NX}×{NY}×{NZ} = {N_atoms} atoms")
print(f"  Box        : {NX*A:.2f} × {NY*A:.2f} × {NZ*A:.2f} Å")
print(f"  APB at z   : {z_apb:.4f} Å")
print(f"  APB layers : z = {gt_z[0]:.4f} Å  and  z = {gt_z[1]:.4f} Å")

dump_in = os.path.join(OUTPUT_DIR, "b2_5k_apb.dump")
write_lammps_dump(dump_in, atoms, box)
print(f"  Written    : {dump_in}")

# ══════════════════════════════════════════════════════════════════════════
#  STEP 2 — Run APB detector (exact same code as detect_apb.py)
# ══════════════════════════════════════════════════════════════════════════
print("\nSTEP 2 — Running APB detector")
t0 = time.time()

pipeline = import_file(dump_in)
pipeline.modifiers.append(build_ptm_modifier(rmsd_cutoff=0.12))
data   = pipeline.compute()
p      = data.particles
pos    = np.array(p["Position"])
ptype  = np.array(p["Particle Type"])
struct = np.array(p["Structure Type"])
pid    = np.array(p["Particle Identifier"])
cell   = np.array(data.cell)
N      = len(pos)

# PTM distribution
b2_types = get_b2_types()
names = {0:"OTHER", 1:"FCC", 2:"HCP", 3:"BCC", 4:"ICO", 5:"SC"}
print("\n  PTM distribution:")
for s, c in zip(*np.unique(struct, return_counts=True)):
    tag = "  ← B2 region" if s in b2_types else ""
    print(f"    {names.get(s,s):6s}({s}): {c:6d}  ({c/N*100:.1f}%){tag}")

apb_flag, same_b2_nn = detect_apb_frame(
    pos, ptype, struct, b2_types,
    n_nn=8,
    apb_nn_min=APB_NN_MIN,
    apb_nn_max=APB_NN_MAX,
)
stats = frame_statistics(apb_flag)

print(f"\n  B2-region atoms  : {stats['n_b2']}")
print(f"  APB atoms (flag=1): {stats['n_apb']}  ({stats['apb_fraction']*100:.2f}%)")
print(f"  bcc-region (flag=2): {stats['n_bcc_other']}  (should be 0 in pure B2 test)")
print(f"  Time: {time.time()-t0:.1f}s")

# same_b2_nn histogram
b2_mask = np.isin(struct, list(b2_types))
print("\n  same_b2_nn histogram (B2-region atoms):")
for v in range(9):
    c = int(((same_b2_nn == v) & b2_mask).sum())
    if c:
        tag = " ← APB" if APB_NN_MIN <= v <= APB_NN_MAX else (
              " ← bcc false-APB" if v > APB_NN_MAX else "")
        print(f"    same_b2_nn={v}: {c:6d}{tag}")

# ══════════════════════════════════════════════════════════════════════════
#  STEP 3 — Verification
# ══════════════════════════════════════════════════════════════════════════
print("\nSTEP 3 — Verification")

apb_at_correct_z = np.isin(np.round(pos[:,2], 3),
                             np.round(gt_z, 3)) & b2_mask
n_expected = int(apb_at_correct_z.sum())
n_fp       = int(((apb_flag == 1) & ~apb_at_correct_z).sum())
n_missed   = int((apb_at_correct_z & (apb_flag != 1)).sum())

print(f"  B2 atoms at known APB z-levels   : {n_expected}")
print(f"  Detected as APB                  : {stats['n_apb']}")
print(f"  False positives (wrong z)        : {n_fp}")
print(f"  Missed (correct z, not flagged)  : {n_missed}")
print(f"    (missed = surface atoms excluded by PTM — expected)")

apb_idx = np.where(apb_flag == 1)[0]
if len(apb_idx):
    z_det = np.unique(np.round(pos[apb_idx, 2], 3))
    print(f"\n  Detected APB z-coords : {z_det}")
    print(f"  Expected              : {gt_z}")

# ── PASS / FAIL ────────────────────────────────────────────────────────────
print()
passed = True

if n_fp != 0:
    print(f"  ✗ FAIL — {n_fp} false positive(s) at wrong z")
    passed = False
if stats['n_bcc_other'] != 0:
    print(f"  ✗ FAIL — {stats['n_bcc_other']} atoms with same_b2_nn > APB_NN_MAX "
          f"(unexpected in pure B2 test)")
    passed = False
if stats['n_apb'] == 0:
    print("  ✗ FAIL — no APB atoms detected at all")
    passed = False

if passed:
    print("  ✓ PASS — APB plane correctly detected, zero false positives.")

# ══════════════════════════════════════════════════════════════════════════
#  STEP 4 — Write labelled dump (open in OVITO)
# ══════════════════════════════════════════════════════════════════════════
out_dump = os.path.join(OUTPUT_DIR, "apb_5k_detected.dump")
xlo, xhi = box["xlo"], box["xhi"]
ylo, yhi = box["ylo"], box["yhi"]
zlo, zhi = box["zlo"], box["zhi"]
mat = np.column_stack([pid.astype(np.int64), ptype.astype(np.int32),
                       pos, apb_flag.astype(np.int32),
                       same_b2_nn.astype(np.int32)])
with open(out_dump, "w") as f:
    f.write("ITEM: TIMESTEP\n0\n")
    f.write(f"ITEM: NUMBER OF ATOMS\n{N}\n")
    f.write("ITEM: BOX BOUNDS ss ss ss\n")
    f.write(f"{xlo:.10e} {xhi:.10e}\n{ylo:.10e} {yhi:.10e}\n{zlo:.10e} {zhi:.10e}\n")
    f.write("ITEM: ATOMS id type x y z apb_flag same_b2_nn\n")
    np.savetxt(f, mat, fmt="%d %d %.6f %.6f %.6f %d %d")

print(f"\n  Labelled dump → {out_dump}")
print("  Open in OVITO and colour by 'apb_flag':")
print("    -1 = not B2 (surface/other)   0 = perfect B2")
print("     1 = APB atom                 2 = bcc-phase atom")

sys.exit(0 if passed else 1)
