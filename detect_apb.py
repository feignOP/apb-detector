#!/usr/bin/env python3
"""
detect_apb.py — General-purpose APB detection for B2 intermetallic alloys
==========================================================================

Runs on any LAMMPS dump file (or glob of dump files) and produces:
  • Per-frame LAMMPS dump files with an ``apb_flag`` column
  • CSV summary (timestep, APB fraction, atom counts)
  • Optional stress/strain plot when a LAMMPS def1 thermo file is supplied

Usage
-----
  # Single dump file
  python detect_apb.py dump.lammpstrj

  # Glob of dump files (quote the pattern)
  python detect_apb.py "GB_AlCo_sig13_*.cfg"

  # With stress/strain plot
  python detect_apb.py "GB_AlCo_sig13_*.cfg" --def1 CoAlNi_tens_100.def1.txt

  # Override atom types (default: type_a=1, type_b=2)
  python detect_apb.py dump.lammpstrj --type-a 2 --type-b 1

  # Change APB threshold (default 4; use 2 for {100} APBs)
  python detect_apb.py dump.lammpstrj --apb-min 4 --apb-max 4

  # Skip reprocessing already-done frames (re-plot only)
  python detect_apb.py "GB_AlCo_sig13_*.cfg" --skip-if-done

apb_flag values
---------------
  -1  not in B2 region (FCC, grain boundary, surface, OTHER)
   0  perfect B2
   1  APB atom
   2  high same-type NN count (e.g. bcc phase-transformed region)
"""

import os, sys, glob, time, argparse, warnings
warnings.filterwarnings("ignore", message=".*OVITO.*PyPI")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── OVITO ──────────────────────────────────────────────────────────────────
try:
    import ovito
    from ovito.io import import_file
    from apb_detector.core import (
        build_ptm_modifier, detect_apb_frame, get_b2_types, frame_statistics
    )
    print(f"OVITO {ovito.version_string}  |  apb_detector loaded")
except ImportError as e:
    sys.exit(f"ERROR: {e}\n"
             "Install dependencies:  pip install -r requirements.txt\n"
             "OVITO:                 conda install -c conda-forge ovito")


# ══════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="Detect Anti-Phase Boundaries in B2 intermetallic alloys "
                    "from LAMMPS dump files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("pattern",
                   help="Path or glob pattern for LAMMPS dump/cfg files "
                        "(e.g. 'dump.lammpstrj' or 'GB_*.cfg')")
    p.add_argument("--output-dir", default="apb_output",
                   help="Directory for output dump files, CSV and plot")
    p.add_argument("--def1",
                   help="LAMMPS thermo def1 file for stress/strain plot "
                        "(columns: _ _ _ -pzz/10000 strain_col …)")
    p.add_argument("--n-frames", type=int, default=None,
                   help="Expected number of frames (used to pick rows from "
                        "def1 file). Auto-detected when omitted.")
    p.add_argument("--type-a", type=int, default=1,
                   help="LAMMPS particle type on sublattice A (e.g. Co=1)")
    p.add_argument("--type-b", type=int, default=2,
                   help="LAMMPS particle type on sublattice B (e.g. Al=2)")
    p.add_argument("--n-nn", type=int, default=8,
                   help="Number of nearest neighbours (8 for B2 first shell)")
    p.add_argument("--apb-min", type=int, default=4,
                   help="Min same-type B2 NN to flag as APB")
    p.add_argument("--apb-max", type=int, default=4,
                   help="Max same-type B2 NN to flag as APB "
                        "(upper bound excludes bcc-phase atoms)")
    p.add_argument("--rmsd-cutoff", type=float, default=0.12,
                   help="PTM RMSD cutoff for structure identification")
    p.add_argument("--skip-if-done", action="store_true",
                   help="If all output dumps exist, skip processing and "
                        "re-plot only")
    p.add_argument("--no-dumps", action="store_true",
                   help="Do not write per-frame dump files (only CSV + plot)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════
def _timestep_from_path(path):
    """Extract integer timestep from a filename like *_<ts>.cfg or *_<ts>.dump."""
    stem = os.path.splitext(os.path.basename(path))[0]
    for part in reversed(stem.replace("-", "_").split("_")):
        try:
            return int(part)
        except ValueError:
            continue
    raise ValueError(f"Cannot extract timestep from filename: {path}")


def write_dump(path, timestep, box_bounds, atom_ids, ptypes, positions,
               apb_flag, same_b2_nn):
    """Write a LAMMPS custom dump with apb_flag and same_b2_nn columns."""
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = box_bounds
    mat = np.column_stack([
        atom_ids.astype(np.int64),
        ptypes.astype(np.int32),
        positions.astype(np.float64),
        apb_flag.astype(np.int32),
        same_b2_nn.astype(np.int32),
    ])
    with open(path, "w") as f:
        f.write(f"ITEM: TIMESTEP\n{timestep}\n")
        f.write(f"ITEM: NUMBER OF ATOMS\n{len(atom_ids)}\n")
        f.write("ITEM: BOX BOUNDS ss ss ss\n")
        f.write(f"{xlo:.10e} {xhi:.10e}\n")
        f.write(f"{ylo:.10e} {yhi:.10e}\n")
        f.write(f"{zlo:.10e} {zhi:.10e}\n")
        f.write("ITEM: ATOMS id type x y z apb_flag same_b2_nn\n")
        np.savetxt(f, mat, fmt="%d %d %.6f %.6f %.6f %d %d")


def load_def1(path, n_frames):
    """
    Load stress/strain from a LAMMPS def1 thermo file.

    Picks ``n_frames`` evenly-spaced rows.  Assumes:
      col 3 (0-indexed) = -pzz/10000  →  True Stress (GPa) = -1 × col3
      col 4             = strain_raw  →  True Strain (%)   = col4 / 10
    """
    data = np.loadtxt(path, skiprows=1)
    idx  = np.round(np.linspace(0, len(data) - 1, n_frames)).astype(int)
    return -1.0 * data[idx, 3], data[idx, 4] / 10.0


def _box_bounds(cell):
    """Extract ((xlo,xhi),(ylo,yhi),(zlo,zhi)) from OVITO cell matrix."""
    origin = cell[:3, 3]
    return (
        (origin[0], origin[0] + cell[0, 0]),
        (origin[1], origin[1] + cell[1, 1]),
        (origin[2], origin[2] + cell[2, 2]),
    )


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Collect input files ───────────────────────────────────────────────
    input_files = sorted(glob.glob(args.pattern), key=_timestep_from_path)
    if not input_files:
        sys.exit(f"ERROR: no files matched pattern '{args.pattern}'")
    print(f"Found {len(input_files)} input file(s)")

    b2_types = get_b2_types()

    # ── Consistency check ─────────────────────────────────────────────────
    input_ts  = {_timestep_from_path(f) for f in input_files}
    output_ts = {_timestep_from_path(f)
                 for f in glob.glob(os.path.join(args.output_dir, "apb_*.dump"))}

    skip = False
    if output_ts and args.skip_if_done:
        if output_ts == input_ts:
            print("All output frames present — skipping processing, re-plotting only.")
            skip = True
        elif output_ts < input_ts:
            missing = sorted(input_ts - output_ts)
            sys.exit(f"ERROR: {len(missing)} output frame(s) missing: {missing}\n"
                     "Delete apb_output/ and re-run, or remove --skip-if-done.")

    # ── Processing loop ───────────────────────────────────────────────────
    results = []   # (timestep, apb_fraction, n_b2, n_apb, n_bcc_other)

    if not skip:
        for i, fpath in enumerate(input_files):
            ts = _timestep_from_path(fpath)
            t0 = time.time()
            print(f"[{i+1:3d}/{len(input_files)}]  t={ts:>9d}  ", end="", flush=True)

            pipeline = import_file(fpath)
            pipeline.modifiers.append(build_ptm_modifier(args.rmsd_cutoff))
            data  = pipeline.compute()
            p     = data.particles

            pos    = np.array(p["Position"])
            ptype  = np.array(p["Particle Type"])
            struct = np.array(p["Structure Type"])
            pid    = np.array(p["Particle Identifier"])
            cell   = np.array(data.cell)

            apb_flag, same_b2_nn = detect_apb_frame(
                pos, ptype, struct, b2_types,
                n_nn=args.n_nn,
                apb_nn_min=args.apb_min,
                apb_nn_max=args.apb_max,
            )
            stats = frame_statistics(apb_flag)
            results.append((ts, stats["apb_fraction"],
                             stats["n_b2"], stats["n_apb"], stats["n_bcc_other"]))

            print(f"B2={stats['n_b2']:7d}  APB={stats['n_apb']:6d} "
                  f"({stats['apb_fraction']*100:.3f}%)  "
                  f"bcc-region={stats['n_bcc_other']:5d}  "
                  f"[{time.time()-t0:.1f}s]")

            if not args.no_dumps:
                out_path = os.path.join(args.output_dir, f"apb_{ts}.dump")
                write_dump(out_path, ts, _box_bounds(cell),
                           pid, ptype, pos, apb_flag, same_b2_nn)

        print()

    else:
        # Re-load fractions from existing dumps
        for f in sorted(glob.glob(os.path.join(args.output_dir, "apb_*.dump")),
                        key=_timestep_from_path):
            ts   = _timestep_from_path(f)
            data = np.loadtxt(f, skiprows=9, usecols=5)
            n_b2       = int((data >= 0).sum())
            n_apb      = int((data == 1).sum())
            n_bcc      = int((data == 2).sum())
            frac       = n_apb / n_b2 if n_b2 > 0 else 0.0
            results.append((ts, frac, n_b2, n_apb, n_bcc))
        results.sort()
        print(f"Loaded {len(results)} frames from existing dumps.\n")

    if not results:
        sys.exit("No results to plot.")

    # ── CSV summary ───────────────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, "apb_summary.csv")
    with open(csv_path, "w") as f:
        f.write("timestep,apb_fraction,n_b2,n_apb,n_bcc_other\n")
        for r in results:
            f.write(f"{r[0]},{r[1]:.6f},{r[2]},{r[3]},{r[4]}\n")
    print(f"CSV  → {csv_path}")

    # ── Plot ──────────────────────────────────────────────────────────────
    frac_arr = np.array([r[1] for r in results]) * 100
    ts_arr   = np.array([r[0] for r in results])

    if args.def1:
        stress, strain = load_def1(args.def1, len(results))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(stress, frac_arr, "b-o", ms=3, lw=1)
        axes[0].set_xlabel("True Stress (GPa)"); axes[0].set_ylabel("APB fraction (%)")
        axes[0].set_title("APB fraction vs True Stress"); axes[0].grid(alpha=0.3)
        axes[1].plot(strain, frac_arr, "r-o", ms=3, lw=1)
        axes[1].set_xlabel("True Strain (%)"); axes[1].set_ylabel("APB fraction (%)")
        axes[1].set_title("APB fraction vs True Strain"); axes[1].grid(alpha=0.3)
    else:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(ts_arr, frac_arr, "b-o", ms=3, lw=1)
        ax.set_xlabel("Timestep"); ax.set_ylabel("APB fraction (%)")
        ax.set_title("APB fraction vs Timestep"); ax.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, "apb_fraction.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot → {plot_path}")
    print(f"\nAPB range: {frac_arr.min():.3f}% → {frac_arr.max():.3f}%")


if __name__ == "__main__":
    main()
