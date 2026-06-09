"""
tests/generate_b2_sample.py
============================
Generate synthetic B2 supercells (with optional APB) as LAMMPS dump files.

These samples are used by test_detection.py to validate the APB detector
against a known ground truth.  They can also be loaded into OVITO directly
for visual inspection.

APB construction
-----------------
An APB is created by TYPE SWAPPING across a plane (not atomic displacement).
For a {001} APB at z = z_apb:
  Lower half (z < z_apb):  corners → type_a,  body-centres → type_b
  Upper half (z ≥ z_apb):  corners → type_b,  body-centres → type_a  ← swapped

This is chemically equivalent to applying an a/2⟨111⟩ shift vector.

Expected detector response
---------------------------
  • lower body-centres at z = (k-0.5)a  →  4 same-type B2 NN  →  APB
  • upper corners      at z = k·a       →  4 same-type B2 NN  →  APB
  • all other interior atoms             →  0 same-type B2 NN  →  perfect B2
  • surface atoms                        →  PTM=OTHER          →  excluded
"""

import numpy as np
import os


def build_b2_supercell(
    nx: int,
    ny: int,
    nz: int,
    a: float = 2.86,
    type_a: int = 1,
    type_b: int = 2,
    apb_layer: int = None,
):
    """
    Build a B2 supercell atom array.

    Parameters
    ----------
    nx, ny, nz : int
        Repeat units along each axis.
    a : float
        Lattice parameter in Å.
    type_a : int
        LAMMPS particle type for the corner sublattice (e.g. Co = 1).
    type_b : int
        LAMMPS particle type for the body-centre sublattice (e.g. Al = 2).
    apb_layer : int or None
        If given, insert a {001} APB by swapping types for all unit cells
        with iz ≥ apb_layer.  E.g. ``apb_layer = nz // 2``.

    Returns
    -------
    atoms : ndarray, shape (2·nx·ny·nz, 5)
        Columns: atom_id, type, x, y, z
    box : dict with keys xlo, xhi, ylo, yhi, zlo, zhi
    z_apb : float or None
        z-coordinate of the APB plane (apb_layer * a), or None.
    """
    rows    = []
    atom_id = 1
    z_apb   = apb_layer * a if apb_layer is not None else None

    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                zc = iz * a
                zb = (iz + 0.5) * a
                in_upper = (apb_layer is not None and iz >= apb_layer)

                t_corner = type_b if in_upper else type_a
                t_bc     = type_a if in_upper else type_b

                rows.append([atom_id, t_corner, ix * a, iy * a, zc])
                atom_id += 1
                rows.append([atom_id, t_bc, (ix + 0.5) * a, (iy + 0.5) * a, zb])
                atom_id += 1

    atoms = np.array(rows)
    box   = dict(xlo=0.0, xhi=nx * a,
                 ylo=0.0, yhi=ny * a,
                 zlo=0.0, zhi=nz * a)
    return atoms, box, z_apb


def write_lammps_dump(path, atoms, box, timestep=0):
    """Write atoms to a LAMMPS custom dump file (ss ss ss boundaries)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"ITEM: TIMESTEP\n{timestep}\n")
        f.write(f"ITEM: NUMBER OF ATOMS\n{len(atoms)}\n")
        f.write("ITEM: BOX BOUNDS ss ss ss\n")
        f.write(f"{box['xlo']:.10e} {box['xhi']:.10e}\n")
        f.write(f"{box['ylo']:.10e} {box['yhi']:.10e}\n")
        f.write(f"{box['zlo']:.10e} {box['zhi']:.10e}\n")
        f.write("ITEM: ATOMS id type x y z\n")
        for row in atoms:
            f.write(f"{int(row[0])} {int(row[1])} "
                    f"{row[2]:.6f} {row[3]:.6f} {row[4]:.6f}\n")


def apb_ground_truth_z(apb_layer, a):
    """
    Return the two z-values where APB atoms sit for a {001} APB.

    These are:
      lower body-centre layer:  (apb_layer - 0.5) * a
      upper corner layer:       apb_layer * a
    """
    return [round((apb_layer - 0.5) * a, 4),
            round( apb_layer        * a, 4)]


if __name__ == "__main__":
    # Quick demo — generate and print summary
    atoms, box, z_apb = build_b2_supercell(14, 14, 14, a=2.86, apb_layer=7)
    out = os.path.join(os.path.dirname(__file__), "output", "b2_5k_apb.dump")
    write_lammps_dump(out, atoms, box)
    print(f"Written {len(atoms)} atoms → {out}")
    print(f"APB at z = {z_apb:.4f} Å")
