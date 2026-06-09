"""
apb_detector.core
=================
General-purpose Anti-Phase Boundary (APB) detection for B2 intermetallic
alloys from LAMMPS molecular-dynamics output.

Works for any B2 structure (CoAl, NiAl, FeAl, TiAl, …) — the alloy
chemistry is not hardcoded.

Detection principle
-------------------
In a perfect B2 lattice every atom has exactly 8 nearest neighbours of the
OPPOSITE type.  At an APB the local sublattice assignment inverts across a
plane, so atoms sitting on the boundary plane see 4 NN on the correct side
(unlike type) and 4 NN on the shifted side (same type):

  Perfect B2  →  same-type B2 NN count = 0
  APB plane   →  same-type B2 NN count = 4
  bcc-region* →  same-type B2 NN count = 8

  * bcc-structured atoms of a single element (e.g. fcc→bcc Co under
    compression) have all 8 NN of the same type and would be false
    positives without the APB_NN_MAX upper-bound guard.

The B2 region is identified by OVITO Polyhedral Template Matching (PTM):
only atoms classified as BCC or B2 by PTM enter the analysis.  This
automatically excludes grain boundaries, surfaces, and FCC regions.

References
----------
Yamaguchi, M. & Umakoshi, Y. (1990). The deformation behaviour of
intermetallic superlattice compounds. Progress in Materials Science, 34(1).
"""

import warnings
warnings.filterwarnings("ignore", message=".*OVITO.*PyPI")

import numpy as np
from scipy.spatial import cKDTree


# ── OVITO PTM type constants ───────────────────────────────────────────────
def _get_ptm_ids():
    from ovito.modifiers import PolyhedralTemplateMatchingModifier as PTM
    fcc_id = int(PTM.Type.FCC)
    bcc_id = int(PTM.Type.BCC)
    try:
        b2_id = int(PTM.Type.B2)
    except AttributeError:
        b2_id = bcc_id          # OVITO < 3.x: B2 appears as BCC
    return fcc_id, bcc_id, b2_id


def build_ptm_modifier(rmsd_cutoff: float = 0.12):
    """Return a configured PTM modifier (FCC + BCC + B2 enabled)."""
    from ovito.modifiers import PolyhedralTemplateMatchingModifier as PTM
    mod = PTM(rmsd_cutoff=rmsd_cutoff)
    mod.structures[PTM.Type.FCC].enabled = True
    mod.structures[PTM.Type.BCC].enabled = True
    try:
        mod.structures[PTM.Type.B2].enabled = True
    except AttributeError:
        pass
    for st in [PTM.Type.HCP, PTM.Type.ICO, PTM.Type.SC]:
        mod.structures[st].enabled = False
    return mod


def detect_apb_frame(
    pos,
    ptype,
    struct,
    b2_types,
    n_nn: int = 8,
    apb_nn_min: int = 4,
    apb_nn_max: int = 4,
):
    """
    Detect APB atoms in a single simulation frame.

    Parameters
    ----------
    pos : array-like, shape (N, 3)
        Cartesian atomic positions.
    ptype : array-like, shape (N,)
        Integer particle-type labels (from LAMMPS `type` column).
    struct : array-like, shape (N,)
        PTM structure-type integer per atom (output of OVITO PTM modifier).
    b2_types : set of int
        PTM type IDs that correspond to the B2 (or BCC) structure.
        Use ``get_b2_types()`` to obtain these automatically.
    n_nn : int, optional
        Number of nearest neighbours to consider.  8 for BCC/B2 first shell.
    apb_nn_min : int, optional
        Minimum same-type B2-NN count to flag an atom as APB.  Default 4.
    apb_nn_max : int, optional
        Maximum same-type B2-NN count to flag as APB.  Default 4.
        Upper bound guards against bcc-phase atoms (same_b2_nn = 8)
        being mis-labelled as APB.

    Returns
    -------
    apb_flag : ndarray, shape (N,), dtype int32
        -1  not in B2 region (FCC, GB, surface, OTHER)
         0  perfect B2
         1  APB atom   (apb_nn_min ≤ same_b2_nn ≤ apb_nn_max)
         2  high same-type count (e.g. phase-transformed bcc)
    same_b2_nn : ndarray, shape (N,), dtype int32
        Per-atom count of same-type neighbours that are also B2-structured.
    """
    pos   = np.asarray(pos,   dtype=np.float64)
    ptype = np.asarray(ptype, dtype=np.int32)
    struct = np.asarray(struct, dtype=np.int32)
    N = len(pos)

    # --- B2 region mask ---------------------------------------------------
    b2_mask = np.isin(struct, list(b2_types))

    # --- 8-NN by rank (strain-independent, no periodic wrapping needed) ---
    k = min(n_nn + 1, N)
    _, nn_idx  = cKDTree(pos).query(pos, k=k, workers=-1)
    nn_idx     = nn_idx[:, 1:]          # (N, n_nn) — exclude self

    nn_types   = ptype[nn_idx]
    nn_in_b2   = b2_mask[nn_idx]
    same_type  = nn_types == ptype[:, np.newaxis]

    same_b2_nn = (same_type & nn_in_b2).sum(axis=1).astype(np.int32)

    # --- Classify ---------------------------------------------------------
    apb_flag = np.full(N, -1, dtype=np.int32)
    apb_flag[b2_mask & (same_b2_nn <  apb_nn_min)]                              = 0
    apb_flag[b2_mask & (same_b2_nn >= apb_nn_min) & (same_b2_nn <= apb_nn_max)] = 1
    apb_flag[b2_mask & (same_b2_nn >  apb_nn_max)]                              = 2

    return apb_flag, same_b2_nn


def get_b2_types():
    """
    Return the set of OVITO PTM integer IDs that map to B2 / BCC structure.

    In OVITO ≥ 3.x both ``PTM.Type.BCC`` and ``PTM.Type.B2`` are available;
    in older builds B2 CoAl is reported as BCC only.
    """
    _, bcc_id, b2_id = _get_ptm_ids()
    return {bcc_id, b2_id}


def frame_statistics(apb_flag):
    """
    Compute summary statistics from an apb_flag array.

    Returns
    -------
    dict with keys: n_b2, n_apb, n_bcc_other, apb_fraction
    """
    n_b2       = int((apb_flag >= 0).sum())
    n_apb      = int((apb_flag == 1).sum())
    n_bcc_other = int((apb_flag == 2).sum())
    frac       = n_apb / n_b2 if n_b2 > 0 else 0.0
    return dict(n_b2=n_b2, n_apb=n_apb, n_bcc_other=n_bcc_other,
                apb_fraction=frac)
