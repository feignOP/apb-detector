"""
apb_detector
============
General-purpose Anti-Phase Boundary (APB) detection for B2 intermetallic
alloys from LAMMPS molecular-dynamics simulations.

Quick start
-----------
>>> from apb_detector.core import detect_apb_frame, get_b2_types, build_ptm_modifier
"""

from .core import detect_apb_frame, get_b2_types, build_ptm_modifier, frame_statistics

__version__ = "1.0.0"
__all__ = ["detect_apb_frame", "get_b2_types", "build_ptm_modifier", "frame_statistics"]
