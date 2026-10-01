"""
uhi_common.py - shared helpers for scene-level masking and Landsat C2 scaling.
"""
import os
import numpy as np
import rasterio

# Landsat Collection 2 Level-2 valid ranges (Science Product Guide)
SR_DN_MIN, SR_DN_MAX = 7273, 43636     # surface reflectance 0-1 (uint16 DN)
LST_MIN_K, LST_MAX_K = 273.15, 345.0   # plausible land surface temperature


def read_mtl(mtl_path):
    """Return MTL key/value pairs as a dict (first occurrence wins, i.e. Landsat 9 block)."""
    out = {}
    with open(mtl_path) as f:
        for line in f:
            if "=" in line:
                k, v = [x.strip().strip('"') for x in line.split("=", 1)]
                out.setdefault(k, v)
    return out


def qa_clear_mask(qa_path, shape):
    """
    Boolean array, True where the pixel is clear.
    Masks fill (bit0), dilated cloud (1), cirrus (2), cloud (3), cloud shadow (4), snow (5).
    Returns None if the QA file is not available.
    """
    if not os.path.exists(qa_path):
        return None
    with rasterio.open(qa_path) as src:
        qa = src.read(1)
    if qa.shape != shape:
        raise ValueError(f"QA_PIXEL shape {qa.shape} != scene shape {shape}")
    bad = 0
    for bit in (0, 1, 2, 3, 4, 5):
        bad |= (qa >> bit) & 1
    return bad == 0
