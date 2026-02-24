"""
Utility functions for the Reynolds solver.

Includes texture (ellipsoidal depression) generation.
"""

import numpy as np


def create_H_with_ellipsoidal_depressions(H0, H_p, Phi_mesh, Z_mesh,
                                           phi_c_flat, Z_c_flat,
                                           A_tex, B_tex):
    """
    Add ellipsoidal depressions to a baseline film thickness field.

    Each depression is an ellipsoidal cap subtracted from the surface,
    increasing the local gap by up to H_p at the centre.

    Parameters
    ----------
    H0 : ndarray, shape (nZ, nPhi)
        Baseline non-dimensional film thickness.
    H_p : float
        Non-dimensional depth of each depression.
    Phi_mesh : ndarray, shape (nZ, nPhi)
        Phi coordinates of the grid.
    Z_mesh : ndarray, shape (nZ, nPhi)
        Z coordinates of the grid (non-dimensional, in [-1, 1]).
    phi_c_flat : 1-D array
        Phi-coordinates of depression centres.
    Z_c_flat : 1-D array
        Z-coordinates of depression centres (same length as phi_c_flat).
    A_tex : float
        Non-dimensional semi-axis of depression in Z direction.
    B_tex : float
        Non-dimensional semi-axis of depression in phi direction.

    Returns
    -------
    H : ndarray, shape (nZ, nPhi)
        Film thickness with depressions added.
    """
    H = H0.copy()

    for k in range(len(phi_c_flat)):
        dphi = Phi_mesh - phi_c_flat[k]
        # Handle periodicity in phi
        dphi = np.mod(dphi + np.pi, 2 * np.pi) - np.pi

        dZ = Z_mesh - Z_c_flat[k]

        r2 = (dphi / B_tex) ** 2 + (dZ / A_tex) ** 2
        mask = r2 <= 1.0
        H[mask] += H_p * (1.0 - r2[mask])

    return H
