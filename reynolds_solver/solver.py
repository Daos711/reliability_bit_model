"""
SOR solver for the non-dimensional Reynolds equation on a (phi, Z) grid.

Steady-state + squeeze-film Reynolds equation (non-dimensional):
    d/dphi [ H^3 dP/dphi ] + (R/L)^2 d/dZ [ H^3 dP/dZ ] = dH/dphi + F_dyn

where F_dyn = xprime * sin(phi) + yprime * cos(phi)

Cavitation (half-Sommerfeld): P = max(P, 0) after each update.
Boundary conditions: P = 0 at Z = -1 and Z = +1 (axial edges).
Periodic in phi.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def _sor_kernel(P, H3_ip, H3_im, H3_jp, H3_jm, A, RHS,
                inv_dphi2, inv_dZ2, omega, tol, max_iter, nZ, nPhi):
    """Gauss-Seidel SOR inner loop with half-Sommerfeld cavitation."""
    delta = 1.0
    n_iter = 0

    for it in range(max_iter):
        max_change = 0.0
        P_max = 0.0

        for j in range(1, nZ - 1):
            for i in range(nPhi):
                ip = (i + 1) % nPhi
                im = (i - 1) % nPhi

                numerator = (
                    H3_ip[j, i] * inv_dphi2 * P[j, ip]
                    + H3_im[j, i] * inv_dphi2 * P[j, im]
                    + H3_jp[j, i] * inv_dZ2 * P[j + 1, i]
                    + H3_jm[j, i] * inv_dZ2 * P[j - 1, i]
                    - RHS[j, i]
                )

                denom = A[j, i]
                if denom == 0.0:
                    continue

                P_gs = numerator / denom
                if P_gs < 0.0:
                    P_gs = 0.0

                P_new = P[j, i] + omega * (P_gs - P[j, i])
                if P_new < 0.0:
                    P_new = 0.0

                change = abs(P_new - P[j, i])
                if change > max_change:
                    max_change = change

                P[j, i] = P_new

                if P_new > P_max:
                    P_max = P_new

        n_iter = it + 1
        delta = max_change / (P_max + 1e-30)

        if delta < tol:
            break

    return P, delta, n_iter


def solve_reynolds(H, d_phi, d_Z, R, L, omega=1.5,
                   xprime=0.0, yprime=0.0,
                   tol=1e-6, max_iter=50000):
    """
    Solve the Reynolds equation using SOR iteration (Numba-accelerated).

    Parameters
    ----------
    H : ndarray, shape (nZ, nPhi)
        Non-dimensional film thickness field.
    d_phi : float
        Grid spacing in phi direction.
    d_Z : float
        Grid spacing in Z direction.
    R : float
        Bearing radius [m].
    L : float
        Bearing length [m].
    omega : float
        SOR relaxation parameter (1 < omega < 2).
    xprime : float
        Non-dimensional squeeze velocity component (multiplies sin(phi)).
    yprime : float
        Non-dimensional squeeze velocity component (multiplies cos(phi)).
    tol : float
        Convergence tolerance on max relative pressure change.
    max_iter : int
        Maximum number of SOR iterations.

    Returns
    -------
    P : ndarray, shape (nZ, nPhi)
        Non-dimensional pressure field (>= 0).
    delta : float
        Final relative residual.
    n_iter : int
        Number of iterations performed.
    """
    nZ, nPhi = H.shape
    P = np.zeros((nZ, nPhi), dtype=np.float64)

    phi = np.arange(nPhi) * d_phi
    RL2 = (R / L) ** 2

    H3 = H ** 3

    # RHS: dH/dphi (central diff, periodic) + dynamic term
    dH_dphi = np.empty_like(H)
    dH_dphi[:, 1:-1] = (H[:, 2:] - H[:, :-2]) / (2.0 * d_phi)
    dH_dphi[:, 0] = (H[:, 1] - H[:, -1]) / (2.0 * d_phi)
    dH_dphi[:, -1] = (H[:, 0] - H[:, -2]) / (2.0 * d_phi)

    sin_phi = np.sin(phi)[np.newaxis, :]
    cos_phi = np.cos(phi)[np.newaxis, :]
    F_dyn = xprime * sin_phi + yprime * cos_phi
    RHS = dH_dphi + F_dyn

    # H^3 at half-nodes (phi direction, periodic)
    H3_ip = np.empty_like(H3)
    H3_im = np.empty_like(H3)
    H3_ip[:, :-1] = 0.5 * (H3[:, :-1] + H3[:, 1:])
    H3_ip[:, -1] = 0.5 * (H3[:, -1] + H3[:, 0])
    H3_im[:, 1:] = 0.5 * (H3[:, 1:] + H3[:, :-1])
    H3_im[:, 0] = 0.5 * (H3[:, 0] + H3[:, -1])

    # H^3 at half-nodes (Z direction, non-periodic)
    H3_jp = np.zeros_like(H3)
    H3_jm = np.zeros_like(H3)
    H3_jp[:-1, :] = 0.5 * (H3[:-1, :] + H3[1:, :])
    H3_jm[1:, :] = 0.5 * (H3[1:, :] + H3[:-1, :])

    inv_dphi2 = 1.0 / (d_phi ** 2)
    inv_dZ2 = RL2 / (d_Z ** 2)

    A = (H3_ip + H3_im) * inv_dphi2 + (H3_jp + H3_jm) * inv_dZ2

    # Ensure contiguous float64 arrays for numba
    P = np.ascontiguousarray(P, dtype=np.float64)
    H3_ip = np.ascontiguousarray(H3_ip, dtype=np.float64)
    H3_im = np.ascontiguousarray(H3_im, dtype=np.float64)
    H3_jp = np.ascontiguousarray(H3_jp, dtype=np.float64)
    H3_jm = np.ascontiguousarray(H3_jm, dtype=np.float64)
    A = np.ascontiguousarray(A, dtype=np.float64)
    RHS = np.ascontiguousarray(RHS, dtype=np.float64)

    P, delta, n_iter = _sor_kernel(
        P, H3_ip, H3_im, H3_jp, H3_jm, A, RHS,
        inv_dphi2, inv_dZ2, omega, tol, max_iter, nZ, nPhi
    )

    return P, delta, n_iter
