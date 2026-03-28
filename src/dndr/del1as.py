from __future__ import annotations

import numpy as np
from scipy.linalg import eig


def del1as(u: np.ndarray, l: np.ndarray, D: np.ndarray, n: int | None = None):
    """
    Python translation of SEC/Del1AS.m.

    Parameters
    ----------
    u : array, shape (N, n0)
        0-Laplacian eigenfunctions evaluated on the sample points.
    l : array, shape (n0,)
        0-Laplacian eigenvalues.
    D : array, shape (N, N)
        Inner-product matrix with u.T @ D @ u ≈ I.
    n : int, optional
        Number of eigenfunctions to use in the antisymmetric SEC frame.

    Returns
    -------
    U, L, D1, G, H, cijk
    """
    u = np.asarray(u, dtype=float)
    l = np.asarray(l, dtype=float).reshape(-1)
    D = np.asarray(D, dtype=float)

    n0 = u.shape[1]
    if n is None:
        n = n0

    Du = D @ u

    # cijk(i,j,k) = <phi_i phi_j, phi_k>
    cijk = np.einsum("ri,rj,rk->ijk", u[:, :n], u[:, :n], Du, optimize=True)

    # Literal translation of the MATLAB tensor algebra.
    l1 = np.tile(l.reshape(-1, 1, 1, 1, 1), (1, n, n, n, n))
    l2 = np.tile(l[:n].reshape(-1, 1, 1, 1, 1), (1, n, n0, n, n))

    h0 = np.tile(cijk[:, :, :, None, None], (1, 1, 1, n, n))
    h0 = np.squeeze(
        np.sum(
            np.transpose(h0, (3, 0, 2, 4, 1))
            * np.transpose(h0, (0, 3, 2, 1, 4))
            * (
                np.transpose(l2, (1, 0, 2, 3, 4))
                + np.transpose(l2, (3, 1, 2, 4, 0))
                - np.transpose(l1, (2, 1, 0, 4, 3))
            ),
            axis=2,
        )
    )

    H = h0 - np.transpose(h0, (1, 0, 2, 3))
    G = h0 + np.transpose(h0, (1, 0, 3, 2)) - np.transpose(h0, (0, 1, 3, 2)) - np.transpose(h0, (1, 0, 2, 3))

    D1 = np.tile(cijk[:, :, :, None, None], (1, 1, 1, n, n))
    lambdas1 = (
        np.transpose(l1, (1, 2, 0, 3, 4)) ** 2
        - np.transpose(l1, (1, 2, 0, 3, 4))
        * (
            l2
            + np.transpose(l2, (1, 0, 2, 3, 4))
            + np.transpose(l2, (1, 3, 2, 0, 4))
            + np.transpose(l2, (1, 3, 2, 4, 0))
        )
    )

    D1 = 2.0 * np.squeeze(
        np.sum(
            (
                np.transpose(D1, (1, 4, 2, 0, 3)) * np.transpose(D1, (4, 1, 2, 3, 0))
                - np.transpose(D1, (4, 1, 2, 0, 3)) * np.transpose(D1, (1, 4, 2, 3, 0))
            )
            * lambdas1,
            axis=2,
        )
    )

    D1 = D1.reshape(n * n, n * n)
    G = G.reshape(n * n, n * n)
    H = H.reshape(n * n, n * n)

    D1 = 0.5 * (D1 + D1.T)
    G = 0.5 * (G + G.T)

    # Sobolev H^1 basis reduction
    Ut, St, _ = np.linalg.svd(D1 + G, full_matrices=False)
    St = np.asarray(St)
    below = np.where(St / St[0] < 1e-3)[0]
    NN = int(below[0]) if below.size else len(St)

    D1proj = Ut[:, :NN].T @ D1 @ Ut[:, :NN]
    D1proj = 0.5 * (D1proj + D1proj.T)

    Gproj = Ut[:, :NN].T @ G @ Ut[:, :NN]
    Gproj = 0.5 * (Gproj + Gproj.T)

    # MATLAB uses eig(D1proj,Gproj)
    vals, vecs = eig(D1proj, Gproj)
    order = np.argsort(np.abs(vals))
    vals = np.real(vals[order])
    vecs = np.real(vecs[:, order])

    U = Ut[:, :NN] @ vecs
    L = vals
    return U, L, D1, G, H, cijk
