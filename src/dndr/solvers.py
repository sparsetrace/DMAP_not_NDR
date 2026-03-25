import numpy as np
from sklearn.manifold import Isomap


def normalize_embedding(Q_ix, *, center=True, scale=True):
    Q_ix = np.asarray(Q_ix, dtype=float).copy()

    if center:
        Q_ix -= Q_ix.mean(axis=0, keepdims=True)

    if scale:
        fro = np.linalg.norm(Q_ix, ord="fro")
        if fro > 0:
            Q_ix /= fro

    return Q_ix


def solve_linear_map(Q_ix, U_in, *, fit_intercept=True, method="lstsq", ridge=0.0, rcond=None):
    """
    Solve Q_ix ~= U_in @ L_nx (+ intercept) in least squares.
    """
    Q_ix = np.asarray(Q_ix, dtype=float)
    U_in = np.asarray(U_in, dtype=float)

    if Q_ix.ndim != 2 or U_in.ndim != 2:
        raise ValueError("Q_ix and U_in must both be 2D arrays.")
    if Q_ix.shape[0] != U_in.shape[0]:
        raise ValueError("Q_ix and U_in must have the same number of samples.")

    N = Q_ix.shape[0]
    A = U_in

    if fit_intercept:
        A = np.hstack([U_in, np.ones((N, 1), dtype=U_in.dtype)])

    if method == "lstsq":
        L_nx, residuals, rank, svals = np.linalg.lstsq(A, Q_ix, rcond=rcond)
    elif method == "pinv":
        L_nx = np.linalg.pinv(A, rcond=rcond) @ Q_ix
        rank = np.linalg.matrix_rank(A)
        svals = np.linalg.svd(A, compute_uv=False)
    elif method == "ridge":
        if ridge <= 0:
            raise ValueError("For method='ridge', ridge must be > 0.")
        ATA = A.T @ A
        ATQ = A.T @ Q_ix
        L_nx = np.linalg.solve(ATA + ridge * np.eye(A.shape[1]), ATQ)
        rank = np.linalg.matrix_rank(A)
        svals = np.linalg.svd(A, compute_uv=False)
    else:
        raise ValueError("method must be one of {'lstsq', 'pinv', 'ridge'}")

    Qhat_ix = A @ L_nx
    resid = Q_ix - Qhat_ix

    info = {
        "Qhat_ix": Qhat_ix,
        "mse": np.mean(resid**2),
        "relative_fro_error": np.linalg.norm(resid, ord="fro")
        / max(np.linalg.norm(Q_ix, ord="fro"), 1e-15),
        "rank": rank,
        "singular_values": svals,
    }
    return L_nx, info


def scan_isomap_to_target(
    R_iX,
    Q_ix,
    ds,
    *,
    isomap_n_neighbors=12,
    isomap_metric="minkowski",
    isomap_p=2,
    fit_intercept=True,
    normalize_target=True,
    normalize_source=True,
    method="lstsq",
    ridge=0.0,
):
    """
    Scan Isomap latent dimension d in ds and fit each Isomap embedding to a fixed target Q_ix.

    Parameters
    ----------
    R_iX : ndarray, shape (N, D)
        Ambient data used to fit Isomap.
    Q_ix : ndarray, shape (N, X)
        Target embedding / target coordinates to be reproduced.
        Examples:
          - ground-truth sheet coordinates
          - Isomap embedding at another dimension
          - UMAP embedding
          - any other reference chart
    ds : sequence of int
        Isomap latent dimensions to try.

    Returns
    -------
    result : dict
        ds      : (S,)
        e_s     : (S,)
        MSE_s   : (S,)
        R_six   : list of fitted outputs in target space, each (N, X)
        U_sin   : list of raw Isomap embeddings, each (N, d_s)
        L_s     : list of fitted maps
        info_s  : fit diagnostics
        Q_ix    : raw target
        Q_use   : normalized target actually used for fitting
    """
    R_iX = np.asarray(R_iX, dtype=np.float32)
    Q_ix = np.asarray(Q_ix, dtype=float)
    ds = np.asarray(ds, dtype=int)
    ds = np.unique(np.sort(ds))

    if R_iX.ndim != 2:
        raise ValueError("R_iX must be a 2D array.")
    if Q_ix.ndim != 2:
        raise ValueError("Q_ix must be a 2D array.")
    if R_iX.shape[0] != Q_ix.shape[0]:
        raise ValueError("R_iX and Q_ix must have the same number of samples.")
    if np.any(ds < 1):
        raise ValueError("All ds must be positive integers.")

    Q_use = normalize_embedding(Q_ix) if normalize_target else Q_ix.copy()

    e_s = np.zeros(len(ds), dtype=float)
    MSE_s = np.zeros(len(ds), dtype=float)
    R_six = []
    U_sin = []
    L_s = []
    info_s = []

    for s, d in enumerate(ds):
        reducer = Isomap(
            n_neighbors=isomap_n_neighbors,
            n_components=int(d),
            metric=isomap_metric,
            p=isomap_p,
        )
        U_in = reducer.fit_transform(R_iX)
        U_use = normalize_embedding(U_in) if normalize_source else np.asarray(U_in, dtype=float)

        L_nx, info = solve_linear_map(
            Q_ix=Q_use,
            U_in=U_use,
            fit_intercept=fit_intercept,
            method=method,
            ridge=ridge,
        )

        e_s[s] = info["relative_fro_error"]
        MSE_s[s] = info["mse"]

        U_sin.append(U_in)
        R_six.append(info["Qhat_ix"])
        L_s.append(L_nx)
        info_s.append(info)

        print(
            f"d={d:>3d} | target_dim={Q_ix.shape[1]:>2d} | "
            f"L shape={L_nx.shape} | "
            f"MSE={info['mse']:.6e} | relF={info['relative_fro_error']:.6f}"
        )

    return {
        "ds": ds,
        "e_s": e_s,
        "MSE_s": MSE_s,
        "R_six": R_six,   # fitted outputs in target space
        "U_sin": U_sin,   # raw Isomap embeddings
        "L_s": L_s,
        "info_s": info_s,
        "Q_ix": Q_ix,
        "Q_use": Q_use,
    }

def _embedding_to_2d_for_plot(U_in):
    U_in = np.asarray(U_in)
    if U_in.ndim != 2:
        raise ValueError("U_in must be 2D.")
    if U_in.shape[1] >= 2:
        return U_in[:, :2]
    return np.column_stack([U_in[:, 0], np.zeros(U_in.shape[0], dtype=U_in.dtype)])

###########

def scan_precomputed_dmap_to_target(
    R_in_max,
    Q_ix,
    ds,
    *,
    fit_intercept=True,
    normalize_target=True,
    normalize_source=True,
    method="lstsq",
    ridge=0.0,
    ):
    R_in_max = np.asarray(R_in_max, dtype=float)
    Q_ix = np.asarray(Q_ix, dtype=float)
    ds = np.asarray(ds, dtype=int)
    ds = np.unique(np.sort(ds))

    if R_in_max.ndim != 2:
        raise ValueError("R_in_max must be a 2D array.")
    if Q_ix.ndim != 2:
        raise ValueError("Q_ix must be a 2D array.")
    if R_in_max.shape[0] != Q_ix.shape[0]:
        raise ValueError("R_in_max and Q_ix must have the same number of samples.")
    if np.any(ds < 1):
        raise ValueError("All ds must be positive integers.")
    if ds.max() > R_in_max.shape[1]:
        raise ValueError("Requested d exceeds the number of precomputed DMAP dimensions.")

    Q_use = normalize_embedding(Q_ix) if normalize_target else Q_ix.copy()

    e_s = np.zeros(len(ds), dtype=float)
    MSE_s = np.zeros(len(ds), dtype=float)
    R_six = []
    U_sin = []
    L_s = []
    info_s = []

    for s, d in enumerate(ds):
        U_in = R_in_max[:, :d]
        U_use = normalize_embedding(U_in) if normalize_source else U_in.copy()

        L_nx, info = solve_linear_map(
            Q_ix=Q_use,
            U_in=U_use,
            fit_intercept=fit_intercept,
            method=method,
            ridge=ridge,
        )

        e_s[s] = info["relative_fro_error"]
        MSE_s[s] = info["mse"]

        U_sin.append(U_in)
        R_six.append(info["Qhat_ix"])
        L_s.append(L_nx)
        info_s.append(info)

        print(
            f"d={d:>4d} | target_dim={Q_ix.shape[1]:>2d} | "
            f"L shape={L_nx.shape} | "
            f"MSE={info['mse']:.6e} | relF={info['relative_fro_error']:.6f}"
        )

    return {
        "ds": ds,
        "e_s": e_s,
        "MSE_s": MSE_s,
        "R_six": R_six,
        "U_sin": U_sin,
        "L_s": L_s,
        "info_s": info_s,
        "Q_ix": Q_ix,
        "Q_use": Q_use,
        "R_in_max": R_in_max,
    }

import numpy as np
import umap


def normalize_embedding(Q_ix, *, center=True, scale=True):
    Q_ix = np.asarray(Q_ix, dtype=float).copy()

    if center:
        Q_ix -= Q_ix.mean(axis=0, keepdims=True)

    if scale:
        fro = np.linalg.norm(Q_ix, ord="fro")
        if fro > 0:
            Q_ix /= fro

    return Q_ix


def solve_linear_map(Q_ix, U_in, *, fit_intercept=True, method="lstsq", ridge=0.0, rcond=None):
    """
    Solve Q_ix ~= U_in @ L_nx (+ intercept) in least squares.
    """
    Q_ix = np.asarray(Q_ix, dtype=float)
    U_in = np.asarray(U_in, dtype=float)

    if Q_ix.ndim != 2 or U_in.ndim != 2:
        raise ValueError("Q_ix and U_in must both be 2D arrays.")
    if Q_ix.shape[0] != U_in.shape[0]:
        raise ValueError("Q_ix and U_in must have the same number of samples.")

    N = Q_ix.shape[0]
    A = U_in

    if fit_intercept:
        A = np.hstack([U_in, np.ones((N, 1), dtype=U_in.dtype)])

    if method == "lstsq":
        L_nx, residuals, rank, svals = np.linalg.lstsq(A, Q_ix, rcond=rcond)
    elif method == "pinv":
        L_nx = np.linalg.pinv(A, rcond=rcond) @ Q_ix
        rank = np.linalg.matrix_rank(A)
        svals = np.linalg.svd(A, compute_uv=False)
    elif method == "ridge":
        if ridge <= 0:
            raise ValueError("For method='ridge', ridge must be > 0.")
        ATA = A.T @ A
        ATQ = A.T @ Q_ix
        L_nx = np.linalg.solve(ATA + ridge * np.eye(A.shape[1]), ATQ)
        rank = np.linalg.matrix_rank(A)
        svals = np.linalg.svd(A, compute_uv=False)
    else:
        raise ValueError("method must be one of {'lstsq', 'pinv', 'ridge'}")

    Qhat_ix = A @ L_nx
    resid = Q_ix - Qhat_ix

    info = {
        "Qhat_ix": Qhat_ix,
        "mse": np.mean(resid**2),
        "relative_fro_error": np.linalg.norm(resid, ord="fro")
        / max(np.linalg.norm(Q_ix, ord="fro"), 1e-15),
        "rank": rank,
        "singular_values": svals,
    }
    return L_nx, info


def scan_umap_to_target(
    R_iX,
    Q_ix,
    ds,
    *,
    umap_n_neighbors=30,
    umap_min_dist=0.05,
    umap_metric="euclidean",
    fit_intercept=True,
    normalize_target=True,
    normalize_source=True,
    method="lstsq",
    ridge=0.0,
    base_random_state=42,
):
    """
    Scan UMAP latent dimension d in ds and fit each UMAP embedding to a fixed target Q_ix.

    Parameters
    ----------
    R_iX : ndarray, shape (N, D)
        Ambient data used to fit UMAP.
    Q_ix : ndarray, shape (N, X)
        Target embedding / target coordinates to be reproduced.
        Examples:
          - ground-truth sheet coordinates
          - Isomap embedding
          - any other reference chart
    ds : sequence of int
        UMAP latent dimensions to try.

    Returns
    -------
    result : dict
        ds      : (S,)
        e_s     : (S,)
        MSE_s   : (S,)
        R_six   : list of fitted outputs in target space, each (N, X)
        U_sin   : list of raw UMAP embeddings, each (N, d_s)
        L_s     : list of fitted maps
        info_s  : fit diagnostics
        Q_ix    : raw target
        Q_use   : normalized target actually used for fitting
    """
    R_iX = np.asarray(R_iX, dtype=np.float32)
    Q_ix = np.asarray(Q_ix, dtype=float)
    ds = np.asarray(ds, dtype=int)
    ds = np.unique(np.sort(ds))

    if R_iX.ndim != 2:
        raise ValueError("R_iX must be a 2D array.")
    if Q_ix.ndim != 2:
        raise ValueError("Q_ix must be a 2D array.")
    if R_iX.shape[0] != Q_ix.shape[0]:
        raise ValueError("R_iX and Q_ix must have the same number of samples.")
    if np.any(ds < 1):
        raise ValueError("All ds must be positive integers.")

    Q_use = normalize_embedding(Q_ix) if normalize_target else Q_ix.copy()

    e_s = np.zeros(len(ds), dtype=float)
    MSE_s = np.zeros(len(ds), dtype=float)
    R_six = []
    U_sin = []
    L_s = []
    info_s = []

    for s, d in enumerate(ds):
        reducer = umap.UMAP(
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            n_components=int(d),
            metric=umap_metric,
            random_state=base_random_state,
        )
        U_in = reducer.fit_transform(R_iX)
        U_use = normalize_embedding(U_in) if normalize_source else np.asarray(U_in, dtype=float)

        L_nx, info = solve_linear_map(
            Q_ix=Q_use,
            U_in=U_use,
            fit_intercept=fit_intercept,
            method=method,
            ridge=ridge,
        )

        e_s[s] = info["relative_fro_error"]
        MSE_s[s] = info["mse"]

        U_sin.append(U_in)
        R_six.append(info["Qhat_ix"])
        L_s.append(L_nx)
        info_s.append(info)

        print(
            f"d={d:>3d} | target_dim={Q_ix.shape[1]:>2d} | "
            f"L shape={L_nx.shape} | "
            f"MSE={info['mse']:.6e} | relF={info['relative_fro_error']:.6f}"
        )

    return {
        "ds": ds,
        "e_s": e_s,
        "MSE_s": MSE_s,
        "R_six": R_six,   # fitted outputs in target space
        "U_sin": U_sin,   # raw UMAP embeddings
        "L_s": L_s,
        "info_s": info_s,
        "Q_ix": Q_ix,
        "Q_use": Q_use,
    }
