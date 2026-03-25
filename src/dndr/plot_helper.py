import numpy as np
import matplotlib.pyplot as plt

def plot_dmap_spectra_overlay(
    L_nx,
    lambdas_n,
    *,
    fit_intercept=False,
    use_abs=True,
    sort_by_lambda=True,
    xscale="log",
    yscale="log",
    ):
    L_nx = np.asarray(L_nx, dtype=float)
    lambdas_n = np.asarray(lambdas_n, dtype=float).ravel()

    if fit_intercept:
        L_modes = L_nx[:-1, :]
    else:
        L_modes = L_nx

    if L_modes.shape[0] != len(lambdas_n):
        raise ValueError("Mismatch between L rows and eigenvalue count.")

    Y = np.abs(L_modes) if use_abs else L_modes

    if sort_by_lambda:
        idx = np.argsort(lambdas_n)
        lambdas_plot = lambdas_n[idx]
        Y = Y[idx, :]
    else:
        lambdas_plot = lambdas_n

    plt.figure(figsize=(7, 5))

    for j in range(Y.shape[1]):
        y = Y[:, j]
        if yscale == "log":
            mask = y > 0
            x = lambdas_plot[mask]
            yy = y[mask]
        else:
            x = lambdas_plot
            yy = y
        plt.plot(x, yy, marker="o", label=f"coord {j}")

    if xscale is not None:
        plt.xscale(xscale)
    if yscale is not None:
        plt.yscale(yscale)

    plt.xlabel(r"DMAP eigenvalue $1-\lambda_n$")
    plt.ylabel(r"$|L_{nj}|$" if use_abs else r"$L_{nj}$")
    plt.legend()
    plt.tight_layout()
    plt.show()

    axes[-1].set_xlabel(r"DMAP eigenvalue $1-\lambda_n$")
    plt.tight_layout()
    plt.show()
    return None
