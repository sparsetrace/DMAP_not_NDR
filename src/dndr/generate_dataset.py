## generate_dataset.py
from __future__ import annotations

import numpy as np


def _lorenz_ode(x: np.ndarray) -> np.ndarray:
    rho = 28.0
    sigma = 10.0
    beta = 8.0 / 3.0
    dx = np.zeros_like(x)
    dx[0, :] = sigma * (x[1, :] - x[0, :])
    dx[1, :] = x[0, :] * (rho - x[2, :]) - x[1, :]
    dx[2, :] = x[0, :] * x[1, :] - beta * x[2, :]
    return dx


def _l63(x0: np.ndarray, T: float, tau: float, D: float):
    t = np.arange(0.0, T + tau, tau)
    N = len(t)
    x = np.zeros((N, *x0.shape), dtype=float)
    x[0, :, :] = x0
    state = x0.copy()

    for i in range(1, N):
        for _ in range(10):
            dt = tau / 10.0
            k1 = dt * _lorenz_ode(state)
            k2 = dt * _lorenz_ode(state + k1 / 2.0)
            k3 = dt * _lorenz_ode(state + k2 / 2.0)
            k4 = dt * _lorenz_ode(state + k3)
            state = state + k1 / 6.0 + k2 / 3.0 + k3 / 3.0 + k4 / 6.0
            state = state + D * np.sqrt(2.0) * np.sqrt(dt) * np.random.randn(*state.shape)
        x[i, :, :] = state

    return x, t


def generate_dataset(N: int, example_name: str, noiselevel: float = 0.0):
    """
    Python translation of GenerateDataSet.m.

    Returns
    -------
    data : array, shape (ambient_dim, N_eff)
    intrinsic : array, shape (intrinsic_dim, N_eff)
    """
    name = example_name.lower()

    if name == "circle":
        theta = 2 * np.pi * np.arange(1, N + 1) / N
        data = np.vstack([np.cos(theta), np.sin(theta), np.zeros_like(theta)])
        intrinsic = np.vstack([np.cos(theta), np.sin(theta)])

    elif name == "sphere":
        X = np.random.randn(3, N)
        X = X / np.sqrt(np.sum(X ** 2, axis=0, keepdims=True))
        data = X

        XX = X[0, :] / (1 - X[2, :])
        YY = X[1, :] / (1 - X[2, :])
        RR = np.sqrt(XX ** 2 + YY ** 2)
        mask = RR > 0
        XX2 = XX.copy()
        YY2 = YY.copy()
        XX2[mask] = XX[mask] * np.log(1 + RR[mask]) / RR[mask]
        YY2[mask] = YY[mask] * np.log(1 + RR[mask]) / RR[mask]
        intrinsic = np.vstack([XX2, YY2])

    elif name == "flattorus":
        NN = int(np.floor(np.sqrt(N)))
        t = np.arange(1, NN + 1) / NN
        NN = len(t)
        theta = np.tile(2 * np.pi * t, NN)
        phi = np.tile(2 * np.pi * t[:, None], (1, NN)).reshape(-1)
        intrinsic = np.vstack([theta, phi])
        data = np.vstack([np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)])

    elif name == "torus":
        NN = int(np.floor(np.sqrt(N)))
        t = np.arange(1, NN + 1) / NN
        NN = len(t)
        theta = np.tile(2 * np.pi * t, NN)
        phi = np.tile(2 * np.pi * t[:, None], (1, NN)).reshape(-1)
        intrinsic = np.vstack([theta, phi])
        R = 2.0
        r = 1.0
        x = (R + r * np.cos(theta)) * np.cos(phi)
        y = (R + r * np.cos(theta)) * np.sin(phi)
        z = r * np.sin(theta)
        data = np.vstack([x, y, z])

    elif name == "mobius":
        NN = int(np.floor(np.sqrt(N)))
        t = np.arange(1, NN + 1) / NN
        NN = len(t)
        theta = np.tile(2 * t - 1, NN)
        phi = np.tile(2 * np.pi * t[:, None], (1, NN)).reshape(-1)
        intrinsic = np.vstack([theta, phi])
        R = 1.0
        r = 1.0
        x = (R + r * theta * np.cos(phi / 2.0) / 2.0) * np.cos(phi)
        y = (R + r * theta * np.cos(phi / 2.0) / 2.0) * np.sin(phi)
        z = r * theta * np.sin(phi / 2.0) / 2.0
        data = np.vstack([x, y, z])

    elif name == "rp2":
        X = np.random.randn(3, N)
        X = X / np.sqrt(np.sum(X ** 2, axis=0, keepdims=True))
        x, y, z = X[0, :], X[1, :], X[2, :]
        data = np.vstack([x * y, x * z, y ** 2 - z ** 2, 2 * y * z])

        XX = X[0, :] / (1 - X[2, :])
        YY = X[1, :] / (1 - X[2, :])
        RR = np.sqrt(XX ** 2 + YY ** 2)
        mask = RR > 0
        XX2 = XX.copy()
        YY2 = YY.copy()
        XX2[mask] = XX[mask] * np.log(1 + RR[mask]) / RR[mask]
        YY2[mask] = YY[mask] * np.log(1 + RR[mask]) / RR[mask]
        intrinsic = np.vstack([XX2, YY2])

    elif name == "kleinbottle":
        NN = int(np.floor(np.sqrt(N)))
        t = np.arange(1, NN + 1) / NN
        NN = len(t)
        theta = np.tile(2 * np.pi * t, NN)
        phi = np.tile(2 * np.pi * t[:, None], (1, NN)).reshape(-1)
        intrinsic = np.vstack([theta, phi])
        R = 2.0
        r = 1.0
        x = (R + r * np.cos(theta)) * np.cos(phi)
        y = (R + r * np.cos(theta)) * np.sin(phi)
        z = r * np.sin(theta) * np.cos(phi / 2.0)
        zz = r * np.sin(theta) * np.sin(phi / 2.0)
        data = np.vstack([x, y, z, zz])

    elif name == "doubletorus":
        NN = int(np.floor(np.sqrt(N / 2.0)))
        t = np.arange(1, NN + 1) / NN
        NN = len(t)
        theta = np.tile(2 * np.pi * t, NN)
        phi = np.tile(2 * np.pi * t[:, None], (1, NN)).reshape(-1)
        R = 2.0
        r = 1.0
        x = (R + r * np.cos(theta)) * np.cos(phi)
        y = (R + r * np.cos(theta)) * np.sin(phi)
        z = r * np.sin(theta)
        mask_left = x < 2
        mask_right = x > -2
        intrinsic = np.vstack([
            np.concatenate([theta[mask_left] - 2 * np.pi, theta[mask_right]]),
            np.concatenate([phi[mask_left], phi[mask_right]])
        ])
        data = np.vstack([
            np.concatenate([x[mask_left] - 2 - np.pi / NN / 4.0, x[mask_right] + 2 + np.pi / NN / 4.0]),
            np.concatenate([y[mask_left], y[mask_right]]),
            np.concatenate([z[mask_left], z[mask_right]])
        ])

    elif name == "l63":
        dt = 0.05
        x, _ = _l63(np.random.randn(3, 1), (N + 10000) * dt, dt, 0.0)
        x = x[10000:, :, 0].T
        data = x
        intrinsic = np.vstack([x[0, :] + x[1, :], x[2, :]])

    else:
        raise ValueError(f"Unknown example_name={example_name!r}")

    data = data + noiselevel * np.random.randn(*data.shape)
    return data, intrinsic
