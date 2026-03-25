import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# Arc-length geometry for an isometric swiss roll
# ============================================================
def _spiral_S(theta, r0=0.5, b=0.6):
    """
    Primitive of sqrt((r0 + b*theta)^2 + b^2) dtheta
    for the Archimedean spiral r(theta)=r0+b*theta.
    """
    theta = np.asarray(theta, dtype=float)
    u = r0 + b * theta
    return (u * np.sqrt(u**2 + b**2) + b**2 * np.arcsinh(u / b)) / (2.0 * b)


def _arc_length(theta0, theta1, r0=0.5, b=0.6):
    return _spiral_S(theta1, r0=r0, b=b) - _spiral_S(theta0, r0=r0, b=b)


def _theta_from_sheet_width(width, theta_min=1.5*np.pi, r0=0.5, b=0.6):
    """
    Find theta_max such that the spiral arc length from theta_min to theta_max
    equals 'width'.
    """
    if width <= 0:
        raise ValueError("width must be > 0")

    lo = theta_min
    hi = theta_min + 1.0

    # Expand until the arc length is large enough
    while _arc_length(theta_min, hi, r0=r0, b=b) < width:
        hi = theta_min + 2.0 * (hi - theta_min)

    # Bisection
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        L = _arc_length(theta_min, mid, r0=r0, b=b)
        if L < width:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


# ============================================================
# Build a roll map from a literal sheet size W x H
# ============================================================
def make_isometric_swiss_roll_map_from_sheet(
    width=18.0,
    height=10.0,
    theta_min=1.5*np.pi,
    r0=0.5,
    b=0.6,
    n_lookup=30000,
):
    """
    Create an isometric swiss roll from a flat sheet of size width x height.

    Flat sheet coordinates are:
        s in [0, width]
        h in [0, height]

    These are physical coordinates on the unrolled sheet.
    """
    theta_max = _theta_from_sheet_width(width, theta_min=theta_min, r0=r0, b=b)

    theta_grid = np.linspace(theta_min, theta_max, n_lookup)
    S_grid = _spiral_S(theta_grid, r0=r0, b=b)
    S0 = S_grid[0]

    return {
        "width": float(width),
        "height": float(height),
        "theta_min": float(theta_min),
        "theta_max": float(theta_max),
        "r0": float(r0),
        "b": float(b),
        "theta_grid": theta_grid,
        "S_grid": S_grid,
        "S0": float(S0),
    }


# ============================================================
# Forward map: literal flat sheet -> rolled sheet
# ============================================================
def sheet_to_swiss_roll(SH, roll_map):
    """
    SH : array of shape (N,2)
         SH[:,0] = s in [0, width]
         SH[:,1] = h in [0, height]

    Returns XYZ and theta.
    """
    SH = np.asarray(SH, dtype=float)
    if SH.ndim != 2 or SH.shape[1] != 2:
        raise ValueError("SH must have shape (N,2)")

    s = SH[:, 0]
    h = SH[:, 1]

    # Convert sheet coordinate s into absolute spiral arc length
    S_abs = roll_map["S0"] + s

    theta = np.interp(S_abs, roll_map["S_grid"], roll_map["theta_grid"])
    r = roll_map["r0"] + roll_map["b"] * theta

    x = r * np.cos(theta)
    z = r * np.sin(theta)
    y = h

    XYZ = np.column_stack([x, y, z])
    return XYZ, theta


# ============================================================
# Inverse map: rolled sheet -> literal flat sheet
# ============================================================
def swiss_roll_to_sheet(XYZ, roll_map, clip=True):
    """
    Inverse map back to literal flat sheet coordinates (s,h).
    """
    XYZ = np.asarray(XYZ, dtype=float)
    if XYZ.ndim != 2 or XYZ.shape[1] != 3:
        raise ValueError("XYZ must have shape (N,3)")

    x = XYZ[:, 0]
    y = XYZ[:, 1]
    z = XYZ[:, 2]

    r = np.sqrt(x**2 + z**2)
    theta = (r - roll_map["r0"]) / roll_map["b"]

    S_abs = _spiral_S(theta, r0=roll_map["r0"], b=roll_map["b"])
    s = S_abs - roll_map["S0"]
    h = y

    SH = np.column_stack([s, h])

    if clip:
        SH[:, 0] = np.clip(SH[:, 0], 0.0, roll_map["width"])
        SH[:, 1] = np.clip(SH[:, 1], 0.0, roll_map["height"])

    return SH


# ============================================================
# Sampling a literal sheet
# ============================================================
def sample_sheet_random(n, width, height, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.uniform(0.0, width, size=n)
    h = rng.uniform(0.0, height, size=n)
    return np.column_stack([s, h])


def sample_sheet_grid(width, height, n_s=120, n_h=40):
    s = np.linspace(0.0, width, n_s)
    h = np.linspace(0.0, height, n_h)
    S, H = np.meshgrid(s, h, indexing="xy")
    return np.column_stack([S.ravel(), H.ravel()])


# ============================================================
# Plotly: literal unrolled sheet, rolled sheet, inverse recovery
# ============================================================
def swiss_roll(
    width=18.0,
    height=10.0,
    theta_min=1.5*np.pi,
    r0=0.5,
    b=0.6,
    n_points=5000,
    seed=0,
    plot=True):
    roll_map = make_isometric_swiss_roll_map_from_sheet(
        width=width,
        height=height,
        theta_min=theta_min,
        r0=r0,
        b=b,
    )

    E_ix = sample_sheet_random(n_points, width=width, height=height, seed=seed)
    XYZ, theta = sheet_to_swiss_roll(E_ix, roll_map)
    SH_back = swiss_roll_to_sheet(XYZ, roll_map)
    color = E_ix[:, 0]  # color by sheet-length coordinate

    if plot:
        fig = make_subplots(
            rows=1,
            cols=3,
            specs=[[{"type": "xy"}, {"type": "scene"}, {"type": "xy"}]],
            subplot_titles=(
                f"Literal unrolled sheet ({width:.1f} × {height:.1f})",
                "Rolled sheet in 3D",
                "Inverse-unrolled recovery",
            ),
            horizontal_spacing=0.06,
        )

        fig.add_trace(
            go.Scattergl(
                x=E_ix[:, 0],
                y=E_ix[:, 1],
                mode="markers",
                marker=dict(size=4, color=color, colorscale="Viridis"),
                showlegend=False,
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter3d(
                x=XYZ[:, 0],
                y=XYZ[:, 1],
                z=XYZ[:, 2],
                mode="markers",
                marker=dict(size=2.5, color=color, colorscale="Viridis"),
                showlegend=False,
            ),
            row=1, col=2
        )

        fig.add_trace(
            go.Scattergl(
                x=SH_back[:, 0],
                y=SH_back[:, 1],
                mode="markers",
                marker=dict(size=4, color=color, colorscale="Viridis"),
                showlegend=False,
            ),
            row=1, col=3
        )

        fig.update_xaxes(title_text="sheet coordinate s", row=1, col=1)
        fig.update_yaxes(title_text="sheet coordinate h", row=1, col=1, scaleanchor="x", scaleratio=1)

        fig.update_xaxes(title_text="sheet coordinate s", row=1, col=3)
        fig.update_yaxes(title_text="sheet coordinate h", row=1, col=3, scaleanchor="x", scaleratio=1)

        fig.update_scenes(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="data",
            row=1, col=2
        )

        fig.update_layout(
            width=1450,
            height=500,
            margin=dict(l=20, r=20, t=60, b=20),
        )

        fig.show()

    err = np.abs(E_ix - SH_back)
    print("sheet size:", (roll_map["width"], roll_map["height"]))
    print("theta range:", (roll_map["theta_min"], roll_map["theta_max"]))
    print("max round-trip error:", err.max())
    print("mean round-trip error:", err.mean())

    return roll_map, E_ix, XYZ, SH_back, color

# Example:
# roll_map, E_ix, E_iX, SH_back, color = swiss_roll(
#    width=60.0,   # this is the literal unrolled sheet length
#    height=10.0,  # literal sheet height
#    r0=0.5, b=0.6, n_points=4000, seed=1, plot=True )
#
# print( color.shape, E_ix.shape, E_iX.shape )
