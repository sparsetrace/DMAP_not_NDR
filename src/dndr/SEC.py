import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any

from scipy.spatial.distance import cdist
from scipy.linalg import eigh
from scipy.sparse.linalg import LinearOperator, lobpcg

import plotly.graph_objects as go


def torus_embed(theta: np.ndarray, phi: np.ndarray, R: float = 2.0, r: float = 0.7) -> np.ndarray:
    theta = np.asarray(theta)
    phi = np.asarray(phi)
    return np.stack(
        [
            (R + r * np.cos(phi)) * np.cos(theta),
            (R + r * np.cos(phi)) * np.sin(theta),
            r * np.sin(phi),
        ],
        axis=-1,
    )


def torus_normal(theta: float, phi: float) -> np.ndarray:
    n = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)], dtype=float)
    return n / np.linalg.norm(n)


def torus_tangent_basis(theta: float, phi: float, R: float = 2.0, r: float = 0.7) -> np.ndarray:
    d_theta = np.array(
        [-(R + r * np.cos(phi)) * np.sin(theta), (R + r * np.cos(phi)) * np.cos(theta), 0.0],
        dtype=float,
    )
    d_phi = np.array(
        [-r * np.sin(phi) * np.cos(theta), -r * np.sin(phi) * np.sin(theta), r * np.cos(phi)],
        dtype=float,
    )
    e1 = d_theta / np.linalg.norm(d_theta)
    d_phi = d_phi - e1 * np.dot(e1, d_phi)
    e2 = d_phi / np.linalg.norm(d_phi)
    return np.column_stack([e1, e2])


def sample_torus_grid(n_theta: int = 20, n_phi: int = 20, R: float = 2.0, r: float = 0.7):
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    X = torus_embed(TH.ravel(), PH.ravel(), R=R, r=r)
    return X, TH.ravel(), PH.ravel()


@dataclass
class SECResult:
    query_point: np.ndarray
    projected_point: np.ndarray
    tangent_basis: np.ndarray
    singular_values: np.ndarray
    raw_field_vectors: np.ndarray
    phi_query: np.ndarray


class SEC:
    """
    Small, working DMAP + SEC prototype for point clouds.

    Design choices for this demo:
    - DMAP/CIDM uses alpha=1 (Coifman-Lafon debiasing).
    - The SEC frame uses the first n_eigs_sec scalar modes.
    - The product expansions in G/E use the first n_eigs_dmap modes; this is crucial for stability.
    - The generalized eigenproblem is solved after the Sobolev-basis reduction described in the
      supplementary note of Mahler et al. (2024).
    - LOBPCG is used on the reduced generalized problem via LinearOperator.
    """

    def __init__(
        self,
        X: np.ndarray,
        intrinsic_dim: int = 2,
        n_eigs_dmap: int = 60,
        n_eigs_sec: int = 8,
        n_sec_fields: int = 8,
        epsilon: Optional[float] = None,
        alpha: float = 1.0,
        tau_ratio: float = 1e-3,
        random_state: int = 0,
    ):
        if alpha != 1.0:
            raise ValueError("This demo currently implements the alpha=1 Coifman-Lafon correction only.")
        self.X = np.asarray(X, dtype=float)
        self.N, self.D = self.X.shape
        self.intrinsic_dim = intrinsic_dim
        self.n_eigs_dmap = min(n_eigs_dmap, self.N)
        self.n_eigs_sec = min(n_eigs_sec, self.n_eigs_dmap)
        self.n_sec_fields = min(n_sec_fields, self.n_eigs_sec * self.n_eigs_sec - 1)
        self.alpha = alpha
        self.tau_ratio = tau_ratio
        self.random_state = random_state
        self.epsilon = epsilon

        self.DMAP()
        self._build_structure_constants()
        self._build_sec_operators()
        self._solve_sec()

    @staticmethod
    def _estimate_epsilon(X: np.ndarray, k: int = 10) -> float:
        D2 = cdist(X, X, metric="sqeuclidean")
        np.fill_diagonal(D2, np.inf)
        kth = np.partition(D2, k - 1, axis=1)[:, k - 1]
        return float(np.median(kth))

    def DMAP(self):
        D2 = cdist(self.X, self.X, metric="sqeuclidean")
        self.epsilon = self.epsilon or self._estimate_epsilon(self.X, k=min(10, self.N - 1))
        K = np.exp(-D2 / self.epsilon)

        q = K.sum(axis=1)
        K_alpha = K / (q[:, None] * q[None, :])
        d = K_alpha.sum(axis=1)

        S = K_alpha / np.sqrt(d[:, None] * d[None, :])
        evals, evecs = eigh(S)
        order = np.argsort(evals)[::-1]
        evals = evals[order][: self.n_eigs_dmap]
        evecs = evecs[:, order][:, : self.n_eigs_dmap]

        psi = evecs / np.sqrt(d[:, None])
        scale = np.sqrt(d.sum())
        Phi = psi * scale

        mu = d / d.sum()
        lambdas = -np.log(np.clip(evals, 1e-14, 1.0)) / self.epsilon
        lambdas[0] = 0.0

        self.kernel_ = K
        self.q_ = q
        self.d_ = d
        self.mu_ = mu
        self.xi_ = evals
        self.Phi_ = Phi
        self.lam_ = lambdas
        self.Fhat_ = self.Phi_.T @ (self.mu_[:, None] * self.X)

        self.Phi_sec_ = self.Phi_[:, : self.n_eigs_sec]
        self.lam_sec_ = self.lam_[: self.n_eigs_sec]
        self.Fhat_sec_ = self.Fhat_[: self.n_eigs_sec]

    def _build_structure_constants(self):
        # c_{ijk} = <phi_i phi_j, phi_k> with k running over all DMAP modes.
        self.c_all_ = np.einsum(
            "ni,nj,nk,n->ijk",
            self.Phi_sec_,
            self.Phi_sec_,
            self.Phi_,
            self.mu_,
            optimize=True,
        )

    def _build_sec_operators(self):
        """
        Build dense SEC operators G and E from operator actions on JxJ coefficient
        matrices A, using the identities
    
            G(A) = 1/2 * ( C_0(A) Λ + C_0(A Λ) - C_1(A) )
    
        and
    
            4 E(A) =
                Λ(F_1(A)-H_1(A)) + (F_1(A)-H_1(A))Λ
              + F_1(ΛA + AΛ) - H_1(ΛA + AΛ)
              + C_1(A)Λ + C_1(AΛ) - Λ C_1(A) - C_1(ΛA)
              + C_2(A) + H_2(A) - F_2(A).
    
        This is dense and meant for small J (e.g. torus demo). It is much less
        error-prone than hand-coding the 4-index contractions.
        """
        J = self.n_eigs_sec
        F = J * J
    
        # c[n,m,s] = <phi_n phi_m, phi_s>
        # first two indices live in the SEC frame (size J),
        # last index runs over all retained scalar DMAP modes.
        c = self.c_all_
        lam_all = self.lam_
        lam_sec = self.lam_sec_
        Lam = np.diag(lam_sec)
    
        def C_p(A, p):
            """
            [C_p(A)]_{nm} = sum_{k,l} c^p_{nmkl} A_{kl},
            where c^p_{nmkl} = sum_s lam_s^p c_{nms} c_{kls}.
            """
            # sigma[s] = sum_{k,l} c[k,l,s] A[k,l]
            sigma = np.einsum("kls,kl->s", c, A, optimize=True)
            # result[n,m] = sum_s lam_s^p c[n,m,s] sigma[s]
            return np.einsum("s,nms->nm", (lam_all ** p) * sigma, c, optimize=True)
    
        def H_p(A, p):
            """
            [H_p(A)]_{nm} = sum_{k,l} c^p_{nkml} A_{kl}
                          = sum_{k,l,s} lam_s^p c[n,k,s] c[m,l,s] A[k,l].
            """
            return np.einsum("nks,mls,kl,s->nm", c, c, A, lam_all ** p, optimize=True)
    
        def F_p(A, p):
            """
            [F_p(A)]_{nm} = sum_{k,l} c^p_{nlmk} A_{kl}
                          = sum_{k,l,s} lam_s^p c[n,l,s] c[m,k,s] A[k,l].
            """
            return np.einsum("nls,mks,kl,s->nm", c, c, A, lam_all ** p, optimize=True)
    
        def G_apply(A):
            # G(A) = 1/2 * ( C_0(A) Λ + C_0(AΛ) - C_1(A) )
            return 0.5 * (C_p(A, 0) @ Lam + C_p(A @ Lam, 0) - C_p(A, 1))
    
        def E_apply(A):
            LA = Lam @ A
            AL = A @ Lam
    
            F1A = F_p(A, 1)
            H1A = H_p(A, 1)
            C1A = C_p(A, 1)
    
            out = (
                Lam @ (F1A - H1A)
                + (F1A - H1A) @ Lam
                + F_p(LA + AL, 1)
                - H_p(LA + AL, 1)
                + C1A @ Lam
                + C_p(AL, 1)
                - Lam @ C1A
                - C_p(LA, 1)
                + C_p(A, 2)
                + H_p(A, 2)
                - F_p(A, 2)
            )
            return 0.25 * out
    
        # Build dense matrix representations on the flattened JxJ coefficient space.
        G = np.zeros((F, F), dtype=float)
        E = np.zeros((F, F), dtype=float)
    
        for col in range(F):
            A = np.zeros((J, J), dtype=float)
            k, l = divmod(col, J)
            A[k, l] = 1.0
    
            GA = G_apply(A)
            EA = E_apply(A)
    
            G[:, col] = GA.reshape(-1)
            E[:, col] = EA.reshape(-1)
    
        # Symmetrize to reduce numerical asymmetry.
        self.G_ = 0.5 * (G + G.T)
        self.E_ = 0.5 * (E + E.T)
    
        # Optional: keep operator handles around for debugging.
        self._C_p = C_p
        self._H_p = H_p
        self._F_p = F_p
        self._G_apply = G_apply
        self._E_apply = E_apply
    
    def _solve_sec(self):
        sob = 0.5 * ((self.E_ + self.G_) + (self.E_ + self.G_).T)
        s_eval, s_vec = eigh(sob)
        keep = s_eval > self.tau_ratio * s_eval.max()
        U = s_vec[:, keep]
    
        Gt = 0.5 * ((U.T @ self.G_ @ U) + (U.T @ self.G_ @ U).T)
        Et = 0.5 * ((U.T @ self.E_ @ U) + (U.T @ self.E_ @ U).T)
    
        g_eval = np.linalg.eigvalsh(Gt)
        if g_eval.min() <= 1e-12:
            Gt = Gt + (1e-10 - g_eval.min()) * np.eye(Gt.shape[0])
    
        m = min(self.n_sec_fields, Gt.shape[0] - 1)
        rng = np.random.default_rng(self.random_state)
        X0 = rng.standard_normal((Gt.shape[0], m))
    
        Aop = LinearOperator(Gt.shape, matvec=lambda x: Et @ x, matmat=lambda X: Et @ X, dtype=float)
        Bop = LinearOperator(Gt.shape, matvec=lambda x: Gt @ x, matmat=lambda X: Gt @ X, dtype=float)
    
        vals, vecs = lobpcg(Aop, X0, B=Bop, largest=False, tol=1e-6, maxiter=300)
        order = np.argsort(vals)
        vals = vals[order]
        vecs = vecs[:, order]
    
        coeffs = U @ vecs
    
        self.sec_evals_ = vals
        self.sec_frame_coeffs_ = coeffs
        self.sec_field_mats_ = [coeffs[:, i].reshape(self.n_eigs_sec, self.n_eigs_sec) for i in range(m)]

    
    def nystrom_phi(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float).reshape(-1)
        d2 = np.sum((self.X - y[None, :]) ** 2, axis=1)
        k = np.exp(-d2 / self.epsilon)
        q_y = np.sum(k)
        k_alpha = k / (q_y * self.q_)
        d_y = np.sum(k_alpha)
        p = k_alpha / d_y
        return (p @ self.Phi_) / self.xi_

    def project_to_manifold(self, y: np.ndarray) -> np.ndarray:
        phi_y = self.nystrom_phi(y)
        return phi_y @ self.Fhat_

    def _eval_operator_field(self, phi_query_sec: np.ndarray, field_index: int) -> np.ndarray:
        Vop = self.sec_operator_mats_[field_index]
        return phi_query_sec @ (Vop @ self.Fhat_sec_)

    def __call__(self, y: np.ndarray, d: Optional[int] = None, n_fields: Optional[int] = None) -> SECResult:
        d = d or self.intrinsic_dim
        n_fields = min(n_fields or self.n_sec_fields, len(self.sec_operator_mats_))

        phi_y = self.nystrom_phi(y)
        x_proj = phi_y @ self.Fhat_
        phi_sec = phi_y[: self.n_eigs_sec]

        raw_fields = np.column_stack([self._eval_operator_field(phi_sec, i) for i in range(n_fields)])
        U, S, _ = np.linalg.svd(raw_fields, full_matrices=False)
        basis = U[:, :d]
        return SECResult(
            query_point=np.asarray(y, dtype=float).reshape(-1),
            projected_point=x_proj,
            tangent_basis=basis,
            singular_values=S,
            raw_field_vectors=raw_fields,
            phi_query=phi_y,
        )

    def plot(self, result: SECResult, filename: Optional[str] = None, vector_scale: float = 0.6):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=self.X[:, 0],
                y=self.X[:, 1],
                z=self.X[:, 2],
                mode="markers",
                marker=dict(size=2.5, opacity=0.45),
                name="training points",
            )
        )

        q = result.query_point
        p = result.projected_point
        fig.add_trace(
            go.Scatter3d(
                x=[q[0]], y=[q[1]], z=[q[2]], mode="markers", marker=dict(size=6), name="query point"
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[p[0]], y=[p[1]], z=[p[2]], mode="markers", marker=dict(size=7), name="Nyström projection"
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[q[0], p[0]], y=[q[1], p[1]], z=[q[2], p[2]], mode="lines", name="projection line"
            )
        )

        for i in range(result.tangent_basis.shape[1]):
            v = vector_scale * result.tangent_basis[:, i]
            fig.add_trace(
                go.Scatter3d(
                    x=[p[0], p[0] + v[0]],
                    y=[p[1], p[1] + v[1]],
                    z=[p[2], p[2] + v[2]],
                    mode="lines+markers",
                    name=f"tangent vector {i + 1}",
                )
            )

        fig.update_layout(
            title="SEC tangent basis on a torus",
            scene=dict(aspectmode="data"),
            margin=dict(l=0, r=0, t=40, b=0),
        )

        if filename:
            fig.write_html(filename, include_plotlyjs="cdn")
        return fig
