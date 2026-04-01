import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from typing import Any, Optional, Tuple

# ------------------------------
# Flax + Optax
# ------------------------------
from flax import linen as nn
from flax import struct
from flax.training import train_state
import optax


# ==============================================================
# Utilities
# ==============================================================
def cosine_alphas_bar(T: int, s: float = 0.008) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Nichol & Dhariwal cosine schedule.

    Returns
    -------
    alpha_t : (T,)
    beta_t  : (T,)
    a_bar_t : (T,)
    """
    steps = jnp.arange(T + 1, dtype=jnp.float32)
    f = jnp.cos(((steps / T + s) / (1.0 + s)) * jnp.pi / 2.0) ** 2
    a_bar = f / f[0]
    a_bar = a_bar[1:]
    alpha = a_bar / jnp.concatenate([jnp.array([1.0], dtype=jnp.float32), a_bar[:-1]])
    beta = 1.0 - alpha
    return alpha, beta, a_bar


def sinusoidal_embedding(t_idx: jnp.ndarray, dim: int) -> jnp.ndarray:
    """Standard transformer-style sinusoidal timestep embedding."""
    t = t_idx.astype(jnp.float32)
    half = dim // 2
    if half == 0:
        return jnp.zeros((t.shape[0], 0), dtype=jnp.float32)
    denom = float(max(half - 1, 1))
    freqs = jnp.exp(-jnp.log(10_000.0) * jnp.arange(half, dtype=jnp.float32) / denom)
    args = t * freqs
    emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
    if dim % 2 == 1:
        emb = jnp.pad(emb, ((0, 0), (0, 1)))
    return emb


def _standardize(X: jnp.ndarray, eps: float = 1e-6) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    mu = jnp.mean(X, axis=0, keepdims=True)
    sigma = jnp.std(X, axis=0, keepdims=True)
    sigma = jnp.where(sigma < eps, 1.0, sigma)
    Xn = (X - mu) / sigma
    return Xn, mu, sigma


# ==============================================================
# Conditional epsilon network
# ==============================================================
class ConditionalEpsMLP(nn.Module):
    hidden_dim: int
    t_embed_dim: int
    cond_dim: int
    x_dim: int
    depth: int = 3

    @nn.compact
    def __call__(self, x_t: jnp.ndarray, cond: jnp.ndarray, t_idx: jnp.ndarray) -> jnp.ndarray:
        """
        x_t  : (B, x_dim)     noisy ambient point
        cond : (B, cond_dim)  DMAP modes / latent condition
        t_idx: (B, 1)         timestep indices
        """
        t_emb = sinusoidal_embedding(t_idx, self.t_embed_dim)
        h_t = nn.Dense(self.hidden_dim)(t_emb)
        h_t = nn.gelu(h_t)

        h_c = nn.Dense(self.hidden_dim)(cond)
        h_c = nn.gelu(h_c)

        h_x = nn.Dense(self.hidden_dim)(x_t)
        h = h_x + h_c + h_t
        h = nn.gelu(h)

        for _ in range(max(self.depth - 1, 1)):
            h_res = h
            h = nn.Dense(self.hidden_dim)(h)
            h = nn.gelu(h + h_t + h_c)
            h = h + h_res

        out = nn.Dense(self.x_dim)(h)
        return out


# ==============================================================
# Train state with EMA
# ==============================================================
@struct.dataclass
class TrainStateEMA(train_state.TrainState):
    ema_params: Any = struct.field(pytree_node=True)

    def apply_gradients(self, *, grads, ema_decay: float):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        new_ema = optax.incremental_update(new_params, self.ema_params, step_size=1.0 - ema_decay)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            ema_params=new_ema,
        )


# ==============================================================
# DDIM decoder
# ==============================================================
class DDIM:
    """
    Conditional deterministic DDIM decoder with a GPLM-like API.

    Train on paired data:
        R_im : (N, M)   diffusion-map modes / latent features
        R_iX : (N, D)   ambient targets (e.g. Swiss-roll points in R^3)

    Then decode new latent points:
        model = DDIM(R_im, R_iX)
        R_aX = model(R_am)                      # full reverse from T-1
        R_aX = model(R_am, t=40, steps=20)     # shorter reverse chain

    Notes
    -----
    * Training is standard epsilon prediction on the ambient points R_iX,
      conditioned on R_im.
    * Sampling / decoding uses DDIM with eta=0 by default, so the reverse
      map is deterministic for a fixed initial noise.
    * For reproducibility, if `key=None` the class uses an internal PRNG state.
    * This is a decoder only. It does not include Nyström OOS encoding.
    """

    def __init__(
        self,
        R_im: jnp.ndarray,
        R_iX: jnp.ndarray,
        *,
        loss: str = "mse",
        training: bool = True,
        T: int = 100,
        hidden_dim: int = 256,
        t_embed_dim: int = 64,
        depth: int = 3,
        learning_rate: float = 1e-3,
        n_iter: int = 20_000,
        batch_size: Optional[int] = 256,
        ema_decay: float = 0.999,
        beta_max: float = 0.02,
        eta: float = 0.0,
        ddim_steps: Optional[int] = None,
        seed: int = 0,
        normalize_cond: bool = True,
        normalize_x: bool = True,
        verbose_every: int = 500,
        eps: float = 1e-5,
    ):
        R_im = jnp.asarray(R_im, dtype=jnp.float32)
        R_iX = jnp.asarray(R_iX, dtype=jnp.float32)

        if R_im.ndim != 2:
            raise ValueError("R_im must have shape (N, M)")
        if R_iX.ndim != 2:
            raise ValueError("R_iX must have shape (N, D)")
        if R_im.shape[0] != R_iX.shape[0]:
            raise ValueError("R_im and R_iX must have the same number of samples")
        if loss.lower() != "mse":
            raise ValueError("Only loss='mse' is implemented in this simplified DDIM")

        self.key = random.PRNGKey(seed)
        self.loss_name = loss.lower()
        self.training = bool(training)
        self.T = int(T)
        self.eta = float(eta)
        self.ddim_steps = None if ddim_steps is None else int(ddim_steps)
        self.verbose_every = int(verbose_every)
        self.eps = float(eps)
        self.M = int(R_im.shape[1])
        self.D = int(R_iX.shape[1])
        self.normalize_cond = bool(normalize_cond)
        self.normalize_x = bool(normalize_x)
        self.batch_size = batch_size
        self.ema_decay = float(ema_decay)

        # Optional normalization
        if self.normalize_cond:
            R_im_n, self.cond_mu, self.cond_sigma = _standardize(R_im)
        else:
            R_im_n = R_im
            self.cond_mu = jnp.zeros((1, self.M), dtype=jnp.float32)
            self.cond_sigma = jnp.ones((1, self.M), dtype=jnp.float32)

        if self.normalize_x:
            R_iX_n, self.x_mu, self.x_sigma = _standardize(R_iX)
        else:
            R_iX_n = R_iX
            self.x_mu = jnp.zeros((1, self.D), dtype=jnp.float32)
            self.x_sigma = jnp.ones((1, self.D), dtype=jnp.float32)

        self.R_im = R_im_n
        self.R_iX = R_iX_n

        # Diffusion schedule
        alpha, beta, a_bar = cosine_alphas_bar(self.T)
        beta = jnp.minimum(beta, beta_max)
        alpha = 1.0 - beta
        a_bar = jnp.cumprod(alpha)
        self.alpha_s = alpha.astype(jnp.float32)
        self.beta_s = beta.astype(jnp.float32)
        self.a_bar_s = a_bar.astype(jnp.float32)

        # Conditional epsilon model
        self.model = ConditionalEpsMLP(
            hidden_dim=hidden_dim,
            t_embed_dim=t_embed_dim,
            cond_dim=self.M,
            x_dim=self.D,
            depth=depth,
        )
        params = self.model.init(
            self.key,
            jnp.zeros((1, self.D), dtype=jnp.float32),
            jnp.zeros((1, self.M), dtype=jnp.float32),
            jnp.zeros((1, 1), dtype=jnp.int32),
        )["params"]

        tx = optax.adam(learning_rate)
        opt_state = tx.init(params)
        self.state = TrainStateEMA(
            step=0,
            apply_fn=self.model.apply,
            params=params,
            tx=tx,
            opt_state=opt_state,
            ema_params=params,
        )

        if self.training:
            self.fit(n_iter=n_iter)

    # ------------------------------
    # data transforms
    # ------------------------------
    def encode_cond(self, R_am: jnp.ndarray) -> jnp.ndarray:
        R_am = jnp.asarray(R_am, dtype=jnp.float32)
        if R_am.ndim == 1:
            R_am = R_am[None, :]
        if R_am.shape[1] != self.M:
            raise ValueError(f"Condition must have shape (B, {self.M})")
        return (R_am - self.cond_mu) / self.cond_sigma

    def decode_x(self, Xn: jnp.ndarray) -> jnp.ndarray:
        return Xn * self.x_sigma + self.x_mu

    # ------------------------------
    # loss / train step
    # ------------------------------
    @staticmethod
    def _loss(params, apply_fn, x_t, cond, t_idx, eps_true):
        eps_pred = apply_fn({"params": params}, x_t, cond, t_idx)
        return jnp.mean((eps_pred - eps_true) ** 2)

    @staticmethod
    @jax.jit
    def _train_step(
        state: TrainStateEMA,
        x0_batch: jnp.ndarray,
        cond_batch: jnp.ndarray,
        key: jax.Array,
        a_bar_s: jnp.ndarray,
        ema_decay: float,
        eps: float,
    ):
        B = x0_batch.shape[0]
        key, k_eps, k_t = random.split(key, 3)
        eps_noise = random.normal(k_eps, shape=x0_batch.shape)
        t_idx = random.randint(k_t, shape=(B, 1), minval=0, maxval=a_bar_s.shape[0])

        a_bar_t = jnp.take(a_bar_s, t_idx.squeeze(-1))[:, None]
        a_bar_t = jnp.clip(a_bar_t, eps, 1.0)
        x_t = jnp.sqrt(a_bar_t) * x0_batch + jnp.sqrt(1.0 - a_bar_t) * eps_noise

        def loss_fn(p):
            return DDIM._loss(p, state.apply_fn, x_t, cond_batch, t_idx, eps_noise)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        new_state = state.apply_gradients(grads=grads, ema_decay=ema_decay)
        return new_state, loss, key

    def fit(self, n_iter: int = 20_000):
        """Train the conditional DDIM decoder on the paired dataset (R_im, R_iX)."""
        N = self.R_im.shape[0]
        for it in range(int(n_iter)):
            if self.batch_size is None or self.batch_size >= N:
                cond_batch = self.R_im
                x0_batch = self.R_iX
            else:
                self.key, k_perm = random.split(self.key)
                idx = random.permutation(k_perm, N)[: self.batch_size]
                cond_batch = self.R_im[idx]
                x0_batch = self.R_iX[idx]

            self.state, loss, self.key = self._train_step(
                self.state,
                x0_batch,
                cond_batch,
                self.key,
                self.a_bar_s,
                self.ema_decay,
                self.eps,
            )

            if self.verbose_every and (it % self.verbose_every == 0 or it == int(n_iter) - 1):
                print(f"iter {it:6d}  loss {float(loss):.6f}", end="\r")

        if self.verbose_every:
            print("\ntraining complete.")
        return self

    # ------------------------------
    # DDIM reverse process
    # ------------------------------
    def _t_schedule(self, t_start: int, *, steps: Optional[int] = None) -> jnp.ndarray:
        t_start = int(t_start)
        if t_start <= 0:
            return jnp.array([0], dtype=jnp.int32)

        S = self.ddim_steps if steps is None else int(steps)
        if S is None or S >= (t_start + 1):
            return jnp.arange(t_start, -1, -1, dtype=jnp.int32)

        grid = jnp.linspace(float(t_start), 0.0, int(S), dtype=jnp.float32)
        ts = jnp.unique(jnp.round(grid).astype(jnp.int32))[::-1]
        if ts[0] != t_start:
            ts = jnp.concatenate([jnp.array([t_start], dtype=jnp.int32), ts])
        if ts[-1] != 0:
            ts = jnp.concatenate([ts, jnp.array([0], dtype=jnp.int32)])
        return ts

    @staticmethod
    def _make_ddim_step(params_ema, apply_fn, a_bar_s, eps: float, eta: float):
        eta = float(eta)

        @jax.jit
        def step(carry, t_pair):
            key, x_t, cond = carry
            t, t_prev = t_pair

            a_bar_t = jnp.clip(a_bar_s[t], eps, 1.0)
            a_bar_prev = jnp.clip(a_bar_s[t_prev], eps, 1.0)

            B = x_t.shape[0]
            t_batch = jnp.full((B, 1), t, dtype=jnp.int32)
            eps_pred = apply_fn({"params": params_ema}, x_t, cond, t_batch)

            sqrt_one_minus = jnp.sqrt(jnp.clip(1.0 - a_bar_t, eps, 1.0))
            x0_hat = (x_t - sqrt_one_minus * eps_pred) / jnp.sqrt(a_bar_t)

            frac = (1.0 - a_bar_prev) / jnp.clip(1.0 - a_bar_t, eps, 1.0)
            inside = 1.0 - a_bar_t / jnp.clip(a_bar_prev, eps, 1.0)
            sigma = eta * jnp.sqrt(jnp.clip(frac * inside, 0.0, 1.0))

            dir_coeff = jnp.sqrt(jnp.clip(1.0 - a_bar_prev - sigma**2, 0.0, 1.0))
            x_prev = jnp.sqrt(a_bar_prev) * x0_hat + dir_coeff * eps_pred

            if eta > 0.0:
                key, k = random.split(key)
                z = random.normal(k, x_t.shape)
                x_prev = x_prev + sigma * z

            return (key, x_prev, cond), x_prev

        return step

    def sample_latent_noise(self, B: int, key: Optional[jax.Array] = None) -> Tuple[jax.Array, jnp.ndarray]:
        if key is None:
            self.key, key = random.split(self.key)
        noise = random.normal(key, (int(B), self.D), dtype=jnp.float32)
        return key, noise

    def predict(
        self,
        R_am: jnp.ndarray,
        t: Optional[int] = None,
        *,
        steps: Optional[int] = None,
        key: Optional[jax.Array] = None,
        x_t: Optional[jnp.ndarray] = None,
        eta: Optional[float] = None,
        return_path: bool = False,
    ):
        """
        Decode new DMAP latent points into ambient points.

        Parameters
        ----------
        R_am : (B, M)
            New DMAP mode vectors.
        t : int, optional
            Starting diffusion step. Default: T-1 (full reverse chain).
        steps : int, optional
            Number of DDIM reverse steps. Default: full chain from t -> 0.
        key : PRNGKey, optional
            RNG key for the initial noise when x_t is not provided.
        x_t : (B, D), optional
            Explicit starting noisy ambient state. If omitted, standard normal
            noise is used.
        eta : float, optional
            DDIM stochasticity. Default uses self.eta. Use eta=0.0 for fully
            deterministic reverse updates given x_t.
        return_path : bool
            If True, return all intermediate states.

        Returns
        -------
        X_hat : (B, D)
            Predicted ambient points.
        or (X_hat, path) if return_path=True.
        """
        cond = self.encode_cond(R_am)
        B = cond.shape[0]
        t = self.T - 1 if t is None else int(t)
        if not (0 <= t < self.T):
            raise ValueError(f"t must be in [0, {self.T - 1}]")

        if x_t is None:
            key, x_t = self.sample_latent_noise(B, key=key)
        else:
            x_t = jnp.asarray(x_t, dtype=jnp.float32)
            if x_t.ndim == 1:
                x_t = x_t[None, :]
            if x_t.shape != (B, self.D):
                raise ValueError(f"x_t must have shape ({B}, {self.D})")
            if key is None:
                self.key, key = random.split(self.key)

        ts = self._t_schedule(t, steps=steps)
        if ts.shape[0] == 1:
            X_hat = self.decode_x(x_t)
            return (X_hat, self.decode_x(x_t)[None, ...]) if return_path else X_hat

        t_pairs = jnp.stack([ts[:-1], ts[1:]], axis=1)
        step = self._make_ddim_step(
            self.state.ema_params,
            self.state.apply_fn,
            self.a_bar_s,
            self.eps,
            self.eta if eta is None else float(eta),
        )

        (final_key, x_final, _), path = jax.lax.scan(step, (key, x_t, cond), xs=t_pairs)
        self.key = final_key
        X_hat = self.decode_x(x_final)
        if return_path:
            path = self.decode_x(path)
            return X_hat, path
        return X_hat

    def __call__(
        self,
        R_am: jnp.ndarray,
        t: Optional[int] = None,
        *,
        steps: Optional[int] = None,
        key: Optional[jax.Array] = None,
        x_t: Optional[jnp.ndarray] = None,
        eta: Optional[float] = None,
        return_path: bool = False,
    ):
        return self.predict(
            R_am,
            t=t,
            steps=steps,
            key=key,
            x_t=x_t,
            eta=eta,
            return_path=return_path,
        )

    # ------------------------------
    # convenience helpers
    # ------------------------------
    def reconstruction_mse(
        self,
        R_im: Optional[jnp.ndarray] = None,
        R_iX: Optional[jnp.ndarray] = None,
        *,
        t: Optional[int] = None,
    ) -> float:
        """Quick reconstruction MSE on a dataset."""
        if R_im is None:
            R_im = self.R_im * self.cond_sigma + self.cond_mu
        if R_iX is None:
            R_iX = self.R_iX * self.x_sigma + self.x_mu
        X_hat = self.predict(R_im, t=t)
        return float(jnp.mean((jnp.asarray(X_hat) - jnp.asarray(R_iX)) ** 2))

    def get_state(self) -> dict:
        """Return a lightweight checkpoint dictionary."""
        return {
            "params": self.state.params,
            "ema_params": self.state.ema_params,
            "opt_state": self.state.opt_state,
            "step": int(self.state.step),
            "cond_mu": self.cond_mu,
            "cond_sigma": self.cond_sigma,
            "x_mu": self.x_mu,
            "x_sigma": self.x_sigma,
            "T": self.T,
            "M": self.M,
            "D": self.D,
            "eta": self.eta,
            "ddim_steps": self.ddim_steps,
        }


__all__ = ["DDIM"]
