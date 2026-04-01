import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from typing import Any, Dict, Optional, Tuple, Union
import optax


class DIAE:
    """
    Diffusion Isometric Autoencoder wrapper around a frozen DDIM decoder.

    The wrapper learns a low-rank linear bottleneck on diffusion-map latents:

        R_im --L(M->d)--> z --U(d->M)--> Rhat_im --frozen DDIM--> Rhat_iX

    and trains L,U by minimizing ambient-space MSE (plus optional extras).

    Parameters
    ----------
    model : DDIM
        A trained DDIM decoder with API compatible with the DDIM.py module.
    R_iX : array, shape (N, D)
        Ambient target points.
    R_im : array, shape (N, M)
        Diffusion-map latents / conditions.
    d : int, default=2
        Bottleneck dimension.
    loss : str or dict, default='mse'
        If 'mse', uses ambient-space mean-squared error only.
        If dict, supported keys are:
            ambient : weight on ambient MSE
            latent  : weight on latent reconstruction MSE
            l2      : weight decay on L and U
    training_sch : dict, optional
        Training schedule / hyperparameters. Supported keys:
            n_iter, learning_rate, batch_size, seed, verbose_every,
            t, steps, eta, init_scale, use_bias, train_x_t
        where train_x_t is one of {'fixed_noise', 'zeros'} or an explicit array.
    training : bool, default=True
        If True, train inside __init__.

    Attributes
    ----------
    L : ndarray, shape (M, d)
        Learned encoder linear map.
    U : ndarray, shape (d, M)
        Learned decoder linear map.
    W : ndarray, shape (M, M)
        Effective low-rank map W = L @ U under row-vector convention.
    """

    def __init__(
        self,
        model: Any,
        R_iX: jnp.ndarray,
        R_im: jnp.ndarray,
        *,
        d: int = 2,
        loss: Union[str, Dict[str, float]] = "mse",
        training_sch: Optional[Dict[str, Any]] = None,
        training: bool = True,
    ):
        self.model = model
        self.R_iX = jnp.asarray(R_iX, dtype=jnp.float32)
        self.R_im = jnp.asarray(R_im, dtype=jnp.float32)

        if self.R_iX.ndim != 2:
            raise ValueError("R_iX must have shape (N, D)")
        if self.R_im.ndim != 2:
            raise ValueError("R_im must have shape (N, M)")
        if self.R_iX.shape[0] != self.R_im.shape[0]:
            raise ValueError("R_iX and R_im must have the same number of rows")

        self.N = int(self.R_iX.shape[0])
        self.D = int(self.R_iX.shape[1])
        self.M = int(self.R_im.shape[1])
        self.d = int(d)

        if hasattr(model, "M") and int(model.M) != self.M:
            raise ValueError(f"model.M={model.M} but R_im has width {self.M}")
        if hasattr(model, "D") and int(model.D) != self.D:
            raise ValueError(f"model.D={model.D} but R_iX has width {self.D}")

        self.loss_cfg = self._parse_loss(loss)
        self.sch = self._parse_training_sch(training_sch)
        self.key = random.PRNGKey(int(self.sch["seed"]))
        self.use_bias = bool(self.sch["use_bias"])
        self.training = bool(training)

        # Fixed DDIM decode schedule used during DIAE training.
        self.train_t = int(self.sch["t"])
        self.train_steps = self.sch["steps"]
        self.train_eta = float(self.sch["eta"])
        self.train_t_pairs = self._make_t_pairs(self.train_t, self.train_steps)

        # Fixed training start state x_t for deterministic optimization.
        self.train_x_t = self._make_train_x_t(self.sch["train_x_t"])

        # Initialize linear maps.
        self.params = self._init_params(scale=float(self.sch["init_scale"]))
        self.tx = optax.adam(float(self.sch["learning_rate"]))
        self.opt_state = self.tx.init(self.params)

        # Build compiled train step.
        self._train_step_jit = self._make_train_step()

        if self.training:
            self.fit(n_iter=int(self.sch["n_iter"]))

    # ------------------------------------------------------------------
    # configuration
    # ------------------------------------------------------------------
    def _parse_loss(self, loss: Union[str, Dict[str, float]]) -> Dict[str, float]:
        if isinstance(loss, str):
            if loss.lower() != "mse":
                raise ValueError("Only loss='mse' or a compatible dict is supported")
            return {"ambient": 1.0, "latent": 0.0, "l2": 0.0}
        cfg = {"ambient": 1.0, "latent": 0.0, "l2": 0.0}
        cfg.update({k: float(v) for k, v in dict(loss).items()})
        return cfg

    def _parse_training_sch(self, training_sch: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        defaults = {
            "n_iter": 5000,
            "learning_rate": 1e-3,
            "batch_size": min(256, self.N),
            "seed": 0,
            "verbose_every": 250,
            "t": int(getattr(self.model, "T", 100) - 1),
            "steps": getattr(self.model, "ddim_steps", None),
            "eta": 0.0,
            "init_scale": 0.05,
            "use_bias": False,
            "train_x_t": "fixed_noise",  # {'fixed_noise', 'zeros'} or explicit array
        }
        if training_sch is not None:
            defaults.update(training_sch)
        return defaults

    # ------------------------------------------------------------------
    # parameter helpers
    # ------------------------------------------------------------------
    def _init_params(self, scale: float = 0.05) -> Dict[str, jnp.ndarray]:
        self.key, k1, k2 = random.split(self.key, 3)
        params = {
            "L": scale * random.normal(k1, (self.M, self.d), dtype=jnp.float32),
            "U": scale * random.normal(k2, (self.d, self.M), dtype=jnp.float32),
        }
        if self.use_bias:
            params["bL"] = jnp.zeros((self.d,), dtype=jnp.float32)
            params["bU"] = jnp.zeros((self.M,), dtype=jnp.float32)
        return params

    @property
    def L(self) -> np.ndarray:
        return np.asarray(self.params["L"])

    @property
    def U(self) -> np.ndarray:
        return np.asarray(self.params["U"])

    @property
    def W(self) -> np.ndarray:
        return np.asarray(self.params["L"] @ self.params["U"])

    # ------------------------------------------------------------------
    # linear bottleneck
    # ------------------------------------------------------------------
    def _apply_linear(self, params: Dict[str, jnp.ndarray], R_m: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        z = R_m @ params["L"]
        if self.use_bias:
            z = z + params["bL"]
        R_hat_m = z @ params["U"]
        if self.use_bias:
            R_hat_m = R_hat_m + params["bU"]
        return z, R_hat_m

    def encode(self, R_m: jnp.ndarray) -> jnp.ndarray:
        R_m = jnp.asarray(R_m, dtype=jnp.float32)
        if R_m.ndim == 1:
            R_m = R_m[None, :]
        z, _ = self._apply_linear(self.params, R_m)
        return z

    def lift(self, R_m: jnp.ndarray) -> jnp.ndarray:
        R_m = jnp.asarray(R_m, dtype=jnp.float32)
        if R_m.ndim == 1:
            R_m = R_m[None, :]
        _, R_hat_m = self._apply_linear(self.params, R_m)
        return R_hat_m

    # ------------------------------------------------------------------
    # frozen DDIM decode
    # ------------------------------------------------------------------
    def _make_t_pairs(self, t: int, steps: Optional[int]) -> jnp.ndarray:
        ts = self.model._t_schedule(int(t), steps=steps)
        ts = jnp.asarray(ts, dtype=jnp.int32)
        if ts.shape[0] <= 1:
            return jnp.zeros((0, 2), dtype=jnp.int32)
        return jnp.stack([ts[:-1], ts[1:]], axis=1)

    def _make_train_x_t(self, train_x_t: Any) -> jnp.ndarray:
        if isinstance(train_x_t, str):
            if train_x_t == "zeros":
                return jnp.zeros((self.N, self.D), dtype=jnp.float32)
            if train_x_t == "fixed_noise":
                k = random.PRNGKey(int(self.sch["seed"]) + 17)
                return random.normal(k, (self.N, self.D), dtype=jnp.float32)
            raise ValueError("train_x_t must be 'fixed_noise', 'zeros', or an explicit array")
        x_t = jnp.asarray(train_x_t, dtype=jnp.float32)
        if x_t.shape != (self.N, self.D):
            raise ValueError(f"Explicit train_x_t must have shape ({self.N}, {self.D})")
        return x_t

    def _default_inference_x_t(self, B: int) -> jnp.ndarray:
        # Deterministic default start state for inference.
        k = random.PRNGKey(int(self.sch["seed"]) + 1009 + int(B))
        return random.normal(k, (int(B), self.D), dtype=jnp.float32)

    def _decode_from_xt(
        self,
        R_hat_m: jnp.ndarray,
        x_t: jnp.ndarray,
        *,
        t_pairs: jnp.ndarray,
        eta: float,
        return_path: bool = False,
    ):
        cond = (R_hat_m - self.model.cond_mu) / self.model.cond_sigma
        x_t = jnp.asarray(x_t, dtype=jnp.float32)

        if t_pairs.shape[0] == 0:
            X_hat = self.model.decode_x(x_t)
            return (X_hat, X_hat[None, ...]) if return_path else X_hat

        step = self.model._make_ddim_step(
            self.model.state.ema_params,
            self.model.state.apply_fn,
            self.model.a_bar_s,
            self.model.eps,
            float(eta),
        )
        dummy_key = random.PRNGKey(0)
        (_, x_final, _), path = jax.lax.scan(step, (dummy_key, x_t, cond), xs=t_pairs)
        X_hat = self.model.decode_x(x_final)
        if return_path:
            return X_hat, self.model.decode_x(path)
        return X_hat

    # ------------------------------------------------------------------
    # objective and optimizer step
    # ------------------------------------------------------------------
    def _make_train_step(self):
        tx = self.tx
        train_t_pairs = self.train_t_pairs
        train_eta = float(self.train_eta)
        ambient_w = float(self.loss_cfg["ambient"])
        latent_w = float(self.loss_cfg["latent"])
        l2_w = float(self.loss_cfg["l2"])
        use_bias = bool(self.use_bias)

        model_cond_mu = self.model.cond_mu
        model_cond_sigma = self.model.cond_sigma
        model_a_bar_s = self.model.a_bar_s
        model_eps = float(self.model.eps)
        model_ema_params = self.model.state.ema_params
        model_apply_fn = self.model.state.apply_fn
        model_x_mu = self.model.x_mu
        model_x_sigma = self.model.x_sigma

        def apply_linear(params: Dict[str, jnp.ndarray], R_m: jnp.ndarray):
            z = R_m @ params["L"]
            if use_bias:
                z = z + params["bL"]
            R_hat_m = z @ params["U"]
            if use_bias:
                R_hat_m = R_hat_m + params["bU"]
            return z, R_hat_m

        def decode_from_xt(R_hat_m: jnp.ndarray, x_t: jnp.ndarray):
            cond = (R_hat_m - model_cond_mu) / model_cond_sigma
            if train_t_pairs.shape[0] == 0:
                return x_t * model_x_sigma + model_x_mu

            step = self.model._make_ddim_step(
                model_ema_params,
                model_apply_fn,
                model_a_bar_s,
                model_eps,
                train_eta,
            )
            dummy_key = random.PRNGKey(0)
            (_, x_final, _), _ = jax.lax.scan(step, (dummy_key, x_t, cond), xs=train_t_pairs)
            return x_final * model_x_sigma + model_x_mu

        @jax.jit
        def train_step(
            params: Dict[str, jnp.ndarray],
            opt_state: optax.OptState,
            R_m_batch: jnp.ndarray,
            R_x_batch: jnp.ndarray,
            x_t_batch: jnp.ndarray,
        ):
            def loss_fn(p):
                z, R_hat_m = apply_linear(p, R_m_batch)
                X_hat = decode_from_xt(R_hat_m, x_t_batch)
                ambient_loss = jnp.mean((X_hat - R_x_batch) ** 2)
                latent_loss = jnp.mean((R_hat_m - R_m_batch) ** 2)
                reg = jnp.mean(p["L"] ** 2) + jnp.mean(p["U"] ** 2)
                if use_bias:
                    reg = reg + jnp.mean(p["bL"] ** 2) + jnp.mean(p["bU"] ** 2)
                total = ambient_w * ambient_loss + latent_w * latent_loss + l2_w * reg
                aux = {
                    "ambient_loss": ambient_loss,
                    "latent_loss": latent_loss,
                    "reg": reg,
                    "z_var": jnp.var(z),
                }
                return total, aux

            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, opt_state = tx.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss, aux

        return train_step

    def fit(self, n_iter: Optional[int] = None):
        n_iter = int(self.sch["n_iter"] if n_iter is None else n_iter)
        batch_size = int(self.sch["batch_size"])
        verbose_every = int(self.sch["verbose_every"])

        for it in range(n_iter):
            if batch_size >= self.N:
                idx = jnp.arange(self.N)
            else:
                self.key, k_perm = random.split(self.key)
                idx = random.permutation(k_perm, self.N)[:batch_size]

            R_m_batch = self.R_im[idx]
            R_x_batch = self.R_iX[idx]
            x_t_batch = self.train_x_t[idx]

            self.params, self.opt_state, loss, aux = self._train_step_jit(
                self.params,
                self.opt_state,
                R_m_batch,
                R_x_batch,
                x_t_batch,
            )

            if verbose_every and (it % verbose_every == 0 or it == n_iter - 1):
                print(
                    f"iter {it:6d}  loss {float(loss):.6f}  "
                    f"ambient {float(aux['ambient_loss']):.6f}",
                    end="\r",
                )

        if verbose_every:
            print("\nDIAE training complete.")
        return self

    # ------------------------------------------------------------------
    # user-facing inference API
    # ------------------------------------------------------------------
    def predict_latents(self, R_am: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Return (z, Rhat_am) for new diffusion-map latents."""
        R_am = jnp.asarray(R_am, dtype=jnp.float32)
        if R_am.ndim == 1:
            R_am = R_am[None, :]
        return self._apply_linear(self.params, R_am)

    def predict(
        self,
        R_am: jnp.ndarray,
        *,
        t: Optional[int] = None,
        steps: Optional[int] = None,
        x_t: Optional[jnp.ndarray] = None,
        eta: Optional[float] = None,
        return_latents: bool = False,
        return_path: bool = False,
    ):
        """
        Decode new latent points through the learned M->d->M map and frozen DDIM.

        Parameters
        ----------
        R_am : (B, M)
            New diffusion-map latents.
        t, steps, x_t, eta : passed to the frozen DDIM sampler.
        return_latents : bool
            If True, also return the bottleneck z and lifted latent Rhat_am.
        return_path : bool
            If True, also return the intermediate DDIM reverse path.
        """
        R_am = jnp.asarray(R_am, dtype=jnp.float32)
        if R_am.ndim == 1:
            R_am = R_am[None, :]
        if R_am.shape[1] != self.M:
            raise ValueError(f"R_am must have shape (B, {self.M})")

        z, R_hat_m = self._apply_linear(self.params, R_am)

        t = int(self.train_t if t is None else t)
        steps = self.train_steps if steps is None else steps
        eta = float(self.train_eta if eta is None else eta)
        t_pairs = self._make_t_pairs(t, steps)

        if x_t is None:
            x_t = self._default_inference_x_t(R_am.shape[0])
        else:
            x_t = jnp.asarray(x_t, dtype=jnp.float32)
            if x_t.ndim == 1:
                x_t = x_t[None, :]
            if x_t.shape != (R_am.shape[0], self.D):
                raise ValueError(f"x_t must have shape ({R_am.shape[0]}, {self.D})")

        if return_path:
            X_hat, path = self._decode_from_xt(R_hat_m, x_t, t_pairs=t_pairs, eta=eta, return_path=True)
            if return_latents:
                return X_hat, z, R_hat_m, path
            return X_hat, path

        X_hat = self._decode_from_xt(R_hat_m, x_t, t_pairs=t_pairs, eta=eta, return_path=False)
        if return_latents:
            return X_hat, z, R_hat_m
        return X_hat

    def __call__(self, R_am: jnp.ndarray, **kwargs):
        return self.predict(R_am, **kwargs)

    # ------------------------------------------------------------------
    # diagnostics / convenience
    # ------------------------------------------------------------------
    def reconstruction_mse(
        self,
        R_iX: Optional[jnp.ndarray] = None,
        R_im: Optional[jnp.ndarray] = None,
        *,
        t: Optional[int] = None,
        steps: Optional[int] = None,
        x_t: Optional[jnp.ndarray] = None,
    ) -> float:
        if R_iX is None:
            R_iX = self.R_iX
        else:
            R_iX = jnp.asarray(R_iX, dtype=jnp.float32)
        if R_im is None:
            R_im = self.R_im
        else:
            R_im = jnp.asarray(R_im, dtype=jnp.float32)
        X_hat = self.predict(R_im, t=t, steps=steps, x_t=x_t)
        return float(jnp.mean((X_hat - R_iX) ** 2))

    def get_state(self) -> Dict[str, Any]:
        return {
            "params": self.params,
            "opt_state": self.opt_state,
            "d": self.d,
            "M": self.M,
            "D": self.D,
            "loss_cfg": self.loss_cfg,
            "training_sch": self.sch,
        }


__all__ = ["DIAE"]
