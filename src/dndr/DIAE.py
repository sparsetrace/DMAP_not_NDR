import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from typing import Any, Dict, Optional, Tuple, Union

try:
    import optax
except Exception:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "optax>=0.2.2"])
    import optax


class DIAE:
    """
    Diffusion Isometric Autoencoder wrapper around a frozen DDIM decoder.

    Core map:
        R_im --L(M->d)--> z --U(d->M)--> Rhat_im --frozen DDIM--> Rhat_iX

    Loss:
        ambient MSE + optional latent MSE + optional L2
        + optional simplified UMAP-style locality loss on z

    Notes
    -----
    * The wrapped DDIM is frozen.
    * Only L, U (and optional biases) are trainable.
    * If UMAP loss is active, this implementation uses full-batch training by
      default because the graph lives on the whole dataset.
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
        self.umap_active = float(self.loss_cfg["umap"]) > 0.0

        # Fixed DDIM decode schedule used during DIAE training.
        self.train_t = int(self.sch["t"])
        self.train_steps = self.sch["steps"]
        self.train_eta = float(self.sch["eta"])
        self.train_t_pairs = self._make_t_pairs(self.train_t, self.train_steps)

        # Fixed training start state x_t for deterministic optimization.
        self.train_x_t = self._make_train_x_t(self.sch["train_x_t"])

        # Optional UMAP graph on the whole dataset.
        self.umap_graph = self._build_umap_graph() if self.umap_active else None

        # Initialize linear maps.
        self.params = self._init_params(scale=float(self.sch["init_scale"]))

        # Optimizer + schedule
        n_iter = int(self.sch["n_iter"])
        base_lr = float(self.sch["learning_rate"])
        lr_schedule_name = str(self.sch.get("lr_schedule", "constant")).lower()
        end_value = float(self.sch.get("end_value", 0.0))

        if lr_schedule_name == "constant":
            lr_schedule = base_lr

        elif lr_schedule_name == "cosine":
            lr_schedule = optax.cosine_decay_schedule(
                init_value=base_lr,
                decay_steps=n_iter,
                alpha=end_value / max(base_lr, 1e-12),
            )

        elif lr_schedule_name == "warmup_cosine":
            warmup_frac = float(self.sch.get("warmup_frac", 0.05))
            warmup_steps = max(1, int(warmup_frac * n_iter))
            lr_schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=base_lr,
                warmup_steps=warmup_steps,
                decay_steps=n_iter,
                end_value=end_value,
            )

        else:
            raise ValueError(
                "training_sch['lr_schedule'] must be one of "
                "{'constant', 'cosine', 'warmup_cosine'}"
            )

        self.lr_schedule = lr_schedule
        self.tx = optax.adam(self.lr_schedule)
        self.opt_state = self.tx.init(self.params)

        self._train_step_jit = self._make_train_step()

        if self.training:
            self.fit(n_iter=n_iter)

    # ------------------------------------------------------------------
    # configuration
    # ------------------------------------------------------------------
    def _parse_loss(self, loss: Union[str, Dict[str, float]]) -> Dict[str, float]:
        defaults = {
            "ambient": 1.0,
            "latent": 0.0,
            "l2": 0.0,
            "umap": 0.0,
            "umap_gamma": 1.0,
            "umap_neg_ratio": 1.0,
        }

        if isinstance(loss, str):
            mode = loss.lower()
            if mode == "mse":
                return defaults
            if mode == "umap":
                cfg = defaults.copy()
                cfg["umap"] = 1.0
                return cfg
            raise ValueError("loss must be 'mse', 'umap', or a dict")

        cfg = defaults.copy()
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

            # learning-rate schedule
            "lr_schedule": "constant",   # {'constant', 'cosine', 'warmup_cosine'}
            "warmup_frac": 0.05,
            "end_value": 0.0,

            # simplified UMAP graph settings
            "umap_k": 15,
            "umap_source": "ambient",    # {'ambient', 'latent'}
            "umap_a": 1.0,
            "umap_b": 1.0,
            "umap_max_neg": 50000,
            "umap_force_full_batch": True,

            # optional precomputed graph or source data
            # "umap_knn_idx": np.ndarray shape (N,k)
            # "umap_knn_dist": np.ndarray shape (N,k)
            # "umap_knn_data": np.ndarray shape (N,p)
        }
        if training_sch is not None:
            defaults.update(training_sch)
        return defaults

    # ------------------------------------------------------------------
    # UMAP graph
    # ------------------------------------------------------------------
    def _build_umap_graph(self) -> Dict[str, jnp.ndarray]:
        knn_idx = self.sch.get("umap_knn_idx", None)
        knn_dist = self.sch.get("umap_knn_dist", None)
        src_name = str(self.sch.get("umap_source", "ambient")).lower()
        X_for_knn = self.sch.get("umap_knn_data", None)

        if X_for_knn is None:
            if src_name == "ambient":
                X_for_knn = np.asarray(self.R_iX)
            elif src_name == "latent":
                X_for_knn = np.asarray(self.R_im)
            else:
                raise ValueError("training_sch['umap_source'] must be 'ambient' or 'latent'")
        else:
            X_for_knn = np.asarray(X_for_knn, dtype=np.float32)

        if X_for_knn.shape[0] != self.N:
            raise ValueError("umap_knn_data must have the same number of rows as R_iX / R_im")

        if knn_idx is None:
            try:
                from sklearn.neighbors import NearestNeighbors
            except Exception:
                import sys, subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
                from sklearn.neighbors import NearestNeighbors

            k = int(self.sch.get("umap_k", 15))
            nbrs = NearestNeighbors(n_neighbors=k + 1)
            nbrs.fit(X_for_knn)
            dist, idx = nbrs.kneighbors(X_for_knn)
            knn_idx = idx[:, 1:]
            knn_dist = dist[:, 1:]
        else:
            knn_idx = np.asarray(knn_idx)
            if knn_idx.ndim != 2 or knn_idx.shape[0] != self.N:
                raise ValueError("umap_knn_idx must have shape (N, k)")
            if knn_dist is None:
                diffs = X_for_knn[knn_idx] - X_for_knn[:, None, :]
                knn_dist = np.linalg.norm(diffs, axis=2)
            else:
                knn_dist = np.asarray(knn_dist)
                if knn_dist.shape != knn_idx.shape:
                    raise ValueError("umap_knn_dist must have the same shape as umap_knn_idx")

        positive = knn_dist[knn_dist > 0]
        sigma = float(np.median(positive)) if positive.size else 1.0
        sigma = max(sigma, 1e-6)

        # Simple affinity proxy for the graph weights.
        w = np.exp(-(knn_dist ** 2) / (sigma ** 2 + 1e-12)).astype(np.float32)
        w = np.clip(w, 1e-3, 1.0)

        rows = np.repeat(np.arange(self.N, dtype=np.int32), knn_idx.shape[1])
        cols = knn_idx.reshape(-1).astype(np.int32)
        edge_w = w.reshape(-1).astype(np.float32)

        # Symmetrize.
        edge_i = np.concatenate([rows, cols], axis=0)
        edge_j = np.concatenate([cols, rows], axis=0)
        edge_w = np.concatenate([edge_w, edge_w], axis=0)

        return {
            "edge_i": jnp.asarray(edge_i, dtype=jnp.int32),
            "edge_j": jnp.asarray(edge_j, dtype=jnp.int32),
            "edge_w": jnp.asarray(edge_w, dtype=jnp.float32),
            "sigma": jnp.asarray(sigma, dtype=jnp.float32),
            "num_edges": int(edge_i.shape[0]),
        }

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
        umap_w = float(self.loss_cfg["umap"])
        umap_gamma = float(self.loss_cfg["umap_gamma"])
        umap_neg_ratio = float(self.loss_cfg["umap_neg_ratio"])
        use_bias = bool(self.use_bias)
        eps_small = 1e-8

        model_cond_mu = self.model.cond_mu
        model_cond_sigma = self.model.cond_sigma
        model_a_bar_s = self.model.a_bar_s
        model_eps = float(self.model.eps)
        model_ema_params = self.model.state.ema_params
        model_apply_fn = self.model.state.apply_fn
        model_x_mu = self.model.x_mu
        model_x_sigma = self.model.x_sigma

        if self.umap_active:
            edge_i = self.umap_graph["edge_i"]
            edge_j = self.umap_graph["edge_j"]
            edge_w = self.umap_graph["edge_w"]
            umap_a = float(self.sch.get("umap_a", 1.0))
            umap_b = float(self.sch.get("umap_b", 1.0))
            max_neg = int(self.sch.get("umap_max_neg", 50000))
            num_edges = int(self.umap_graph["num_edges"])
            num_neg = int(min(max_neg, max(1, int(num_edges * umap_neg_ratio))))
        else:
            edge_i = edge_j = edge_w = None
            umap_a = umap_b = 1.0
            num_neg = 0

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
            key: jax.Array,
        ):
            if umap_w > 0.0:
                key, k1, k2 = random.split(key, 3)
                neg_i = random.randint(k1, shape=(num_neg,), minval=0, maxval=R_m_batch.shape[0])
                neg_offset = random.randint(k2, shape=(num_neg,), minval=1, maxval=R_m_batch.shape[0])
                neg_j = (neg_i + neg_offset) % R_m_batch.shape[0]
            else:
                neg_i = neg_j = None

            def loss_fn(p):
                z, R_hat_m = apply_linear(p, R_m_batch)
                X_hat = decode_from_xt(R_hat_m, x_t_batch)

                ambient_loss = jnp.mean((X_hat - R_x_batch) ** 2)
                latent_loss = jnp.mean((R_hat_m - R_m_batch) ** 2)

                reg = jnp.mean(p["L"] ** 2) + jnp.mean(p["U"] ** 2)
                if use_bias:
                    reg = reg + jnp.mean(p["bL"] ** 2) + jnp.mean(p["bU"] ** 2)

                if umap_w > 0.0:
                    # Positive edges
                    z_pos_i = z[edge_i]
                    z_pos_j = z[edge_j]
                    d2_pos = jnp.sum((z_pos_i - z_pos_j) ** 2, axis=1)
                    q_pos = 1.0 / (1.0 + umap_a * jnp.power(d2_pos + eps_small, umap_b))
                    umap_attr = -jnp.mean(edge_w * jnp.log(jnp.clip(q_pos, eps_small, 1.0)))

                    # Negative samples
                    z_neg_i = z[neg_i]
                    z_neg_j = z[neg_j]
                    d2_neg = jnp.sum((z_neg_i - z_neg_j) ** 2, axis=1)
                    q_neg = 1.0 / (1.0 + umap_a * jnp.power(d2_neg + eps_small, umap_b))
                    umap_rep = -jnp.mean(jnp.log(jnp.clip(1.0 - q_neg, eps_small, 1.0)))

                    umap_loss = umap_attr + umap_gamma * umap_rep
                else:
                    umap_attr = jnp.array(0.0, dtype=jnp.float32)
                    umap_rep = jnp.array(0.0, dtype=jnp.float32)
                    umap_loss = jnp.array(0.0, dtype=jnp.float32)

                total = (
                    ambient_w * ambient_loss
                    + latent_w * latent_loss
                    + l2_w * reg
                    + umap_w * umap_loss
                )

                aux = {
                    "ambient_loss": ambient_loss,
                    "latent_loss": latent_loss,
                    "reg": reg,
                    "umap_attr": umap_attr,
                    "umap_rep": umap_rep,
                    "umap_loss": umap_loss,
                    "z_var": jnp.var(z),
                }
                return total, aux

            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, opt_state = tx.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss, aux, key

        return train_step

    def fit(self, n_iter: Optional[int] = None):
        n_iter = int(self.sch["n_iter"] if n_iter is None else n_iter)
        batch_size = int(self.sch["batch_size"])
        verbose_every = int(self.sch["verbose_every"])

        if self.umap_active and bool(self.sch.get("umap_force_full_batch", True)):
            batch_size = self.N

        for it in range(n_iter):
            if batch_size >= self.N:
                idx = jnp.arange(self.N)
            else:
                self.key, k_perm = random.split(self.key)
                idx = random.permutation(k_perm, self.N)[:batch_size]

            R_m_batch = self.R_im[idx]
            R_x_batch = self.R_iX[idx]
            x_t_batch = self.train_x_t[idx]

            self.params, self.opt_state, loss, aux, self.key = self._train_step_jit(
                self.params,
                self.opt_state,
                R_m_batch,
                R_x_batch,
                x_t_batch,
                self.key,
            )

            if verbose_every and (it % verbose_every == 0 or it == n_iter - 1):
                if callable(self.lr_schedule):
                    lr_now = float(self.lr_schedule(it))
                else:
                    lr_now = float(self.lr_schedule)

                msg = (
                    f"iter {it:6d}  loss {float(loss):.6f}  "
                    f"ambient {float(aux['ambient_loss']):.6f}  "
                    f"latent {float(aux['latent_loss']):.6f}  "
                    f"umap {float(aux['umap_loss']):.6f}  "
                    f"lr {lr_now:.6e}"
                )
                print(msg, end="\r")

        if verbose_every:
            print("\nDIAE training complete.")
        return self

    # ------------------------------------------------------------------
    # user-facing inference API
    # ------------------------------------------------------------------
    def predict_latents(self, R_am: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
        state = {
            "params": self.params,
            "opt_state": self.opt_state,
            "d": self.d,
            "M": self.M,
            "D": self.D,
            "loss_cfg": self.loss_cfg,
            "training_sch": self.sch,
        }
        if self.umap_active:
            state["umap_graph"] = self.umap_graph
        return state


__all__ = ["DIAE"]
