"""Structured probabilistic lap-time network (design doc §2, Network A).

The target is the lap time in percent above the driver's qualifying lap,
``y = 100 · (lap_time / q_s − 1)``, decomposed into physically meaningful terms:

    y = base(circuit, quali pace, grid, race distance)
      + conditions(track temp, compound)                       (shared by all circuits)
      + fuel_rate(circuit) · laps_remaining                    (rate ≥ 0)
      + offset(compound, circuit, fresh)
      + deg(tyre_age | compound, circuit, fresh)               (non-decreasing, convex in age)
      + traffic(gap ahead, leader, circuit)
      + interaction(everything except tyre age and temp)       (bounded to ±0.5 %)
      + ε,  ε ~ N(0, σ(context)²)

Identifiability rule: effects measurable *within* a race (fuel, tyre wear, traffic) may vary
by circuit; race-level conditions (track temperature) enter only through terms shared across
circuits. With 1–3 races per circuit, a circuit-specific temperature effect is confounded with
everything else that differed between those races (2024 validation: the network learned a
Lusail "hotter = faster" slope from 2021 vs 2023, really the 2023 18-lap stint limit, and
extrapolated it to +3.5 s at a cooler 2024 race).

``deg`` is a non-negative combination of a linear term and softened hinges at 10/20/30 laps,
so it can bend upwards (a tyre cliff) but never predicts that older tyres get faster, which
keeps counterfactual long stints physically plausible. The interaction term catches what the
additive terms miss (e.g. qualifying gap × circuit) without seeing tyre age, so the
monotonicity guarantee holds for the full prediction. Circuits are embedded; 10% of training
laps see the "unknown circuit" embedding so unseen circuits get a sensible prior.

Training is two-stage, because a jointly trained noise head learns training-season noise
that doesn't transfer to a new season (overconfident σ, and early stopping on NLL halts the
mean after a few epochs):

1. **Mean**: Huber loss on the mean terms. The epoch count is chosen by within-race MAE on
   the last training season (same protocol as the LightGBM baseline), then models are
   refitted on all training seasons.
2. **Uncertainty**: leave-one-season-out residuals of the mean (fit on the other training
   seasons, predict the held-out one) train the σ head by Gaussian NLL with everything else
   frozen, so σ reflects out-of-season error while staying heteroscedastic.

An ensemble of seeds is combined as a Gaussian mixture (mean of means, total variance).
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from strategist.eval.metrics import within_mae
from strategist.models.features import COMPOUNDS

log = logging.getLogger(__name__)

UNKNOWN = 0  # circuit index for unseen circuits
KNOTS = (10.0, 20.0, 30.0)
SIGMA_FLOOR = 0.05  # percent of the qualifying lap (~0.045 s at 90 s)
INTERACTION_BOUND = 0.5  # percent

CONTEXT_FEATURES = ["quali_gap_pct", "quali_missing", "grid_position", "track_temp", "total_laps"]
TEMP = CONTEXT_FEATURES.index("track_temp")
RACE_CONTEXT = [i for i in range(len(CONTEXT_FEATURES)) if i != TEMP]  # may meet the circuit


@dataclass
class Batch:
    circuit: torch.Tensor   # long
    compound: torch.Tensor  # long
    context: torch.Tensor   # standardised CONTEXT_FEATURES
    tyre_life: torch.Tensor
    laps_remaining: torch.Tensor
    fresh: torch.Tensor
    gap: torch.Tensor
    leader: torch.Tensor
    fuel_frac: torch.Tensor

    def subset(self, idx: torch.Tensor) -> "Batch":
        return Batch(**{k: v[idx] for k, v in self.__dict__.items()})


def mlp(n_in: int, hidden: int, n_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                         nn.Linear(hidden, n_out))


class LapNetModule(nn.Module):
    def __init__(self, n_circuits: int, emb_dim: int = 8, compound_dim: int = 4, hidden: int = 32,
                 base_init: float = 7.0, tau: float = 3.0, interaction: bool = True):
        super().__init__()
        self.tau = tau
        self.circuit_emb = nn.Embedding(n_circuits, emb_dim)
        self.compound_emb = nn.Embedding(len(COMPOUNDS), compound_dim)
        nr = len(RACE_CONTEXT)
        tyre_in = emb_dim + compound_dim + 1  # + fresh

        self.base = mlp(emb_dim + nr, hidden, 1)
        self.conditions = mlp(compound_dim + 1, hidden // 2, 1)  # no circuit input, by design
        self.fuel = nn.Linear(emb_dim, 1)
        self.offset = mlp(tyre_in, hidden, 1)
        self.deg = mlp(tyre_in, hidden, 1 + len(KNOTS))
        self.traffic = mlp(emb_dim + 3, hidden, 1)
        self.interaction = mlp(emb_dim + compound_dim + nr + 4, hidden, 1) if interaction else None
        self.sigma = mlp(emb_dim + compound_dim + 6, hidden, 1)

        with torch.no_grad():
            self.base[-1].bias.fill_(base_init)
            self.fuel.bias.fill_(np.log(np.expm1(0.8)))   # ≈0.08 %/lap ≈ 0.07 s/lap at 90 s
            self.deg[-1].bias.fill_(-1.0)
            if self.interaction is not None:
                self.interaction[-1].weight.mul_(0.1)
                self.interaction[-1].bias.zero_()

    def basis(self, age: torch.Tensor) -> torch.Tensor:
        """[age, τ·softplus((age − k)/τ) for each knot] / 10, all non-decreasing and convex."""
        hinges = [self.tau * F.softplus((age - k) / self.tau) for k in KNOTS]
        return torch.stack([age, *hinges], dim=-1) / 10.0

    def forward(self, b: Batch) -> dict[str, torch.Tensor]:
        ec, ek = self.circuit_emb(b.circuit), self.compound_emb(b.compound)
        temp = b.context[:, TEMP].unsqueeze(-1)
        race_ctx = b.context[:, RACE_CONTEXT]
        tyre_ctx = torch.cat([ec, ek, b.fresh.unsqueeze(-1)], dim=-1)
        near = torch.exp(-b.gap).unsqueeze(-1)
        leader = b.leader.unsqueeze(-1)

        base = self.base(torch.cat([ec, race_ctx], dim=-1)).squeeze(-1)
        conditions = self.conditions(torch.cat([ek, temp], dim=-1)).squeeze(-1)
        fuel = F.softplus(self.fuel(ec)).squeeze(-1) * b.laps_remaining / 10.0
        offset = self.offset(tyre_ctx).squeeze(-1)
        deg = (F.softplus(self.deg(tyre_ctx)) * self.basis(b.tyre_life)).sum(-1)
        traffic = self.traffic(torch.cat([ec, near, (b.gap / 10).unsqueeze(-1), leader],
                                         dim=-1)).squeeze(-1)
        if self.interaction is not None:  # no tyre age (monotonicity) and no temp (identifiability)
            x = torch.cat([ec, ek, race_ctx, b.fuel_frac.unsqueeze(-1), near, leader,
                           b.fresh.unsqueeze(-1)], dim=-1)
            interaction = INTERACTION_BOUND * torch.tanh(self.interaction(x).squeeze(-1))
        else:
            interaction = torch.zeros_like(base)
        sig_in = torch.cat([ec, ek, (b.tyre_life / 30).unsqueeze(-1), near, leader,
                            b.fuel_frac.unsqueeze(-1), b.context[:, :2]], dim=-1)
        sigma = F.softplus(self.sigma(sig_in)).squeeze(-1) + SIGMA_FLOOR
        mu = base + conditions + fuel + offset + deg + traffic + interaction
        return {"mu": mu, "sigma": sigma, "base": base, "conditions": conditions, "fuel": fuel,
                "offset": offset, "deg": deg, "traffic": traffic, "interaction": interaction}


def gaussian_nll(y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return (0.5 * ((y - mu) / sigma) ** 2 + torch.log(sigma)).mean()


class LapTimeNet:
    name = "lap_net"

    def __init__(self, emb_dim: int = 8, hidden: int = 32, interaction: bool = True,
                 lr: float = 3e-3, weight_decay: float = 1e-4, batch_size: int = 1024,
                 max_epochs: int = 40, patience: int = 8, huber_delta: float = 1.0,
                 circuit_dropout: float = 0.1, sigma_epochs: int = 15, ensemble: int = 1,
                 seed: int = 42, threads: int = 4):
        self.hp = dict(emb_dim=emb_dim, hidden=hidden, interaction=interaction)
        self.lr, self.weight_decay, self.batch_size = lr, weight_decay, batch_size
        self.max_epochs, self.patience, self.huber_delta = max_epochs, patience, huber_delta
        self.circuit_dropout, self.sigma_epochs = circuit_dropout, sigma_epochs
        self.ensemble, self.seed, self.threads = ensemble, seed, threads

    # ---- data -------------------------------------------------------------------------------
    def _batch(self, df: pd.DataFrame) -> Batch:
        ctx = (df[CONTEXT_FEATURES].astype(float) - self.mean) / self.std
        f = lambda col: torch.tensor(df[col].astype(float).to_numpy(), dtype=torch.float32)
        return Batch(
            circuit=torch.tensor(df["circuit"].map(self.circuit_index).fillna(UNKNOWN)
                                 .astype(int).to_numpy()),
            compound=torch.tensor(df["compound"].map({c: i for i, c in enumerate(COMPOUNDS)})
                                  .fillna(1).astype(int).to_numpy()),
            context=torch.tensor(ctx.fillna(0.0).to_numpy(), dtype=torch.float32),
            tyre_life=f("tyre_life"), laps_remaining=f("laps_remaining"),
            fresh=f("fresh_tyre").nan_to_num(1.0), gap=f("gap_ahead_s"), leader=f("is_leader"),
            fuel_frac=f("fuel_frac"),
        )

    @staticmethod
    def _target(df: pd.DataFrame) -> torch.Tensor:
        return torch.tensor(100.0 * (df["ratio"].to_numpy() - 1.0), dtype=torch.float32)

    def _new_module(self, train: pd.DataFrame) -> LapNetModule:
        return LapNetModule(len(self.circuit_index) + 1,
                            base_init=float(self._target(train).median()), **self.hp)

    # ---- stage 1: mean ----------------------------------------------------------------------
    def _train_mean(self, train: pd.DataFrame, seed: int, epochs: int,
                    val: pd.DataFrame | None = None) -> tuple[LapNetModule, int]:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        module = self._new_module(train)
        mean_params = [p for n, p in module.named_parameters() if not n.startswith("sigma.")]
        opt = torch.optim.AdamW(mean_params, lr=self.lr, weight_decay=self.weight_decay)
        # per-epoch decay independent of the total, so a refit for the early-stopped epoch
        # count follows the same trajectory as the run that chose it
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.96)
        xb, y = self._batch(train), self._target(train)
        xv = self._batch(val) if val is not None else None
        best, best_epoch, best_state, stale = np.inf, epochs, None, 0
        for epoch in range(1, epochs + 1):
            module.train()
            for idx in torch.tensor(rng.permutation(len(y))).split(self.batch_size):
                b = xb.subset(idx)
                drop = torch.rand(len(idx)) < self.circuit_dropout
                b.circuit = torch.where(drop, torch.full_like(b.circuit, UNKNOWN), b.circuit)
                loss = F.huber_loss(module(b)["mu"], y[idx], delta=self.huber_delta)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(mean_params, 5.0)
                opt.step()
            sched.step()
            if val is None:
                continue
            module.eval()
            with torch.no_grad():
                mu = (1.0 + module(xv)["mu"].numpy() / 100.0) * val["q_s"].to_numpy()
            score = within_mae(val, val["lap_time_s"], pd.Series(mu, index=val.index))
            if score < best - 1e-4:
                best, best_epoch, stale = score, epoch, 0
                best_state = {k: t.clone() for k, t in module.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is not None:
            module.load_state_dict(best_state)
        return module, best_epoch

    # ---- stage 2: uncertainty ---------------------------------------------------------------
    def _fit_sigma(self, module: LapNetModule, laps: pd.DataFrame, resid: torch.Tensor,
                   seed: int) -> None:
        """Train only the σ head on out-of-season residuals (percent units)."""
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        params = list(module.sigma.parameters())
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        xb = self._batch(laps)
        module.eval()  # no dropout/batch-norm here, but keep embeddings fixed
        for _ in range(self.sigma_epochs):
            for idx in torch.tensor(rng.permutation(len(resid))).split(self.batch_size):
                sigma = module(xb.subset(idx))["sigma"]
                loss = gaussian_nll(resid[idx], torch.zeros_like(sigma), sigma)
                opt.zero_grad()
                loss.backward()
                opt.step()

    def fit(self, train: pd.DataFrame) -> "LapTimeNet":
        torch.set_num_threads(self.threads)
        self.circuit_index = {c: i + 1 for i, c in enumerate(sorted(train["circuit"].unique()))}
        self.mean = train[CONTEXT_FEATURES].astype(float).mean()
        self.std = train[CONTEXT_FEATURES].astype(float).std().replace(0.0, 1.0)

        seasons = sorted(train["year"].unique())
        if len(seasons) > 1:
            inner_train, inner_val = train[train["year"] < seasons[-1]], train[train["year"] == seasons[-1]]
            _, self.epochs = self._train_mean(inner_train, self.seed, self.max_epochs, inner_val)
        else:
            self.epochs = self.max_epochs // 2
        log.info("lap_net: %d epochs (early stopping on %d)", self.epochs, seasons[-1])

        # out-of-season residuals for the σ head
        resid_parts, resid_laps = [], []
        for s in seasons if len(seasons) > 1 else []:
            fold_train, fold_eval = train[train["year"] != s], train[train["year"] == s]
            fold, _ = self._train_mean(fold_train, self.seed, self.epochs)
            with torch.no_grad():
                fold.eval()
                mu = fold(self._batch(fold_eval))["mu"]
            resid_parts.append(self._target(fold_eval) - mu)
            resid_laps.append(fold_eval)
        log.info("lap_net: σ head from %d out-of-season residuals", sum(len(r) for r in resid_parts))
        if resid_parts:  # kept for the simulator's split of race-level vs lap-level variance
            self.oos_residuals = pd.concat(resid_laps)[["race_id", "driver", "q_s"]].assign(
                resid_pct=torch.cat(resid_parts).numpy())

        self.modules = []
        for k in range(self.ensemble):
            module, _ = self._train_mean(train, self.seed + k, self.epochs)
            if resid_parts:
                self._fit_sigma(module, pd.concat(resid_laps), torch.cat(resid_parts), self.seed + k)
            self.modules.append(module)
        return self

    # ---- prediction -------------------------------------------------------------------------
    def _forward(self, df: pd.DataFrame) -> list[dict[str, torch.Tensor]]:
        b = self._batch(df)
        outs = []
        with torch.no_grad():
            for m in self.modules:
                m.eval()
                outs.append(m(b))
        return outs

    def predict_ratio_pct(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        outs = self._forward(df)
        mus = torch.stack([o["mu"] for o in outs])
        var = torch.stack([o["sigma"] ** 2 for o in outs])
        mu = mus.mean(0)
        total = (var + mus**2).mean(0) - mu**2
        return mu.numpy(), total.clamp_min(SIGMA_FLOOR**2).sqrt().numpy()

    def predict_dist(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        mu, sigma = self.predict_ratio_pct(df)
        q = df["q_s"].to_numpy()
        return (pd.Series((1.0 + mu / 100.0) * q, index=df.index),
                pd.Series(sigma / 100.0 * q, index=df.index))

    def predict(self, df: pd.DataFrame) -> pd.Series:
        return self.predict_dist(df)[0]

    def predict_batch(self, b: Batch, members: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Mixture mean and σ (percent of q_s) for a pre-built batch: the simulator's fast path.
        ``members`` limits the ensemble (cheaper screening)."""
        with torch.no_grad():
            outs = [m(b) for m in self.modules[:members]]
        mus = torch.stack([o["mu"] for o in outs])
        var = torch.stack([o["sigma"] ** 2 for o in outs])
        mu = mus.mean(0)
        return mu, ((var + mus**2).mean(0) - mu**2).clamp_min(SIGMA_FLOOR**2).sqrt()

    def static_inputs(self, df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        """Circuit index and standardised context for one row per car (constant within a race)."""
        b = self._batch(df.assign(tyre_life=1.0, laps_remaining=0.0, fresh_tyre=1.0,
                                  gap_ahead_s=10.0, is_leader=0.0, fuel_frac=0.0))
        return b.circuit, b.context

    # ---- persistence ------------------------------------------------------------------------
    def save(self, path) -> None:
        torch.save({
            "init": {k: getattr(self, k) for k in ("lr", "weight_decay", "batch_size",
                                                     "max_epochs", "patience", "huber_delta",
                                                     "circuit_dropout", "sigma_epochs",
                                                     "ensemble", "seed", "threads")} | self.hp,
            "circuit_index": self.circuit_index, "mean": self.mean.to_dict(),
            "std": self.std.to_dict(), "epochs": self.epochs,
            "states": [m.state_dict() for m in self.modules],
            "oos_residuals": getattr(self, "oos_residuals", None),
        }, path)

    @classmethod
    def load(cls, path) -> "LapTimeNet":
        blob = torch.load(path, weights_only=False)
        net = cls(**blob["init"])
        net.circuit_index, net.epochs = blob["circuit_index"], blob["epochs"]
        net.mean, net.std = pd.Series(blob["mean"]), pd.Series(blob["std"])
        net.oos_residuals = blob["oos_residuals"]
        net.modules = []
        for state in blob["states"]:
            m = LapNetModule(len(net.circuit_index) + 1, **net.hp)
            m.load_state_dict(state)
            m.eval()
            net.modules.append(m)
        return net

    def components(self, df: pd.DataFrame) -> pd.DataFrame:
        """Additive terms in seconds (ensemble mean), for explanations."""
        outs = self._forward(df)
        q = df["q_s"].to_numpy() / 100.0
        parts = ["base", "conditions", "fuel", "offset", "deg", "traffic", "interaction"]
        return pd.DataFrame({p: torch.stack([o[p] for o in outs]).mean(0).numpy() * q
                             for p in parts}, index=df.index)
