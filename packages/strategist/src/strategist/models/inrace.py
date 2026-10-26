"""In-race adaptation: update lap-time forecasts from the laps already run (design doc §2, v2).

``lap_net`` forecasts a race before it starts. Most of its error is race-level and persists
through the race (2021–24 out-of-season residuals: 65% of the variance is shared by the whole
field, 12% is a driver's offset in that race, and the first and second halves of a race
correlate at 0.9), so the laps already run say a lot about the laps to come. Both models here
forecast the **residual** of lap_net, ``r = 100·(lap_time/q_s − 1) − μ_lap_net`` (percent of
the qualifying lap), for every future lap from the clean laps up to a cutoff, given the
strategy each driver runs:

- ``InRaceGP``: a Gaussian process whose covariance spells out where the error comes from::

      r = μ0 + β·age + γ·e
        + c                 race: shared by the whole field
        + a_p·e             after a start or restart: the field runs slow for a few laps (cold
                            tyres, bunched cars), by an amount drawn per green-flag period p;
                            e = exp(−(laps since green − 1)/λ)
        + t·lap             race trend: e.g. lap_net's fuel effect off for this race
        + d(lap)            race drift: track evolution, wind, temperature (OU in lap number)
        + k_c + b_c·age     compound in this race: lap_net's error on this compound's pace
                            and wear, shared by every driver on it
        + o_i + u_i(lap)    driver: race-long offset and short-term pace swings (OU)
        + j_s + s_s·age     stint: set-specific offset and wear-rate error
        + ε

  The posterior is exact (Gaussian conditioning), so the forecast is calibrated if the
  structure is right, and the compound term carries information across drivers: a field
  running long on the hard tells the model how the hard wears today before a driver fits it.
  The hyperparameters are fitted by marginal likelihood on training races. The restart and
  trend terms were added after the first validation run (2025): without them the GP read slow
  laps after the start or a safety car (+1.5–3.6 s at Baku, Monaco and Singapore) as the
  race's pace, and could not
  extrapolate the steady drifts against lap_net seen in many races (Suzuka −0.8 → −2.4 s,
  Monza +0.1 → −1.8 s); the mean-reverting drift term can't.

- ``InRaceGRU``: a neural alternative with no structure imposed. A GRU reads each driver's
  race so far (residuals, tyre age, stint, compound, the field's mean residual per lap), and
  a head maps its state plus a future lap's tyre state and horizon to a mean and σ. Trained
  by Gaussian NLL at random cutoffs; epochs chosen on the last training season.

Both take only the laps up to the cutoff (``obs``) and the future laps' strategy inputs
(``target``, residual dropped), so a forecast cannot see the laps it predicts.

Training residuals must be out-of-sample (``LapTimeNet.oos_residuals``: each season predicted
by a lap_net fitted on the other seasons), or the race-level error the models learn to
remove would look smaller than it is on a new race.
"""

import logging

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

log = logging.getLogger(__name__)

ROW_COLUMNS = ["race_id", "driver", "stint", "lap_number", "tyre_life", "compound", "hardness",
               "total_laps", "since_green", "green_period"]
AGE_SCALE = 10.0  # tyre age in units of 10 laps inside the GP
LAP_SCALE = 50.0  # race trend per 50 laps
SIGMA_FLOOR = 0.05  # percent, as lap_net


def green_flag_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Per (race, lap): laps since the start or the last restart (1 = first green lap after
    it, 0 = the start lap or a lap under SC, VSC or red flag, as the leader saw it) and a
    green-flag period counter."""
    from strategist.sim.params import lap_status  # the simulator's leader-based status

    status = lap_status(laps)
    parts = []
    for race_id, st in status.groupby(level=0):
        st = st.droplevel(0).sort_index()
        lap = st.index.to_numpy()
        marker = (lap == 1) | (st.to_numpy() != "GREEN")
        last = np.maximum.accumulate(np.where(marker, lap, 0))
        prev = np.concatenate([[0], last[:-1]])
        period = np.cumsum(np.concatenate([[False], marker[:-1] & ~marker[1:]]))
        parts.append(pd.DataFrame({"race_id": race_id, "lap_number": lap,
                                   "since_green": np.where(marker, 0, lap - prev),
                                   "green_period": period}))
    return pd.concat(parts, ignore_index=True)


def known_at_cutoff(target: pd.DataFrame, flags: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    """Restart inputs for laps after the cutoff as known at the cutoff: no further
    neutralisation assumed (``flags``: this race's green_flag_laps rows)."""
    upto = flags[flags["lap_number"] <= cutoff].sort_values("lap_number")
    if upto.empty:
        last_marker, period = 1, 0
    else:
        row = upto.iloc[-1]
        if row["since_green"] == 0:  # cutoff lap is the start or under a code: restart next
            last_marker, period = cutoff, int(row["green_period"]) + 1
        else:
            last_marker, period = cutoff - int(row["since_green"]), int(row["green_period"])
    return target.assign(since_green=target["lap_number"] - last_marker, green_period=period)


def residual_frame(laps: pd.DataFrame, mu_pct: np.ndarray | pd.Series) -> pd.DataFrame:
    """Rows the in-race models use: strategy inputs plus lap_net's residual (percent)."""
    out = laps[ROW_COLUMNS + ["q_s"]].copy()
    out["resid"] = 100.0 * (laps["ratio"].to_numpy() - 1.0) - np.asarray(mu_pct)
    return out


def oos_frame(net, train: pd.DataFrame) -> pd.DataFrame:
    """Training rows with lap_net's out-of-season residuals (see module docstring)."""
    oos = net.oos_residuals
    if oos is None or "lap_number" not in oos:
        raise ValueError("lap_net has no per-lap out-of-season residuals; refit it")
    keys = ["race_id", "driver", "lap_number"]
    merged = train.merge(oos[keys + ["resid_pct"]], on=keys, how="inner", validate="1:1")
    out = merged[ROW_COLUMNS + ["q_s"]].copy()
    out["resid"] = merged["resid_pct"].to_numpy()
    return out


# ---- Gaussian process --------------------------------------------------------------------------

GP_PARAMS = {  # name: initial value (percent of the qualifying lap; lengths in laps)
    "race": 1.3, "opening": 1.0, "opening_len": 3.0, "trend": 0.5, "drift": 0.3, "drift_len": 20.0, "compound": 0.2, "compound_wear": 0.2,
    "driver": 0.6, "swing": 0.5, "swing_len": 3.0, "stint": 0.2, "stint_wear": 0.3, "noise": 0.6,
}


def _race_tensors(df: pd.DataFrame) -> dict[str, torch.Tensor]:
    """Integer keys and covariates for one race's rows (keys only compared within the race)."""
    t = lambda a, dtype=torch.float64: torch.as_tensor(np.asarray(a), dtype=dtype)
    return {
        "driver": t(pd.factorize(df["driver"])[0], torch.long),
        "stint": t(pd.factorize(df["driver"].astype(str) + "/" + df["stint"].astype(str))[0], torch.long),
        "compound": t(pd.factorize(df["compound"])[0], torch.long),
        "period": t(df["green_period"].astype(int), torch.long),
        "since": t(df["since_green"].astype(float)),
        "lap": t(df["lap_number"].astype(float)),
        "age": t(df["tyre_life"].astype(float) / AGE_SCALE),
    }


class InRaceGP:
    name = "in_race_gp"

    def __init__(self, steps: int = 300, races_per_step: int = 8, lr: float = 0.03, seed: int = 0):
        self.steps, self.races_per_step, self.lr, self.seed = steps, races_per_step, lr, seed
        self.log_params = {k: torch.tensor(np.log(v), dtype=torch.float64) for k, v in GP_PARAMS.items()}
        self.mean0 = torch.tensor(0.0, dtype=torch.float64)
        self.wear0 = torch.tensor(0.0, dtype=torch.float64)
        self.open0 = torch.tensor(0.0, dtype=torch.float64)

    @property
    def params(self) -> dict[str, float]:
        return {k: float(v.exp()) for k, v in self.log_params.items()} | {
            "mean": float(self.mean0), "mean_wear": float(self.wear0), "mean_opening": float(self.open0)}

    def _cov(self, a: dict, b: dict) -> torch.Tensor:
        p = {k: v.exp() for k, v in self.log_params.items()}
        same = lambda key: (a[key][:, None] == b[key][None, :]).to(torch.float64)
        dlap = (a["lap"][:, None] - b["lap"][None, :]).abs()
        wear = a["age"][:, None] * b["age"][None, :]
        # keys are unique within a race; the stint key implies the driver
        open_a, open_b = self._opening(a), self._opening(b)
        return (p["race"] ** 2
                + same("period") * p["opening"] ** 2 * open_a[:, None] * open_b[None, :]
                + p["trend"] ** 2 * (a["lap"][:, None] / LAP_SCALE) * (b["lap"][None, :] / LAP_SCALE)
                + p["drift"] ** 2 * torch.exp(-dlap / p["drift_len"])
                + same("compound") * (p["compound"] ** 2 + p["compound_wear"] ** 2 * wear)
                + same("driver") * (p["driver"] ** 2 + p["swing"] ** 2 * torch.exp(-dlap / p["swing_len"]))
                + same("stint") * (p["stint"] ** 2 + p["stint_wear"] ** 2 * wear))

    def _opening(self, x: dict) -> torch.Tensor:
        return torch.exp(-(x["since"] - 1.0).clamp_min(0.0) / self.log_params["opening_len"].exp())

    def _mean(self, x: dict) -> torch.Tensor:
        return self.mean0 + self.wear0 * x["age"] + self.open0 * self._opening(x)

    def _noise(self) -> torch.Tensor:
        return self.log_params["noise"].exp() ** 2

    def neg_log_marginal(self, x: dict, r: torch.Tensor) -> torch.Tensor:
        K = self._cov(x, x) + self._noise() * torch.eye(len(r), dtype=torch.float64)
        L = torch.linalg.cholesky(K)
        z = torch.cholesky_solve((r - self._mean(x))[:, None], L)[:, 0]
        return 0.5 * ((r - self._mean(x)) @ z) + torch.log(torch.diagonal(L)).sum()

    def fit(self, frame: pd.DataFrame) -> "InRaceGP":
        races = [(_race_tensors(g), torch.as_tensor(g["resid"].to_numpy(), dtype=torch.float64))
                 for _, g in frame.groupby("race_id", sort=True)]
        params = [*self.log_params.values(), self.mean0, self.wear0, self.open0]
        for p in params:
            p.requires_grad_(True)
        opt = torch.optim.Adam(params, lr=self.lr)
        rng = np.random.default_rng(self.seed)
        for step in range(self.steps):
            batch = rng.choice(len(races), size=min(self.races_per_step, len(races)), replace=False)
            loss = sum(self.neg_log_marginal(*races[i]) for i in batch) / sum(len(races[i][1]) for i in batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 50 == 0:
                log.info("in_race_gp: step %d, NLL per lap %.3f", step, float(loss.detach()))
        for p in params:
            p.requires_grad_(False)
        log.info("in_race_gp: %s", {k: round(v, 3) for k, v in self.params.items()})
        return self

    def predict(self, obs: pd.DataFrame, target: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and σ (percent) of the residual on ``target`` laps given ``obs`` laps
        of the same race (``obs`` may be empty: the prior)."""
        both = pd.concat([obs[ROW_COLUMNS], target[ROW_COLUMNS]], ignore_index=True)
        x = _race_tensors(both)
        n = len(obs)
        xo = {k: v[:n] for k, v in x.items()}
        xt = {k: v[n:] for k, v in x.items()}
        with torch.no_grad():
            prior_var = torch.diagonal(self._cov(xt, xt)) + self._noise()
            mu = self._mean(xt)
            if n == 0:
                return mu.numpy(), prior_var.sqrt().numpy()
            r = torch.as_tensor(obs["resid"].to_numpy(), dtype=torch.float64)
            K = self._cov(xo, xo) + self._noise() * torch.eye(n, dtype=torch.float64)
            L = torch.linalg.cholesky(K)
            Kot = self._cov(xo, xt)
            mu = mu + Kot.T @ torch.cholesky_solve((r - self._mean(xo))[:, None], L)[:, 0]
            v = torch.linalg.solve_triangular(L, Kot, upper=False)
            var = (prior_var - (v**2).sum(0)).clamp_min(SIGMA_FLOOR**2)
        return mu.numpy(), var.sqrt().numpy()

    def state(self) -> dict:
        return self.params

    @classmethod
    def from_state(cls, state: dict) -> "InRaceGP":
        gp = cls()
        gp.log_params = {k: torch.tensor(np.log(state[k]), dtype=torch.float64) for k in GP_PARAMS}
        gp.mean0 = torch.tensor(state["mean"], dtype=torch.float64)
        gp.wear0 = torch.tensor(state["mean_wear"], dtype=torch.float64)
        gp.open0 = torch.tensor(state.get("mean_opening", 0.0), dtype=torch.float64)
        return gp


# ---- GRU ---------------------------------------------------------------------------------------

STEP_FEATURES = 9  # see _Sequences
HEAD_FEATURES = 9  # see InRaceGRU._head_inputs
SINCE_CAP = 10.0  # laps since green, capped
AGE_NORM, STINT_NORM, HARD_NORM = 40.0, 4.0, 6.0


class _Sequences:
    """One sequence per driver-race over lap numbers 1..T (T = longest race in the frame).

    ``present``: the lap has a row (a clean lap whose strategy inputs are known). ``observed``:
    its residual may be used as an input (in training every row; when forecasting, only laps
    up to the cutoff). Step inputs (zero unless observed): residual, observed flag, tyre age,
    stint number, compound hardness, laps since green; always: lap fraction, the field's mean residual on that
    lap and the share of the field observed. Step t describes lap t only, so the GRU state
    after k steps knows laps 1..k.
    """

    def __init__(self, frame: pd.DataFrame, n_laps: int, observed: np.ndarray | None = None):
        seq_keys = frame["race_id"].astype(str) + "|" + frame["driver"].astype(str)
        codes, self.keys = pd.factorize(seq_keys)
        self.T = T = int(n_laps)
        S = len(self.keys)
        lap = frame["lap_number"].to_numpy().astype(int) - 1
        if (lap >= T).any():
            raise ValueError(f"lap beyond the sequence length {T}")
        obs = np.ones(len(frame), bool) if observed is None else np.asarray(observed, bool)

        def grid(values, fill=0.0, where=None):
            g = np.full((S, T), fill, dtype=np.float32)
            w = slice(None) if where is None else where
            g[codes[w], lap[w]] = np.asarray(values)[w]
            return g

        self.present = grid(np.ones(len(frame)))
        self.observed = grid(np.ones(len(frame)), where=obs)
        self.age = grid(frame["tyre_life"].to_numpy())
        self.stint = grid(frame["stint"].to_numpy(), fill=-1.0)
        self.hard = grid(frame["hardness"].fillna(3.0).to_numpy())
        self.since = grid(frame["since_green"].clip(upper=SINCE_CAP).to_numpy())
        resid = frame["resid"].to_numpy() if "resid" in frame else np.zeros(len(frame))
        self.resid = grid(np.nan_to_num(resid), where=obs)
        race_of_seq = pd.Series(self.keys).str.split("|").str[0].to_numpy()
        self.total = (frame.groupby(codes)["total_laps"].first().reindex(range(S))
                      .fillna(T).to_numpy().astype(np.float32))
        # field summaries per race and lap, from observed rows only
        race_codes, race_names = pd.factorize(race_of_seq)
        drivers = np.bincount(race_codes)
        r_idx = race_codes[codes[obs]]
        sums = np.zeros((len(race_names), T), dtype=np.float64)
        counts = np.zeros((len(race_names), T), dtype=np.float64)
        np.add.at(sums, (r_idx, lap[obs]), np.nan_to_num(resid[obs]))
        np.add.at(counts, (r_idx, lap[obs]), 1.0)
        mean = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        self.field_mean = mean[race_codes].astype(np.float32)
        self.field_share = (counts / drivers[:, None])[race_codes].astype(np.float32)
        self.codes, self.lap_index = codes, lap

    def step_inputs(self) -> torch.Tensor:
        o = self.observed
        frac = np.arange(1, self.T + 1, dtype=np.float32)[None, :] / self.total[:, None]
        x = np.stack([self.resid * o, o, self.age / AGE_NORM * o, self.stint.clip(0) / STINT_NORM * o,
                      self.hard / HARD_NORM * o, self.since / SINCE_CAP * o,
                      np.broadcast_to(frac, o.shape),
                      self.field_mean, self.field_share], -1)
        return torch.as_tensor(x)


class _GRUModule(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.gru = nn.GRU(STEP_FEATURES, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + HEAD_FEATURES, hidden), nn.SiLU(),
                                  nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2))

    def states(self, x: torch.Tensor) -> torch.Tensor:
        """GRU state after k = 0..T steps: (S, T + 1, H)."""
        out, _ = self.gru(x)
        return torch.cat([torch.zeros_like(out[:, :1]), out], 1)


class InRaceGRU:
    name = "in_race_gru"

    def __init__(self, hidden: int = 32, lr: float = 3e-3, weight_decay: float = 1e-4,
                 max_epochs: int = 200, patience: int = 20, cutoffs_per_seq: int = 4,
                 batch_size: int = 128, seed: int = 0, threads: int = 4):
        self.hidden, self.lr, self.weight_decay = hidden, lr, weight_decay
        self.max_epochs, self.patience, self.cutoffs_per_seq = max_epochs, patience, cutoffs_per_seq
        self.batch_size, self.seed, self.threads = batch_size, seed, threads

    # ---- shared pieces --------------------------------------------------------------------
    @staticmethod
    def _head_inputs(seq: _Sequences, idx: np.ndarray, k: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        """Head features for every lap of sequences ``idx`` at cutoffs ``k`` (laps completed),
        plus the mask of laps to score (clean and after the cutoff)."""
        T = seq.T
        seen = (seq.observed[idx] > 0) & (np.arange(T)[None, :] < k[:, None])
        steps = np.arange(T)[None, :]
        # the driver's last observed lap before the cutoff
        last = np.where(seen, steps, -1).max(1)
        has = last >= 0
        cur_stint = np.where(has, seq.stint[idx, last.clip(0)], -2.0)
        cur_age = np.where(has, seq.age[idx, last.clip(0)], 0.0)
        n_obs = seen.sum(1)
        total = seq.total[idx][:, None]
        same = (seq.stint[idx] == cur_stint[:, None]).astype(np.float32)
        age = seq.age[idx]
        z = np.stack([
            (steps + 1 - k[:, None]) / total,
            same,
            age / AGE_NORM,
            same * (age - cur_age[:, None]) / AGE_NORM,
            seq.hard[idx] / HARD_NORM,
            (steps + 1) / total,
            np.broadcast_to((n_obs / T)[:, None], (len(idx), T)),
            np.broadcast_to(has[:, None].astype(np.float32), (len(idx), T)),
            seq.since[idx] / SINCE_CAP,
        ], -1).astype(np.float32)
        mask = (seq.present[idx] > 0) & (steps >= k[:, None])
        return torch.as_tensor(z), mask

    def _forward(self, module: _GRUModule, x: torch.Tensor, seq: _Sequences, idx: np.ndarray,
                 k: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        h = module.states(x)[torch.arange(len(idx)), torch.as_tensor(k)]  # (B, H)
        z, mask = self._head_inputs(seq, idx, k)
        out = module.head(torch.cat([h[:, None, :].expand(-1, seq.T, -1), z], -1))
        mu = out[..., 0]
        sigma = F.softplus(out[..., 1]) + SIGMA_FLOOR
        return mu, sigma, mask

    def _loss(self, module, seq: _Sequences, x_all: torch.Tensor, idx: np.ndarray,
              k: np.ndarray) -> torch.Tensor:
        mu, sigma, mask = self._forward(module, x_all[idx], seq, idx, k)
        y = torch.as_tensor(seq.resid[idx])
        m = torch.as_tensor(mask)
        nll = 0.5 * ((y - mu) / sigma) ** 2 + torch.log(sigma)
        return nll[m].mean()

    def _sample_cutoffs(self, seq: _Sequences, idx: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
        rep = np.repeat(idx, self.cutoffs_per_seq)
        hi = np.maximum(seq.total[rep].astype(int) - 1, 1)
        return rep, (rng.random(len(rep)) * hi).astype(int)

    @staticmethod
    def _fixed_cutoffs(seq: _Sequences, fracs=(0.1, 0.25, 0.5, 0.75)) -> tuple[np.ndarray, np.ndarray]:
        idx = np.repeat(np.arange(len(seq.keys)), len(fracs))
        k = np.round(np.tile(fracs, len(seq.keys)) * seq.total[idx]).astype(int)
        return idx, k

    def _train(self, frame: pd.DataFrame, epochs: int, val: pd.DataFrame | None = None) -> tuple[_GRUModule, int]:
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        seq = _Sequences(frame, self.T)
        x_all = seq.step_inputs()
        module = _GRUModule(self.hidden)
        opt = torch.optim.AdamW(module.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if val is not None:
            vseq = _Sequences(val, self.T)
            vx = vseq.step_inputs()
            vidx, vk = self._fixed_cutoffs(vseq)
        best, best_epoch, best_state, stale = np.inf, epochs, None, 0
        for epoch in range(1, epochs + 1):
            module.train()
            order = rng.permutation(len(seq.keys))
            for start in range(0, len(order), self.batch_size):
                idx, k = self._sample_cutoffs(seq, order[start:start + self.batch_size], rng)
                loss = self._loss(module, seq, x_all, idx, k)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(module.parameters(), 5.0)
                opt.step()
            if val is None:
                continue
            module.eval()
            with torch.no_grad():
                score = float(self._loss(module, vseq, vx, vidx, vk))
            if score < best - 1e-4:
                best, best_epoch, stale = score, epoch, 0
                best_state = {n: t.clone() for n, t in module.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is not None:
            module.load_state_dict(best_state)
        return module, best_epoch

    def fit(self, frame: pd.DataFrame) -> "InRaceGRU":
        torch.set_num_threads(self.threads)
        self.T = int(frame["total_laps"].max())
        seasons = sorted(frame["race_id"].str[:4].astype(int).unique())
        inner_train = frame[frame["race_id"].str[:4].astype(int) < seasons[-1]]
        inner_val = frame[frame["race_id"].str[:4].astype(int) == seasons[-1]]
        _, self.epochs = self._train(inner_train, self.max_epochs, inner_val)
        log.info("in_race_gru: %d epochs (early stopping on %d)", self.epochs, seasons[-1])
        self.module, _ = self._train(frame, self.epochs)
        self.module.eval()
        return self

    def predict(self, obs: pd.DataFrame, target: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Mean and σ (percent) of the residual on ``target`` laps given ``obs`` laps of the same
        race; the cutoff is the last lap before the first target lap. Target residuals, if
        present, are never read."""
        cutoff = int(target["lap_number"].min()) - 1
        rows = pd.concat([obs, target.drop(columns="resid", errors="ignore")], ignore_index=True)
        observed = np.arange(len(rows)) < len(obs)
        T = max(self.T, int(rows["lap_number"].max()))
        seq = _Sequences(rows, T, observed)
        idx = np.arange(len(seq.keys))
        k = np.full(len(idx), cutoff)
        with torch.no_grad():
            mu, sigma, _ = self._forward(self.module, seq.step_inputs(), seq, idx, k)
        tcodes, tlap = seq.codes[~observed], seq.lap_index[~observed]
        return mu[tcodes, tlap].numpy().astype(float), sigma[tcodes, tlap].numpy().astype(float)
