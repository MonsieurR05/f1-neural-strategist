"""Non-neural lap-time baselines.

All three predict the normalised target ``ratio = lap_time / q_s`` (see ``features.prepare``)
and return lap time in seconds. Uncertainty is added model-agnostically by the evaluation
harness, so every model is scored with the same calibration procedure.

1. ``CircuitPace``: race pace = qualifying pace × a per-circuit ratio. No tyres, fuel or
   traffic: the "knows nothing about strategy" floor.
2. ``FuelDegLinear``: the standard analyst model, fuel-corrected linear degradation per
   compound, fitted per circuit and shrunk towards a pooled fit so circuits with little
   data (or a missing compound) borrow strength.
3. ``LightGBMLapTime``: gradient-boosted trees on every input feature.
"""

from typing import Protocol

import lightgbm as lgb
import numpy as np
import pandas as pd

from strategist.models.features import CATEGORICAL_FEATURES, COMPOUNDS, NUMERIC_FEATURES


class LapTimeModel(Protocol):
    name: str

    def fit(self, train: pd.DataFrame) -> "LapTimeModel": ...

    def predict(self, df: pd.DataFrame) -> pd.Series: ...


class CircuitPace:
    name = "circuit_pace"

    def fit(self, train: pd.DataFrame) -> "CircuitPace":
        self.ratio_by_circuit = train.groupby("circuit")["ratio"].median()
        self.ratio_global = float(train["ratio"].median())
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        ratio = df["circuit"].map(self.ratio_by_circuit).fillna(self.ratio_global)
        return ratio * df["q_s"]


class FuelDegLinear:
    """ratio = b0 + b_fuel·laps_remaining + Σ_c 1[c]·(a_c + g_c·tyre_life), per circuit.

    Per-circuit coefficients minimise ‖y − Xb‖² + ‖Λ^½ (b − b_pooled)‖², a ridge pull towards
    the pooled fit (intercept free). Λ is expressed in pseudo-laps: ``prior_laps`` laps' worth
    of evidence for the pooled value of each coefficient.
    """

    name = "fuel_deg_linear"
    TERMS = ["intercept", "laps_remaining", "is_MEDIUM", "is_HARD",
             "age_SOFT", "age_MEDIUM", "age_HARD"]

    def __init__(self, prior_laps: float = 300.0):
        self.prior_laps = prior_laps

    @staticmethod
    def design(df: pd.DataFrame) -> np.ndarray:
        cols = [np.ones(len(df)), df["laps_remaining"].to_numpy(float)]
        is_c = {c: (df["compound"] == c).to_numpy(float) for c in COMPOUNDS}
        cols += [is_c["MEDIUM"], is_c["HARD"]]
        cols += [is_c[c] * df["tyre_life"].to_numpy(float) for c in COMPOUNDS]
        return np.column_stack(cols)

    def fit(self, train: pd.DataFrame) -> "FuelDegLinear":
        # pooled: one intercept per circuit, shared slopes and compound offsets
        X = self.design(train)
        circuits = sorted(train["circuit"].unique())
        fe = pd.get_dummies(pd.Categorical(train["circuit"], categories=circuits), dtype=float)
        pooled, *_ = np.linalg.lstsq(np.column_stack([fe.to_numpy(), X[:, 1:]]),
                                     train["ratio"].to_numpy(), rcond=None)
        intercepts = pooled[: len(circuits)]
        self.pooled = np.concatenate([[np.median(intercepts)], pooled[len(circuits):]])

        penalty = self.prior_laps * np.mean(X**2, axis=0)
        penalty[0] = 0.0
        self.coef = {}
        for circuit, g in train.groupby("circuit"):
            Xc, yc = self.design(g), g["ratio"].to_numpy()
            prior = self.pooled.copy()
            prior[0] = intercepts[circuits.index(circuit)]
            A = Xc.T @ Xc + np.diag(penalty)
            A[0, 0] += 1e-9
            self.coef[circuit] = np.linalg.solve(A, Xc.T @ yc + penalty * prior)
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        X = self.design(df)
        B = np.vstack([self.coef.get(c, self.pooled) for c in df["circuit"]])
        return pd.Series(np.einsum("ij,ij->i", X, B), index=df.index) * df["q_s"]

    def coefficients(self) -> pd.DataFrame:
        rows = {"(pooled)": self.pooled, **self.coef}
        return pd.DataFrame(rows, index=self.TERMS).T


class LightGBMLapTime:
    """Early stopping uses the last training season as an inner validation set, then the model
    is refitted on all training seasons with that number of rounds. The evaluation season is
    never seen during fitting or model selection.

    With ``circuit_offset`` the boosting starts from each circuit's median training ratio, so
    trees model deviations rather than re-learning circuit levels (unseen circuits start from
    the global median). Settings were chosen on an inner split (fit 2021–22, score 2023):
    see docs/reports.
    """

    name = "lightgbm"

    def __init__(self, params: dict | None = None, max_rounds: int = 5000,
                 early_stopping_rounds: int = 200, seed: int = 42,
                 drop_features: list[str] | None = None, circuit_offset: bool = False):
        drop = set(drop_features or [])
        self.FEATURES = [f for f in NUMERIC_FEATURES + CATEGORICAL_FEATURES if f not in drop]
        self.circuit_offset = circuit_offset
        self.params = {
            "objective": "regression_l2", "learning_rate": 0.03, "num_leaves": 31,
            "min_data_in_leaf": 100, "feature_fraction": 0.8, "bagging_fraction": 0.8,
            "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1, "seed": seed,
            "deterministic": True, "force_col_wise": True, "num_threads": 4,
            **(params or {}),
        }
        self.max_rounds = max_rounds
        self.early_stopping_rounds = early_stopping_rounds

    def _init_score(self, df: pd.DataFrame) -> np.ndarray | None:
        if not self.circuit_offset:
            return None
        return df["circuit"].map(self.ratio_by_circuit).fillna(self.ratio_global).to_numpy()

    def _dataset(self, df: pd.DataFrame, reference=None) -> lgb.Dataset:
        cats = [c for c in CATEGORICAL_FEATURES if c in self.FEATURES]
        return lgb.Dataset(self._matrix(df), label=df["ratio"], reference=reference,
                           init_score=self._init_score(df), categorical_feature=cats,
                           free_raw_data=False)

    def _matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.FEATURES].copy()
        for col in CATEGORICAL_FEATURES:
            if col in X:
                X[col] = pd.Categorical(X[col], categories=self.categories[col])
        return X

    def fit(self, train: pd.DataFrame) -> "LightGBMLapTime":
        self.categories = {c: sorted(train[c].unique()) for c in CATEGORICAL_FEATURES}
        self.ratio_by_circuit = train.groupby("circuit")["ratio"].median()
        self.ratio_global = float(train["ratio"].median())
        last = train["year"].max()
        inner_train, inner_val = train[train["year"] < last], train[train["year"] == last]
        if len(inner_train) and len(inner_val):
            dtrain = self._dataset(inner_train)
            booster = lgb.train(self.params, dtrain, self.max_rounds,
                                valid_sets=[self._dataset(inner_val, reference=dtrain)],
                                callbacks=[lgb.early_stopping(self.early_stopping_rounds,
                                                              verbose=False)])
            self.rounds = booster.best_iteration
        else:
            self.rounds = 500
        self.booster = lgb.train(self.params, self._dataset(train), self.rounds)
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        ratio = self.booster.predict(self._matrix(df))
        offset = self._init_score(df)
        if offset is not None:
            ratio = ratio + offset
        return pd.Series(ratio, index=df.index) * df["q_s"]


def make_model(name: str, params: dict | None = None, seed: int = 42) -> LapTimeModel:
    params = dict(params or {})
    if name == CircuitPace.name:
        return CircuitPace()
    if name == FuelDegLinear.name:
        return FuelDegLinear(**params)
    if name == LightGBMLapTime.name:
        max_rounds = params.pop("max_rounds", 5000)
        stop = params.pop("early_stopping_rounds", 200)
        drop = params.pop("drop_features", None)
        offset = params.pop("circuit_offset", False)
        return LightGBMLapTime(params, max_rounds, stop, seed=seed, drop_features=drop,
                               circuit_offset=offset)
    if name == "lap_net":
        from strategist.models.lapnet import LapTimeNet
        return LapTimeNet(**params, seed=seed)
    raise ValueError(f"unknown model {name!r}")
