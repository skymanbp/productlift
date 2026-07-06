"""Train and register a model. Orchestration — Milestone 3.

The canonical flow, in the order that keeps it honest:
  seeded product split (test touched ONCE) → baseline first → GBDT with the
  M2 target encoder fit on train only → calibrate on a fold the GBDT never saw
  → compare against the baseline → register model + ModelCard.

    python -m scripts.train
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from app.config import get_params, get_settings
from app.features.encoders import SmoothedTargetEncoder
from app.models.baseline import build_baseline, fit_predict_proba
from app.models.calibrate import calibrate
from app.models.gbdt import build_gbdt, compute_scale_pos_weight, fit_gbdt
from app.models.registry import ModelCard, save_model
from evaluation.metrics import pr_auc

TARGET = "is_underperforming"
ID_COL = "product_id"
CATEGORY = "product_category_name"


def _seeded_product_split(
    n_rows: int,
    *,
    test_fraction: float,
    valid_fraction: float,
    calib_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random split over PRODUCTS (rows are product-level). Temporal leakage was
    already handled when the matrix was built — features pre-as_of, label post-as_of
    — so a seeded random split across products is legitimate here. Returns
    (train, valid, calib, test) positional indices."""
    order = np.random.default_rng(seed).permutation(n_rows)
    n_test = round(n_rows * test_fraction)
    n_valid = round(n_rows * valid_fraction)
    n_calib = round(n_rows * calib_fraction)
    test = order[:n_test]
    valid = order[n_test : n_test + n_valid]
    calib = order[n_test + n_valid : n_test + n_valid + n_calib]
    train = order[n_test + n_valid + n_calib :]
    return train, valid, calib, test


def main() -> None:
    settings = get_settings()
    params = get_params()
    matrix = pd.read_parquet(settings.processed_dir / "modeling_matrix.parquet")

    y = matrix[TARGET]
    # product_id is an identifier, not a feature: encoding it lets the model
    # memorize products (inflated offline PR-AUC) and can never generalize to a
    # product it hasn't seen — which is the whole point of the model.
    X = matrix.drop(columns=[TARGET, ID_COL])
    feature_names = list(X.columns)

    seed = int(params["seed"])
    holdout = params["model"]["holdout"]
    idx_tr, idx_va, idx_ca, idx_te = _seeded_product_split(
        len(X),
        test_fraction=holdout["test_fraction"],
        valid_fraction=holdout["valid_fraction"],
        calib_fraction=holdout["calib_fraction"],
        seed=seed,
    )
    X_tr, y_tr = X.iloc[idx_tr], y.iloc[idx_tr]
    X_va, y_va = X.iloc[idx_va], y.iloc[idx_va]
    X_ca, y_ca = X.iloc[idx_ca], y.iloc[idx_ca]
    X_te, y_te = X.iloc[idx_te], y.iloc[idx_te]
    print(
        f"split: train {len(X_tr)} | valid {len(X_va)} | calib {len(X_ca)} | "
        f"test {len(X_te)} (seed={seed})"
    )

    numeric = X.select_dtypes("number").columns.tolist()
    categorical = [CATEGORY]

    # --- Baseline first (always) ---
    baseline = build_baseline(numeric, categorical, seed=seed)
    base_valid = pr_auc(y_va, fit_predict_proba(baseline, X_tr, y_tr, X_va))
    print(f"valid PR-AUC — prevalence {y_va.mean():.3f} | baseline {base_valid:.3f}")

    # --- GBDT: numeric passthrough + M2 smoothed target encoder, fit on train only ---
    features = ColumnTransformer(
        [
            ("num", "passthrough", numeric),
            (
                "cat",
                SmoothedTargetEncoder(
                    CATEGORY, smoothing=params["features"]["target_encoding_smoothing"]
                ),
                [CATEGORY],
            ),
        ]
    )
    # pandas output end to end: the GBDT then sees the SAME feature names at fit
    # and predict time (ndarray at predict while names existed at fit triggers
    # sklearn's feature-name warnings — a real train/serve-skew smell, not noise).
    features.set_output(transform="pandas")
    X_tr_f = features.fit_transform(X_tr, y_tr)
    X_va_f = features.transform(X_va)

    spw = (
        compute_scale_pos_weight(y_tr)
        if params["model"]["handle_imbalance"] == "scale_pos_weight"
        else None
    )
    g = params["model"]["gbdt"]
    gbdt = build_gbdt(
        spw,
        seed=seed,
        n_estimators=g["n_estimators"],
        learning_rate=g["learning_rate"],
        num_leaves=g["num_leaves"],
    )
    gbdt = fit_gbdt(
        gbdt, X_tr_f, y_tr, X_va_f, y_va, early_stopping_rounds=g["early_stopping_rounds"]
    )
    gbdt_valid = pr_auc(y_va, gbdt.predict_proba(X_va_f)[:, 1])
    print(f"valid PR-AUC — gbdt {gbdt_valid:.3f} (best_iter={gbdt.best_iteration_})")

    # --- Calibrate on a fold the GBDT never trained or early-stopped on ---
    head = calibrate(gbdt, features.transform(X_ca), y_ca) if params["model"]["calibrate"] else gbdt
    # Pre-fitted steps packaged as one Pipeline: serving applies the exact same
    # transform as training (train/serve symmetry via the ModelCard contract).
    model = Pipeline([("features", features), ("clf", head)])

    # --- The test set, touched exactly once ---
    test_baseline = pr_auc(y_te, baseline.predict_proba(X_te)[:, 1])
    test_gbdt = pr_auc(y_te, model.predict_proba(X_te)[:, 1])
    print(
        f"test PR-AUC — prevalence {y_te.mean():.3f} | baseline {test_baseline:.3f} | "
        f"gbdt+calib {test_gbdt:.3f}"
    )

    card = ModelCard(
        name="productlift_gbdt_calibrated",
        params={"model": params["model"], "tuning": params["tuning"], "seed": seed},
        metrics={
            "valid_pr_auc_baseline": float(base_valid),
            "valid_pr_auc_gbdt": float(gbdt_valid),
            "test_pr_auc_baseline": float(test_baseline),
            "test_pr_auc_gbdt_calibrated": float(test_gbdt),
            "test_prevalence": float(y_te.mean()),
        },
        feature_names=feature_names,
        data_window={"train_end": params["split"]["train_end"]},
    )
    save_model(model, card, settings.model_path)
    print(f"Saved model + card -> {settings.model_path}")


if __name__ == "__main__":
    main()
