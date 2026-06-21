"""
models.py — Model definitions, depth-3 tree figure, walk-forward backtest.

Model lineup (CLAUDE.md §6):
  Baseline 1  : D-1 same-hour naïve          (price_lag_24h)
  Baseline 2  : D-7 same-hour naïve           (price_lag_168h)  ← honest DA benchmark
  Selected    : Ridge with merit-order terms   (transparent, interpretable)
  Challenger  : LightGBM                       (nonlinearity check / driver validator)
  Figure only : DecisionTreeRegressor depth=3  (visualises merit-order kink — NOT selected)

Selection rule (CLAUDE.md §6):
  Default to Ridge unless LightGBM beats it by a margin that justifies losing
  interpretability. The decision and reasoning are stated explicitly in the report.

Metrics: MAE and RMSE (EUR/MWh). No MAPE — prices cross zero (undefined/explosive).
"""

import os
from typing import Optional

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, plot_tree

from config import OOS_START, RANDOM_SEED, TEST_START

_ROOT      = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
FIGURES_DIR = os.path.join(_ROOT, "figures")
OUTPUTS_DIR = os.path.join(_ROOT, "outputs")

# ---------------------------------------------------------------------------
# Feature column sets per model
# ---------------------------------------------------------------------------

# Ridge: cyclic calendar encodings (not raw ordinals — linear model needs
# these to represent the non-monotonic hour/week/month effects without
# imposing a linear trend across their integer values).
RIDGE_FEATURES = [
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
    "month_sin","month_cos",
    "is_weekend", "is_holiday",
    "load_forecast_mw", "wind_forecast_mw", "solar_forecast_mw",
    "residual_load_mw", "residual_load_sq", "residual_load_high",
    "price_lag_24h", "price_lag_168h",
    "rolling_resid_mean_7d", "rolling_resid_std_7d",
    "ttf_eur_mwh",
    "ttf_resid_interaction",  # A3 (optional): lets Ridge re-slope with gas
]

# LightGBM + tree: raw ordinal calendar is fine for tree-based models;
# cyclic encodings are redundant and excluded.
LGBM_FEATURES = [
    "hour", "dow", "month",
    "is_weekend", "is_holiday",
    "load_forecast_mw", "wind_forecast_mw", "solar_forecast_mw",
    "residual_load_mw", "residual_load_sq", "residual_load_high",
    "price_lag_24h", "price_lag_168h",
    "rolling_resid_mean_7d", "rolling_resid_std_7d",
    "ttf_eur_mwh",
]


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def make_ridge(alpha: float = 10.0) -> Pipeline:
    """
    Ridge regression with StandardScaler.
    Alpha=10 — light L2 regularisation; chosen via 5-fold time-series CV on
    first 3 years of data (alphas tested: 0.1, 1, 10, 100, 1000).
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=alpha)),
    ])


def make_lgbm() -> lgb.LGBMRegressor:
    """
    LightGBM challenger — used as a nonlinearity check and driver validator,
    NOT as the selected model unless MAE improvement justifies the opacity.
    n_jobs=1 + random_state for full reproducibility.
    """
    return lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        n_jobs=1,
        verbose=-1,
    )


def make_tree(max_depth: int = 3) -> DecisionTreeRegressor:
    """
    Depth-3 decision tree — interpretability figure ONLY, not a selected model.
    High bias (blocky step function) but splits literally show the merit-order kink.
    """
    return DecisionTreeRegressor(max_depth=max_depth, random_state=RANDOM_SEED)


# ---------------------------------------------------------------------------
# Baseline predictors (no fitting — direct lookup from feature columns)
# ---------------------------------------------------------------------------

def predict_naive_24h(X: pd.DataFrame) -> pd.Series:
    return X["price_lag_24h"].rename("y_pred")


def predict_naive_168h(X: pd.DataFrame) -> pd.Series:
    return X["price_lag_168h"].rename("y_pred")


# ---------------------------------------------------------------------------
# Depth-3 tree figure  (CLAUDE.md §6 — "render as a figure, don't select it")
# ---------------------------------------------------------------------------

def plot_merit_order_tree(X: pd.DataFrame, y: pd.Series) -> str:
    """
    Fit a depth-3 DecisionTreeRegressor on the full pre-OOS training set,
    then save two figures:
      figures/merit_order_tree.png    — sklearn tree diagram
      figures/price_vs_resid_tree.png — scatter with tree decision boundaries

    Returns the path to the tree diagram.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Pre-OOS training data only
    mask = X.index < OOS_START
    X_train = X.loc[mask, LGBM_FEATURES]
    y_train = y.loc[mask]

    tree = make_tree(max_depth=3)
    tree.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # Figure 1: sklearn tree diagram
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(22, 9))
    plot_tree(
        tree,
        feature_names=LGBM_FEATURES,
        ax=ax,
        filled=True,
        rounded=True,
        fontsize=8,
        precision=1,
        impurity=False,
    )
    ax.set_title(
        "Depth-3 Decision Tree — DE-LU Day-Ahead Price (EUR/MWh)\n"
        "Trained on full pre-OOS history.  For visualisation only — not the selected model.",
        fontsize=11,
    )
    tree_path = os.path.join(FIGURES_DIR, "merit_order_tree.png")
    fig.savefig(tree_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {tree_path}")

    # ------------------------------------------------------------------
    # Figure 2: price vs residual load — scatter + tree decision boundaries
    # Shows the kinked supply stack that the tree (and Ridge) are fitting.
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sample 8 000 points for readability
    rng = np.random.default_rng(RANDOM_SEED)
    idx_sample = rng.choice(len(X_train), size=min(8_000, len(X_train)), replace=False)
    x_plot = X_train["residual_load_mw"].iloc[idx_sample].values
    y_plot = y_train.iloc[idx_sample].values

    ax.scatter(x_plot, y_plot, alpha=0.15, s=6, color="steelblue", label="Observed")

    # Overlay tree prediction across residual load range
    resid_grid = np.linspace(X_train["residual_load_mw"].min(),
                              X_train["residual_load_mw"].max(), 400)
    # Build a synthetic feature row: median values for all other features
    median_row = X_train.median()
    grid_df = pd.DataFrame(
        np.tile(median_row.values, (400, 1)), columns=LGBM_FEATURES
    )
    grid_df["residual_load_mw"] = resid_grid
    grid_df["residual_load_sq"] = resid_grid ** 2 / 1e6
    from config import RESID_HIGH_MW
    grid_df["residual_load_high"] = np.maximum(resid_grid - RESID_HIGH_MW, 0)

    y_tree_pred = tree.predict(grid_df)
    ax.plot(resid_grid, y_tree_pred, color="firebrick", lw=2.5,
            label="Depth-3 tree (step function)")

    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Residual Load (MW)  =  Load − Wind − Solar", fontsize=11)
    ax.set_ylabel("Day-Ahead Price (EUR/MWh)", fontsize=11)
    ax.set_title("Merit-Order Kink: Price vs Residual Load\n"
                 "Tree step function highlights the scarcity / negative-price regimes",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    scatter_path = os.path.join(FIGURES_DIR, "price_vs_resid_tree.png")
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {scatter_path}")

    # Print the tree splits for the report
    print("\nDepth-3 tree splits (text):")
    from sklearn.tree import export_text
    print(export_text(tree, feature_names=LGBM_FEATURES, decimals=1))

    return tree_path


# ---------------------------------------------------------------------------
# LightGBM feature-importance figure (REVISION_PLAN.md B1)
# Substantiates (or corrects) the "residual load + price lags dominate" claim
# made throughout the report — this is the one chart that backs it.
# ---------------------------------------------------------------------------

def plot_feature_importance_lgbm(X: pd.DataFrame, y: pd.Series) -> str:
    """
    Fit LightGBM on the pre-Test training set (all rows before TEST_START =
    2025-01-01), plot gain-based feature importances as a sorted bar chart.

    Returns the path to the saved figure.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)

    mask = X.index < TEST_START
    X_train = X.loc[mask, LGBM_FEATURES]
    y_train = y.loc[mask]

    model = make_lgbm()
    model.fit(X_train, y_train)

    importances = model.booster_.feature_importance(importance_type="gain")
    order = np.argsort(importances)
    feat_sorted = [LGBM_FEATURES[i] for i in order]
    imp_sorted  = importances[order]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#d6604d" if f in ("price_lag_24h", "price_lag_168h") else "#2166ac"
              for f in feat_sorted]
    ax.barh(feat_sorted, imp_sorted, color=colors, alpha=0.85)
    ax.set_xlabel("Gain-based importance (LightGBM)", fontsize=11)
    ax.set_title(
        "LightGBM Feature Importance — DE-LU Day-Ahead Price\n"
        "Trained on full pre-Test history (2019-2024). Red = price lags, blue = fundamentals.",
        fontsize=11,
    )
    ax.grid(axis="x", alpha=0.3)
    path = os.path.join(FIGURES_DIR, "feature_importance_lgbm.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")

    print("\nLightGBM feature importance (gain, sorted):")
    for f, imp in zip(reversed(feat_sorted), reversed(list(imp_sorted))):
        print(f"  {f:<24s} {imp:>10.0f}")

    return path


# ---------------------------------------------------------------------------
# Metrics helpers (used by step 6 validation and step 7 OOS)
# ---------------------------------------------------------------------------

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def skill_score(mae_model: float, mae_baseline: float) -> float:
    """Skill score vs baseline: 1 − MAE_model / MAE_baseline. Higher = better."""
    return float(1 - mae_model / mae_baseline)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from features import build_features

    print("Building features …")
    X, y = build_features()

    print("\nFitting depth-3 tree and saving figures …")
    plot_merit_order_tree(X, y)

    print("\nModel lineup summary:")
    print(f"  Ridge features  : {len(RIDGE_FEATURES)}")
    print(f"  LightGBM features: {len(LGBM_FEATURES)}")
    print(f"  Baselines       : lag_24h, lag_168h (no fitting)")
