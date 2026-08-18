"""
selection_benchmark.py — Does multi-criteria model selection actually beat
single-metric selection?

The system's central claim is that ranking candidate models by the composite
score S (services/model_runner.py) produces better model choices than ranking
by RMSE alone. That claim is currently asserted, not measured. This module
measures it.

Method
------
For every series, and for every selection policy:

  1. Split the series into a SELECTION region and a HELD-OUT TEST window.
     Models only ever see the selection region.
  2. Run the full model registry on the selection region. Each model returns
     its cross-validated metrics (computed inside the selection region) and a
     forecast covering exactly the test window.
  3. Each policy picks one model using ONLY the cross-validated metrics —
     never the test window.
  4. Score the picked model's forecast against the test window.

Step 1 is the methodological fix this benchmark exists to provide. Selecting
the best of ~19 models on cross-validated error and then REPORTING that same
cross-validated error is upward-biased: the minimum of many noisy estimates is
optimistic (the "winner's curse"). When the spread between models is small —
as it is on low-signal retail series — that bias can exceed the differences
being interpreted. A test window that never touches selection removes it.

Metric
------
Realized error is reported as MASE (mean absolute scaled error), scaled by the
in-sample seasonal-naive MAE. RMSE cannot be averaged across series of
different scales; MASE can, which is why the M-competitions use scaled errors.
MASE < 1 means the forecast beat a seasonal-naive baseline.

Policies compared
-----------------
  composite       the system's score S                     (the claim)
  rmse_only       lowest cross-validated RMSE              (the alternative)
  mae_only        lowest cross-validated MAE
  always_<Model>  a fixed model, never adaptive            (does selection help at all?)
  naive           last value carried forward               (reference)
  seasonal_naive  last season carried forward              (the MASE denominator)
  oracle          lowest REALIZED test error               (unachievable lower bound)

`oracle` is not a policy anyone can run — it is the regret floor. The gap
between a policy and the oracle is how much accuracy the selection rule leaves
on the table, and it is the number the paper should report.

Statistics
----------
Across series, policies are compared with a Friedman test (do the policies
differ at all?) followed by pairwise Wilcoxon signed-rank tests with Holm
correction — the standard Demsar protocol for comparing methods over multiple
datasets. Diebold-Mariano is provided for per-series pairwise comparison,
where it is the appropriate test.

Usage
-----
    # Against ingested M5 series:
    python -m benchmarks.selection_benchmark --data-dir data/m5 \
        --date-col date --target-col sales --horizon 28 --out results.csv

    # Restrict to a quick subset while iterating:
    python -m benchmarks.selection_benchmark --data-dir data/m5 \
        --max-files 20 --skip-dl --out results.csv

    # Weight sensitivity sweep (no model retraining — reuses cached runs):
    python -m benchmarks.selection_benchmark --data-dir data/m5 \
        --weight-sweep --out results.csv
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# The system's shipped weights (services/model_runner.py). Kept here as the
# sweep's centre point rather than imported, so the sweep still runs if the
# production weights are changed.
DEFAULT_WEIGHTS: dict[str, float] = {
    "rmse": 0.40,
    "mae": 0.20,
    "mape": 0.20,
    "stability": 0.10,
    "speed": 0.10,
    "r2": -0.10,
}


# ── Scale-free error metrics ─────────────────────────────────────────────────

def seasonal_naive_mae(train: np.ndarray, season: int) -> float:
    """
    In-sample MAE of a seasonal-naive forecast — the MASE denominator.

    Falls back to a lag-1 naive denominator when the series is too short for
    the requested season, and to a small positive constant for a constant
    series (where any denominator would otherwise be zero).
    """
    if len(train) > season:
        diffs = np.abs(train[season:] - train[:-season])
    elif len(train) > 1:
        diffs = np.abs(np.diff(train))
    else:
        return 1e-6
    denom = float(np.mean(diffs)) if diffs.size else 0.0
    return denom if denom > 1e-9 else 1e-6


def mase(actual: np.ndarray, predicted: np.ndarray, denom: float) -> float:
    """Mean absolute scaled error against a precomputed in-sample denominator."""
    if len(actual) == 0 or len(predicted) == 0:
        return float("inf")
    n = min(len(actual), len(predicted))
    return float(np.mean(np.abs(actual[:n] - predicted[:n])) / denom)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    if len(actual) == 0 or len(predicted) == 0:
        return float("inf")
    n = min(len(actual), len(predicted))
    return float(np.sqrt(np.mean((actual[:n] - predicted[:n]) ** 2)))


# ── Composite score recomputation (mirrors model_runner._robust_scale) ───────

def robust_scale(values: Iterable[float]) -> float:
    """
    Tukey-fence normalisation denominator (Q3 + 1.5*IQR).

    Reimplemented here — rather than imported — so the weight sweep can rescore
    cached runs without re-running any models. Deliberately mirrors
    services/model_runner.py::_robust_scale, including its fallback behaviour,
    so swept scores are directly comparable to production scores.
    """
    finite = [v for v in values if v is not None and np.isfinite(v)]
    if not finite:
        return 1e-6
    arr = np.asarray(finite, dtype=float)
    q1, q3 = np.percentile(arr, [25, 75])
    fence = q3 + 1.5 * (q3 - q1)
    if fence > 1e-6:
        return float(fence)
    return float(arr.max()) or 1e-6


# Normalization schemes.
#
# "raw" reproduces the shipped scoring in services/model_runner.py exactly.
# Measured on 70 real M5 level-9 series, it has two structural defects:
#
#   1. The Tukey fence bounds the DENOMINATOR, not the RATIO. Training time
#      spans six orders of magnitude across the registry (KNN ~0.0001s vs
#      SARIMA ~118s), so a slow model's "normalized" speed reaches 43.6 while
#      normalized RMSE — spanning a factor of ~1.2 — compresses into 0.48-1.03.
#      A 79x range mismatch. Result: speed accounts for 88.3% of the score's
#      discriminating power at a nominal weight of 0.10, while RMSE accounts
#      for 3.5% at a nominal weight of 0.40. The stated 4:1 preference for
#      accuracy operates as roughly 1:25 against it.
#
#   2. r2 is the sole term never divided by a fence at all — five normalized
#      quantities summed with one raw one. It therefore carries influence
#      comparable to RMSE (3.5%) on a quarter of the weight, and takes over as
#      the dominant driver whenever the speed weight is reduced.
#
# "clipped" caps every fence-normalized term at 1.0 and puts r2 on the same
# footing. The cap needs no tuning — the Tukey fence already DEFINES the
# outlier boundary (Q3 + 1.5*IQR), so a value beyond it is an outlier by
# construction and "worst in class" is the honest reading. Same robust-
# statistics reasoning preprocessor.py already applies to input outliers.
#
# "balanced" adds the second half of the fix. Capping bounds the runaway but
# does NOT equalize influence, because the terms still OCCUPY different amounts
# of [0, 1]: after capping, speed spans the full range (slow models pinned at
# 1.0, fast ones near 0) while RMSE — models being genuinely close on accuracy
# — occupies only ~0.66-0.83. Speed therefore still out-influences RMSE at a
# quarter of the weight. "balanced" min-max rescales each capped term across
# the candidate set so every criterion spans exactly [0, 1], making effective
# influence equal to nominal weight by construction. Order matters: cap FIRST
# (so a single catastrophic model cannot set the scale, the original reason the
# Tukey fence exists), then rescale.
#
# Keeping all three is deliberate — they decompose the fix into two independent
# effects, so the paper can report how much each contributes.
NORMALIZATION_SCHEMES = ("raw", "clipped", "balanced")
_CAPPED_SCHEMES = ("clipped", "balanced")


def _normalized_terms(r: dict, fences: dict, scheme: str) -> dict[str, float]:
    """
    Per-term normalized values for one model. Single source of truth for both
    scoring and the decomposition diagnostic, so they can never disagree.
    """
    speed = (r["train_time"] / fences["train"] + r["pred_time"] / fences["pred"]) / 2
    terms = {
        "rmse": r["cv_rmse"] / fences["rmse"],
        "mae": r["cv_mae"] / fences["mae"],
        "mape": r["cv_mape"] / fences["mape"],
        "stability": r["stability"] / fences["stab"],
        "speed": speed,
        # Raw R2 under the shipped scheme; fence-normalized like everything
        # else under "clipped". Shifted by the observed minimum first because
        # R2 is signed (negative R2 is common and valid on noisy retail data)
        # and dividing a signed quantity by a positive fence would flip the
        # direction of the penalty for negative values.
        "r2": r["cv_r2"] if scheme == "raw"
              else (r["cv_r2"] - fences["r2_min"]) / fences["r2"],
    }
    if scheme in _CAPPED_SCHEMES:
        terms = {k: min(v, 1.0) for k, v in terms.items()}
    return terms


def _all_normalized_terms(usable: Sequence[dict], scheme: str) -> dict[str, dict[str, float]]:
    """
    {model: {term: normalized_value}} for the whole candidate set.

    Cross-model work lives here rather than in _normalized_terms because the
    "balanced" rescale needs each term's range across ALL candidates, which no
    per-model function can know.
    """
    fences = _compute_fences(usable, scheme)
    per_model = {r["model"]: _normalized_terms(r, fences, scheme) for r in usable}

    if scheme == "balanced":
        for term in DEFAULT_WEIGHTS:
            values = [m[term] for m in per_model.values()]
            lo, hi = min(values), max(values)
            span = hi - lo
            for model_terms in per_model.values():
                # A term identical across every candidate carries no information
                # about which model to pick; flatten it to 0 rather than let a
                # near-zero span amplify floating-point noise into a ranking.
                model_terms[term] = ((model_terms[term] - lo) / span
                                     if span > 1e-12 else 0.0)
    return per_model


def _compute_fences(usable: Sequence[dict], scheme: str) -> dict:
    fences = {
        "rmse": robust_scale(r["cv_rmse"] for r in usable),
        "mae": robust_scale(r["cv_mae"] for r in usable),
        "mape": robust_scale(r["cv_mape"] for r in usable),
        "train": robust_scale(r["train_time"] for r in usable),
        "pred": robust_scale(r["pred_time"] for r in usable),
        "stab": robust_scale(r["stability"] for r in usable),
        "r2_min": 0.0,
        "r2": 1.0,
    }
    if scheme in _CAPPED_SCHEMES:
        r2_values = [r["cv_r2"] for r in usable if np.isfinite(r["cv_r2"])]
        r2_min = min(r2_values) if r2_values else 0.0
        fences["r2_min"] = r2_min
        # Fence over the shifted (non-negative) R2 values, matching how every
        # other term is scaled; guarded so an all-equal R2 set cannot divide by 0.
        fences["r2"] = robust_scale(v - r2_min for v in r2_values) if r2_values else 1.0
    return fences


def recompute_composite(
    records: Sequence[dict],
    weights: dict[str, float] | None = None,
    scheme: str = "raw",
) -> dict[str, float]:
    """
    Recompute the composite score for every successful model under arbitrary
    weights, from the metrics already returned by a completed run.

    Args:
        scheme : "raw" reproduces the shipped model_runner.py scoring exactly.
                 "clipped" caps every fence-normalized term at 1.0 and puts r2
                 on the same footing — see NORMALIZATION_SCHEMES above.

    Returns {model_name: score}. Models missing a usable forecast are omitted
    rather than assigned infinity, so callers can distinguish "scored badly"
    from "never produced a candidate".
    """
    if scheme not in NORMALIZATION_SCHEMES:
        raise ValueError(f"scheme must be one of {NORMALIZATION_SCHEMES}, got {scheme!r}")

    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    usable = [r for r in records if r.get("ok")]
    if not usable:
        return {}

    return {
        model: float(sum(w[t] * v for t, v in terms.items()))
        for model, terms in _all_normalized_terms(usable, scheme).items()
    }


# ── Term-dominance diagnostic ────────────────────────────────────────────────
#
# The weight sweep alone cannot explain composite's failure: zeroing speed and
# stability together recovers only ~25% of the gap to rmse_only, and the a
# priori suspects (mape, r2) turned out to move the WRONG direction when swept.
# So the failure is not "one weight is set wrong" — it must be structural.
#
# Hypothesis: the six terms are normalized to comparable DENOMINATORS (each
# divided by its own Tukey fence), but that does not give them comparable
# RANGES. RMSE across 19 models on one series typically spans a factor of
# ~1.2, so after normalization the accuracy terms cluster tightly around 1.0.
# Training time spans microseconds (KNN) to ~120s (SARIMA) — six orders of
# magnitude — and the fence caps the denominator, not the ratio, so a slow
# model's normalized speed can run to 20+ while a fast model's is ~0. A term's
# WEIGHT sets its nominal influence; its NORMALIZED SPREAD across candidates
# sets its actual influence on which model wins. These need not agree, and the
# composite's 0.40/0.10 split on paper may not be a 4:1 split in practice.
#
# This is answerable directly from persisted records, at zero retraining cost.

def decompose_composite_terms(
    raw: Sequence[SeriesResult],
    weights: dict[str, float] | None = None,
    scheme: str = "raw",
) -> pd.DataFrame:
    """
    Long-format per-(series, model, term) breakdown of the composite score,
    using the IDENTICAL normalization recompute_composite() uses — same Tukey
    fences, same formula, just not summed — so this never drifts out of sync
    with the actual scoring path.

    Columns: series_id, model, term, raw_value, normalized_value, weight,
    weighted_contribution (= normalized_value * weight; these six sum to the
    model's composite score for that series).
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    raw_key = {"rmse": "cv_rmse", "mae": "cv_mae", "mape": "cv_mape",
               "stability": "stability", "r2": "cv_r2"}   # speed has two raw inputs

    rows: list[dict] = []
    for result in raw:
        usable = [r for r in result.records if r.get("ok")]
        if not usable:
            continue

        by_model = _all_normalized_terms(usable, scheme)
        for r in usable:
            for term, norm_val in by_model[r["model"]].items():
                rows.append({
                    "series_id": result.series_id,
                    "model": r["model"],
                    "term": term,
                    "raw_value": r.get(raw_key.get(term, "train_time")),
                    "normalized_value": float(norm_val),
                    "weight": w[term],
                    "weighted_contribution": float(norm_val * w[term]),
                })
    return pd.DataFrame(rows)


def summarize_term_dominance(decomposed: pd.DataFrame) -> pd.DataFrame:
    """
    Per term: how much its WEIGHTED CONTRIBUTION varies across candidate
    models within a series, averaged over series. A term with a near-zero
    spread cannot change which model wins no matter its weight (every
    candidate gets penalized about equally); a term with a large spread can
    decide the ranking even with a small nominal weight. Sorted so the term
    actually driving model selection is first, regardless of what its weight
    on paper suggests.
    """
    per_series = (
        decomposed.groupby(["series_id", "term"])
        .agg(
            weight=("weight", "first"),
            normalized_spread=("normalized_value", lambda s: float(s.max() - s.min())),
            contribution_spread=("weighted_contribution", lambda s: float(s.max() - s.min())),
        )
        .reset_index()
    )
    summary = (
        per_series.groupby("term")
        .agg(
            weight=("weight", "first"),
            mean_normalized_spread=("normalized_spread", "mean"),
            mean_contribution_spread=("contribution_spread", "mean"),
        )
        .reset_index()
        .sort_values("mean_contribution_spread", ascending=False)
        .reset_index(drop=True)
    )
    # How much of the TOTAL score spread (sum across terms) each term accounts
    # for — the "effective weight" implied by the data, comparable directly to
    # the nominal `weight` column.
    total = summary["mean_contribution_spread"].sum()
    summary["effective_share"] = (
        summary["mean_contribution_spread"] / total if total > 1e-12 else np.nan
    )
    return summary


def explain_pick_vs_alternative(
    raw: Sequence[SeriesResult],
    alternative_model: str,
    weights: dict[str, float] | None = None,
    scheme: str = "raw",
) -> pd.DataFrame:
    """
    For every series where composite's actual pick differs from
    `alternative_model` (and the alternative was a usable candidate), report
    each term's weighted-contribution GAP between them: alternative's
    contribution minus the pick's. A large positive gap on a term means that
    term is why the alternative lost — it scored much worse than the pick on
    that specific term, enough to outweigh the others.

    This directly answers "why does composite never pick SARIMA, even when it
    has the lowest RMSE" — it isolates exactly which term(s) SARIMA loses on.
    Pass scheme="balanced" to ask the same question AFTER the scale-mismatch
    fix — useful when that fix closes only part of the gap and a different
    term (e.g. stability) turns out to be the real remaining driver.
    """
    decomposed = decompose_composite_terms(raw, weights, scheme)
    rows: list[dict] = []

    for result in raw:
        scores = recompute_composite(result.records, weights, scheme)
        if not scores or alternative_model not in scores:
            continue
        pick = min(scores, key=lambda m: scores[m])
        if pick == alternative_model:
            continue

        sub = decomposed[decomposed["series_id"] == result.series_id]
        pick_row = sub[sub["model"] == pick].set_index("term")["weighted_contribution"]
        alt_row = sub[sub["model"] == alternative_model].set_index("term")["weighted_contribution"]

        for term in DEFAULT_WEIGHTS:
            rows.append({
                "series_id": result.series_id,
                "picked_model": pick,
                "alternative_model": alternative_model,
                "term": term,
                "pick_contribution": float(pick_row.get(term, np.nan)),
                "alt_contribution": float(alt_row.get(term, np.nan)),
                "gap": float(alt_row.get(term, np.nan) - pick_row.get(term, np.nan)),
            })

    return pd.DataFrame(rows)


def compare_normalization_schemes(
    raw: Sequence[SeriesResult],
    weights: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rescore every series under each normalization scheme, holding weights,
    models, forecasts and held-out data fixed. The ONLY thing that varies is
    how the six criteria are scaled — so any difference in realized MASE is
    attributable to the scaling alone.

    Returns (per_series, summary). The comparison is paired by construction
    (same series, same candidate pool), so the Wilcoxon test in `summary` is
    the appropriate significance check.

    The clip threshold is fixed at 1.0 on the a priori Tukey argument, NOT
    selected by trying thresholds on this data — choosing it by best observed
    MASE here would be exactly the winner's curse this benchmark exists to
    eliminate.
    """
    rows: list[dict] = []
    for result in raw:
        usable = {r["model"]: r for r in result.records if r.get("ok")}
        if not usable:
            continue
        entry = {"series_id": result.series_id}
        for scheme in NORMALIZATION_SCHEMES:
            scores = recompute_composite(result.records, weights, scheme)
            if not scores:
                continue
            pick = min(scores, key=lambda m: scores[m])
            entry[f"{scheme}_model"] = pick
            entry[f"{scheme}_mase"] = usable[pick]["test_mase"]
        # Best achievable pick, for context on how much of the gap is closed.
        entry["oracle_mase"] = min(r["test_mase"] for r in usable.values())
        rows.append(entry)

    per_series = pd.DataFrame(rows)
    if per_series.empty:
        return per_series, pd.DataFrame()

    summary_rows = []
    for scheme in NORMALIZATION_SCHEMES:
        col = f"{scheme}_mase"
        if col not in per_series:
            continue
        vals = per_series[col].replace([np.inf, -np.inf], np.nan).dropna()
        summary_rows.append({
            "scheme": scheme,
            "mean_mase": float(vals.mean()),
            "median_mase": float(vals.median()),
            "pct_beats_naive": float((vals < 1.0).mean() * 100),
            "mean_regret_vs_oracle": float(
                (per_series[col] - per_series["oracle_mase"]).mean()),
            "n_series": int(len(vals)),
        })
    summary = pd.DataFrame(summary_rows)

    # Paired comparison of each alternative scheme against the shipped one.
    # Always reported, including when a scheme changes nothing — "0 picks
    # changed" is a real result, not an absence of one.
    from scipy import stats

    comparisons = []
    for scheme in NORMALIZATION_SCHEMES:
        if scheme == "raw" or f"{scheme}_mase" not in per_series.columns:
            continue
        paired = per_series[["raw_mase", f"{scheme}_mase"]].replace(
            [np.inf, -np.inf], np.nan).dropna()
        changed = int((per_series["raw_model"] != per_series[f"{scheme}_model"]).sum())
        row = {
            "scheme": scheme,
            "picks_changed": changed,
            "n_series": int(len(per_series)),
            "wins_vs_raw": int((paired[f"{scheme}_mase"] < paired["raw_mase"]).sum()),
            "losses_vs_raw": int((paired["raw_mase"] < paired[f"{scheme}_mase"]).sum()),
            "mean_mase_delta": float(
                (paired[f"{scheme}_mase"] - paired["raw_mase"]).mean()) if len(paired) else np.nan,
            "wilcoxon_p": 1.0,
        }
        if len(paired) >= 3 and not np.allclose(paired["raw_mase"], paired[f"{scheme}_mase"]):
            try:
                with np.errstate(invalid="ignore", divide="ignore"):
                    _, p = stats.wilcoxon(paired["raw_mase"], paired[f"{scheme}_mase"])
                row["wilcoxon_p"] = float(p) if np.isfinite(p) else 1.0
            except ValueError:
                pass
        comparisons.append(row)

    summary.attrs["comparisons"] = comparisons
    return per_series, summary


# ── Candidate scoring configurations ─────────────────────────────────────────
#
# The measured diagnosis on 70 M5 level-9 series: decomposing why the shipped
# score passes over SARIMA (the model rmse_only picks in 44/70 series, and which
# is genuinely top-3 on HELD-OUT error in 44/70), three terms push against it —
# mape (+0.081), speed (+0.088), stability (+0.045) — while only rmse (-0.025),
# mae (-0.016) and r2 (-0.005) push for it. The harmful terms outweigh the
# helpful ones 4.6:1, which is why re-weighting and re-normalizing each moved
# realized MASE by only ~1%. The defect is WHICH criteria are included, not how
# they are scaled:
#
#   mape      — asymmetric (punishes over-forecasting harder than under-) and
#               unstable near zero, so on low-volume days it systematically
#               favours flat under-forecasts over genuine seasonal ones. This
#               project's own results already documented MAPE at 294-399%.
#   speed     — a COST criterion, not an accuracy one. Including it trades
#               forecast quality for compute time by construction.
#   stability — its discontinuity term |forecast[0] - last_actual| penalises a
#               model for not anchoring to the final observation, rewarding
#               persistence-like forecasts over statistical ones.
#
# These configurations test that diagnosis. Naming each one makes the comparison
# a stated hypothesis rather than a search over arbitrary weight vectors.
_ZERO = {k: 0.0 for k in DEFAULT_WEIGHTS}

CANDIDATE_CONFIGS: dict[str, dict] = {
    # The shipped scoring, and the same weights under the fixed normalisation —
    # isolates how much is scaling versus criteria choice.
    "shipped":          {"weights": DEFAULT_WEIGHTS, "scheme": "raw"},
    "shipped_balanced": {"weights": DEFAULT_WEIGHTS, "scheme": "balanced"},

    # Leave-one-criterion-out: which single removal helps most?
    "drop_mape":      {"weights": {**DEFAULT_WEIGHTS, "mape": 0.0}, "scheme": "balanced"},
    "drop_speed":     {"weights": {**DEFAULT_WEIGHTS, "speed": 0.0}, "scheme": "balanced"},
    "drop_stability": {"weights": {**DEFAULT_WEIGHTS, "stability": 0.0}, "scheme": "balanced"},

    # The diagnosis's actual prediction: drop all three harmful criteria.
    "accuracy_only": {"weights": {**_ZERO, "rmse": 0.40, "mae": 0.20, "r2": -0.10},
                      "scheme": "balanced"},
    # Same idea without the in-sample R2 term, which is itself CV-fitted.
    "rmse_mae":      {"weights": {**_ZERO, "rmse": 0.60, "mae": 0.40}, "scheme": "balanced"},
    # Degenerate case: should reproduce the rmse_only policy exactly, and serves
    # as a correctness check on the whole configuration machinery.
    "rmse_pure":     {"weights": {**_ZERO, "rmse": 1.0}, "scheme": "balanced"},
}

# Not a composite configuration — the single-metric baseline the composite must
# beat, evaluated through the identical held-out path for a fair comparison.
REFERENCE_CONFIG = "reference_rmse_only"


# Compute budgets offered as SELECTABLE candidates rather than reported as a
# tuned constant. Reading the frontier across every series and then quoting its
# minimum would choose a hyperparameter on the evaluation data — the same
# winner's curse this module already corrects for model choice, weight vectors
# and configurations. Budgets therefore compete inside the identical nested
# machinery, so a reported budget is one that was chosen on training folds and
# scored on held-out ones.
#
# Grid refined between 1s and 5s: the raw frontier had a single point at 2s in
# that interval, so "2.0s is optimal" claimed more precision than the spacing
# could support.
BUDGET_GRID: tuple[float, ...] = (
    0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 30.0, float("inf"),
)


def _budget_config_name(budget: float) -> str:
    return "budget_unlimited" if np.isinf(budget) else f"budget_{budget:g}s"


def select_within_budget(records: Sequence[dict], budget: float,
                         cost_field: str = "train_time") -> dict | None:
    """
    Lowest-CV-RMSE model whose cost fits the budget; if none fits, the cheapest
    available. Shared by the frontier and the nested selection so both express
    exactly the same policy.
    """
    usable = [r for r in records if r.get("ok")]
    if not usable:
        return None
    affordable = [r for r in usable if r.get(cost_field, 0.0) <= budget]
    if affordable:
        return min(affordable, key=lambda r: r["cv_rmse"])
    return min(usable, key=lambda r: r.get(cost_field, 0.0))


def evaluate_configs(
    raw: Sequence[SeriesResult],
    configs: dict[str, dict] | None = None,
    budgets: Sequence[float] | None = BUDGET_GRID,
) -> pd.DataFrame:
    """
    Realized held-out MASE for every (series, candidate) pair.

    Nothing varies but the selection rule — same models, same forecasts, same
    held-out windows — so differences are attributable to the rule alone.
    Candidates are the weighted-sum configurations AND, when `budgets` is
    given, one budget-constrained policy per budget, so the two families are
    compared and selected on equal terms.

    Returns long format: series_id, config, model, mase.
    """
    configs = configs or CANDIDATE_CONFIGS
    rows: list[dict] = []

    for result in raw:
        usable = {r["model"]: r for r in result.records if r.get("ok")}
        if not usable:
            continue

        for name, cfg in configs.items():
            scores = recompute_composite(result.records, cfg["weights"], cfg["scheme"])
            if not scores:
                continue
            pick = min(scores, key=lambda m: scores[m])
            rows.append({"series_id": result.series_id, "config": name,
                         "model": pick, "mase": usable[pick]["test_mase"]})

        for budget in (budgets or ()):
            pick = select_within_budget(result.records, budget)
            if pick is not None:
                rows.append({"series_id": result.series_id,
                             "config": _budget_config_name(budget),
                             "model": pick["model"], "mase": pick["test_mase"]})

        # The baseline, computed directly rather than through the composite so
        # it cannot inherit any quirk of the scoring machinery.
        best_rmse = min(usable.values(), key=lambda r: r["cv_rmse"])
        rows.append({"series_id": result.series_id, "config": REFERENCE_CONFIG,
                     "model": best_rmse["model"], "mase": best_rmse["test_mase"]})
        # Unachievable floor, for regret.
        oracle = min(usable.values(), key=lambda r: r["test_mase"])
        rows.append({"series_id": result.series_id, "config": "oracle",
                     "model": oracle["model"], "mase": oracle["test_mase"]})

    return pd.DataFrame(rows)


def nested_config_selection(
    raw: Sequence[SeriesResult],
    configs: dict[str, dict] | None = None,
    n_folds: int = 5,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Honest estimate of "pick the best configuration, then use it".

    Choosing a configuration by its score on all 70 series and then reporting
    that same score is the winner's curse one level up — the exact bias this
    benchmark exists to remove, and it has already appeared twice in this
    project (model selection on CV folds, and weight tuning on the evaluation
    set). So: K-fold over SERIES. Within each fold, the configuration is chosen
    using only the OTHER folds, then scored on the held-out fold it never saw.

    Returns (per_fold, summary):
      per_fold — which configuration each fold selected and what it achieved,
                 alongside the reference baseline on the same fold.
      summary  — for every configuration, its in-sample mean (optimistic) and
                 the honest nested estimate for the selection PROCEDURE.
    """
    evaluated = evaluate_configs(raw, configs)
    if evaluated.empty:
        return pd.DataFrame(), pd.DataFrame()

    wide = evaluated.pivot(index="series_id", columns="config", values="mase")
    wide = wide.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if len(wide) < n_folds:
        return pd.DataFrame(), pd.DataFrame()

    selectable = [c for c in wide.columns
                  if c not in (REFERENCE_CONFIG, "oracle")]

    series_ids = np.array(wide.index)
    rng = np.random.RandomState(seed)
    folds = np.array_split(rng.permutation(len(series_ids)), n_folds)

    fold_rows: list[dict] = []
    for i, test_idx in enumerate(folds):
        test_ids = series_ids[test_idx]
        train = wide.drop(index=test_ids)
        test = wide.loc[test_ids]

        chosen = train[selectable].mean().idxmin()
        fold_rows.append({
            "fold": i,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "selected_config": chosen,
            "train_mase": float(train[chosen].mean()),
            "test_mase": float(test[chosen].mean()),
            "reference_test_mase": float(test[REFERENCE_CONFIG].mean()),
            "oracle_test_mase": float(test["oracle"].mean()),
        })

    per_fold = pd.DataFrame(fold_rows)
    # Per-fold margin against the baseline. Reporting only the mean would hide
    # that a "win" can rest on some folds losing — which is exactly the case at
    # the granularities where the margin is small.
    per_fold["margin_vs_reference"] = (
        per_fold["reference_test_mase"] - per_fold["test_mase"])

    summary_rows = [{
        "config": c,
        "in_sample_mean_mase": float(wide[c].mean()),
        "median_mase": float(wide[c].median()),
        "pct_beats_naive": float((wide[c] < 1.0).mean() * 100),
        "n_series": int(len(wide)),
    } for c in wide.columns]
    summary = pd.DataFrame(summary_rows).sort_values("in_sample_mean_mase")

    # The honest number: performance of the SELECTION PROCEDURE, not of the
    # configuration that happened to look best overall.
    margins = per_fold["margin_vs_reference"].to_numpy()
    n = len(margins)
    # Fold-level spread. With k folds this is a coarse interval, but quoting a
    # bare point estimate for a margin smaller than the fold-to-fold variation
    # would overstate what k folds can resolve.
    se = float(margins.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    summary.attrs["nested_test_mase"] = float(per_fold["test_mase"].mean())
    summary.attrs["nested_reference_mase"] = float(per_fold["reference_test_mase"].mean())
    summary.attrs["nested_oracle_mase"] = float(per_fold["oracle_test_mase"].mean())
    summary.attrs["margin_mean"] = float(margins.mean())
    summary.attrs["margin_se"] = se
    summary.attrs["margin_ci95"] = (
        (float(margins.mean() - 1.96 * se), float(margins.mean() + 1.96 * se))
        if np.isfinite(se) else (float("nan"), float("nan")))
    summary.attrs["folds_won"] = int((margins > 0).sum())
    summary.attrs["folds_total"] = n
    # A margin whose interval spans zero is not a demonstrated improvement,
    # however favourable the point estimate looks.
    summary.attrs["margin_excludes_zero"] = bool(
        np.isfinite(se) and (margins.mean() - 1.96 * se) > 0)
    summary.attrs["selection_stability"] = per_fold["selected_config"].value_counts().to_dict()
    best_in_sample = summary.iloc[0]["config"]
    summary.attrs["optimism_bias"] = float(
        per_fold["test_mase"].mean() - wide[best_in_sample].mean())
    return per_fold, summary


def config_disagreement(
    raw: Sequence[SeriesResult],
    configs: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """
    How often does each configuration actually pick a DIFFERENT model from the
    RMSE-only baseline, and what does it cost when it does?

    Calibrates how strong "no configuration beats rmse_only" really is. If the
    surviving configurations diverge from the baseline on only a handful of
    series, the claim is thinner than it sounds — they are near-copies of the
    baseline rather than genuine alternatives that lost a fair fight. A reviewer
    will ask this; better to answer it first.
    """
    evaluated = evaluate_configs(raw, configs)
    if evaluated.empty:
        return pd.DataFrame()

    models = evaluated.pivot(index="series_id", columns="config", values="model")
    mases = evaluated.pivot(index="series_id", columns="config", values="mase")
    ref = REFERENCE_CONFIG

    rows = []
    for config in models.columns:
        if config == ref:
            continue
        differs = models[config] != models[ref]
        n_diff = int(differs.sum())
        delta = (mases[config] - mases[ref])
        rows.append({
            "config": config,
            "n_series": int(len(models)),
            "n_differs_from_baseline": n_diff,
            "pct_differs": float(n_diff / len(models) * 100),
            # Cost/benefit restricted to the series where it actually deviated —
            # averaging over series where the pick was identical just dilutes it.
            "mean_mase_delta_when_differs": float(delta[differs].mean()) if n_diff else 0.0,
            "wins_when_differs": int((delta[differs] < 0).sum()) if n_diff else 0,
            "losses_when_differs": int((delta[differs] > 0).sum()) if n_diff else 0,
        })
    return pd.DataFrame(rows).sort_values("n_differs_from_baseline", ascending=False)


# ── Constrained selection: cost as a budget, not a penalty ───────────────────
#
# The measured failure of the shipped score is not that training time was
# weighted wrongly, but that it was weighted AT ALL. Folding a cost criterion
# into a weighted sum makes it trade against accuracy at every margin, at an
# exchange rate nobody chose deliberately: at a nominal weight of 0.10, speed
# ended up dictating 88% of selections, because normalized training time spans
# six orders of magnitude while normalized accuracy spans a factor of ~1.2.
#
# Standard multi-criteria practice expresses a cost requirement as a CONSTRAINT
# instead: "the most accurate model that trains within T seconds". That has
# three advantages over a penalty term. The trade-off becomes an explicit,
# auditable budget rather than an implied exchange rate; accuracy is optimized
# exactly, not diluted; and the resulting accuracy-versus-latency frontier is
# something an operator can actually reason about when choosing T.

def constrained_selection_frontier(
    raw: Sequence[SeriesResult],
    budgets: Sequence[float] | None = None,
    cost_field: str = "train_time",
) -> pd.DataFrame:
    """
    Accuracy/latency frontier: for each budget T, select the lowest-CV-RMSE
    model whose cost is within T, and report the realized held-out MASE.

    Selection uses ONLY cross-validated error and the cost field — never the
    held-out window — so every point on the frontier is an achievable operating
    point, not hindsight. Series with no model inside the budget fall back to
    the cheapest available model, which is what a real system would have to do.

    Returns one row per budget: budget_seconds, mean/median MASE,
    pct_beats_naive, the mean selected-model cost, and how often the budget
    actually bound (i.e. excluded the model RMSE alone would have chosen).

    NOTE the frontier is monotone in `mean_selected_cv_rmse` — relaxing the
    budget can only widen the choice set, so the selection objective can only
    improve — but it is NOT guaranteed monotone in realized MASE, because a
    model with lower cross-validated error can still generalize worse. A budget
    that improves held-out accuracy is therefore acting as a regularizer: it
    excludes models that overfit the CV folds. Both columns are reported so
    that effect is visible rather than averaged away.
    """
    if budgets is None:
        # Sorted so the frontier reads monotonically; BUDGET_GRID ends at inf,
        # which would otherwise land before the 60s/120s points.
        budgets = sorted({0.01, *BUDGET_GRID, 60.0, 120.0})

    rows = []
    for budget in budgets:
        mases, costs, cv_rmses, bound, fallbacks = [], [], [], 0, 0
        for result in raw:
            usable = [r for r in result.records if r.get("ok")]
            if not usable:
                continue
            unconstrained = min(usable, key=lambda r: r["cv_rmse"])
            # Same policy the nested selection scores, so the frontier and the
            # honest estimate can never describe different rules.
            pick = select_within_budget(result.records, budget, cost_field)
            if not any(r.get(cost_field, 0.0) <= budget for r in usable):
                # Nothing fit; select_within_budget fell back to the cheapest
                # model. Counted, not hidden — the budget was not honoured.
                fallbacks += 1
            if pick["model"] != unconstrained["model"]:
                bound += 1
            mases.append(pick["test_mase"])
            costs.append(float(pick.get(cost_field, 0.0)))
            cv_rmses.append(float(pick["cv_rmse"]))

        if not mases:
            continue
        arr = np.asarray(mases, dtype=float)
        arr = arr[np.isfinite(arr)]
        rows.append({
            "budget_seconds": budget,
            "mean_mase": float(arr.mean()),
            "median_mase": float(np.median(arr)),
            "pct_beats_naive": float((arr < 1.0).mean() * 100),
            # The selection objective. Monotone in the budget by construction —
            # if this ever rises as the budget relaxes, the selector is broken.
            "mean_selected_cv_rmse": float(np.mean(cv_rmses)),
            "mean_selected_cost": float(np.mean(costs)),
            "max_selected_cost": float(np.max(costs)),
            "pct_budget_binding": float(bound / len(mases) * 100),
            "n_fallback": fallbacks,
            "n_series": int(len(arr)),
        })
    return pd.DataFrame(rows)


def compare_penalty_vs_constraint(
    raw: Sequence[SeriesResult],
    weights: dict[str, float] | None = None,
    scheme: str = "raw",
    cost_field: str = "train_time",
) -> pd.DataFrame:
    """
    Head-to-head at matched cost: the shipped weighted-sum score versus a
    constraint tuned to spend the SAME average budget.

    This is the fair comparison. Showing that constrained selection is more
    accurate than the composite is uninteresting if it simply spends more
    compute; the question is whether, at equal cost, expressing the requirement
    as a budget beats blending it into the score.
    """
    # What the shipped score actually achieves, and what it actually costs.
    comp_mases, comp_costs = [], []
    for result in raw:
        usable = {r["model"]: r for r in result.records if r.get("ok")}
        if not usable:
            continue
        scores = recompute_composite(result.records, weights, scheme)
        if not scores:
            continue
        pick = usable[min(scores, key=lambda m: scores[m])]
        comp_mases.append(pick["test_mase"])
        comp_costs.append(float(pick.get(cost_field, 0.0)))

    if not comp_mases:
        return pd.DataFrame()

    composite_cost = float(np.mean(comp_costs))
    frontier = constrained_selection_frontier(raw, cost_field=cost_field)

    # The tightest budget whose realized average cost still matches or undercuts
    # what the composite spent — i.e. the constraint that is no more expensive.
    affordable = frontier[frontier["mean_selected_cost"] <= composite_cost + 1e-9]
    matched = (affordable.sort_values("mean_mase").iloc[0]
               if not affordable.empty else frontier.iloc[0])

    comp_arr = np.asarray(comp_mases, dtype=float)
    comp_arr = comp_arr[np.isfinite(comp_arr)]
    unlimited = frontier[np.isinf(frontier["budget_seconds"])]

    rows = [{
        "method": "composite (cost as penalty)",
        "mean_mase": float(comp_arr.mean()),
        "mean_cost_seconds": composite_cost,
        "budget_seconds": np.nan,
    }, {
        "method": "constrained (cost as budget, matched)",
        "mean_mase": float(matched["mean_mase"]),
        "mean_cost_seconds": float(matched["mean_selected_cost"]),
        "budget_seconds": float(matched["budget_seconds"]),
    }]
    if not unlimited.empty:
        rows.append({
            "method": "unconstrained rmse_only (reference)",
            "mean_mase": float(unlimited.iloc[0]["mean_mase"]),
            "mean_cost_seconds": float(unlimited.iloc[0]["mean_selected_cost"]),
            "budget_seconds": float("inf"),
        })
    return pd.DataFrame(rows)


def summarize_explanation(explained: pd.DataFrame) -> pd.DataFrame:
    """
    Per term, across every series where the alternative lost: mean gap, and
    the share of series where this term had the LARGEST gap (i.e. was the
    single biggest reason the alternative lost that series). The second
    number is the one to lead with — it directly answers "which term is most
    often the deciding factor".
    """
    if explained.empty:
        return pd.DataFrame()

    n_series = explained["series_id"].nunique()
    biggest = (
        explained.loc[explained.groupby("series_id")["gap"].idxmax(), "term"]
        .value_counts()
        .reindex(DEFAULT_WEIGHTS.keys(), fill_value=0)
    )
    summary = (
        explained.groupby("term")["gap"].mean().reindex(DEFAULT_WEIGHTS.keys())
        .to_frame("mean_gap")
        .assign(
            pct_series_biggest_reason=lambda d: (biggest / n_series * 100).reindex(d.index).to_numpy(),
        )
        .sort_values("pct_series_biggest_reason", ascending=False)
        .reset_index()
        .rename(columns={"index": "term"})
    )
    return summary


# ── Per-series evaluation ────────────────────────────────────────────────────

@dataclass
class SeriesResult:
    """Everything measured for one series — enough to rescore without retraining."""
    series_id: str
    n_obs: int
    horizon: int
    season: int
    mase_denom: float
    test_actual: list[float]
    records: list[dict] = field(default_factory=list)
    dl_skipped: bool = False
    # Tail of the SELECTION region, kept so the naive baselines can be rebuilt
    # during re-analysis. Without it, --from-raw would have to fall back to the
    # held-out actuals, which would let the naive forecast see the very window
    # it is being scored against.
    train_tail: list[float] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "series_id": self.series_id, "n_obs": self.n_obs, "horizon": self.horizon,
            "season": self.season, "mase_denom": self.mase_denom,
            "test_actual": self.test_actual, "records": self.records,
            "dl_skipped": self.dl_skipped, "train_tail": self.train_tail,
        }

    @classmethod
    def from_json(cls, d: dict) -> "SeriesResult":
        return cls(**d)


def _default_runner(series: pd.Series, horizon: int, skip_dl: bool):
    """
    Bridge to the production orchestrator.

    Imported lazily so this module can be imported — and its analysis logic
    tested — without pulling in torch/prophet/statsmodels.
    """
    from services.model_runner import run_all_models
    return run_all_models(series, forecast_horizon=horizon, include_dl=not skip_dl)


# ── Series-level parallelism ─────────────────────────────────────────────────
#
# Within a single series, model_runner already runs all 19 models concurrently
# on a thread pool — but per-series wall time is then bounded by the SLOWEST
# model, not the total. On real M5 data that slowest model is SARIMA/SARIMAX's
# (p,d,q)x(P,D,Q,m) grid search, measured at 77-120s per series, which sits on
# the critical path and cannot be shortened by adding more workers to it.
#
# Series, however, are completely independent, so processing several at once is
# embarrassingly parallel and scales close to linearly with core count. That is
# the only large speedup available here — and it is why a GPU would not help:
# the critical path is a CPU-bound statsmodels grid search with no GPU code path
# at all, and the deep learning models are small enough that device-transfer
# overhead tends to cancel any GPU gain.
#
# Processes rather than threads, because each worker starts its own internal
# thread pool and nesting thread pools would just oversubscribe the GIL.

def _worker_init(threads_per_series: int) -> None:
    """
    Runs once per worker process.

    Shrinks model_runner's internal thread pool so that
    (n_jobs x threads_per_series) stays near the physical core count. Without
    this, 4 parallel series x 8 internal workers = 32 CPU-bound fits competing
    for ~12 cores, and context-switching thrash eats most of the speedup.

    Patches the module attribute rather than editing services/model_runner.py,
    so nothing in the request path changes.
    """
    from services import model_runner
    model_runner.MAX_WORKERS = max(2, threads_per_series)


def _evaluate_one(task: tuple) -> tuple:
    """
    Picklable top-level worker. Returns (series_id, SeriesResult | None, error).

    Exceptions are returned rather than raised so one bad series cannot abort
    the pool — matching how model_runner isolates a single failing model.
    """
    series_id, series, horizon, season, skip_dl = task
    try:
        return series_id, evaluate_series(
            series, series_id, horizon, season, skip_dl, _default_runner,
        ), None
    except Exception as exc:
        return series_id, None, str(exc)


def evaluate_series(
    series: pd.Series,
    series_id: str,
    horizon: int,
    season: int = 7,
    skip_dl: bool = False,
    runner: Callable = _default_runner,
) -> SeriesResult:
    """
    Run the full registry on the selection region and score every model's
    forecast against the held-out test window.

    The test window is the final `horizon` observations and is never passed to
    the runner — this is what makes the reported errors unbiased.
    """
    if len(series) <= horizon + 10:
        raise ValueError(
            f"series {series_id!r} has {len(series)} observations, too short for a "
            f"{horizon}-step held-out window"
        )

    train = series.iloc[:-horizon]
    test = series.iloc[-horizon:]
    train_values = train.to_numpy(dtype=float)
    test_values = test.to_numpy(dtype=float)
    denom = seasonal_naive_mae(train_values, season)

    results, run_info = runner(train, horizon, skip_dl)

    records: list[dict] = []
    for r in results:
        forecast = r.get("forecast") or []
        ok = (
            r.get("status") == "success"
            and len(forecast) >= horizon
            and np.all(np.isfinite(np.asarray(forecast[:horizon], dtype=float)))
        )
        rec = {
            "model": r.get("model_name", "?"),
            "ok": bool(ok),
            "status": r.get("status"),
            # Cross-validated metrics from the SELECTION region only.
            "cv_rmse": float(r.get("rmse", np.inf)),
            "cv_mae": float(r.get("mae", np.inf)),
            "cv_mape": float(r.get("mape", np.inf)),
            "cv_r2": float(r.get("r2", 0.0) or 0.0),
            "cv_score": float(r.get("adjusted_score", np.inf)),
            "stability": float(r.get("stability_penalty", np.inf)),
            # model_runner falls back to elapsed_sec when a model doesn't
            # report train_time_sec — mirrored here so speed stays comparable.
            "train_time": float(r.get("train_time_sec", r.get("elapsed_sec", 0.0)) or 0.0),
            "pred_time": float(r.get("pred_time_sec", 0.0) or 0.0),
        }
        if ok:
            fc = np.asarray(forecast[:horizon], dtype=float)
            rec["test_mase"] = mase(test_values, fc, denom)
            rec["test_rmse"] = rmse(test_values, fc)
            rec["forecast"] = [float(v) for v in fc]
        else:
            rec["test_mase"] = float("inf")
            rec["test_rmse"] = float("inf")
            rec["forecast"] = []
        records.append(rec)

    return SeriesResult(
        series_id=series_id,
        n_obs=len(series),
        horizon=horizon,
        season=season,
        mase_denom=denom,
        test_actual=[float(v) for v in test_values],
        records=records,
        dl_skipped=bool(run_info.get("dl_skipped")),
        # _naive_forecasts needs only the last `season` points, but keep a
        # little headroom so re-analysis isn't boxed in by today's baselines.
        train_tail=[float(v) for v in train_values[-max(2 * season, 60):]],
    )


# ── Selection policies ───────────────────────────────────────────────────────

def _naive_forecasts(train_tail: np.ndarray, horizon: int, season: int) -> dict[str, np.ndarray]:
    """Reference forecasts that involve no model selection at all."""
    last = float(train_tail[-1]) if train_tail.size else 0.0
    naive = np.full(horizon, last, dtype=float)
    if train_tail.size >= season:
        cycle = train_tail[-season:]
        seasonal = np.array([cycle[i % season] for i in range(horizon)], dtype=float)
    else:
        seasonal = naive.copy()
    return {"naive": naive, "seasonal_naive": seasonal}


def apply_policies(
    result: SeriesResult,
    train_tail: np.ndarray,
    weights: dict[str, float] | None = None,
    fixed_models: Sequence[str] = (),
    scheme: str = "raw",
) -> dict[str, dict]:
    """
    Resolve every selection policy to a chosen model and its realized error.

    Returns {policy: {"model": str, "mase": float, "rmse": float}}.
    Policies that cannot be resolved (e.g. a fixed model that failed on this
    series) are omitted, and the aggregation step drops series where any
    compared policy is missing, so every comparison stays paired.
    """
    usable = [r for r in result.records if r["ok"]]
    out: dict[str, dict] = {}
    if not usable:
        return out

    def _record(policy: str, rec: dict) -> None:
        out[policy] = {"model": rec["model"], "mase": rec["test_mase"], "rmse": rec["test_rmse"]}

    # Multi-criteria: the system's own score, recomputed so the weight sweep
    # and the shipped configuration go through identical code.
    scores = recompute_composite(result.records, weights, scheme)
    if scores:
        best = min(scores, key=lambda m: scores[m])
        match = next((r for r in usable if r["model"] == best), None)
        if match:
            _record("composite", match)

    # Single-metric alternatives.
    _record("rmse_only", min(usable, key=lambda r: r["cv_rmse"]))
    _record("mae_only", min(usable, key=lambda r: r["cv_mae"]))

    # Non-adaptive baselines: does choosing per series beat always using one model?
    for name in fixed_models:
        match = next((r for r in usable if r["model"] == name), None)
        if match:
            _record(f"always_{name}", match)

    # Reference forecasts.
    actual = np.asarray(result.test_actual, dtype=float)
    for name, fc in _naive_forecasts(train_tail, result.horizon, result.season).items():
        out[name] = {
            "model": name,
            "mase": mase(actual, fc, result.mase_denom),
            "rmse": rmse(actual, fc),
        }

    # Unachievable lower bound — uses the test window, so it is a floor, not a policy.
    _record("oracle", min(usable, key=lambda r: r["test_mase"]))
    return out


# ── Statistical comparison ───────────────────────────────────────────────────

def diebold_mariano(
    actual: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    horizon: int = 1,
    power: int = 1,
) -> tuple[float, float]:
    """
    Diebold-Mariano test with the Harvey-Leybourne-Newbold small-sample
    correction, for comparing two forecasts on ONE series.

    Returns (statistic, two-sided p-value). A negative statistic favours A.
    Across MULTIPLE series, use the Friedman/Wilcoxon protocol below instead —
    DM assumes a single loss sequence over time.
    """
    from scipy import stats

    n = min(len(actual), len(pred_a), len(pred_b))
    if n < 3:
        return float("nan"), float("nan")
    a, b, y = pred_a[:n], pred_b[:n], actual[:n]
    d = np.abs(y - a) ** power - np.abs(y - b) ** power

    d_mean = float(np.mean(d))
    # Newey-West style long-run variance, truncated at the forecast horizon.
    gamma0 = float(np.mean((d - d_mean) ** 2))
    lrv = gamma0
    for lag in range(1, min(horizon, n - 1)):
        cov = float(np.mean((d[lag:] - d_mean) * (d[:-lag] - d_mean)))
        lrv += 2 * cov
    if lrv <= 0:
        return float("nan"), float("nan")

    dm = d_mean / np.sqrt(lrv / n)
    # HLN correction for short samples.
    correction = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    dm_corrected = dm * correction
    p = 2 * (1 - stats.t.cdf(abs(dm_corrected), df=n - 1))
    return float(dm_corrected), float(p)


def compare_policies(df: pd.DataFrame, metric: str = "mase") -> dict:
    """
    Demsar protocol for comparing selection policies across many series:
    Friedman omnibus test, average ranks, then Holm-corrected pairwise
    Wilcoxon signed-rank tests.

    `df` is the long frame from run_benchmark(): one row per (series, policy).
    """
    from scipy import stats

    wide = df.pivot(index="series_id", columns="policy", values=metric)
    wide = wide.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if wide.shape[0] < 3 or wide.shape[1] < 2:
        return {"error": f"need >=3 complete series and >=2 policies, got {wide.shape}"}

    policies = list(wide.columns)
    ranks = wide.rank(axis=1)
    avg_ranks = ranks.mean().sort_values()

    friedman_stat, friedman_p = stats.friedmanchisquare(*[wide[p].to_numpy() for p in policies])

    pairwise = []
    for a, b in itertools.combinations(policies, 2):
        try:
            with np.errstate(invalid="ignore", divide="ignore"):
                stat, p = stats.wilcoxon(wide[a], wide[b])
        except ValueError:      # identical vectors -> no differences to rank
            stat, p = float("nan"), 1.0
        # Two policies that always pick the same model give a zero-variance
        # differential; that is "no evidence of a difference", not a result.
        if not np.isfinite(p):
            p = 1.0
        pairwise.append({
            "policy_a": a, "policy_b": b,
            f"median_{metric}_a": float(wide[a].median()),
            f"median_{metric}_b": float(wide[b].median()),
            "a_wins": int((wide[a] < wide[b]).sum()),
            "b_wins": int((wide[b] < wide[a]).sum()),
            "statistic": float(stat), "p_raw": float(p),
        })

    # Holm-Bonferroni step-down correction across the pairwise family.
    pairwise.sort(key=lambda r: r["p_raw"])
    m = len(pairwise)
    running = 0.0
    for i, row in enumerate(pairwise):
        adjusted = min(1.0, row["p_raw"] * (m - i))
        running = max(running, adjusted)      # enforce monotonicity
        row["p_holm"] = running

    return {
        "n_series": int(wide.shape[0]),
        "metric": metric,
        "friedman_statistic": float(friedman_stat),
        "friedman_p": float(friedman_p),
        "average_ranks": {k: float(v) for k, v in avg_ranks.items()},
        "median_by_policy": {p: float(wide[p].median()) for p in policies},
        "pairwise": pairwise,
    }


# ── Orchestration ────────────────────────────────────────────────────────────

def load_series_from_dir(
    data_dir: Path,
    date_col: str,
    target_col: str,
    pattern: str = "*.csv",
    max_files: int | None = None,
    exclude: Sequence[str] = ("manifest",),
    cap_outliers: bool = True,
) -> list[tuple[str, pd.Series]]:
    """
    Load each CSV in `data_dir` through the system's own preprocessor, so the
    benchmark measures the deployed pipeline rather than a parallel one.

    `cap_outliers` defaults to True because that is what the deployed system
    does — but it is exposed because it is a methodological choice, not an
    implementation detail: IQR capping rewrites the forecast target, and on
    promotion-driven retail data it can clip genuine demand spikes rather than
    noise. Report which setting produced the published numbers, and ideally
    report both.
    """
    from services.preprocessor import preprocess

    paths = sorted(p for p in data_dir.glob(pattern)
                   if not any(tok in p.name for tok in exclude))
    if max_files is not None:
        paths = paths[:max_files]

    loaded: list[tuple[str, pd.Series]] = []
    for path in paths:
        try:
            df = pd.read_csv(path)
            series, _ = preprocess(df, date_col=date_col, target_col=target_col,
                                   cap_outliers=cap_outliers)
            loaded.append((path.stem, series))
        except Exception as exc:
            logger.warning("skipping %s: %s", path.name, exc)
    return loaded


def run_benchmark(
    series_list: Sequence[tuple[str, pd.Series]],
    horizon: int,
    season: int = 7,
    skip_dl: bool = False,
    weights: dict[str, float] | None = None,
    fixed_models: Sequence[str] = ("XGBoost", "ARIMA", "Prophet", "KNN"),
    runner: Callable = _default_runner,
    n_jobs: int = 1,
    threads_per_series: int = 4,
) -> tuple[pd.DataFrame, list[SeriesResult]]:
    """
    Evaluate every series and resolve every policy.

    Returns (long_frame, raw_results). The raw results are returned so a weight
    sweep can rescore without retraining anything — model training dominates
    runtime, and the sweep needs none of it.

    Args:
        n_jobs             : number of series to evaluate in parallel. 1 keeps
                             the original sequential path (and is required when
                             passing a custom `runner`, since worker processes
                             re-import this module and cannot receive closures).
        threads_per_series : size of each worker's internal model thread pool
                             when n_jobs > 1. Keep n_jobs * threads_per_series
                             at or below the physical core count.
    """
    rows: list[dict] = []
    raw: list[SeriesResult] = []

    def _collect(series_id: str, series: pd.Series, result: SeriesResult) -> None:
        raw.append(result)
        train_tail = series.iloc[:-horizon].to_numpy(dtype=float)
        for policy, info in apply_policies(result, train_tail, weights, fixed_models).items():
            rows.append({
                "series_id": series_id, "policy": policy, "model": info["model"],
                "mase": info["mase"], "rmse": info["rmse"],
                "n_obs": result.n_obs, "dl_skipped": result.dl_skipped,
            })

    if n_jobs <= 1:
        for i, (series_id, series) in enumerate(series_list, 1):
            logger.info("[%d/%d] %s (%d obs)", i, len(series_list), series_id, len(series))
            try:
                result = evaluate_series(series, series_id, horizon, season, skip_dl, runner)
            except Exception as exc:
                logger.warning("  failed: %s", exc)
                continue
            _collect(series_id, series, result)
        return pd.DataFrame(rows), raw

    # Parallel path. BLAS reads its thread-count env vars at import time, and
    # spawned children import numpy before any initializer runs — so these must
    # be set in the PARENT, before the pool is created, for children to inherit
    # them. Without this each worker's BLAS spawns its own thread fan-out on top
    # of everything else.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")

    by_id = dict(series_list)
    tasks = [(sid, s, horizon, season, skip_dl) for sid, s in series_list]
    done = 0

    with ProcessPoolExecutor(
        max_workers=n_jobs, initializer=_worker_init, initargs=(threads_per_series,),
    ) as pool:
        for series_id, result, error in pool.map(_evaluate_one, tasks):
            done += 1
            if error is not None:
                logger.warning("[%d/%d] %s failed: %s", done, len(tasks), series_id, error)
                continue
            logger.info("[%d/%d] %s (%d obs)", done, len(tasks), series_id, result.n_obs)
            _collect(series_id, by_id[series_id], result)

    return pd.DataFrame(rows), raw


def save_raw(raw: Sequence[SeriesResult], path: Path) -> None:
    """
    Persist per-series records so any later re-analysis is free.

    Model training dominates runtime and produces these records; every
    downstream question (different weights, new policies, another metric) needs
    only the records, not the models. Without this they die with the process and
    a single extra weight grid costs another full multi-hour run.
    """
    path.write_text(json.dumps([r.to_json() for r in raw]))


def load_raw(path: Path) -> list[SeriesResult]:
    """Reload records written by save_raw() for retraining-free re-analysis."""
    return [SeriesResult.from_json(d) for d in json.loads(path.read_text())]


def weight_sweep(
    raw: Sequence[SeriesResult],
    series_lookup: dict[str, np.ndarray] | None = None,
    grids: dict[str, Sequence[float]] | None = None,
) -> pd.DataFrame:
    """
    Rescore cached runs under varying composite weights — no model retraining.

    Answers the question a reviewer will ask about the shipped weights
    (0.40/0.20/0.20/0.10/0.10/-0.10): are the rankings they produce stable, or
    does the winner move under small perturbations? Varies one weight at a time
    from the shipped configuration.

    All six weights are swept by default. `mape` and `r2` matter most: r2 is
    computed IN-SAMPLE on the CV folds, so a model that overfits those folds is
    directly rewarded by the -0.10 term and then generalises worse on the
    held-out window — the winner's curse operating inside the scoring function
    itself. MAPE is asymmetric (it punishes over-forecasting harder than
    under-forecasting) and is unstable near zero-valued actuals, so it can
    quietly favour whichever model biases low.
    """
    grids = grids or {
        "rmse": [0.20, 0.30, 0.40, 0.50, 0.60],
        "mae": [0.00, 0.10, 0.20, 0.30],
        "mape": [0.00, 0.05, 0.10, 0.20, 0.30],
        "stability": [0.00, 0.05, 0.10, 0.20, 0.30],
        "speed": [0.00, 0.05, 0.10, 0.20, 0.30],
        "r2": [-0.30, -0.20, -0.10, 0.00],
    }

    rows: list[dict] = []
    for key, values in grids.items():
        for value in values:
            weights = {**DEFAULT_WEIGHTS, key: value}
            for result in raw:
                scores = recompute_composite(result.records, weights)
                if not scores:
                    continue
                best = min(scores, key=lambda m: scores[m])
                match = next((r for r in result.records if r["model"] == best), None)
                if match:
                    rows.append({
                        "swept_weight": key, "value": value,
                        "series_id": result.series_id,
                        "model": best, "mase": match["test_mase"],
                    })
    return pd.DataFrame(rows)


# Reference forecasts, not selection policies. They are not drawn from the
# trained model pool, so the oracle (a minimum over that pool) is not a floor
# for them and "regret vs oracle" is undefined — reported as NaN rather than as
# a misleading negative number. A reference baseline BEATING the oracle is a
# real and important finding: it means no trained model beat doing nothing.
REFERENCE_POLICIES = frozenset({"naive", "seasonal_naive"})


def summarize(df: pd.DataFrame, metric: str = "mase") -> pd.DataFrame:
    """
    Per-policy summary: central tendency, regret against the oracle floor, and
    how often the policy beat a seasonal-naive forecast.

    MASE < 1 means the forecast beat the in-sample seasonal-naive benchmark, so
    `pct_beats_naive` is the honest headline number: if it is low, the finding
    is that this data is not forecastable by these models, regardless of which
    selection rule wins the internal comparison.
    """
    wide = df.pivot(index="series_id", columns="policy", values=metric)
    wide = wide.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if wide.empty:
        return pd.DataFrame()

    rows = []
    has_oracle = "oracle" in wide.columns
    for policy in wide.columns:
        row = {
            "policy": policy,
            f"mean_{metric}": float(wide[policy].mean()),
            f"median_{metric}": float(wide[policy].median()),
            "n_series": int(wide.shape[0]),
        }
        if metric == "mase":
            row["pct_beats_naive"] = float((wide[policy] < 1.0).mean() * 100)
        if has_oracle and policy not in REFERENCE_POLICIES:
            row["mean_regret_vs_oracle"] = float((wide[policy] - wide["oracle"]).mean())
            row["pct_series_optimal"] = float((wide[policy] <= wide["oracle"] + 1e-12).mean() * 100)
        else:
            row["mean_regret_vs_oracle"] = float("nan")
            row["pct_series_optimal"] = float("nan")
        rows.append(row)

    return pd.DataFrame(rows).sort_values(f"mean_{metric}").reset_index(drop=True)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure whether multi-criteria model selection beats single-metric selection.",
    )
    parser.add_argument("--data-dir", type=Path, required=False)
    parser.add_argument("--from-raw", type=Path, default=None,
                        help="Re-analyse a previous run's persisted .raw.json records "
                             "instead of training. Every weight grid, policy, and "
                             "statistic is recomputed in seconds with no model fitting.")
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--target-col", default="sales")
    parser.add_argument("--horizon", type=int, default=28,
                        help="Held-out test window length (default 28, matching M5).")
    parser.add_argument("--season", type=int, default=7,
                        help="Seasonal period for the MASE denominator (default 7 = weekly).")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--skip-dl", action="store_true",
                        help="Skip deep learning models for a faster run.")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Series to evaluate in parallel (default 1). Series are "
                             "independent, so this scales near-linearly with cores and is "
                             "the only large speedup available — the per-series critical "
                             "path is SARIMA's CPU-bound grid search, which more internal "
                             "workers cannot shorten. Try 4 on a 12-core machine.")
    parser.add_argument("--threads-per-series", type=int, default=4,
                        help="Internal model-thread pool size per worker when --n-jobs > 1. "
                             "Keep n_jobs * threads_per_series at or below your core count.")
    parser.add_argument("--no-cap-outliers", action="store_true",
                        help="Disable the preprocessor's IQR outlier capping. Capping "
                             "rewrites the forecast target and can clip real promotional "
                             "spikes; report which setting produced your published numbers.")
    parser.add_argument("--weight-sweep", action="store_true",
                        help="Also run the composite-weight sensitivity sweep.")
    parser.add_argument("--constrained", action="store_true",
                        help="Treat training cost as a BUDGET rather than a score penalty: "
                             "report the accuracy/latency frontier, and compare the shipped "
                             "weighted-sum score against a constraint spending the same "
                             "average cost. Requires no retraining.")
    parser.add_argument("--disagreement", action="store_true",
                        help="How often each configuration picks a different model from the "
                             "RMSE-only baseline, and what it costs when it does. Calibrates "
                             "how strong the 'nothing beats rmse_only' result actually is.")
    parser.add_argument("--select-config", action="store_true",
                        help="Compare named scoring configurations and report an honest "
                             "nested-CV estimate of choosing one. Selection happens on "
                             "training folds of SERIES only, so the reported number is "
                             "not inflated by picking the configuration on the same data "
                             "it is scored against.")
    parser.add_argument("--config-folds", type=int, default=5,
                        help="Folds over series for --select-config (default 5).")
    parser.add_argument("--config-seed", type=int, default=0,
                        help="Fold-assignment seed for --select-config (default 0).")
    parser.add_argument("--compare-normalization", action="store_true",
                        help="Rescore every series under both the shipped ('raw') and "
                             "the clipped normalization, holding weights and models fixed. "
                             "Isolates how much of composite's failure is caused by term "
                             "scaling alone. Requires no retraining.")
    parser.add_argument("--scheme", choices=NORMALIZATION_SCHEMES, default="raw",
                        help="Normalization used for the `composite` policy, and for "
                             "--decompose / --explain-vs (default 'raw' = the shipped "
                             "scoring). Use 'balanced' to ask why a model still loses "
                             "AFTER the scale-mismatch fix.")
    parser.add_argument("--decompose", action="store_true",
                        help="Break the composite score into its six weighted terms and "
                             "report which term actually drives model selection. A term's "
                             "nominal weight sets its intended influence; its spread across "
                             "candidates sets its real one, and the two can disagree sharply.")
    parser.add_argument("--explain-vs", type=str, default=None, metavar="MODEL",
                        help="With --decompose: for every series where composite picked "
                             "something else, show which term cost MODEL the selection "
                             "(e.g. --explain-vs SARIMA).")
    parser.add_argument("--out", type=Path, default=Path("benchmark_results.csv"))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.data_dir is None and args.from_raw is None:
        parser.error("one of --data-dir or --from-raw is required")

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    raw_path = args.out.with_suffix(".raw.json")

    if args.from_raw is not None:
        # Re-analysis path: no data loading, no training. Everything below runs
        # off the persisted records from a previous run.
        raw = load_raw(args.from_raw)
        print(f"Loaded {len(raw)} cached series records from {args.from_raw} "
              f"(no models retrained)")
        rows: list[dict] = []
        for result in raw:
            # Must be the SELECTION-region tail, never the held-out actuals —
            # the naive baselines are forecasts and cannot be allowed to see the
            # window they are scored on.
            tail = np.asarray(result.train_tail, dtype=float)
            for policy, info in apply_policies(result, tail).items():
                rows.append({
                    "series_id": result.series_id, "policy": policy,
                    "model": info["model"], "mase": info["mase"],
                    "rmse": info["rmse"], "n_obs": result.n_obs,
                    "dl_skipped": result.dl_skipped,
                })
        df = pd.DataFrame(rows)
    else:
        series_list = load_series_from_dir(
            args.data_dir, args.date_col, args.target_col, max_files=args.max_files,
            cap_outliers=not args.no_cap_outliers,
        )
        if not series_list:
            print(f"No usable series found in {args.data_dir}", file=sys.stderr)
            return 1
        print(f"Loaded {len(series_list)} series from {args.data_dir}")

        df, raw = run_benchmark(
            series_list, horizon=args.horizon, season=args.season, skip_dl=args.skip_dl,
            n_jobs=args.n_jobs, threads_per_series=args.threads_per_series,
        )
        # Persist BEFORE any analysis, so a crash in reporting can never destroy
        # hours of training.
        save_raw(raw, raw_path)
        print(f"Saved {len(raw)} raw series records to {raw_path} "
              f"(re-analyse later with --from-raw, no retraining)")

    if df.empty:
        print("No series produced results.", file=sys.stderr)
        return 1

    df.to_csv(args.out, index=False)

    summary = summarize(df)
    print("\n=== Policy comparison (MASE; lower is better, <1 beats seasonal-naive) ===")
    print(summary.to_string(index=False))

    stats_out = compare_policies(df)
    stats_path = args.out.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats_out, indent=2))
    if "error" not in stats_out:
        print(f"\nFriedman: chi2={stats_out['friedman_statistic']:.3f} "
              f"p={stats_out['friedman_p']:.4g} over {stats_out['n_series']} series")
        print("Average ranks:",
              ", ".join(f"{k}={v:.2f}" for k, v in stats_out["average_ranks"].items()))
        key = [r for r in stats_out["pairwise"]
               if {r["policy_a"], r["policy_b"]} == {"composite", "rmse_only"}]
        if key:
            r = key[0]
            print(f"\ncomposite vs rmse_only: {r['a_wins']}-{r['b_wins']} wins, "
                  f"p_holm={r['p_holm']:.4g}")
    print(f"\nWrote {args.out} and {stats_path}")

    if args.disagreement:
        disagree = config_disagreement(raw)
        if disagree.empty:
            print("\nNo configurations to compare.")
        else:
            dis_path = args.out.with_suffix(".disagreement.csv")
            disagree.to_csv(dis_path, index=False)
            print("\n=== How far each configuration actually departs from rmse_only ===")
            print("(a config that rarely differs is a near-copy of the baseline,")
            print(" not a genuine alternative that lost a fair comparison)")
            print(disagree.to_string(index=False))
            print(f"Wrote {dis_path}")

    if args.constrained:
        frontier = constrained_selection_frontier(raw)
        fr_path = args.out.with_suffix(".frontier.csv")
        frontier.to_csv(fr_path, index=False)
        print("\n=== Accuracy / latency frontier (cost as a budget) ===")
        print("Most accurate model by CV RMSE that trains within the budget.")
        print(frontier.to_string(index=False))

        head_to_head = compare_penalty_vs_constraint(raw, scheme=args.scheme)
        if not head_to_head.empty:
            print("\n=== Cost as penalty vs cost as budget, at matched spend ===")
            print(head_to_head.to_string(index=False))
        print(f"Wrote {fr_path}")

    if args.select_config:
        per_fold, cfg_summary = nested_config_selection(
            raw, n_folds=args.config_folds, seed=args.config_seed)
        if cfg_summary.empty:
            print("\nNot enough complete series to run configuration selection.")
        else:
            cfg_path = args.out.with_suffix(".configs.csv")
            cfg_summary.to_csv(cfg_path, index=False)
            fold_path = args.out.with_suffix(".config_folds.csv")
            per_fold.to_csv(fold_path, index=False)

            print("\n=== Scoring configurations (in-sample; OPTIMISTIC) ===")
            print(cfg_summary.to_string(index=False))

            a = cfg_summary.attrs
            print(f"\n=== Nested {args.config_folds}-fold estimate (the honest number) ===")
            print("Configuration chosen on training folds only, scored on the held-out fold.")
            print(per_fold.to_string(index=False))
            print(f"\nselection procedure : {a['nested_test_mase']:.4f} MASE")
            print(f"rmse_only baseline  : {a['nested_reference_mase']:.4f} MASE")
            print(f"oracle floor        : {a['nested_oracle_mase']:.4f} MASE")
            lo, hi = a["margin_ci95"]
            print(f"margin vs baseline  : {a['margin_mean']:+.4f} "
                  f"(95% CI {lo:+.4f} to {hi:+.4f}, won {a['folds_won']}/{a['folds_total']} folds)")
            if a["margin_excludes_zero"]:
                verdict = "BEATS single-metric selection (CI excludes zero)"
            elif a["margin_mean"] > 0:
                verdict = ("is better on average but NOT demonstrated — "
                           "the confidence interval spans zero")
            else:
                verdict = "does NOT beat single-metric selection"
            print(f"=> the corrected composite {verdict}")
            print(f"\ncandidate chosen per fold: {a['selection_stability']}")
            print(f"optimism from selecting in-sample: {a['optimism_bias']:+.4f} MASE")
            print(f"Wrote {cfg_path} and {fold_path}")

    if args.compare_normalization:
        per_series, norm_summary = compare_normalization_schemes(raw)
        norm_path = args.out.with_suffix(".normalization.csv")
        per_series.to_csv(norm_path, index=False)
        print("\n=== Normalization scheme comparison (identical models, weights, data) ===")
        print(norm_summary.to_string(index=False))
        comparisons = norm_summary.attrs.get("comparisons", [])
        if comparisons:
            print("\nPaired against the shipped 'raw' scheme:")
            print(pd.DataFrame(comparisons).to_string(index=False))
        print(f"Wrote {norm_path}")

    if args.decompose:
        decomposed = decompose_composite_terms(raw, scheme=args.scheme)
        dec_path = args.out.with_suffix(".terms.csv")
        decomposed.to_csv(dec_path, index=False)

        dominance = summarize_term_dominance(decomposed)
        print(f"\n=== Which term actually drives selection (scheme={args.scheme}) ===")
        print("(nominal weight vs the spread it creates across candidate models;")
        print(" a term with no spread cannot change the winner at any weight)")
        print(dominance.to_string(index=False))
        print(f"Wrote {dec_path}")

        if args.explain_vs:
            explained = explain_pick_vs_alternative(raw, args.explain_vs, scheme=args.scheme)
            if explained.empty:
                print(f"\n{args.explain_vs} was never a losing candidate "
                      f"under scheme={args.scheme} — nothing to explain.")
            else:
                exp_path = args.out.with_suffix(f".explain_{args.scheme}.csv")
                explained.to_csv(exp_path, index=False)
                n = explained["series_id"].nunique()
                print(f"\n=== Why composite (scheme={args.scheme}) passed over "
                      f"{args.explain_vs} ({n} series) ===")
                print("(positive gap = the term where it lost ground to the picked model)")
                print(summarize_explanation(explained).to_string(index=False))
                print(f"Wrote {exp_path}")

    if args.weight_sweep:
        sweep = weight_sweep(raw)
        sweep_path = args.out.with_suffix(".weights.csv")
        sweep.to_csv(sweep_path, index=False)
        pivot = sweep.groupby(["swept_weight", "value"])["mase"].mean().reset_index()
        print("\n=== Weight sensitivity (mean MASE of the selected model) ===")
        print(pivot.to_string(index=False))
        print(f"Wrote {sweep_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
