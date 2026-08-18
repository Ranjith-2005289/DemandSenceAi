"""
Tests for the research benchmark tooling (benchmarks/).

These cover the parts that produce numbers reported in the paper, so a silent
regression here would corrupt published results rather than just break a
feature. Nothing here trains a real model — evaluate_series is exercised
through an injected fake runner, mirroring the fake_fast_registry approach
already used in test_model_runner_dl_gate.py.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks.m5_ingest import (
    AGGREGATION_LEVELS,
    aggregate_prices,
    aggregate_sales,
    build_calendar_features,
    to_long_frames,
)
from benchmarks.selection_benchmark import (
    DEFAULT_WEIGHTS,
    apply_policies,
    compare_policies,
    diebold_mariano,
    evaluate_series,
    mase,
    recompute_composite,
    robust_scale,
    run_benchmark,
    seasonal_naive_mae,
    summarize,
    weight_sweep,
)


# ── Synthetic M5 fixtures ────────────────────────────────────────────────────

N_DAYS = 56          # 8 weeks
ITEMS = ["FOODS_1_001", "FOODS_1_002", "HOBBIES_1_001", "HOBBIES_1_002"]
STORES = ["CA_1", "TX_1"]


@pytest.fixture
def m5_calendar() -> pd.DataFrame:
    dates = pd.date_range("2011-01-29", periods=N_DAYS, freq="D")
    return pd.DataFrame({
        "date": dates,
        "wm_yr_wk": 11101 + np.arange(N_DAYS) // 7,
        "d": [f"d_{i + 1}" for i in range(N_DAYS)],
        # An event every 10th day -> promo_flag should be 1 on those days only.
        "event_name_1": [("SuperBowl" if i % 10 == 0 else None) for i in range(N_DAYS)],
        "snap_CA": (np.arange(N_DAYS) % 2).astype(int),
        "snap_TX": (np.arange(N_DAYS) % 3 == 0).astype(int),
        "snap_WI": np.zeros(N_DAYS, dtype=int),
    })


@pytest.fixture
def m5_sales() -> pd.DataFrame:
    rng = np.random.RandomState(0)
    rows = []
    for item in ITEMS:
        cat = item.split("_")[0]
        dept = "_".join(item.split("_")[:2])
        for store in STORES:
            rows.append({
                "id": f"{item}_{store}_evaluation",
                "item_id": item, "dept_id": dept, "cat_id": cat,
                "store_id": store, "state_id": store.split("_")[0],
                **{f"d_{i + 1}": int(rng.randint(0, 20)) for i in range(N_DAYS)},
            })
    return pd.DataFrame(rows)


@pytest.fixture
def m5_prices() -> pd.DataFrame:
    rows = []
    for item in ITEMS:
        for store in STORES:
            for wk in range(11101, 11101 + (N_DAYS + 6) // 7):
                rows.append({"store_id": store, "item_id": item,
                             "wm_yr_wk": wk, "sell_price": 5.0 + (wk - 11101) * 0.1})
    return pd.DataFrame(rows)


# ── m5_ingest ────────────────────────────────────────────────────────────────

def test_aggregate_sales_preserves_total_across_levels(m5_sales):
    """Aggregating to any level must conserve total units — a wrong groupby
    silently inflating or dropping sales would corrupt every downstream number."""
    day_cols = [c for c in m5_sales.columns if c.startswith("d_")]
    grand_total = m5_sales[day_cols].to_numpy().sum()

    for level in (1, 3, 5, 9, 12):
        agg, cols = aggregate_sales(m5_sales, level)
        assert agg[cols].to_numpy().sum() == grand_total, f"level {level} lost sales"


def test_aggregate_sales_series_counts(m5_sales):
    assert len(aggregate_sales(m5_sales, 1)[0]) == 1                      # total
    assert len(aggregate_sales(m5_sales, 3)[0]) == len(STORES)            # store
    assert len(aggregate_sales(m5_sales, 12)[0]) == len(ITEMS) * len(STORES)


def test_day_columns_sorted_numerically_not_lexicographically(m5_sales):
    """d_10 must not sort before d_2 — lexicographic ordering would scramble
    the date mapping and silently shuffle every series in time."""
    _, day_cols = aggregate_sales(m5_sales, 1)
    assert day_cols[:11] == [f"d_{i}" for i in range(1, 12)]


def test_calendar_events_are_not_labelled_as_promotions(m5_calendar):
    """M5's event_name_1 holds HOLIDAYS (Super Bowl, Easter), not retailer
    promotions. Naming it promo_flag would auto-route it to SARIMAX/XGBoost as
    promotional exog and make any 'real promo features' claim false."""
    from services.preprocessor import PROMO_KEYWORDS

    feats = build_calendar_features(m5_calendar, level=9)
    assert "calendar_event" in feats.columns
    assert "promo_flag" not in feats.columns
    assert feats["calendar_event"].sum() == len([i for i in range(N_DAYS) if i % 10 == 0])
    # The event column must not be picked up by the exog promo detector.
    assert not any(kw in "calendar_event" for kw in PROMO_KEYWORDS)


def test_markdown_promo_flag_detects_price_drops():
    from benchmarks.m5_ingest import compute_markdown_promo_flag

    price = np.full(200, 10.0)
    price[100:110] = 8.0                       # a 20% markdown
    flag = compute_markdown_promo_flag(price, window=56, threshold=0.05)

    assert flag[100:110].all(), "markdown period not detected"
    assert not flag[:100].any(), "flagged before the markdown"
    assert set(np.unique(flag)) <= {0, 1}


def test_markdown_promo_flag_ignores_constant_price():
    """A product whose price never moves must yield zero promos, not noise."""
    from benchmarks.m5_ingest import compute_markdown_promo_flag

    assert compute_markdown_promo_flag(np.full(200, 7.5)).sum() == 0


def test_markdown_promo_flag_ignores_small_fluctuations():
    from benchmarks.m5_ingest import compute_markdown_promo_flag

    rng = np.random.RandomState(0)
    price = 10.0 + rng.randn(300) * 0.02       # ~0.2% jitter, far below 5%
    assert compute_markdown_promo_flag(price, threshold=0.05).sum() == 0


def test_markdown_baseline_survives_a_sustained_discount():
    """Regression: a rolling MEDIAN baseline lost a 50-day markdown from day 28
    onward, because the discount became the majority of its own window. A
    rolling max must hold the regular price for the whole event."""
    from benchmarks.m5_ingest import compute_markdown_promo_flag

    price = np.full(300, 20.0)
    price[150:200] = 12.0                      # 50 days at -40%
    flag = compute_markdown_promo_flag(price, window=56, threshold=0.05)
    assert flag[150:200].all(), "sustained markdown lost mid-period"
    assert not flag[:150].any()


def test_permanent_price_cut_is_a_known_bounded_false_positive():
    """Documents the accepted trade-off: a permanent cut reads as promotional
    until it ages out of the window, and no longer."""
    from benchmarks.m5_ingest import compute_markdown_promo_flag

    price = np.full(300, 20.0)
    price[100:] = 15.0                         # permanent -25% reprice
    flag = compute_markdown_promo_flag(price, window=56, threshold=0.05)

    assert flag[100:150].all()                 # flagged while the old price lingers
    assert not flag[160:].any()                # and correctly stops after that


def test_aggregate_prices_returns_none_at_total_level(m5_sales, m5_prices):
    """A mean price across every product is not a meaningful signal, so level 1
    must emit no price exog rather than a misleading one."""
    assert aggregate_prices(m5_sales, m5_prices, level=1) is None
    assert aggregate_prices(m5_sales, m5_prices, level=9) is not None


def test_to_long_frames_shape_and_exog_naming(m5_sales, m5_calendar, m5_prices):
    """Emitted column names are load-bearing: preprocessor.py matches exog
    columns by keyword, so renaming these silently disables the exog path."""
    from services.preprocessor import PRICE_KEYWORDS, PROMO_KEYWORDS

    frames, manifest = to_long_frames(m5_sales, m5_calendar, m5_prices, level=9)

    assert len(frames) == 4                      # 2 stores x 2 departments
    assert len(manifest) == 4

    frame = next(iter(frames.values()))
    assert len(frame) == N_DAYS
    assert list(frame["date"]) == list(m5_calendar["date"])
    assert frame["sales"].notna().all()

    assert any(kw in "sell_price" for kw in PRICE_KEYWORDS)
    assert any(kw in "promo_flag" for kw in PROMO_KEYWORDS)
    assert "sell_price" in frame.columns
    assert "promo_flag" in frame.columns         # derived from price markdowns
    assert "calendar_event" in frame.columns     # holidays, descriptive only
    assert "snap" in frame.columns               # store level -> state resolvable


def test_promo_flag_only_emitted_where_price_exists(m5_sales, m5_calendar, m5_prices):
    """The promo proxy is derived FROM price. Emitting it without price would
    assert a signal that was never measured — level 1 has no price."""
    frames, _ = to_long_frames(m5_sales, m5_calendar, m5_prices, level=1)
    frame = next(iter(frames.values()))
    assert "sell_price" not in frame.columns
    assert "promo_flag" not in frame.columns
    assert "calendar_event" in frame.columns     # still emitted, still descriptive


def test_to_long_frames_max_series_keeps_highest_volume(m5_sales, m5_calendar, m5_prices):
    all_frames, _ = to_long_frames(m5_sales, m5_calendar, m5_prices, level=12)
    capped, manifest = to_long_frames(m5_sales, m5_calendar, m5_prices,
                                      level=12, max_series=3)
    assert len(capped) == 3
    totals = {sid: f["sales"].sum() for sid, f in all_frames.items()}
    top3 = sorted(totals, key=lambda k: -totals[k])[:3]
    assert set(capped) == set(top3)


def test_to_long_frames_min_nonzero_filter(m5_sales, m5_calendar, m5_prices):
    """The filter that keeps level-12 benchmarking away from all-zero series."""
    sparse = m5_sales.copy()
    day_cols = [c for c in sparse.columns if c.startswith("d_")]
    sparse.loc[0, day_cols] = 0                  # make one item-store all zeros

    frames, _ = to_long_frames(sparse, m5_calendar, m5_prices,
                               level=12, min_nonzero_frac=0.5)
    assert len(frames) == len(ITEMS) * len(STORES) - 1


def test_trim_leading_zeros_drops_pre_launch_period(m5_sales, m5_calendar, m5_prices):
    """M5 introduces products over time; leading zeros mean 'not stocked yet',
    not zero demand. Training on them teaches a flat-zero regime that never
    existed."""
    sales = m5_sales.copy()
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    # Zero out the first 20 days for every row of one item-store.
    sales.loc[:, day_cols[:20]] = 0

    untrimmed, _ = to_long_frames(sales, m5_calendar, m5_prices, level=12)
    trimmed, manifest = to_long_frames(sales, m5_calendar, m5_prices, level=12,
                                       trim_leading_zeros=True)

    assert all(len(f) == N_DAYS for f in untrimmed.values())
    assert all(len(f) < N_DAYS for f in trimmed.values())
    assert (manifest["trimmed_leading"] >= 20).all()
    # First retained observation must be a real sale.
    for frame in trimmed.values():
        assert frame["sales"].iloc[0] != 0


def test_trim_leading_zeros_drops_never_sold_series(m5_sales, m5_calendar, m5_prices):
    sales = m5_sales.copy()
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    sales.loc[0, day_cols] = 0                   # this item-store never sells

    frames, _ = to_long_frames(sales, m5_calendar, m5_prices, level=12,
                               trim_leading_zeros=True)
    assert len(frames) == len(ITEMS) * len(STORES) - 1


def test_trim_leading_zeros_keeps_exog_aligned(m5_sales, m5_calendar, m5_prices):
    """Trimming must slice date/price/promo/snap identically to sales — a
    misaligned exog column would silently shift the promo signal in time."""
    sales = m5_sales.copy()
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    sales.loc[:, day_cols[:15]] = 0

    frames, _ = to_long_frames(sales, m5_calendar, m5_prices, level=9,
                               trim_leading_zeros=True)
    frame = next(iter(frames.values()))
    offset = N_DAYS - len(frame)

    expected_dates = list(m5_calendar["date"].iloc[offset:])
    assert list(frame["date"]) == expected_dates
    expected_events = m5_calendar["event_name_1"].notna().astype(int).iloc[offset:]
    assert list(frame["calendar_event"]) == list(expected_events)
    assert len(frame["sell_price"]) == len(frame)
    assert len(frame["promo_flag"]) == len(frame)


def test_price_lookup_is_indexed_not_scanned(m5_sales, m5_calendar, m5_prices):
    """Price must be resolved via a prebuilt index. Filtering the price frame
    per series is quadratic and makes level-12 ingestion unusable."""
    frames, manifest = to_long_frames(m5_sales, m5_calendar, m5_prices, level=12)
    assert manifest["has_price"].all()
    for frame in frames.values():
        assert frame["sell_price"].notna().all()


def test_all_twelve_levels_run(m5_sales, m5_calendar, m5_prices):
    for level in AGGREGATION_LEVELS:
        frames, manifest = to_long_frames(m5_sales, m5_calendar, m5_prices, level=level)
        assert frames, f"level {level} produced no series"
        assert len(manifest) == len(frames)


# ── Metrics ──────────────────────────────────────────────────────────────────

def test_seasonal_naive_mae_known_value():
    # Perfectly periodic series -> seasonal-naive has zero error -> guarded to 1e-6.
    periodic = np.array([1.0, 2, 3, 1, 2, 3, 1, 2, 3])
    assert seasonal_naive_mae(periodic, season=3) == pytest.approx(1e-6)
    # Linear ramp, season 1 -> every step differs by exactly 1.
    assert seasonal_naive_mae(np.arange(10, dtype=float), season=1) == pytest.approx(1.0)


def test_mase_is_one_when_error_equals_denominator():
    actual = np.array([10.0, 10, 10])
    predicted = np.array([12.0, 12, 12])
    assert mase(actual, predicted, denom=2.0) == pytest.approx(1.0)
    assert mase(actual, predicted, denom=4.0) == pytest.approx(0.5)


def test_robust_scale_matches_tukey_fence():
    values = [1.0, 2.0, 3.0, 4.0]
    q1, q3 = np.percentile(values, [25, 75])
    assert robust_scale(values) == pytest.approx(q3 + 1.5 * (q3 - q1))


def test_robust_scale_ignores_outlier_dominance():
    """The property the production normaliser exists for: one catastrophic
    model must not set the scale for everyone else."""
    normal = [100.0, 110, 120, 130]
    with_outlier = normal + [8.23e8]
    assert robust_scale(with_outlier) < 1000      # max-normalisation would give 8.23e8


def test_robust_scale_handles_empty_and_nonfinite():
    assert robust_scale([]) == 1e-6
    assert robust_scale([np.inf, np.nan]) == 1e-6


# ── Composite recomputation ──────────────────────────────────────────────────

def _record(model, cv_rmse, train_time=0.01, pred_time=0.01, **kw):
    base = {
        "model": model, "ok": True, "status": "success",
        "cv_rmse": cv_rmse, "cv_mae": cv_rmse * 0.8, "cv_mape": 0.1,
        "cv_r2": 0.5, "cv_score": 0.0, "stability": 1.0,
        "train_time": train_time, "pred_time": pred_time,
        "test_mase": 1.0, "test_rmse": cv_rmse, "forecast": [1.0],
    }
    base.update(kw)
    return base


def test_recompute_composite_ranks_by_accuracy_when_costs_equal():
    records = [_record("A", 100.0), _record("B", 200.0), _record("C", 300.0)]
    scores = recompute_composite(records)
    assert scores["A"] < scores["B"] < scores["C"]


def test_speed_weight_can_override_accuracy():
    """Reproduces the mechanism behind the paper's headline result: when
    candidate accuracies are nearly tied, the 0.10-weighted speed term decides
    the winner. Setting speed to zero must hand it back to the accurate model."""
    records = [
        _record("Accurate", 100.0, train_time=2.0, pred_time=0.01),
        _record("Fast", 105.0, train_time=0.0001, pred_time=0.01),
    ]
    shipped = recompute_composite(records, DEFAULT_WEIGHTS)
    assert min(shipped, key=lambda m: shipped[m]) == "Fast"

    no_speed = recompute_composite(records, {**DEFAULT_WEIGHTS, "speed": 0.0})
    assert min(no_speed, key=lambda m: no_speed[m]) == "Accurate"


def test_recompute_composite_skips_failed_models():
    records = [_record("A", 100.0), {**_record("B", 1.0), "ok": False}]
    assert set(recompute_composite(records)) == {"A"}


def test_recompute_composite_empty_input():
    assert recompute_composite([]) == {}
    assert recompute_composite([{**_record("A", 1.0), "ok": False}]) == {}


# ── Policies ─────────────────────────────────────────────────────────────────

def _series_result(records, horizon=3, actual=(10.0, 10.0, 10.0)):
    from benchmarks.selection_benchmark import SeriesResult
    return SeriesResult(
        series_id="s1", n_obs=50, horizon=horizon, season=7, mase_denom=2.0,
        test_actual=list(actual), records=records,
    )


def test_oracle_is_the_lower_bound():
    records = [
        _record("A", 100.0, test_mase=0.9),
        _record("B", 200.0, test_mase=0.4),
        _record("C", 300.0, test_mase=1.5),
    ]
    result = _series_result(records)
    policies = apply_policies(result, np.array([10.0] * 20))

    assert policies["oracle"]["model"] == "B"
    model_policies = [p for p in policies
                      if p not in ("naive", "seasonal_naive", "oracle")]
    for policy in model_policies:
        assert policies["oracle"]["mase"] <= policies[policy]["mase"] + 1e-12


def test_rmse_only_and_composite_can_disagree():
    records = [
        _record("Accurate", 100.0, train_time=2.0, test_mase=0.5),
        _record("Fast", 105.0, train_time=0.0001, test_mase=0.8),
    ]
    policies = apply_policies(_series_result(records), np.array([10.0] * 20))
    assert policies["rmse_only"]["model"] == "Accurate"
    assert policies["composite"]["model"] == "Fast"
    # And the disagreement is visible in realized error, which is the point.
    assert policies["composite"]["mase"] > policies["rmse_only"]["mase"]


def test_fixed_model_policies_resolve_when_present():
    records = [_record("XGBoost", 100.0), _record("KNN", 120.0)]
    policies = apply_policies(_series_result(records), np.array([10.0] * 20),
                              fixed_models=("XGBoost", "Prophet"))
    assert policies["always_XGBoost"]["model"] == "XGBoost"
    assert "always_Prophet" not in policies      # absent, not silently substituted


def test_naive_baselines_are_always_available():
    policies = apply_policies(_series_result([_record("A", 1.0)]),
                              np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))
    assert "naive" in policies and "seasonal_naive" in policies
    assert np.isfinite(policies["naive"]["mase"])


def test_apply_policies_with_no_usable_models():
    records = [{**_record("A", 1.0), "ok": False}]
    assert apply_policies(_series_result(records), np.array([1.0] * 10)) == {}


# ── Held-out evaluation ──────────────────────────────────────────────────────

def _fake_runner_factory(forecasts: dict[str, float]):
    """A runner that returns one constant-valued forecast per named model."""
    def _runner(series, horizon, skip_dl):
        results = []
        for i, (name, value) in enumerate(forecasts.items()):
            results.append({
                "model_name": name, "status": "success",
                "rmse": 10.0 + i, "mae": 8.0 + i, "mape": 0.1, "r2": 0.5,
                "adjusted_score": 0.5 + i * 0.1, "stability_penalty": 1.0,
                "train_time_sec": 0.01, "pred_time_sec": 0.01, "elapsed_sec": 0.02,
                "forecast": [value] * horizon,
            })
        return results, {"dl_skipped": False}
    return _runner


def test_evaluate_series_never_shows_models_the_test_window():
    """The core methodological guarantee: the runner must receive exactly the
    series minus the held-out window."""
    seen = {}

    def _spy(series, horizon, skip_dl):
        seen["n"] = len(series)
        seen["last"] = float(series.iloc[-1])
        return _fake_runner_factory({"M": 5.0})(series, horizon, skip_dl)

    values = np.arange(100, dtype=float)
    series = pd.Series(values, index=pd.date_range("2023-01-01", periods=100, freq="D"))

    result = evaluate_series(series, "s", horizon=10, runner=_spy)

    assert seen["n"] == 90                    # 100 - 10 held out
    assert seen["last"] == 89.0               # last training point, not 99
    assert result.test_actual == [float(v) for v in values[-10:]]


def test_evaluate_series_scores_against_held_out_truth():
    series = pd.Series(
        np.full(60, 10.0), index=pd.date_range("2023-01-01", periods=60, freq="D"),
    )
    # Constant series -> seasonal-naive denominator is the 1e-6 guard.
    result = evaluate_series(
        series, "s", horizon=5, runner=_fake_runner_factory({"Exact": 10.0, "Off": 20.0}),
    )
    by_model = {r["model"]: r for r in result.records}
    assert by_model["Exact"]["test_rmse"] == pytest.approx(0.0)
    assert by_model["Off"]["test_rmse"] == pytest.approx(10.0)
    assert by_model["Exact"]["test_mase"] < by_model["Off"]["test_mase"]


def test_evaluate_series_rejects_too_short_series():
    series = pd.Series(np.arange(12, dtype=float),
                       index=pd.date_range("2023-01-01", periods=12, freq="D"))
    with pytest.raises(ValueError, match="too short"):
        evaluate_series(series, "s", horizon=10, runner=_fake_runner_factory({"M": 1.0}))


def test_evaluate_series_marks_nonfinite_forecasts_unusable():
    def _bad_runner(series, horizon, skip_dl):
        return [{
            "model_name": "Broken", "status": "success", "rmse": 1.0, "mae": 1.0,
            "mape": 0.1, "r2": 0.5, "adjusted_score": 0.1, "stability_penalty": 1.0,
            "train_time_sec": 0.01, "pred_time_sec": 0.01,
            "forecast": [float("nan")] * horizon,
        }], {"dl_skipped": False}

    series = pd.Series(np.arange(60, dtype=float),
                       index=pd.date_range("2023-01-01", periods=60, freq="D"))
    result = evaluate_series(series, "s", horizon=5, runner=_bad_runner)
    assert result.records[0]["ok"] is False
    assert not np.isfinite(result.records[0]["test_mase"])


# ── End-to-end aggregation and statistics ────────────────────────────────────

def _many_series(n=12):
    rng = np.random.RandomState(1)
    out = []
    for i in range(n):
        values = 100 + 10 * np.sin(np.arange(80) * 2 * np.pi / 7) + rng.randn(80)
        out.append((f"s{i}", pd.Series(
            values, index=pd.date_range("2023-01-01", periods=80, freq="D"))))
    return out


def test_run_benchmark_produces_paired_policy_rows():
    df, raw = run_benchmark(
        _many_series(), horizon=7,
        runner=_fake_runner_factory({"A": 100.0, "B": 105.0, "C": 95.0}),
        fixed_models=("A",),
    )
    assert not df.empty
    assert len(raw) == 12
    expected = {"composite", "rmse_only", "mae_only", "always_A",
                "naive", "seasonal_naive", "oracle"}
    assert expected <= set(df["policy"].unique())
    # Every policy must be present for every series, or the paired tests break.
    counts = df.groupby("policy")["series_id"].nunique()
    assert (counts == 12).all()


def test_summarize_reports_regret_against_oracle():
    from benchmarks.selection_benchmark import REFERENCE_POLICIES

    df, _ = run_benchmark(
        _many_series(), horizon=7,
        runner=_fake_runner_factory({"A": 100.0, "B": 105.0, "C": 95.0}),
    )
    summary = summarize(df).set_index("policy")

    assert summary.loc["oracle", "mean_regret_vs_oracle"] == pytest.approx(0.0)

    # Regret is only defined for policies that select from the trained pool.
    selection = summary.drop(index=list(REFERENCE_POLICIES), errors="ignore")
    assert (selection["mean_regret_vs_oracle"] >= -1e-9).all()

    # Reference forecasts get NaN rather than a misleading negative regret.
    for policy in REFERENCE_POLICIES:
        if policy in summary.index:
            assert np.isnan(summary.loc[policy, "mean_regret_vs_oracle"])


def test_summarize_reports_beats_naive_share():
    """The honest headline number: MASE < 1 means the forecast beat the
    seasonal-naive benchmark. A low share is itself the finding."""
    df, _ = run_benchmark(
        _many_series(), horizon=7,
        runner=_fake_runner_factory({"A": 100.0, "B": 105.0, "C": 95.0}),
    )
    summary = summarize(df)
    assert "pct_beats_naive" in summary.columns
    assert (summary["pct_beats_naive"].between(0.0, 100.0)).all()


def test_compare_policies_runs_full_demsar_protocol():
    df, _ = run_benchmark(
        _many_series(), horizon=7,
        runner=_fake_runner_factory({"A": 100.0, "B": 105.0, "C": 95.0}),
    )
    out = compare_policies(df)
    assert "error" not in out
    assert out["n_series"] == 12
    assert 0.0 <= out["friedman_p"] <= 1.0
    assert set(out["average_ranks"]) == set(df["policy"].unique())
    # Holm correction must be monotone non-decreasing and never below raw p.
    ps = [r["p_holm"] for r in out["pairwise"]]
    assert ps == sorted(ps)
    assert all(r["p_holm"] >= r["p_raw"] - 1e-12 for r in out["pairwise"])


def test_compare_policies_needs_enough_series():
    df = pd.DataFrame([
        {"series_id": "a", "policy": "x", "mase": 1.0},
        {"series_id": "a", "policy": "y", "mase": 2.0},
    ])
    assert "error" in compare_policies(df)


def test_run_benchmark_n_jobs_one_is_the_sequential_path():
    """n_jobs=1 must keep accepting an injected runner. Worker processes
    re-import this module and cannot receive closures, so the sequential path is
    the only one custom runners can use."""
    df, raw = run_benchmark(
        _many_series(4), horizon=7,
        runner=_fake_runner_factory({"A": 100.0, "B": 95.0}), n_jobs=1,
    )
    assert len(raw) == 4
    assert not df.empty


def test_parallel_worker_shrinks_internal_thread_pool(monkeypatch):
    """n_jobs * threads_per_series must stay near the core count, or parallel
    series oversubscribe the CPU and the speedup is eaten by context switching."""
    import importlib
    import types

    # A stand-in avoids pulling torch/statsmodels/prophet in just to check an
    # integer. Both the sys.modules entry AND the package attribute must be
    # patched: `_worker_init` does `from services import model_runner`, which
    # reads the attribute off the already-imported package when another test
    # has loaded the real module first, and only falls through to sys.modules
    # otherwise. Patching one alone makes this test order-dependent.
    fake_module = types.SimpleNamespace(MAX_WORKERS=8)
    monkeypatch.setitem(sys.modules, "services.model_runner", fake_module)
    monkeypatch.setattr(importlib.import_module("services"), "model_runner",
                        fake_module, raising=False)

    from benchmarks.selection_benchmark import _worker_init
    _worker_init(3)
    assert fake_module.MAX_WORKERS == 3

    _worker_init(1)          # never drop below 2 — one worker serialises the run
    assert fake_module.MAX_WORKERS == 2


def test_evaluate_one_returns_errors_instead_of_raising():
    """One unusable series must not abort the whole pool, mirroring how
    model_runner isolates a single failing model."""
    from benchmarks.selection_benchmark import _evaluate_one

    too_short = pd.Series(np.arange(12, dtype=float),
                          index=pd.date_range("2023-01-01", periods=12, freq="D"))
    series_id, result, error = _evaluate_one(("bad", too_short, 10, 7, True))
    assert series_id == "bad"
    assert result is None
    assert "too short" in error


def _realistic_registry_records():
    """
    19 models with the measured M5 level-9 cost profile.

    The composition is what matters: only 3 of 19 models are slow
    (SARIMA/SARIMAX/ARIMA grid searches at ~105-120s), so BOTH quartiles of the
    training-time distribution land among the fast models, the IQR is tiny, and
    the Tukey fence ends up far below the slow models' actual cost. Their
    normalized speed then runs to double digits. A fixture with a higher
    slow-model share does NOT reproduce this — Q3 rises into the slow group and
    the fence grows with it.

    Accuracy is deliberately realistic too: RMSE spans only ~1.25x across the
    whole registry, and MAPE/R2 use the ranges actually observed in the
    project's own results table (MAPE 2.9-4.0, R2 -0.41 to 0.03). Every term
    varies, so none is silently excluded from the influence comparison.
    """
    specs = [  # name, cv_rmse, train_s, pred_s, mape, r2, stability
        ("SARIMA",               1100.0, 118.0,   1.10,    3.377, -0.048, 0.92),
        ("SARIMAX",              1140.0, 112.0,   1.09,    3.251, -0.413, 1.04),
        ("ARIMA",                1130.0, 105.0,   4.16,    3.517, -0.014, 0.97),
        ("SVM",                  1125.0,   0.0016, 0.0006, 3.213,  0.003, 1.31),
        ("ExponentialSmoothing", 1129.0,   0.0014, 0.0006, 3.368, -0.011, 1.28),
        ("CrostonSBA",           1141.0,   0.0002, 0.00001, 3.349, -0.036, 1.33),
        ("HoltWinters",          1145.0,   0.1764, 0.0043, 3.397, -0.040, 1.19),
        ("MovingAverage",        1158.0,   0.00005, 0.0004, 3.400, -0.062, 1.35),
        ("KNN",                  1162.0,   0.0001, 0.2010, 3.002, -0.078, 1.30),
        ("Prophet",              1215.0,   0.3744, 0.0080, 2.944, -0.168, 1.02),
        ("XGBoost",              1250.0,   0.4200, 0.0081, 3.246,  0.026, 1.11),
        ("RandomForest",         1270.0,   0.4800, 0.3019, 3.995, -0.221, 1.16),
        ("CNN1D",                1340.0,   0.2500, 0.0400, 3.610, -0.290, 1.22),
        ("TCN",                  1345.0,   0.3000, 0.0500, 3.640, -0.300, 1.24),
        ("RNN",                  1350.0,   0.3500, 0.0500, 3.670, -0.310, 1.26),
        ("GRU",                  1355.0,   0.3800, 0.0500, 3.700, -0.320, 1.27),
        ("LSTM",                 1360.0,   0.4000, 0.0500, 3.730, -0.330, 1.29),
        ("BiLSTM",               1370.0,   0.4500, 0.0600, 3.760, -0.340, 1.32),
        ("Transformer",          1380.0,   0.5000, 0.0600, 3.790, -0.350, 1.34),
    ]
    return [
        _record(n, rmse, train_time=t, pred_time=p,
                cv_mape=mape, cv_r2=r2, stability=stab)
        for n, rmse, t, p, mape, r2, stab in specs
    ]


def test_raw_scheme_is_the_default_and_unchanged():
    """The 'raw' scheme must keep reproducing the shipped model_runner.py
    scoring exactly, or every previously reported number silently shifts."""
    records = [
        _record("A", 100.0, train_time=2.0),
        _record("B", 105.0, train_time=0.001),
    ]
    assert recompute_composite(records) == recompute_composite(records, scheme="raw")


def test_clipped_scheme_bounds_every_term_at_one():
    """The defect being fixed: a fence bounds the denominator, not the ratio,
    so normalized speed reached 43.6 on real data while normalized RMSE spanned
    0.48-1.03. Under 'clipped' nothing may exceed 1.0."""
    from benchmarks.selection_benchmark import decompose_composite_terms

    result = _series_result(_realistic_registry_records())

    raw_terms = decompose_composite_terms([result], scheme="raw")
    clipped_terms = decompose_composite_terms([result], scheme="clipped")

    # Measured on real data: speed reached 43.6 while rmse spanned 0.48-1.03.
    raw_speed = raw_terms[raw_terms["term"] == "speed"]["normalized_value"]
    raw_rmse = raw_terms[raw_terms["term"] == "rmse"]["normalized_value"]
    assert raw_speed.max() > 5.0, "fixture should reproduce runaway speed"
    assert (raw_speed.max() - raw_speed.min()) > 10 * (raw_rmse.max() - raw_rmse.min())

    assert clipped_terms["normalized_value"].max() <= 1.0 + 1e-12
    assert clipped_terms["normalized_value"].min() >= 0.0

    # 'balanced' additionally makes every term span exactly [0, 1].
    balanced_terms = decompose_composite_terms([result], scheme="balanced")
    for term in DEFAULT_WEIGHTS:
        vals = balanced_terms[balanced_terms["term"] == term]["normalized_value"]
        assert vals.min() == pytest.approx(0.0)
        assert vals.max() == pytest.approx(1.0)


def test_balanced_scheme_preserves_ranking_within_each_term():
    """Rescaling may change term SCALES but must never reorder models within a
    term — that would change which model is better on that criterion."""
    from benchmarks.selection_benchmark import decompose_composite_terms

    result = _series_result(_realistic_registry_records())
    clipped = decompose_composite_terms([result], scheme="clipped")
    balanced = decompose_composite_terms([result], scheme="balanced")

    for term in DEFAULT_WEIGHTS:
        c = clipped[clipped["term"] == term].set_index("model")["normalized_value"]
        b = balanced[balanced["term"] == term].set_index("model")["normalized_value"]
        assert list(c.sort_values().index) == list(b.reindex(c.index).sort_values().index), (
            f"{term}: rescaling reordered models")


def test_clipped_scheme_rebalances_effective_influence():
    """Under 'raw', speed out-influences rmse despite a quarter of the weight
    (measured: 88.3% vs 3.5% effective share on real M5 data). Clipping must
    restore rmse to the dominant position its 0.40 weight implies."""
    from benchmarks.selection_benchmark import (
        decompose_composite_terms, summarize_term_dominance,
    )

    result = _series_result(_realistic_registry_records())

    def share(scheme):
        return summarize_term_dominance(
            decompose_composite_terms([result], scheme=scheme)
        ).set_index("term")["effective_share"]

    raw_share, clipped_share, balanced_share = share("raw"), share("clipped"), share("balanced")

    # The defect: speed out-influences rmse at a quarter of the weight.
    assert raw_share["speed"] > raw_share["rmse"], "fixture should reproduce the defect"

    # Capping bounds the runaway but does NOT by itself restore accuracy's
    # priority — the terms still occupy different fractions of [0, 1].
    assert clipped_share["speed"] < raw_share["speed"]

    # Rescaling does: effective influence converges on nominal weight share.
    assert balanced_share["rmse"] > balanced_share["speed"], "balanced did not rebalance"
    total_weight = sum(abs(w) for w in DEFAULT_WEIGHTS.values())
    for term, weight in DEFAULT_WEIGHTS.items():
        assert balanced_share[term] == pytest.approx(abs(weight) / total_weight, abs=1e-6), (
            f"{term}: effective share should equal its nominal weight share")


def test_clipped_scheme_normalizes_r2_which_raw_leaves_unscaled():
    """r2 is the one term the shipped formula never divides by a fence — five
    normalized quantities summed with one raw one. Clipped must fix that."""
    from benchmarks.selection_benchmark import decompose_composite_terms

    records = [
        _record("A", 100.0, cv_r2=-3.5),      # very negative R2 is valid on noisy data
        _record("B", 110.0, cv_r2=0.4),
        _record("C", 120.0, cv_r2=0.1),
    ]
    result = _series_result(records)

    raw_r2 = decompose_composite_terms([result], scheme="raw")
    raw_r2 = raw_r2[raw_r2["term"] == "r2"]["normalized_value"]
    clipped_r2 = decompose_composite_terms([result], scheme="clipped")
    clipped_r2 = clipped_r2[clipped_r2["term"] == "r2"]["normalized_value"]

    assert raw_r2.min() < 0, "raw scheme should pass R2 through unscaled and signed"
    assert clipped_r2.min() >= 0.0 and clipped_r2.max() <= 1.0
    # Ordering must be preserved: higher R2 stays higher after shifting/scaling.
    assert clipped_r2.iloc[0] < clipped_r2.iloc[2] < clipped_r2.iloc[1]


def test_compare_normalization_is_paired_and_isolates_scaling():
    """Only the scaling may vary between schemes — same series, same candidate
    pool, same held-out data — so any MASE difference is attributable to it."""
    from benchmarks.selection_benchmark import compare_normalization_schemes

    records = [
        _record("SARIMA", 1100.0, train_time=118.0, pred_time=1.10, test_mase=0.80),
        _record("XGBoost", 1250.0, train_time=60.0, pred_time=0.01, test_mase=1.10),
        _record("KNN", 1300.0, train_time=0.0001, pred_time=0.20, test_mase=1.30),
    ]
    per_series, summary = compare_normalization_schemes([_series_result(records)])

    from benchmarks.selection_benchmark import NORMALIZATION_SCHEMES

    assert len(per_series) == 1
    assert set(summary["scheme"]) == set(NORMALIZATION_SCHEMES)
    # The oracle floor is scheme-independent and must bound every scheme.
    assert per_series["oracle_mase"].iloc[0] == pytest.approx(0.80)
    for scheme in NORMALIZATION_SCHEMES:
        assert per_series[f"{scheme}_mase"].iloc[0] >= per_series["oracle_mase"].iloc[0]


def test_compare_normalization_handles_no_usable_models():
    from benchmarks.selection_benchmark import compare_normalization_schemes

    result = _series_result([{**_record("A", 1.0), "ok": False}])
    per_series, summary = compare_normalization_schemes([result])
    assert per_series.empty and summary.empty


def test_invalid_scheme_rejected():
    with pytest.raises(ValueError, match="scheme must be one of"):
        recompute_composite([_record("A", 1.0)], scheme="nonsense")


def test_decomposed_terms_sum_to_the_composite_score():
    """The decomposition must reconstruct the score exactly, or the diagnostic
    is describing a different formula than the one selecting models."""
    from benchmarks.selection_benchmark import decompose_composite_terms

    records = [
        _record("Slow", 100.0, train_time=120.0, pred_time=1.0),
        _record("Fast", 105.0, train_time=0.0001, pred_time=0.001),
        _record("Mid", 110.0, train_time=1.0, pred_time=0.05),
    ]
    result = _series_result(records)
    scores = recompute_composite(records)
    decomposed = decompose_composite_terms([result])

    for model, expected in scores.items():
        got = decomposed[decomposed["model"] == model]["weighted_contribution"].sum()
        assert got == pytest.approx(expected), f"{model} decomposition != score"

    # Same guarantee must hold under the clipped scheme.
    clipped_scores = recompute_composite(records, scheme="clipped")
    clipped_decomposed = decompose_composite_terms([result], scheme="clipped")
    for model, expected in clipped_scores.items():
        got = clipped_decomposed[
            clipped_decomposed["model"] == model]["weighted_contribution"].sum()
        assert got == pytest.approx(expected), f"{model} clipped decomposition != score"


def test_term_dominance_detects_scale_mismatch():
    """Reproduces the real-data condition: RMSE spans a factor of ~1.2 across
    models while train time spans six orders of magnitude. Speed should then
    dominate selection despite carrying a quarter of RMSE's nominal weight."""
    from benchmarks.selection_benchmark import (
        decompose_composite_terms, summarize_term_dominance,
    )

    records = [
        _record("SARIMA", 1100.0, train_time=120.0, pred_time=1.0),
        _record("XGBoost", 1250.0, train_time=60.0, pred_time=0.01),
        _record("KNN", 1300.0, train_time=0.0001, pred_time=0.2),
        _record("SVM", 1280.0, train_time=0.002, pred_time=0.001),
        _record("MovingAverage", 1290.0, train_time=0.00005, pred_time=0.0004),
    ]
    dominance = summarize_term_dominance(
        decompose_composite_terms([_series_result(records)])
    ).set_index("term")

    # Nominal weights say accuracy dominates 4:1 over speed.
    assert dominance.loc["rmse", "weight"] == 0.40
    assert dominance.loc["speed", "weight"] == 0.10
    # The data says otherwise: speed creates the larger real spread.
    assert (dominance.loc["speed", "mean_contribution_spread"]
            > dominance.loc["rmse", "mean_contribution_spread"]), (
        "speed should out-influence rmse under six-orders-of-magnitude time spread")
    assert dominance["effective_share"].sum() == pytest.approx(1.0)


def test_term_dominance_no_spread_means_no_influence():
    """A term identical across every candidate cannot change the winner at any
    weight — its effective share must be ~0 even though its weight is 0.40."""
    from benchmarks.selection_benchmark import (
        decompose_composite_terms, summarize_term_dominance,
    )

    records = [
        _record("A", 100.0, train_time=0.1),
        _record("B", 100.0, train_time=5.0),
        _record("C", 100.0, train_time=50.0),
    ]
    dominance = summarize_term_dominance(
        decompose_composite_terms([_series_result(records)])
    ).set_index("term")

    assert dominance.loc["rmse", "mean_contribution_spread"] == pytest.approx(0.0)
    assert dominance.loc["rmse", "effective_share"] == pytest.approx(0.0)
    assert dominance.loc["speed", "mean_contribution_spread"] > 0.0


def test_explain_vs_isolates_the_deciding_term():
    """SARIMA has the best RMSE but the worst train time. The explainer must
    name speed — not RMSE — as the reason it lost."""
    from benchmarks.selection_benchmark import (
        explain_pick_vs_alternative, summarize_explanation,
    )

    # Same realistic spread as the dominance test: a two-model fixture is not
    # enough, because the Tukey fence needs a populated distribution before the
    # normalized ranges diverge the way they do on real data.
    records = [
        _record("SARIMA", 1100.0, train_time=120.0, pred_time=1.0),
        _record("XGBoost", 1250.0, train_time=60.0, pred_time=0.01),
        _record("KNN", 1300.0, train_time=0.0001, pred_time=0.2),
        _record("SVM", 1280.0, train_time=0.002, pred_time=0.001),
        _record("MovingAverage", 1290.0, train_time=0.00005, pred_time=0.0004),
    ]
    explained = explain_pick_vs_alternative([_series_result(records)], "SARIMA")

    assert not explained.empty, "SARIMA should have lost despite the best RMSE"
    assert "SARIMA" not in explained["picked_model"].unique()

    summary = summarize_explanation(explained)
    assert summary.iloc[0]["term"] == "speed"
    assert summary.iloc[0]["pct_series_biggest_reason"] == pytest.approx(100.0)
    # SARIMA genuinely wins on accuracy — its RMSE and MAE gaps must be negative.
    by_term = summary.set_index("term")
    assert by_term.loc["rmse", "mean_gap"] < 0
    assert by_term.loc["mae", "mean_gap"] < 0
    # And it loses on speed by more than it wins on accuracy combined.
    assert by_term.loc["speed", "mean_gap"] > abs(
        by_term.loc["rmse", "mean_gap"] + by_term.loc["mae", "mean_gap"])


def test_explain_vs_respects_the_normalization_scheme():
    """The scale-mismatch fix closed only ~1% of the gap on real data, so the
    explainer must be able to ask 'why does it STILL lose' under 'balanced' —
    a different term can dominate once speed is no longer running away."""
    from benchmarks.selection_benchmark import (
        explain_pick_vs_alternative, summarize_explanation,
    )

    result = _series_result(_realistic_registry_records())

    raw_expl = explain_pick_vs_alternative([result], "SARIMA", scheme="raw")
    bal_expl = explain_pick_vs_alternative([result], "SARIMA", scheme="balanced")

    assert not raw_expl.empty
    assert summarize_explanation(raw_expl).iloc[0]["term"] == "speed"

    # Under 'balanced' the deciding term must no longer be speed-by-runaway;
    # whatever it is, the decomposition must still be internally consistent.
    if not bal_expl.empty:
        bal_summary = summarize_explanation(bal_expl)
        assert set(bal_summary["term"]) == set(DEFAULT_WEIGHTS)
        assert bal_summary["pct_series_biggest_reason"].sum() == pytest.approx(100.0)


def test_explain_vs_skips_series_where_it_was_already_picked():
    from benchmarks.selection_benchmark import explain_pick_vs_alternative

    records = [
        _record("Best", 100.0, train_time=0.001),
        _record("Worse", 500.0, train_time=90.0),
    ]
    assert explain_pick_vs_alternative([_series_result(records)], "Best").empty


def test_explain_vs_unknown_model_is_empty_not_an_error():
    from benchmarks.selection_benchmark import explain_pick_vs_alternative

    result = _series_result([_record("A", 100.0), _record("B", 120.0)])
    assert explain_pick_vs_alternative([result], "NotAModel").empty


def _varied_raw(n=30, seed=0):
    """Series whose model rankings genuinely differ, so configuration selection
    has something to discriminate between."""
    rng = np.random.RandomState(seed)
    out = []
    for i in range(n):
        records = []
        for r in _realistic_registry_records():
            rec = dict(r)
            rec["cv_rmse"] *= 1.0 + rng.randn() * 0.05
            rec["cv_mae"] = rec["cv_rmse"] * 0.78
            rec["cv_mape"] *= 1.0 + rng.randn() * 0.05
            rec["test_mase"] = 0.9 + rng.rand() * 0.6
            records.append(rec)
        out.append(_series_result(records))
        out[-1].series_id = f"s{i}"
    return out


def test_rmse_pure_config_reproduces_the_baseline_exactly():
    """A configuration weighting only RMSE must select exactly what rmse_only
    selects. If it doesn't, the configuration machinery is broken and every
    other comparison built on it is meaningless."""
    from benchmarks.selection_benchmark import REFERENCE_CONFIG, evaluate_configs

    evaluated = evaluate_configs(_varied_raw(12))
    wide = evaluated.pivot(index="series_id", columns="config", values="model")
    assert (wide["rmse_pure"] == wide[REFERENCE_CONFIG]).all(), (
        "rmse_pure must reproduce the single-metric baseline")


def test_rmse_mae_config_may_diverge_from_the_baseline():
    """ONLY rmse_pure is required to reproduce the RMSE-only baseline.

    rmse_mae weights RMSE and MAE together, so it must be free to disagree
    whenever those two metrics rank models differently — which they do on real
    data. Asserting otherwise would be wrong, and it would be easy to believe
    because the other fixtures here set cv_mae = cv_rmse * 0.78, a perfect
    linear relationship under which the two can never disagree.
    """
    from benchmarks.selection_benchmark import REFERENCE_CONFIG, evaluate_configs

    # A wins on RMSE but is worst on MAE; B is a strong compromise.
    records = [
        _record("A", 100.0, cv_mae=100.0, train_time=0.1),
        _record("B", 102.0, cv_mae=50.0, train_time=0.1),
        _record("C", 104.0, cv_mae=40.0, train_time=0.1),
    ]
    result = _series_result(records)
    picks = evaluate_configs([result]).set_index("config")["model"]

    assert picks["rmse_pure"] == "A" == picks[REFERENCE_CONFIG], (
        "rmse_pure must always reproduce the RMSE-only baseline")
    assert picks["rmse_mae"] == "B", (
        "rmse_mae should follow the combined RMSE+MAE optimum, not RMSE alone")


def test_evaluate_configs_includes_reference_and_oracle():
    from benchmarks.selection_benchmark import (
        CANDIDATE_CONFIGS, REFERENCE_CONFIG, evaluate_configs,
    )

    evaluated = evaluate_configs(_varied_raw(8))
    configs = set(evaluated["config"].unique())
    assert set(CANDIDATE_CONFIGS) <= configs
    assert REFERENCE_CONFIG in configs and "oracle" in configs

    # The oracle is a floor: no configuration may beat it on any series.
    wide = evaluated.pivot(index="series_id", columns="config", values="mase")
    for config in CANDIDATE_CONFIGS:
        assert (wide[config] >= wide["oracle"] - 1e-12).all()


def test_nested_selection_never_scores_a_config_on_its_own_fold():
    """The whole point: a configuration is chosen from training folds only. If
    selection could see the test fold, the estimate would be inflated exactly
    the way selecting a model on its own CV folds is."""
    from benchmarks.selection_benchmark import evaluate_configs, nested_config_selection

    raw = _varied_raw(30)
    per_fold, summary = nested_config_selection(raw, n_folds=5, seed=0)

    assert len(per_fold) == 5
    assert per_fold["n_test"].sum() == 30
    assert (per_fold["n_train"] + per_fold["n_test"] == 30).all()

    # Reconstruct fold 0's choice from the training folds and confirm it matches.
    evaluated = evaluate_configs(raw)
    wide = evaluated.pivot(index="series_id", columns="config", values="mase")
    selectable = [c for c in wide.columns if c not in ("oracle", "reference_rmse_only")]
    rng = np.random.RandomState(0)
    folds = np.array_split(rng.permutation(len(wide)), 5)
    test_ids = np.array(wide.index)[folds[0]]
    expected = wide.drop(index=test_ids)[selectable].mean().idxmin()
    assert per_fold.iloc[0]["selected_config"] == expected


def test_nested_estimate_is_not_better_than_in_sample_best():
    """Selecting in-sample is optimistic by construction; the nested estimate
    must expose that rather than hide it."""
    from benchmarks.selection_benchmark import nested_config_selection

    per_fold, summary = nested_config_selection(_varied_raw(30), n_folds=5, seed=1)
    best_in_sample = summary.iloc[0]["in_sample_mean_mase"]
    assert summary.attrs["nested_test_mase"] >= best_in_sample - 1e-9
    assert summary.attrs["optimism_bias"] >= -1e-9


def test_nested_selection_reports_stability_and_baselines():
    from benchmarks.selection_benchmark import nested_config_selection

    per_fold, summary = nested_config_selection(_varied_raw(25), n_folds=5, seed=2)
    a = summary.attrs
    assert sum(a["selection_stability"].values()) == 5
    assert a["nested_oracle_mase"] <= a["nested_test_mase"] + 1e-9
    assert a["nested_oracle_mase"] <= a["nested_reference_mase"] + 1e-9


def test_nested_selection_handles_too_few_series():
    from benchmarks.selection_benchmark import nested_config_selection

    per_fold, summary = nested_config_selection(_varied_raw(3), n_folds=5)
    assert per_fold.empty and summary.empty


def test_candidate_configs_weights_are_complete():
    """Every configuration must specify all six weights, or recompute_composite
    silently falls back to the shipped default for the missing ones."""
    from benchmarks.selection_benchmark import CANDIDATE_CONFIGS, NORMALIZATION_SCHEMES

    for name, cfg in CANDIDATE_CONFIGS.items():
        assert set(cfg["weights"]) == set(DEFAULT_WEIGHTS), f"{name}: incomplete weights"
        assert cfg["scheme"] in NORMALIZATION_SCHEMES, f"{name}: bad scheme"


def test_budgets_compete_as_selectable_candidates():
    """Reading the frontier over every series and quoting its minimum would
    choose a hyperparameter on the evaluation data — the same winner's curse
    corrected elsewhere. Budgets must therefore be candidates the nested
    machinery selects, not a constant reported after the fact."""
    from benchmarks.selection_benchmark import BUDGET_GRID, evaluate_configs

    configs = set(evaluate_configs(_varied_raw(12))["config"].unique())
    for budget in BUDGET_GRID:
        name = ("budget_unlimited" if np.isinf(budget) else f"budget_{budget:g}s")
        assert name in configs, f"{name} missing from candidate set"


def test_budget_candidates_can_be_disabled():
    from benchmarks.selection_benchmark import evaluate_configs

    configs = set(evaluate_configs(_varied_raw(8), budgets=None)["config"].unique())
    assert not any(c.startswith("budget_") for c in configs)


def test_unlimited_budget_candidate_equals_the_baseline():
    """budget_unlimited applies no constraint, so it must reproduce rmse_only
    exactly — a correctness check on the budget policy itself."""
    from benchmarks.selection_benchmark import REFERENCE_CONFIG, evaluate_configs

    picks = evaluate_configs(_varied_raw(15)).pivot(
        index="series_id", columns="config", values="model")
    assert (picks["budget_unlimited"] == picks[REFERENCE_CONFIG]).all()


def test_frontier_and_nested_selection_share_one_policy():
    """If the frontier and the nested estimate described different rules, the
    honest number would not describe the operating point being recommended."""
    from benchmarks.selection_benchmark import (
        constrained_selection_frontier, evaluate_configs,
    )

    raw = _varied_raw(20)
    frontier = constrained_selection_frontier(raw, budgets=[2.0])
    candidates = evaluate_configs(raw)
    via_candidates = candidates[candidates["config"] == "budget_2s"]["mase"].mean()
    assert frontier.iloc[0]["mean_mase"] == pytest.approx(via_candidates)


def test_nested_selection_reports_margin_uncertainty():
    """A margin smaller than the fold-to-fold spread is not a demonstrated
    improvement; the interval must be reported alongside the point estimate."""
    from benchmarks.selection_benchmark import nested_config_selection

    per_fold, summary = nested_config_selection(_varied_raw(40), n_folds=5, seed=0)
    a = summary.attrs

    assert "margin_vs_reference" in per_fold.columns
    lo, hi = a["margin_ci95"]
    assert lo <= a["margin_mean"] <= hi
    assert a["folds_total"] == 5
    assert 0 <= a["folds_won"] <= 5
    # The flag must agree with the interval it summarises.
    assert a["margin_excludes_zero"] == (lo > 0)
    # And the margin must reconcile with the two headline numbers.
    assert a["margin_mean"] == pytest.approx(
        a["nested_reference_mase"] - a["nested_test_mase"])


def test_margin_flag_is_false_when_interval_spans_zero():
    """Guards the specific failure mode: a favourable point estimate reported
    as a win when the folds disagree."""
    from benchmarks.selection_benchmark import nested_config_selection

    _, summary = nested_config_selection(_varied_raw(30), n_folds=5, seed=3)
    a = summary.attrs
    if a["folds_won"] < a["folds_total"]:
        lo, _ = a["margin_ci95"]
        if lo <= 0:
            assert a["margin_excludes_zero"] is False


def test_frontier_budgets_are_reported_in_order():
    """inf sits at the end of BUDGET_GRID, so an unsorted default would print
    the unlimited row before the 60s/120s rows and make the frontier unreadable."""
    from benchmarks.selection_benchmark import constrained_selection_frontier

    budgets = constrained_selection_frontier(_varied_raw(6))["budget_seconds"].to_numpy()
    assert np.all(np.diff(budgets) > 0), f"budgets not ascending: {budgets}"


def test_budget_grid_is_refined_between_one_and_five_seconds():
    """The original grid had a single point between 1s and 5s, so quoting an
    optimum there implied precision the spacing could not support."""
    from benchmarks.selection_benchmark import BUDGET_GRID

    mid = [b for b in BUDGET_GRID if 1.0 <= b <= 5.0]
    assert len(mid) >= 5, f"grid too coarse between 1s and 5s: {mid}"


def test_disagreement_counts_only_genuine_departures():
    """A configuration that always picks what rmse_only picks is a near-copy,
    not a defeated alternative — the report must make that visible."""
    from benchmarks.selection_benchmark import config_disagreement

    disagree = config_disagreement(_varied_raw(20)).set_index("config")

    # rmse_pure is the baseline by construction, so it can never depart from it.
    assert disagree.loc["rmse_pure", "n_differs_from_baseline"] == 0
    assert disagree.loc["rmse_pure", "mean_mase_delta_when_differs"] == 0.0
    # The shipped score picks very differently, so it must depart often.
    assert disagree.loc["shipped", "n_differs_from_baseline"] > 0
    assert (disagree["pct_differs"].between(0.0, 100.0)).all()


def test_constrained_frontier_respects_the_budget():
    """Every selected model must be inside the budget, unless nothing fits — in
    which case the fallback must be counted, not hidden."""
    from benchmarks.selection_benchmark import constrained_selection_frontier

    raw = _varied_raw(15)
    # 1e-05 is below the cheapest model in the fixture (MovingAverage, 5e-05),
    # so nothing fits and every series must fall back.
    frontier = constrained_selection_frontier(raw, budgets=[1e-05, 1.0, float("inf")])

    assert len(frontier) == 3
    tight = frontier[frontier["budget_seconds"] == 1e-05].iloc[0]
    assert tight["n_fallback"] == 15
    # The fallback is the cheapest model, so the budget is still exceeded — that
    # must be reported honestly rather than presented as an achievable point.
    assert tight["max_selected_cost"] > 1e-05
    # A generous budget binds on nobody and must reproduce rmse_only exactly.
    loose = frontier[np.isinf(frontier["budget_seconds"])].iloc[0]
    assert loose["pct_budget_binding"] == pytest.approx(0.0)
    assert loose["n_fallback"] == 0
    # The loosest budget must be best on the SELECTION OBJECTIVE. It need not be
    # best on realized MASE — see test_budget_can_act_as_a_regularizer.
    assert loose["mean_selected_cv_rmse"] <= frontier["mean_selected_cv_rmse"].min() + 1e-9


def test_constrained_frontier_matches_rmse_only_when_unbounded():
    """With no budget, constrained selection IS rmse_only. If these diverge,
    the constraint machinery is altering selection when it should not."""
    from benchmarks.selection_benchmark import (
        REFERENCE_CONFIG, constrained_selection_frontier, evaluate_configs,
    )

    raw = _varied_raw(18)
    unbounded = constrained_selection_frontier(raw, budgets=[float("inf")]).iloc[0]
    baseline = evaluate_configs(raw)
    baseline = baseline[baseline["config"] == REFERENCE_CONFIG]["mase"].mean()
    assert unbounded["mean_mase"] == pytest.approx(baseline)


def test_constrained_frontier_is_monotone_in_the_selection_objective():
    """Relaxing the budget only widens the choice set, so the quantity actually
    being minimised — cross-validated RMSE — can never get worse."""
    from benchmarks.selection_benchmark import constrained_selection_frontier

    frontier = constrained_selection_frontier(
        _varied_raw(25), budgets=[0.01, 1.0, 10.0, 200.0])
    cv = frontier.sort_values("budget_seconds")["mean_selected_cv_rmse"].to_numpy()
    assert np.all(np.diff(cv) <= 1e-9), "selection objective worsened as budget relaxed"


def test_budget_can_act_as_a_regularizer():
    """Realized MASE is NOT required to be monotone in the budget: a model with
    lower cross-validated error can generalize worse, so excluding expensive
    models can improve held-out accuracy. Documented because the natural
    assumption is the opposite, and asserting monotonicity here would encode a
    false property and hide a real effect."""
    from benchmarks.selection_benchmark import constrained_selection_frontier

    frontier = constrained_selection_frontier(
        _varied_raw(25), budgets=[0.01, 1.0, 10.0, 200.0]
    ).sort_values("budget_seconds")

    assert np.all(np.diff(frontier["mean_selected_cv_rmse"]) <= 1e-9)  # objective monotone

    # In this fixture the LOOSEST budget is not the most accurate: allowing the
    # lowest-CV-error models in actively hurts held-out performance. That is the
    # regularization effect, and it is the reason MASE monotonicity must not be
    # asserted anywhere in this module.
    best_budget = frontier.loc[frontier["mean_mase"].idxmin(), "budget_seconds"]
    loosest = frontier["budget_seconds"].max()
    assert best_budget < loosest, "expected a tighter budget to beat the unbounded one"
    assert frontier["mean_mase"].max() > frontier["mean_mase"].min() + 1e-6


def test_penalty_vs_constraint_compares_at_matched_cost():
    """Beating the composite by simply spending more compute would prove
    nothing — the constrained arm must not cost more than the composite."""
    from benchmarks.selection_benchmark import compare_penalty_vs_constraint

    result = compare_penalty_vs_constraint(_varied_raw(20)).set_index("method")
    penalty = result.loc["composite (cost as penalty)"]
    matched = result.loc["constrained (cost as budget, matched)"]

    assert matched["mean_cost_seconds"] <= penalty["mean_cost_seconds"] + 1e-9
    assert np.isfinite(matched["mean_mase"]) and np.isfinite(penalty["mean_mase"])


def test_constrained_handles_no_usable_models():
    from benchmarks.selection_benchmark import (
        compare_penalty_vs_constraint, constrained_selection_frontier,
    )

    result = _series_result([{**_record("A", 1.0), "ok": False}])
    assert constrained_selection_frontier([result]).empty
    assert compare_penalty_vs_constraint([result]).empty


def test_raw_records_round_trip_through_disk(tmp_path):
    """Training produces these records and dominates runtime; every later
    question needs only the records. Losing them costs another full run."""
    from benchmarks.selection_benchmark import load_raw, save_raw

    _, raw = run_benchmark(
        _many_series(5), horizon=7,
        runner=_fake_runner_factory({"A": 100.0, "B": 95.0, "C": 105.0}),
    )
    path = tmp_path / "records.raw.json"
    save_raw(raw, path)
    restored = load_raw(path)

    assert len(restored) == len(raw)
    for before, after in zip(raw, restored):
        assert after.series_id == before.series_id
        assert after.horizon == before.horizon
        assert after.mase_denom == pytest.approx(before.mase_denom)
        assert after.test_actual == pytest.approx(before.test_actual)
        assert after.train_tail == pytest.approx(before.train_tail)
        assert len(after.records) == len(before.records)

    # And the sweep must produce identical results from restored records.
    grid = {"speed": [0.0, 0.3], "r2": [-0.3, 0.0]}
    assert weight_sweep(raw, grids=grid)["mase"].tolist() == pytest.approx(
        weight_sweep(restored, grids=grid)["mase"].tolist()
    )


def test_train_tail_is_selection_region_not_held_out_data():
    """The naive baselines are forecasts. If train_tail carried held-out values,
    naive/seasonal_naive would be scored against data they had already seen."""
    values = np.arange(200, dtype=float)
    series = pd.Series(values, index=pd.date_range("2023-01-01", periods=200, freq="D"))

    result = evaluate_series(series, "s", horizon=28, season=7,
                             runner=_fake_runner_factory({"A": 5.0}))

    assert result.train_tail, "train_tail not populated"
    # Last training point is index 171 (200 - 28 - 1); nothing at or beyond 172.
    assert max(result.train_tail) == pytest.approx(171.0)
    assert min(result.test_actual) == pytest.approx(172.0)
    assert all(v < min(result.test_actual) for v in result.train_tail)


def test_reanalysis_reproduces_the_original_naive_baselines():
    """--from-raw must rebuild naive/seasonal_naive identically to the live run,
    which is only possible because train_tail is persisted."""
    from benchmarks.selection_benchmark import load_raw, save_raw
    import tempfile

    series_list = _many_series(5)
    df_live, raw = run_benchmark(
        series_list, horizon=7,
        runner=_fake_runner_factory({"A": 100.0, "B": 95.0}),
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "r.json"
        save_raw(raw, path)
        restored = load_raw(path)

    live = df_live[df_live["policy"].isin(["naive", "seasonal_naive"])]
    live = live.sort_values(["series_id", "policy"])["mase"].to_numpy()

    rebuilt = []
    for result in sorted(restored, key=lambda r: r.series_id):
        policies = apply_policies(result, np.asarray(result.train_tail, dtype=float))
        for name in sorted(["naive", "seasonal_naive"]):
            rebuilt.append(policies[name]["mase"])

    assert np.allclose(live, rebuilt), "re-analysis changed the naive baselines"


def test_weight_sweep_covers_mape_and_r2_by_default():
    """r2 is computed IN-SAMPLE on CV folds, so the -0.10 term rewards models
    that overfit those folds — the prime suspect for composite's failure. It
    must be swept, not assumed."""
    _, raw = run_benchmark(
        _many_series(4), horizon=7,
        runner=_fake_runner_factory({"A": 100.0, "B": 95.0}),
    )
    swept = set(weight_sweep(raw)["swept_weight"].unique())
    assert {"rmse", "mae", "mape", "stability", "speed", "r2"} == swept


def test_weight_sweep_rescoring_needs_no_retraining():
    _, raw = run_benchmark(
        _many_series(4), horizon=7,
        runner=_fake_runner_factory({"A": 100.0, "B": 105.0, "C": 95.0}),
    )
    sweep = weight_sweep(raw, {}, grids={"speed": [0.0, 0.5], "rmse": [0.4]})
    assert not sweep.empty
    assert set(sweep["swept_weight"].unique()) == {"speed", "rmse"}
    assert len(sweep[(sweep["swept_weight"] == "speed") & (sweep["value"] == 0.0)]) == 4


def test_diebold_mariano_detects_a_clear_winner():
    rng = np.random.RandomState(0)
    actual = rng.randn(60) * 5 + 100
    good = actual + rng.randn(60) * 0.1
    bad = actual + rng.randn(60) * 10
    stat, p = diebold_mariano(actual, good, bad, horizon=1)
    assert stat < 0            # negative favours the first forecast
    assert p < 0.05


def test_diebold_mariano_on_identical_forecasts():
    actual = np.arange(20, dtype=float)
    stat, p = diebold_mariano(actual, actual.copy(), actual.copy())
    assert np.isnan(stat) and np.isnan(p)


def test_diebold_mariano_too_short():
    stat, p = diebold_mariano(np.array([1.0]), np.array([1.0]), np.array([2.0]))
    assert np.isnan(stat) and np.isnan(p)
