"""
Regression tests for the Phase 1 data-health fixes (DATA_HEALTH_DESIGN.md §2.1).
Each fault used to change the data silently; the probes that found them are
in benchmarks/data_health/parse_probes.py.
"""
import numpy as np
import pandas as pd
import pytest

from routers.upload import _detect_numeric_columns, _infer_frequency as upload_frequency
from services.preprocessor import _parse_dates, _parse_target, preprocess


# ── P1: number formats decided per column ──────────────────────────────────────

@pytest.mark.parametrize(
    "values, expected",
    [
        (["1,234", "2,500", "3,999"], [1234, 2500, 3999]),          # US thousands (was 1.234 …)
        (["$1,234.50", "$999.99"], [1234.5, 999.99]),               # US with cents
        (["1.234,56", "12,5", "3,25"], [1234.56, 12.5, 3.25]),      # European decimals
        (["1.234", "2.500"], [1234, 2500]),                         # European thousands (unchanged)
        (["12.5", "7", "3.25"], [12.5, 7.0, 3.25]),                 # plain decimals
    ],
)
def test_number_format_is_decided_for_the_whole_column(values, expected):
    assert _parse_target(pd.Series(values), "Revenue").tolist() == pytest.approx(expected)


# ── P2: day-first vs month-first ──────────────────────────────────────────────

def test_day_first_monthly_dates_keep_their_months():
    months = pd.date_range("2021-01-01", periods=36, freq="MS")
    parsed = _parse_dates(pd.Series(months.strftime("%d/%m/%Y")))
    assert parsed.dt.month.nunique() == 12
    assert parsed.iloc[2] == pd.Timestamp("2021-03-01")
    assert parsed.attrs["date_order"] == "day-first"
    assert parsed.attrs["date_order_ambiguous"] is True


def test_month_first_monthly_dates_still_month_first():
    months = pd.date_range("2021-01-01", periods=36, freq="MS")
    parsed = _parse_dates(pd.Series(months.strftime("%m/%d/%Y")))
    assert (parsed.reset_index(drop=True) == pd.Series(months)).all()


def test_day_first_daily_dates_are_unambiguous():
    days = pd.date_range("2023-01-01", periods=60)
    parsed = _parse_dates(pd.Series(days.strftime("%d/%m/%Y")))
    assert (parsed.reset_index(drop=True) == pd.Series(days)).all()
    assert parsed.attrs["date_order_ambiguous"] is False


def test_ambiguous_order_is_reported_as_a_warning():
    months = pd.date_range("2021-01-01", periods=36, freq="MS")
    df = pd.DataFrame({"Date": months.strftime("%d/%m/%Y"), "Sales": np.arange(36) + 100.0})
    series, report = preprocess(df, "Date", "Sales")
    assert len(series) == 36
    assert any("day-first" in w for w in report["warnings"])


def test_upload_frequency_uses_the_same_date_reading():
    months = pd.date_range("2021-01-01", periods=36, freq="MS")
    df = pd.DataFrame({"Date": months.strftime("%d/%m/%Y")})
    assert upload_frequency(df, "Date") == "monthly"


# ── P4: isolated typo dates ───────────────────────────────────────────────────

def test_typo_date_far_outside_the_data_is_excluded_not_interpolated():
    dates = pd.date_range("2023-01-01", periods=365).strftime("%Y-%m-%d").tolist()
    dates[200] = "2032-07-20"
    df = pd.DataFrame({"Date": dates, "Sales": 100.0})
    series, report = preprocess(df, "Date", "Sales")
    assert series.index.max() == pd.Timestamp("2023-12-31")
    assert len(series) == 365
    assert any("2032-07-20" in w for w in report["warnings"])


def test_genuine_long_history_is_not_trimmed():
    dates = pd.date_range("2015-01-01", "2023-12-31", freq="MS")
    df = pd.DataFrame({"Date": dates, "Sales": 100.0})
    series, report = preprocess(df, "Date", "Sales")
    assert len(series) == len(dates)
    assert not any("excluded" in w for w in report["warnings"])


# ── P7: no global capping by default ──────────────────────────────────────────

def test_recurring_seasonal_peaks_are_kept():
    rng = np.random.default_rng(0)
    d = pd.date_range("2022-01-01", "2023-12-31", freq="D")
    y = rng.normal(100, 8, len(d))
    y[np.asarray(d.month) == 12] *= 3
    series, _ = preprocess(pd.DataFrame({"Date": d, "Sales": y}), "Date", "Sales")
    assert series[series.index.month == 12].mean() == pytest.approx(y[np.asarray(d.month) == 12].mean())


# ── P8: periods with no records are zero for summed records ───────────────────

def test_days_without_transactions_count_as_zero():
    rows = [("2023-01-01", 2), ("2023-01-01", 3), ("2023-01-02", 4)]
    rows += [(str(d.date()), 1) for d in pd.date_range("2023-01-04", periods=40) for _ in range(2)]
    df = pd.DataFrame(rows, columns=["Date", "Qty"])   # 2023-01-03 has no rows
    series, report = preprocess(df, "Date", "Qty")
    assert series.loc["2023-01-03"] == 0
    assert series.loc["2023-01-01"] == 5
    assert any("counted as 0" in w for w in report["warnings"])


def test_gaps_in_one_row_per_day_data_are_still_imputed():
    d = pd.date_range("2023-01-01", periods=60).delete(30)
    df = pd.DataFrame({"Date": d, "Sales": 100.0})
    series, _ = preprocess(df, "Date", "Sales")
    assert series.iloc[30] == pytest.approx(100.0)


# ── P9: an all-blank period stays missing ─────────────────────────────────────

def test_blank_values_on_a_duplicated_date_are_not_turned_into_zero():
    dates = ["2023-01-01", "2023-01-01"] + [str(d.date()) for d in pd.date_range("2023-01-02", periods=40) for _ in range(2)]
    sales = [np.nan, np.nan] + [5.0] * 80
    series, _ = preprocess(pd.DataFrame({"Date": dates, "Sales": sales}), "Date", "Sales")
    assert series.loc["2023-01-01"] != 0


# ── P10: daylight-saving repeats ──────────────────────────────────────────────

def test_repeated_dst_hour_is_averaged_not_summed():
    idx = pd.date_range("2023-10-20", "2023-11-15", freq="h", tz="America/New_York")
    df = pd.DataFrame({"Date": idx.strftime("%Y-%m-%d %H:%M"), "Sales": 100.0})
    series, report = preprocess(df, "Date", "Sales")
    assert series.max() == 100.0
    assert any("appeared twice" in w for w in report["warnings"])


def test_hourly_transactions_are_still_summed():
    h = pd.date_range("2024-01-01", periods=200, freq="h")
    df = pd.DataFrame({"Date": np.repeat(h, 3).strftime("%Y-%m-%d %H:%M"), "Sales": 1.0})
    series, _ = preprocess(df, "Date", "Sales")
    assert series.iloc[0] == 3.0


# ── P11: target names matched as whole words ──────────────────────────────────

def test_target_detection_uses_whole_words():
    rng = np.random.default_rng(0)
    cols = ["Paid_Amount", "Liquid_Litres", "Holiday_Sales", "Turkey_Sales", "Units_Sold",
            "Store_ID", "StoreID", "customerid", "Transaction ID", "zip_code", "Latitude", "Promo_Flag"]
    df = pd.DataFrame({c: rng.normal(100, 20, 50) for c in cols})
    kept = set(_detect_numeric_columns(df, []))
    assert {"Paid_Amount", "Liquid_Litres", "Holiday_Sales", "Turkey_Sales", "Units_Sold"} <= kept
    assert not kept & {"Store_ID", "StoreID", "customerid", "Transaction ID", "zip_code", "Latitude", "Promo_Flag"}


# ── Clean data stays quiet ────────────────────────────────────────────────────

def test_clean_daily_data_produces_no_warnings():
    d = pd.date_range("2022-01-01", periods=500)
    y = 100 + 10 * np.sin(np.arange(500) / 7)
    series, report = preprocess(pd.DataFrame({"Date": d.strftime("%Y-%m-%d"), "Sales": y}), "Date", "Sales")
    assert report["warnings"] == []
    assert np.allclose(series.values, y)


# ── Generalisation fixes (benchmarks/data_health/generalization.py) ───────────

def test_level_metric_duplicates_are_averaged_and_gaps_not_zeroed():
    d = pd.date_range("2023-01-01", periods=120).delete(60)
    df = pd.DataFrame({"Date": np.repeat(d, 3), "Temperature": 22.0})
    series, report = preprocess(df, "Date", "Temperature")
    assert report["metric_kind"] == "level"
    assert series.min() == pytest.approx(22.0) and series.max() == pytest.approx(22.0)


def test_isolated_extra_zero_spike_is_replaced():
    d = pd.date_range("2023-01-01", periods=300)
    y = 100 + 10 * np.sin(np.arange(300) / 7)
    y[200] = y[200] * 10
    series, report = preprocess(pd.DataFrame({"Date": d, "Sales": y}), "Date", "Sales")
    assert series.iloc[200] < 200
    assert any("data errors" in w for w in report["warnings"])


def test_black_friday_style_peak_is_not_treated_as_an_error():
    d = pd.date_range("2023-01-01", periods=300)
    y = 100 + 10 * np.sin(np.arange(300) / 7)
    y[200] = y[200] * 5
    series, report = preprocess(pd.DataFrame({"Date": d, "Sales": y}), "Date", "Sales")
    assert series.iloc[200] == pytest.approx(y[200])


def test_irregular_intraday_timestamps_are_aggregated_not_dropped():
    rng = np.random.default_rng(0)
    t = pd.Timestamp("2024-01-01") + pd.to_timedelta(np.sort(rng.integers(0, 60 * 24 * 30, 2000)), unit="min")
    df = pd.DataFrame({"Timestamp": t.strftime("%Y-%m-%d %H:%M"), "Amount": 1.0})
    series, _ = preprocess(df, "Timestamp", "Amount")
    assert series.sum() == pytest.approx(2000)     # nothing lost


def test_sparse_hourly_sales_empty_hours_are_zero():
    # Sales in a random half of the hours (a perfectly regular every-other-hour
    # file would legitimately be a 2-hourly series).
    rng = np.random.default_rng(0)
    hours = pd.date_range("2024-01-01", periods=24 * 30, freq="h")
    sold = hours[rng.random(len(hours)) < 0.5]
    df = pd.DataFrame({"Time": sold.strftime("%Y-%m-%d %H:%M"), "Qty": 1.0})
    series, _ = preprocess(df, "Time", "Qty")
    assert series.sum() == pytest.approx(len(sold))
    assert series.mean() == pytest.approx(len(sold) / len(series))


def test_weekly_data_dated_on_mondays_survives_a_gap():
    d = pd.date_range("2020-01-06", periods=150, freq="W-MON").delete([40, 41])
    series, _ = preprocess(pd.DataFrame({"Week": d, "Sales": 100.0}), "Week", "Sales")
    assert len(series) == 150 and series.notna().all()


def test_trading_days_do_not_get_weekends():
    d = pd.bdate_range("2022-01-03", periods=300).delete([20, 80])
    series, _ = preprocess(pd.DataFrame({"Date": d, "Close": 100.0}), "Date", "Close")
    assert not (series.index.dayofweek >= 5).any()
    assert len(series) == 300


def test_leading_zero_decimal_is_not_european_thousands():
    assert _parse_target(pd.Series(["$0.125", "$0.500", "$0.875"]), "Price").tolist() == pytest.approx([0.125, 0.5, 0.875])
