"""
Column detection on upload, modelled on real files that exposed bugs:
- Walmart_Sales.csv: 'Weekly_Sales' and 'Holiday_Flag' were detected as DATE
  columns (numbers read as nanoseconds since 1970), so the sales column could
  not be chosen; 'Store' was not offered as a group; Temperature was suggested.
- avocado.csv: pandas-3 text columns ('type') were never offered as groups;
  the saved index 'Unnamed: 0' was offered as a target; AveragePrice suggested.
- DMart export: UnitPrice was suggested over Quantity/Total.
"""
import numpy as np
import pandas as pd

from routers.upload import (
    _detect_date_columns,
    _detect_grouping_columns,
    _detect_numeric_columns,
    _target_score,
    _try_parse_date_series,
)


def _detect(df):
    dates = list(_detect_date_columns(df))
    numeric = _detect_numeric_columns(df, dates)
    groups = _detect_grouping_columns(df, dates, list(numeric))
    best = max(numeric, key=lambda c: _target_score(c, numeric[c])) if numeric else None
    return dates, numeric, groups, best


def _walmart_like():
    rng = np.random.default_rng(0)
    weeks = pd.date_range("2010-02-05", periods=30, freq="W-FRI")
    rows = [
        (s, d.strftime("%d-%m-%Y"), float(rng.normal(1.5e6, 1e5)), int(d.month == 11),
         float(rng.normal(60, 10)), float(rng.normal(3, 0.2)), 211.0, 8.1)
        for s in range(1, 6) for d in weeks
    ]
    return pd.DataFrame(rows, columns=["Store", "Date", "Weekly_Sales", "Holiday_Flag",
                                       "Temperature", "Fuel_Price", "CPI", "Unemployment"])


def test_walmart_sales_column_is_a_target_not_a_date():
    dates, numeric, groups, best = _detect(_walmart_like())
    assert dates == ["Date"]
    assert "Weekly_Sales" in numeric
    assert best == "Weekly_Sales"
    assert groups == ["Store"]
    assert "Store" not in numeric


def test_avocado_text_groups_and_index_column():
    rng = np.random.default_rng(1)
    d = pd.date_range("2015-01-04", periods=20, freq="W-SUN")
    rows = [(i, x.strftime("%Y-%m-%d"), float(rng.uniform(1, 2)), float(rng.uniform(1e5, 1e6)),
             float(rng.uniform(1e4, 1e5)), t, x.year, r)
            for i, (x, t, r) in enumerate((x, t, r) for x in d for t in ["conventional", "organic"] for r in ["Albany", "TotalUS"])]
    df = pd.DataFrame(rows, columns=["Unnamed: 0", "Date", "AveragePrice", "Total Volume", "Total Bags", "type", "year", "region"])
    df["type"] = df["type"].astype("string")
    df["region"] = df["region"].astype("string")
    dates, numeric, groups, best = _detect(df)
    assert "Date" in dates
    assert "Unnamed: 0" not in numeric and "year" not in numeric
    assert best == "Total Volume"
    assert set(groups) == {"type", "region"}


def test_dmart_prefers_quantity_or_total_over_unit_price():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "Date": pd.date_range("2023-01-01", periods=200, freq="h").strftime("%d-%m-%Y %H:%M"),
        "UnitPrice": rng.uniform(10, 500, 200),
        "Quantity": rng.integers(1, 6, 200),
        "Total": rng.uniform(10, 2500, 200),
    })
    *_, best = _detect(df)
    assert best in {"Quantity", "Total"}


def test_numeric_columns_only_parse_as_real_timestamps():
    assert _try_parse_date_series(pd.Series([1643690.9, 1641957.4, 1611968.2])) is None   # sales
    assert _try_parse_date_series(pd.Series([0, 1, 0, 0])) is None                        # flags
    assert _try_parse_date_series(pd.Series([1_700_000_000, 1_700_086_400])) is not None   # epoch seconds
    parsed = _try_parse_date_series(pd.Series([20230105, 20230106, 20230107]))             # yyyymmdd
    assert parsed is not None and parsed.iloc[0] == pd.Timestamp("2023-01-05")


def test_integer_year_column_is_a_date_for_annual_data():
    df = pd.DataFrame({"Year": list(range(2000, 2024)), "Revenue": np.arange(24) * 10.0 + 100})
    dates, numeric, _, best = _detect(df)
    assert dates == ["Year"]
    assert best == "Revenue"
