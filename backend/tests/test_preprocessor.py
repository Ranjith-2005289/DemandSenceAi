"""
Regression tests for two real bugs found and fixed this session:

1. pandas 3.0 removed `infer_datetime_format` entirely (passing it raises
   TypeError). Every date-parsing call using it was silently failing —
   swallowed by a broad except — which meant ANY dataset without a format
   matching the small explicit fallback list (e.g. "04/19/19 08:46") could
   never be detected as a date column at all, even a column literally named
   "Order Date".
2. A too-broad "at" substring name-hint plus an unbounded Unix-timestamp
   fallback meant unrelated numeric columns (Temperature, Rate...) were
   misdetected as date columns, and small numbers falsely "parsed" as
   valid dates near 1970-01-01.
3. Selecting the same column for both date_col and target_col produced a
   cryptic 'DataFrame' object has no attribute 'str' crash instead of a
   clear validation error.
"""
import numpy as np
import pandas as pd
import pytest

from services.preprocessor import _parse_dates, preprocess


def test_parses_us_datetime_with_two_digit_year():
    """The exact format that exposed the infer_datetime_format removal bug."""
    s = pd.Series(["04/19/19 08:46", "04/07/19 22:30", "04/12/19 14:38", "04/30/19 09:27"])
    parsed = _parse_dates(s)
    assert parsed.notna().all()
    assert parsed.iloc[0] == pd.Timestamp("2019-04-19 08:46:00")


def test_parses_mixed_formats_within_one_column():
    """format='mixed' fix — the old `mixed=True` was never valid pandas syntax."""
    s = pd.Series(["04/19/19 08:46", "2023-01-01", "2023-06-15"])
    parsed = _parse_dates(s)
    assert parsed.notna().sum() >= 2


def test_iso_dates_still_work():
    s = pd.Series(pd.date_range("2023-01-01", periods=30).strftime("%Y-%m-%d"))
    parsed = _parse_dates(s)
    assert parsed.notna().all()


def test_same_column_for_date_and_target_raises_clear_error():
    df = pd.DataFrame({"Date": ["2023-01-01", "2023-01-02", "2023-01-03"], "Sales": [1, 2, 3]})
    with pytest.raises(ValueError, match="can't both be"):
        preprocess(df, date_col="Date", target_col="Date")


def test_preprocess_end_to_end_on_clean_data():
    df = pd.DataFrame({
        "Date": pd.date_range("2023-01-01", periods=40).strftime("%Y-%m-%d"),
        "Sales": np.arange(100, 140),
    })
    series, report = preprocess(df, date_col="Date", target_col="Sales")
    assert len(series) == 40
    assert isinstance(series.index, pd.DatetimeIndex)
