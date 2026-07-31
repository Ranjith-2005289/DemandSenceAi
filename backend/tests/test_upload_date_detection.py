"""
Regression tests for the date-detection false-positive bug: a too-broad
"at" substring name-hint plus an unbounded Unix-timestamp fallback meant
unrelated numeric columns (Temperature, Rate...) were misdetected as date
columns, because small numbers falsely "parsed" as valid dates near
1970-01-01 and "at" matched inside unrelated words like "temperature".
"""
import numpy as np
import pandas as pd

from routers.upload import _detect_date_columns


def test_temperature_column_not_misdetected_as_date():
    df = pd.DataFrame({"Temperature": [42.5, 38.2, 55.0, 61.3, 33.9] * 20})
    detected = _detect_date_columns(df)
    assert "Temperature" not in detected


def test_rate_and_category_columns_not_misdetected_as_date():
    df = pd.DataFrame({
        "Rate": [0.05, 0.1, 0.15, 0.2] * 20,
        "Category": ["A", "B", "C", "D"] * 20,
    })
    detected = _detect_date_columns(df)
    assert "Rate" not in detected
    assert "Category" not in detected


def test_plain_date_string_column_still_detected():
    df = pd.DataFrame({"Date": pd.date_range("2023-01-01", periods=80).strftime("%Y-%m-%d")})
    detected = _detect_date_columns(df)
    assert "Date" in detected


def test_created_at_column_still_detected():
    df = pd.DataFrame({
        "created_at": pd.date_range("2023-01-01", periods=80).strftime("%Y-%m-%d %H:%M:%S")
    })
    detected = _detect_date_columns(df)
    assert "created_at" in detected


def test_real_unix_seconds_timestamp_still_detected():
    real_ts = pd.Series([int(pd.Timestamp(d).timestamp()) for d in pd.date_range("2023-01-01", periods=80)])
    df = pd.DataFrame({"event_timestamp": real_ts})
    detected = _detect_date_columns(df)
    assert "event_timestamp" in detected


def test_order_date_us_format_detected():
    """The exact real-world dataset that exposed the pandas 3.0 regression."""
    df = pd.DataFrame({
        "Order Date": ["04/19/19 08:46", "04/07/19 22:30", "04/12/19 14:38", "04/30/19 09:27"] * 10
    })
    detected = _detect_date_columns(df)
    assert "Order Date" in detected
