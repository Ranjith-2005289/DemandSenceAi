"""Regression tests reusing the exact scenario manually verified earlier this session."""
import numpy as np
import pandas as pd

from services.product_analytics import compute_group_summary


def _make_demo_products_df():
    rng = np.random.RandomState(0)
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    rows = []
    for product, base, slope in [("Widget A", 100, 2), ("Widget B", 50, -1), ("Widget C", 10, 0)]:
        for i, d in enumerate(dates):
            rows.append({"Date": d, "Product": product, "Sales": base + slope * i + rng.randn()})
    rows.append({"Date": dates[0], "Product": "Widget D", "Sales": 5})
    rows.append({"Date": dates[1], "Product": "Widget D", "Sales": 6})
    return pd.DataFrame(rows)


def test_growing_and_declining_trends_detected():
    df = _make_demo_products_df()
    result = compute_group_summary(df, "Date", "Sales", "Product", top_n=10)

    by_name = {g["group_value"]: g for g in result["groups"]}
    assert by_name["Widget A"]["trend"] == "growing"
    assert by_name["Widget B"]["trend"] == "declining"
    assert by_name["Widget C"]["trend"] == "flat"


def test_sparse_group_labeled_insufficient_data():
    df = _make_demo_products_df()
    result = compute_group_summary(df, "Date", "Sales", "Product", top_n=10)
    by_name = {g["group_value"]: g for g in result["groups"]}
    assert by_name["Widget D"]["trend"] == "insufficient_data"
    assert by_name["Widget D"]["growth_rate_pct"] is None


def test_ranking_and_share_pct():
    df = _make_demo_products_df()
    result = compute_group_summary(df, "Date", "Sales", "Product", top_n=10)
    assert result["total_groups"] == 4
    assert result["groups"][0]["rank"] == 1
    # Ranked descending by total
    totals = [g["total"] for g in result["groups"]]
    assert totals == sorted(totals, reverse=True)
    # Shares should sum to ~100%
    assert abs(sum(g["share_pct"] for g in result["groups"]) - 100) < 0.5
