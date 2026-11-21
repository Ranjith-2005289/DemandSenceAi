"""
Generalisation check for the Phase 1 rules: run many dataset shapes through
preprocess() and compare the output with what a careful analyst would expect.
Each case states its expectation; the script reports PASS / RISK per case and
the warnings a user would get (clean data should get none).

Run from backend/:  ./venv/bin/python benchmarks/data_health/generalization.py
"""
import sys
import logging
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # backend/
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from services.preprocessor import preprocess

rng = np.random.default_rng(42)
CASES = []


def case(name, kind, expect):
    def wrap(fn):
        CASES.append((name, kind, expect, fn))
        return fn
    return wrap


def seasonal(n, level=100, noise=5):
    t = np.arange(n)
    return level + 10 * np.sin(2 * np.pi * t / 7) + rng.normal(0, noise, n)


# ── Clean datasets of many shapes: expect no warnings and unchanged values ────

@case("Daily ISO, retail sales", "clean", "unchanged, no warnings")
def _():
    d = pd.date_range("2022-01-01", periods=600)
    y = seasonal(600)
    return pd.DataFrame({"Date": d.strftime("%Y-%m-%d"), "Sales": y}), "Date", "Sales", y


@case("Weekly (Sunday-anchored)", "clean", "unchanged, no warnings")
def _():
    d = pd.date_range("2019-01-06", periods=260, freq="W-SUN")
    y = seasonal(260, 5000, 200)
    return pd.DataFrame({"Week": d, "Weekly_Sales": y}), "Week", "Weekly_Sales", y


@case("Monthly, 10 years", "clean", "unchanged, no warnings")
def _():
    d = pd.date_range("2014-01-01", periods=120, freq="MS")
    y = seasonal(120, 1e5, 3e3)
    return pd.DataFrame({"Month": d.strftime("%Y-%m"), "Revenue": y}), "Month", "Revenue", y


@case("Quarterly '2023-Q1' strings", "clean", "unchanged, no warnings")
def _():
    q = [f"{y}-Q{i}" for y in range(2012, 2024) for i in range(1, 5)]
    y = seasonal(len(q), 1e6, 2e4)
    return pd.DataFrame({"Quarter": q, "Revenue": y}), "Quarter", "Revenue", y


@case("Hourly energy load", "clean", "unchanged, no warnings")
def _():
    d = pd.date_range("2024-01-01", periods=24 * 90, freq="h")
    y = 500 + 100 * np.sin(2 * np.pi * np.arange(len(d)) / 24) + rng.normal(0, 10, len(d))
    return pd.DataFrame({"Timestamp": d.strftime("%Y-%m-%d %H:%M"), "kWh": y}), "Timestamp", "kWh", y


@case("European file: dd/mm/yyyy + '1.234,56'", "clean", "unchanged, no warnings")
def _():
    d = pd.date_range("2023-01-01", periods=300)
    y = np.round(seasonal(300, 1500, 50), 2)
    txt = [f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") for v in y]
    return pd.DataFrame({"Datum": d.strftime("%d/%m/%Y"), "Umsatz": txt}), "Datum", "Umsatz", y


@case("US file: mm/dd/yyyy + '$1,234.56'", "clean", "unchanged, no warnings")
def _():
    d = pd.date_range("2023-01-01", periods=300)
    y = np.round(seasonal(300, 1500, 50), 2)
    return pd.DataFrame({"Date": d.strftime("%m/%d/%Y"), "Revenue": [f"${v:,.2f}" for v in y]}), "Date", "Revenue", y


@case("Profit with negative days", "clean", "unchanged, no warnings")
def _():
    d = pd.date_range("2023-01-01", periods=300)
    y = rng.normal(50, 80, 300)
    return pd.DataFrame({"Date": d, "Profit": y}), "Date", "Profit", y


@case("Store × product panel, all days present", "clean", "daily totals, no warnings")
def _():
    d = pd.date_range("2023-01-01", periods=200)
    rows = [(x, s, p, float(rng.poisson(20))) for x in d for s in "ABC" for p in range(4)]
    df = pd.DataFrame(rows, columns=["Date", "Store", "Product", "Units"])
    return df, "Date", "Units", df.groupby("Date")["Units"].sum().values


# ── Shapes where a Phase 1 rule could misfire ─────────────────────────────────

@case("Sensor panel (level metric), one date missing", "edge", "missing date imputed (~22 °C), not 0")
def _():
    d = pd.date_range("2023-01-01", periods=120).delete(60)
    df = pd.DataFrame({"Date": np.repeat(d, 3), "Sensor": np.tile(["s1", "s2", "s3"], len(d)),
                       "Temperature": rng.normal(22, 1, 3 * len(d))})
    return df, "Date", "Temperature", None


@case("Company sales, export missed a whole week", "edge", "ambiguous: 0 (closed) or unknown (outage)")
def _():
    d = pd.date_range("2023-01-01", periods=200).delete(range(150, 157))
    rows = [(x, s, float(rng.poisson(30))) for x in d for s in "AB"]
    return pd.DataFrame(rows, columns=["Date", "Store", "Sales"]), "Date", "Sales", None


@case("Single-row daily data, store closed 3 weeks", "edge", "ambiguous: 0 (closed) or unknown")
def _():
    d = pd.date_range("2023-01-01", periods=300).delete(range(100, 121))
    return pd.DataFrame({"Date": d, "Sales": seasonal(len(d))}), "Date", "Sales", None


@case("A few pilot rows, then 4-month pause, then regular data", "edge", "pilot rows kept or dropped with a warning")
def _():
    d = list(pd.date_range("2022-01-01", periods=3)) + list(pd.date_range("2022-05-15", periods=300))
    return pd.DataFrame({"Date": d, "Sales": seasonal(len(d))}), "Date", "Sales", None


@case("One typo spike (extra zero: 1000 instead of 100)", "edge", "spike handled")
def _():
    d = pd.date_range("2023-01-01", periods=300)
    y = seasonal(300); y[290] *= 10
    return pd.DataFrame({"Date": d, "Sales": y}), "Date", "Sales", None


@case("Low-volume hourly transactions (rarely 2 in one hour)", "edge", "summed, not averaged")
def _():
    hours = pd.date_range("2024-01-01", periods=24 * 60, freq="h")
    picked = hours[rng.random(len(hours)) < 0.5]
    doubles = picked[rng.random(len(picked)) < 0.01]
    ts = np.concatenate([picked, doubles])
    df = pd.DataFrame({"Time": pd.DatetimeIndex(ts).strftime("%Y-%m-%d %H:%M"), "Qty": 1.0})
    return df, "Time", "Qty", None


@case("Minute-stamped transactions (POS export)", "edge", "aggregated to a usable period")
def _():
    t = pd.Timestamp("2024-01-01") + pd.to_timedelta(np.sort(rng.integers(0, 60 * 24 * 90, 4000)), unit="min")
    return pd.DataFrame({"Timestamp": t.strftime("%Y-%m-%d %H:%M"), "Amount": rng.gamma(2, 20, 4000)}), "Timestamp", "Amount", None


@case("Prices with 3 decimals as text ('$0.125')", "edge", "0.125, not 125")
def _():
    d = pd.date_range("2023-01-01", periods=100)
    y = np.round(rng.uniform(0.1, 0.9, 100), 3)
    return pd.DataFrame({"Date": d, "Unit_Price": [f"${v:.3f}" for v in y]}), "Date", "Unit_Price", y


@case("Stock closing prices, trading days only", "edge", "no weekend rows invented")
def _():
    d = pd.bdate_range("2022-01-03", periods=500).delete([20, 80, 140, 250, 330])
    y = 100 + np.cumsum(rng.normal(0, 1, len(d)))
    return pd.DataFrame({"Date": d, "Close": y}), "Date", "Close", None


def main():
    print(f"{'Case':58s} {'Result':6s}  Detail")
    for name, kind, expect, fn in CASES:
        df, dc, tc, truth = fn()
        try:
            s, rep = preprocess(df, dc, tc)
        except Exception as e:
            print(f"{name:58s} ERROR   {e}")
            continue
        warns = rep["warnings"]
        detail = f"rows {len(df)}→{len(s)}"
        if truth is not None:
            ok = len(s) == len(truth) and np.allclose(s.values, truth, rtol=1e-6, atol=1e-6)
            status = "PASS" if ok and not warns else "RISK"
            if not ok:
                detail += f" | values differ (mean {np.mean(truth):.3f} → {s.mean():.3f})"
        else:
            status = "INFO"
            detail += f" | min {s.min():.3f} max {s.max():.3f} mean {s.mean():.3f}"
        print(f"{name:58s} {status:6s}  {detail}")
        print(f"{'':58s}         expect: {expect}")
        for w in warns:
            print(f"{'':58s}         warning: {w[:150]}")


if __name__ == "__main__":
    main()
