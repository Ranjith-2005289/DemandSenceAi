"""Fast fault probes: run only the backend's parsing/preprocessing on crafted inputs."""
import sys, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # backend/
logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd
from services.preprocessor import preprocess, _parse_target, _parse_dates

rng = np.random.default_rng(1)
def show(name, df, d="Date", t="Sales", **kw):
    try:
        s, rep = preprocess(df, d, t, **kw)
        print(f"\n## {name}\n  rows in={len(df)} out={len(s)} range={s.index.min().date()}..{s.index.max().date()} mean={s.mean():.2f}")
        for st in rep["steps"][3:8]: print("   ", st)
        print("   warnings:", rep["warnings"])
        return s, rep
    except Exception as e:
        print(f"\n## {name}\n  ERROR: {e}")

# P1: US thousands separators parsed as European decimals?
vals = pd.Series(["1,234", "2,500", "3,999", "1,050", "8,765"])
print("P1 US-thousands strings ->", _parse_target(vals, "Revenue").tolist())
vals2 = pd.Series(["$1,234.50", "$2,500.00", "$999.99"])
print("P1b with cents ->", _parse_target(vals2, "Revenue").tolist())

# P2: monthly data written day-first (01/MM/YYYY)
months = pd.date_range("2021-01-01", periods=36, freq="MS")
df = pd.DataFrame({"Date": months.strftime("%d/%m/%Y"), "Sales": rng.normal(100, 10, 36).round()})
parsed = _parse_dates(df["Date"])
print("\nP2 day-first monthly: first 5 parsed ->", [str(x.date()) for x in parsed[:5]], " unique months:", parsed.dt.month.nunique())

# P3: business-day data with a few holidays missing
bd = pd.bdate_range("2023-01-02", "2024-06-28")
bd = bd.delete([10, 60, 120, 200, 300])
show("P3 business days, 5 holidays absent", pd.DataFrame({"Date": bd, "Sales": rng.normal(100, 10, len(bd)).round()}))

# P4: one typo date years in the future
d = pd.date_range("2023-01-01", periods=365, freq="D").strftime("%Y-%m-%d").tolist()
d[200] = "2032-07-20"
show("P4 single typo date (2032)", pd.DataFrame({"Date": d, "Sales": rng.normal(100, 10, 365).round()}))

# P5: level metric (temperature) with duplicate timestamps from two sensors
d = pd.date_range("2023-01-01", periods=200, freq="D")
df = pd.DataFrame({"Date": np.repeat(d, 2), "Sales": np.tile(rng.normal(22, 2, 200), 2).round(1)})
show("P5 two sensors per day, level metric (should average, not sum)", df)

# P6: exact duplicate rows (export copied twice for one month)
d = pd.date_range("2023-01-01", periods=365, freq="D")
df = pd.DataFrame({"Date": d, "Sales": rng.normal(100, 5, 365).round()})
df = pd.concat([df, df.iloc[300:330]]).sort_values("Date")
s, _ = show("P6 one month of rows duplicated", df)
print("   mean of duplicated month vs rest:", round(s.iloc[300:330].mean(), 1), "vs", round(s.iloc[:300].mean(), 1))

# P7: strong seasonal peak (December x3) — does IQR capping remove real peaks?
d = pd.date_range("2022-01-01", "2023-12-31", freq="D")
base = rng.normal(100, 8, len(d)); base[d.month == 12] *= 3
s, rep = show("P7 December peak x3", pd.DataFrame({"Date": d, "Sales": base.round()}))
print("   Dec mean raw vs after preprocessing:", round(base[d.month == 12].mean(), 1), "vs", round(s[s.index.month == 12].mean(), 1))

# P8: exponential growth — recent values capped as outliers?
d = pd.date_range("2022-01-01", "2023-12-31", freq="D")
g = 50 * np.exp(np.linspace(0, 2.3, len(d))) * rng.normal(1, 0.05, len(d))
s, rep = show("P8 exponential growth (10x over 2 years)", pd.DataFrame({"Date": d, "Sales": g.round()}))
print("   last-30-day mean raw vs after:", round(g[-30:].mean(), 1), "vs", round(s[-30:].mean(), 1))

# P9: hourly data — smoothing removes real spikes?
h = pd.date_range("2024-01-01", periods=24 * 60, freq="h")
y = np.asarray(100 + 30 * np.sin(2 * np.pi * h.hour / 24)) + rng.normal(0, 5, len(h))
y[np.asarray(h.hour == 18)] += 60   # real evening rush every day
s, rep = show("P9 hourly with daily 6pm rush", pd.DataFrame({"Date": h, "Sales": y}))
print("   6pm mean raw vs after:", round(y[np.asarray(h.hour == 18)].mean(), 1), "vs", round(s[s.index.hour == 18].mean(), 1))

# P10: NaN targets on a date that has duplicates -> becomes 0?
df = pd.DataFrame({"Date": ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"] + pd.date_range("2023-01-03", periods=40).strftime("%Y-%m-%d").tolist(),
                   "Sales": [np.nan, np.nan, 5, 6] + [10.0] * 40})
s, _ = show("P10 date whose only values are blank, with duplicates present", df)
print("   value on 2023-01-01 ->", s.iloc[0])
