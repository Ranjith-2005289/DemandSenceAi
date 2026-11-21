"""
Fault-impact experiment: run the real DemandSense pipeline (preprocess ->
run_all_models -> select_best) on synthetic daily series with a known future,
once as the app handles each fault today and once with the correct handling.
Score = MASE of the selected model's 28-day forecast against the TRUE future
(scaled by in-sample weekly-naive error of the clean history). Lower is better.
"""
import sys, json, time, logging, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # backend/
logging.disable(logging.CRITICAL); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from services.preprocessor import preprocess
from services.model_runner import run_all_models
from services.best_model import select_best

H = 28
def base(n, level=60, seed=0, start="2022-01-01", weekly=0.25, yearly=0.15, trend=0.2):
    rng = np.random.default_rng(seed)
    d = pd.date_range(start, periods=n + H, freq="D")
    t = np.arange(n + H)
    mu = np.asarray(level * (1 + trend * t / (n + H))) * (1 + weekly * np.isin(d.dayofweek, [4, 5])) * (1 + yearly * np.sin(2 * np.pi * np.asarray(d.dayofyear) / 365.25))
    y = rng.poisson(mu).astype(float)
    return d, y, mu

def mase(truth, fc, hist):
    scale = np.mean(np.abs(hist[7:] - hist[:-7]))
    return float(np.mean(np.abs(np.asarray(truth) - np.asarray(fc)[:len(truth)])) / scale)

def run_series(s, h=H):
    s = s.asfreq("D") if s.index.freq is None else s
    res, info = run_all_models(s, forecast_horizon=h)
    best = select_best(res, series=s)
    return best["forecast"], best["best_model_name"]

def run_df(df, h=H, **kw):
    s, rep = preprocess(df, "Date", "Sales", **kw)
    fc, name = run_series(s, h)
    return fc, name, s

out = []
def record(scn, variant, fc, name, truth, clean_hist, note=""):
    m = mase(truth, fc, clean_hist)
    ref_mean = mase(truth, np.full(H, np.mean(clean_hist[-56:])), clean_hist)
    bias = (float(np.sum(np.asarray(fc)[:len(truth)])) - float(np.sum(truth))) / max(float(np.sum(truth)), 1e-9)
    out.append(dict(scenario=scn, variant=variant, model=name, mase=round(m, 3), bias_pct=round(100 * bias, 1), flat_recent_mean_mase=round(ref_mean, 3), note=note))
    print(json.dumps(out[-1]), flush=True)

def main():
    N = 700
    # A. clean reference
    d, y, _ = base(N, seed=1)
    fc, name, _ = run_df(pd.DataFrame({"Date": d[:N], "Sales": y[:N]}))
    record("A clean", "as-is", fc, name, y[N:], y[:N])

    # B. low-volume sales, days with no sales absent from the file
    d, y, _ = base(N, level=2.0, seed=2)
    mask = y[:N] > 0
    df = pd.DataFrame({"Date": d[:N][mask], "Sales": y[:N][mask]})
    fc, name, s = run_df(df)
    record("B no-sale days absent", "as-is (interpolated)", fc, name, y[N:], y[:N], f"{(~mask).sum()} zero days missing")
    fc, name = run_series(pd.Series(y[:N], index=d[:N]))
    record("B no-sale days absent", "fixed (zero-filled)", fc, name, y[N:], y[:N])

    # C. December peak, forecasting December
    d, y, _ = base(N, seed=3, start="2021-12-04")   # history ends 2023-11-03.. adjust below
    NC = 1060
    d = pd.date_range(end="2023-12-28", periods=NC + H, freq="D")
    rng = np.random.default_rng(3)
    mu = 80 * (1 + 0.25 * np.isin(d.dayofweek, [4, 5])); mu = mu * np.where(np.asarray(d.month) == 12, 2.6, 1.0)
    y = rng.poisson(mu).astype(float)
    df = pd.DataFrame({"Date": d[:NC], "Sales": y[:NC]})
    fc, name, _ = run_df(df)
    record("C seasonal peak (Dec x2.6)", "as-is (IQR capping)", fc, name, y[NC:], y[:NC], "2 prior Decembers in history")
    fc, name, _ = run_df(df, cap_outliers=False)
    record("C seasonal peak (Dec x2.6)", "fixed (no global capping)", fc, name, y[NC:], y[:NC])

    # D. stockouts: three 10-day runs of zero in the last 6 months
    d, y, _ = base(N, seed=4)
    yh = y[:N].copy()
    for st in (N - 170, N - 110, N - 45): yh[st:st + 10] = 0
    fc, name, _ = run_df(pd.DataFrame({"Date": d[:N], "Sales": yh}))
    record("D stockout zeros", "as-is", fc, name, y[N:], y[:N])
    s = pd.Series(yh, index=d[:N]).replace(0, np.nan).interpolate()
    fc, name = run_series(s)
    record("D stockout zeros", "fixed (masked as missing)", fc, name, y[N:], y[:N])

    # E. level shift: demand drops 40% 120 days before the end and stays there
    d, y, mu = base(N, seed=5, trend=0)
    rng = np.random.default_rng(55)
    mu2 = np.array(mu, dtype=float); mu2[N - 120:] *= 0.6
    y = rng.poisson(mu2).astype(float)
    fc, name, _ = run_df(pd.DataFrame({"Date": d[:N], "Sales": y[:N]}))
    record("E level shift -40%", "as-is (full history)", fc, name, y[N:], y[:N])
    fc, name = run_series(pd.Series(y[N - 120:N], index=d[N - 120:N]))
    record("E level shift -40%", "fixed (train after break)", fc, name, y[N:], y[:N])

    # F. partial last period: export taken mid-morning, last day has 25% of sales
    d, y, _ = base(N, seed=6)
    yh = y[:N].copy(); yh[-1] = round(yh[-1] * 0.25)
    fc, name, _ = run_df(pd.DataFrame({"Date": d[:N], "Sales": yh}))
    record("F partial last day", "as-is", fc, name, y[N:], y[:N])
    fc, name = run_series(pd.Series(yh[:-1], index=d[:N - 1]), h=H + 1)
    record("F partial last day", "fixed (drop partial day)", fc[1:], name, y[N:], y[:N])

    # G. one month of rows exported twice (recent)
    d, y, _ = base(N, seed=7)
    df = pd.DataFrame({"Date": d[:N], "Sales": y[:N]})
    dup = pd.concat([df, df.iloc[N - 60:N - 30]]).sort_values("Date", kind="stable")
    fc, name, _ = run_df(dup)
    record("G duplicated export rows", "as-is (summed)", fc, name, y[N:], y[:N])
    fc, name, _ = run_df(dup.drop_duplicates())
    record("G duplicated export rows", "fixed (exact duplicates dropped)", fc, name, y[N:], y[:N])

    # H. no learnable pattern (like the retail_sales_dataset): iid noise around 7
    rng = np.random.default_rng(8)
    d = pd.date_range("2023-01-01", periods=366 + H, freq="D")
    y = rng.gamma(2.8, 2.6, len(d)).round()
    fc, name, _ = run_df(pd.DataFrame({"Date": d[:366], "Sales": y[:366]}))
    record("H pure noise", "as-is", fc, name, y[366:], y[:366], "reference = flat mean")

    # I. intermittent demand (~75% zero days)
    rng = np.random.default_rng(9)
    d = pd.date_range("2022-01-01", periods=N + H, freq="D")
    y = (rng.random(N + H) < 0.25) * rng.poisson(4, N + H).astype(float)
    fc, name, _ = run_df(pd.DataFrame({"Date": d[:N], "Sales": y[:N]}))
    record("I intermittent (75% zeros)", "as-is", fc, name, y[N:], y[:N])

    json.dump(out, open(__file__.replace(".py", "_results.json"), "w"), indent=1)
    print("DONE")

if __name__ == "__main__":
    main()
