import sys, logging, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # backend/
logging.disable(logging.CRITICAL); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from services.preprocessor import preprocess
from services.model_runner import run_all_models
from services.best_model import select_best

def main():
    H, NC = 28, 1060
    d = pd.date_range(end="2023-12-28", periods=NC + H, freq="D")
    rng = np.random.default_rng(3)
    mu = 80 * (1 + 0.25 * np.isin(d.dayofweek, [4, 5])) * np.where(np.asarray(d.month) == 12, 2.6, 1.0)
    y = rng.poisson(mu).astype(float)
    s, _ = preprocess(pd.DataFrame({"Date": d[:NC], "Sales": y[:NC]}), "Date", "Sales", cap_outliers=False)
    res, info = run_all_models(s, forecast_horizon=H)
    best = select_best(res, series=s)
    print("truth Dec mean %.0f | Nov mean %.0f | backtest origins (cut dates):" % (y[NC:].mean(), y[NC-28:NC].mean()),
          [str(s.index[c].date()) for c in info.get("backtest_origins_idx", [])] or info.get("backtest_origins"))
    for r in sorted(res, key=lambda r: r.get("rank", 99)):
        fc = r.get("forecast") or []
        if fc: print(f"  {r['model_name']:22s} backtest MASE={r.get('mase', float('nan')):.3f}  Dec forecast mean={np.mean(fc):6.0f}")
    print("selected:", best["best_model_name"])

if __name__ == "__main__":
    main()
