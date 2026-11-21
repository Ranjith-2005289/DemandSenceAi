# Data-health benchmarks

Evidence for `DATA_HEALTH_DESIGN.md`. Run from `backend/` with the venv:

| Script | What it shows | Runtime |
|---|---|---|
| `parse_probes.py` | Parsing/preprocessing faults on crafted inputs (number formats, date order, typo dates, duplicates, capping) | seconds |
| `fault_impact.py` | End-to-end accuracy (MASE, bias) of the real pipeline per fault, as-is vs correctly handled | ~5 min |
| `seasonal_blind_spot.py` | Per-model December forecasts vs truth; shows the backtest ranks peak-aware models low | ~1 min |

```bash
cd backend && ./venv/bin/python benchmarks/data_health/parse_probes.py
```
