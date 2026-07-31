# DemandSense AI — Interview Prep

An end-to-end reference for talking about this project in interviews: what it is, how it's built, and a complete answer bank for the questions most likely to come up. Everything below reflects the actual implementation, not aspirational description.

---

## 1. About the project

DemandSense AI is a full-stack retail demand forecasting system. A user uploads any time-series dataset — CSV, Excel, or JSON — and the backend automatically detects the date and target columns, cleans the data, and trains **19 forecasting models** in parallel: seven statistical models (ARIMA, SARIMA, SARIMAX, Holt-Winters, Exponential Smoothing, Moving Average, Prophet), four classical machine learning models (XGBoost, Random Forest, KNN, SVM), seven deep learning models (RNN, LSTM, GRU, BiLSTM, CNN-1D, TCN, Transformer), and one intermittent-demand model (Croston-SBA, for sparse, zero-inflated demand).

Rather than picking the model with the lowest error alone, the system ranks all successfully trained candidates using a **composite score** that also weighs mean absolute error, mean absolute percentage error, forecast stability, and computation time — because a single metric like RMSE can be misleading in practice (this is demonstrated concretely below). On top of the forecast, a retrieval-augmented LLM layer answers user questions and generates a structured business report, explicitly constrained so it can never contradict the numbers the system itself computed.

**Stack:** FastAPI (Python) backend, React 18 + Vite frontend, scikit-learn/XGBoost/statsmodels/Prophet/PyTorch for modeling, Google Gemini + ChromaDB for the LLM/RAG layer, no database — local disk storage only. 36 automated tests (pytest) cover the core logic.

The engineering emphasis throughout has been **honesty over polish**: real forecast uncertainty derived from each model's own error rather than a fixed percentage, real historical chart data instead of fabricated placeholder noise, and an LLM layer that is architecturally prevented from contradicting the system's own deterministic labels.

---

## 2. Project at a glance

| Layer | Stack |
|---|---|
| Backend | FastAPI (Python), Uvicorn ASGI server |
| Frontend | React 18 + Vite, Recharts, Axios, Tailwind CSS |
| Statistical models | statsmodels (ARIMA/SARIMA/SARIMAX/Holt-Winters), Prophet |
| Classical ML | scikit-learn (KNN, SVM), XGBoost, Random Forest |
| Deep learning | PyTorch (Keras fallback if PyTorch is unavailable) |
| LLM / RAG | Google Gemini (google-genai SDK), ChromaDB vector store |
| Storage | Local disk only — no database |
| Testing | pytest — 36 fast unit/logic tests + slow integration tests |

Repo structure: `backend/routers` (API endpoints), `backend/services` (preprocessing, orchestration, scoring, RAG, session persistence), `backend/models` (one file per model family), `frontend/src/components` (upload flow, charts, chatbot, insights page).

---

## 3. System architecture

**Request flow, end to end:**

1. **Upload** (`POST /upload`) — file saved to disk with a UUID-prefixed name; column detection runs via dtype inspection plus keyword matching against curated vocabularies (date-like, target-like, grouping-like, price-like, promo-like names), plus a cascading date-parsing strategy: pandas native inference → explicit format list → Unix timestamp (only if the raw values fall in a plausible 2000–2100 calendar-year range) → mixed-format fallback.
2. **Preprocess** — parse dates and target, sum duplicate timestamps, reindex to a complete calendar at the inferred frequency, fill gaps (time-interpolation → forward-fill → backward-fill), IQR-based outlier capping, optional exogenous-feature alignment.
3. **Forecast** (`POST /forecast`) — the cleaned series is submitted to a shared `ThreadPoolExecutor`; every registered model trains concurrently, each under its own timeout by category (statistical / ML / DL).
4. **Evaluate** — every model is scored with walk-forward cross-validation (never a single train/test split), and a composite score picks the winner.
5. **Respond** — full leaderboard, the winner's forecast, a real uncertainty interval, and real historical data.

### Q: Why FastAPI over Flask/Django?
Async support for I/O-bound work, automatic request validation via Pydantic models (catches malformed requests before they reach business logic), and built-in OpenAPI docs — useful given how many optional parameters a forecast request can carry (horizon, group columns, feature columns, country code, session ID, multi-target list).

### Q: Why a thread pool instead of multiprocessing for concurrent model training?
Most of these libraries (NumPy, statsmodels, PyTorch, XGBoost) release the GIL during their actual numeric computation, so threads still get real parallelism for the expensive part, without the serialization overhead and memory duplication of spawning separate processes for 19 models per request.

---

## 4. "Why doesn't this use a database?"

**Straight answer:** because nothing in the current scope needs one. There are no user accounts, no cross-user queries, no need to join or aggregate records relationally. Uploaded files live on disk under `uploads/`, preprocessed series under `processed/`, and session state (the latest forecast per session, chatbot history) is kept in an in-memory dict for speed, mirrored to a JSON file per session under `session_data/` so a backend restart doesn't lose active sessions.

### Q: So when would you add one?
Persistent multi-user accounts, a history/analytics view across past forecasts, or running multiple backend instances behind a load balancer (the in-memory cache is local to one process — it wouldn't be shared across instances). SQLite would be the first reach, not Postgres immediately — a single-file, zero-ops fit for a system that otherwise has no external services besides the LLM API.

### Q: Isn't a JSON-file session store fragile?
For a single-process deployment, no — writes are atomic (write to a temp file, then rename), and it's purely a restart-safety net; the in-memory dict is always the fast path. This was stress-tested directly: killing the in-memory cache mid-session and confirming both the forecast cache and an active chat conversation (including LLM context) survived and resumed correctly, without replaying every message through the API.

---

## 5. The 19-model ensemble

No single forecasting technique wins across all data — well established by forecasting competitions like M4/M5. So instead of committing to one model family, the system trains all of them and lets an auditable scoring rule decide.

- **Statistical (7):** ARIMA, SARIMA, SARIMAX, Holt-Winters, Exponential Smoothing, Moving Average, Prophet
- **Machine learning (4):** XGBoost, Random Forest, KNN, SVM
- **Deep learning (7):** RNN, LSTM, GRU, BiLSTM, CNN-1D, TCN, Transformer
- **Intermittent demand (1):** Croston-SBA

### Q: Why detrend-and-residual for tree/instance models?
XGBoost, Random Forest, KNN, and SVM can't extrapolate outside the range of target values they saw during training — on a genuinely trending series they just plateau at the training max. Fix: fit a simple linear trend by OLS, train the model to predict the *residual* (target minus trend), then add the extrapolated trend back at forecast time. The residual stays roughly stationary regardless of how far the level has trended.

> **Real trade-off, not assumed:** the first attempt at fixing this used *differencing* instead of detrend-and-residual, benchmarked across 10 random seeds before shipping — and it made things worse (compounding recursive-forecast error). Caught before shipping, pivoted to detrend-and-residual, re-benchmarked, confirmed the improvement. Good material for "tell me about a time you were wrong" — the fix was tested and reverted before it ever shipped.

### Q: Why gate deep learning by row count instead of always trying it?
DL models need substantially more observations than statistical or tree models to generalize instead of memorizing noise. Training an LSTM on 80 rows and reporting its RMSE next to ARIMA's, with no caveat, is actively misleading — a user could pick the "best" model by the number without knowing it never had enough data to be reliable. The gate is frequency-aware (daily vs. weekly vs. monthly data need different minimums), and the decision is returned explicitly in the API response (`dl_skipped: true`, with the reason), not just logged silently.

---

## 6. Model selection: the composite score

This is the single most interview-worthy design decision in the project — real, defensible statistical engineering, not a framework default.

```
S = 0.40·RMSE̅ + 0.20·MAE̅ + 0.20·MAPE̅ + 0.10·Stab̅ + 0.10·Speed̅ − 0.10·R²

where X̅ = X / (Q3 + 1.5·IQR), computed across all successfully-trained candidates
```

### Q: Why not just rank by RMSE?
Because a single metric hides real trade-offs — a model can have the best RMSE and still be a bad practical choice (unstable forecast, orders of magnitude slower, poor MAE/MAPE). In one real benchmark, XGBoost had the *lowest raw RMSE* of 12 candidates but ranked **9th** under the composite score, because its training time was roughly three orders of magnitude higher than KNN's and its forecast was comparatively less stable on that series. KNN won overall. The full leaderboard stays visible precisely so a user who only cares about RMSE can still see and pick the RMSE-optimal model directly.

### Q: Why normalize by a Tukey fence (Q3 + 1.5·IQR) instead of the raw max?
Because a single catastrophically bad model in the batch (a real case: SARIMA hitting ~823M RMSE on an irregular dataset) becomes the denominator for *every* model if normalized by max — it compresses every other model's normalized error toward zero, so the 80%-accuracy-weighted terms stop differentiating between the actually-good models, and the ranking gets decided almost by accident on the remaining 20% (stability/speed). Robust statistics fix this the same way outlier capping does elsewhere in the pipeline — the fence is set by the bulk of the distribution, not the single worst point.

### Cross-validation
Every model is evaluated with **walk-forward, expanding-window cross-validation** — never a random train/test split, since that leaks future information into training for time series. Fold count is adaptive: `k = min(5, max(2, n_obs // 100))` — more data earns more folds (less risk of a model looking best by luck on one split), capped so re-fitting cost stays bounded for expensive models.

> **Real bug this replaced:** the fold-count logic used to be `n_splits = 2 if n_obs >= 20 else 2` — both branches evaluate to `2`. Every model in the project was always evaluated on a fixed 2-fold split regardless of dataset size, silently, across 8 files and 13 call sites, until this was found and fixed.

---

## 7. Data pipeline

**Feature engineering (ML/DL models):**
- Lags: 1, 2, 3, 7, 14, 21, 28, 30 days
- Rolling stats: mean/std/min/max over 7/14/30-day windows
- EWM: exponentially weighted moving average, spans 7 and 14
- Calendar: day-of-week, month, quarter, day-of-year, week-of-year, month-boundary flags, sine/cosine cyclical encodings

**Normalization:** robust scaler (median/IQR-based) for XGBoost/RF/KNN/SVM — chosen over standard mean-variance scaling because retail series often have heavy-tailed spikes even after capping. Min-max scaling for deep learning models, fit on train only, standard practice for recurrent/conv nets.

### Q: Walk me through the date-parsing logic.
Cascading strategy, cheapest/most-common first: (1) pandas native inference, (2) explicit format list cycling through common patterns, (3) Unix-timestamp interpretation — but only if the raw numeric values fall in a plausible 2000–2100 calendar-year range in seconds, otherwise arbitrary numeric columns like temperature or price false-positive as "valid" dates near 1970-01-01, (4) mixed-format fallback as a last resort. This ordering matters — a real bug was found and fixed where a "Weekly_Sales" numeric column was silently being reinterpreted as a single-day Unix timestamp before the range check existed.

---

## 8. Exogenous features (price/promotion)

XGBoost, Random Forest, and SARIMAX can use real price and promotion columns from the uploaded file, when present — not just lags of the target.

### Q: The forecast horizon is the future — you don't know the future price or promo. How did you handle that?
A different rule per feature type, chosen deliberately rather than one blanket assumption: **price** carries forward its last known value (reasonable — prices are usually stable over a short horizon); **promotion flags** default to zero for the forecast horizon (assuming *no* promotion, rather than guessing one continues or recurs — assuming a promo continues would be a real guess dressed up as data). This was an explicit design decision, surfaced as a choice, picking the more honest option over the one that would make a demo look flashier.

Real, measured impact: on a synthetic benchmark with a 15-day promo cycle, SARIMAX's RMSE improved from **11.7 → 3.56** with real exogenous data wired in versus calendar-only features, because the promo signal was previously invisible to it entirely.

---

## 9. Explainability: feature importance + LLM/RAG

> **Direct answer if asked about SHAP:** this system does *not* use SHAP or LIME. The honest answer to "how do you explain predictions" is: (1) native gain/impurity-based feature importance straight from XGBoost/Random Forest's own training — free, no extra computation — and (2) a retrieval-augmented LLM layer that explains the forecast in natural language. Post-hoc attribution (SHAP) is a natural future extension to name if asked what's next.

### Q: Why RAG instead of just prompting an LLM directly?
To ground the model in two separate, verifiable sources instead of letting it reason freely: (1) **deterministic context** — real metrics, the winning model, forecast min/max/avg, a trend label computed by a fixed percentage threshold — built directly from the pipeline's own numbers, never re-derived by the LLM; (2) **retrieved context** — for qualitative/advice questions only ("how do I reduce stockouts"), the top-k most similar passages from a 13-document curated retail knowledge base, via cosine similarity over Gemini embeddings stored in ChromaDB.

### Q: Why not use retrieval for numeric questions too, like "which product sold best"?
Because retrieval-based grounding for a fact the system already knows exactly would only add hallucination risk, not reduce it. Numeric/factual questions are answered straight from deterministic pipeline output; retrieval is reserved for genuinely qualitative advice where there's no ground truth to compute directly.

### Q: How do you stop the LLM from contradicting the system's own numbers?
Explicit prompt-level constraints. The trend classifier uses a fixed ±5% threshold — if a change is inside that band it's labeled "flat," and the prompt instructs the model never to call it "declining" or "growing" even if the underlying number is technically negative or positive. Same pattern for system capabilities — the prompt states plainly which models used real exogenous features *this run*, so the LLM can't imply a capability wasn't actually exercised. A real case was found and fixed where the LLM's recommendation contradicted the system's own trend label before this constraint existed.

---

## 10. Real bugs fixed (interview gold)

These are the strongest answers to "tell me about a challenging bug" or "tell me about a time you found a problem no one asked you to look for" — specific, verifiable, with root cause, fix, and verification for each.

**87-minute hang**
- *Symptom:* a forecast request that should time out in ~35 seconds instead hung for 87 minutes.
- *Root cause:* `with ThreadPoolExecutor(...) as executor:` calls `shutdown(wait=True)` on exit — which blocks the whole HTTP request until every submitted thread finishes, even threads already logically marked as "timed out."
- *Fix:* manual `executor.shutdown(wait=False, cancel_futures=True)` in a `finally` block. Verified with a deliberately-stuck fake model (90s sleep) — request now correctly returns at the ~35s global timeout instead of hanging.

**Outlier-distortion in scoring**
- *Symptom:* a model with a worse RMSE was winning "best model" over one with a better RMSE.
- *Root cause:* score normalization divided by the raw batch maximum; one catastrophic outlier (SARIMA at ~823M RMSE) became the denominator for everyone, compressing all other models' normalized scores to near-zero noise.
- *Fix:* Tukey-fence normalization (Q3 + 1.5·IQR) instead of raw max. Verified the fix actually changes the ranking outcome on a reconstructed 18-model composition.

**Stale frontend state**
- *Symptom:* uploading a second file with no date column produced a cryptic 400 error — "Date column 'date' not found" — even though the user never typed "date" anywhere.
- *Root cause:* the upload handler only ever *set* the date/target column selections when the new file's auto-detected suggestions were non-empty — it never reset them first. A "date" column selected for an earlier upload silently survived into the new file's config screen, and "Run Forecast" only checked truthiness, not validity against the current file.
- *Fix:* reset all four column selections at the start of every upload. Verified by reproducing the exact scenario (upload with dates → upload without) and confirming the dropdown now correctly blanks and the button correctly disables.

**Fabricated frontend data**
- *Symptom:* none reported — found by reading the code, not by a bug report. The strongest "no one asked me to check this" story available.
- *Root cause:* the "Historical & Forecast" chart's blue line — the first thing a user sees — was 100% `Math.random()`, invented client-side, because the backend never returned real historical data. The confidence band was a flat `forecast × 1.1` guess, unrelated to which model won or its real uncertainty.
- *Fix:* backend now returns real historical series (capped to 365 points) and real per-model intervals derived from each model's own cross-validated RMSE (Random Forest keeps its own tree-variance-based interval, more informative than the generic fallback). Verified live in the browser against the real uploaded dataset.

**pandas 3.0 migration**
- *Root cause:* `infer_datetime_format` was removed entirely in pandas 3.0 (silently caught by a broad `except`, quietly disabling the fast date-parsing path for every dataset); separately, `mixed=True` was never valid pandas syntax at all — should have been `format="mixed"`, a pre-existing bug unrelated to the version bump.
- *Fix:* removed the deprecated arg, corrected the mixed-format call. Found via direct reproduction with a real "04/19/19 08:46"-style dataset that was failing silently.

---

## 11. Honest limitations

Interviewers respect "here's what it doesn't do and why" far more than a project that claims to do everything.

- **No auth or rate-limiting** — every endpoint, including the metered LLM ones, is open. Fine for a single-user/demo deployment; a real prerequisite before any public exposure.
- **No hierarchical reconciliation** — store→region→total forecasts aren't numerically consistent with each other. Explicitly scoped out as disproportionate to the rest of the project.
- **Exogenous features limited to price + one promo flag**, detected by column name — not arbitrary user-specified variables, and no holiday-flag detection from the data itself (Prophet's own country-based holiday calendar is separate and unaffected).
- **No intermittent-demand model beyond Croston-SBA** — fine for sparse demand, but the other 18 models still assume reasonably continuous data.
- **MAPE is unstable near zero-valued actuals** — observed directly in a real benchmark (values of 300%+ on a low-volume dataset); a known, documented limitation of the metric itself, not a bug.
- **Fixed hyperparameters** for XGBoost/RandomForest/DL — not tuned per dataset (only the statistical models and KNN/SVM search their own hyperparameters via CV).

---

## 12. "How would you scale this?"

### Q: Multiple backend instances / load balancer?
Move the in-memory forecast cache and session state to a shared store (Redis, or Postgres if relational queries become useful) — it's currently local to one process by design, fine for one instance but not shared across a fleet.

### Q: Larger datasets / more concurrent forecast requests?
The per-model timeout and adaptive CV fold count already bound cost as a function of data size rather than a fixed budget; the next step would be a task queue (Celery/RQ) instead of synchronous request-blocking training, so a slow 19-model run doesn't hold an HTTP connection open the whole time.

### Q: Deployment size concern?
`torch` alone adds ~400MB to the backend install — a real, named constraint when picking a hosting platform, solvable with a CPU-only wheel via a custom pip index if the platform doesn't need CUDA.

### Q: What's the actual bottleneck today?
Wall-clock training time for the full 19-model ensemble, dominated by the deep-learning family when they're not gated out, and by grid-searched statistical models (SARIMA/SARIMAX order search) on larger seasonal periods — both already have explicit cost-bounding logic (frequency gating, a max-seasonal-period threshold that skips grid search) rather than running unbounded.

---

## 13. Rapid-fire technical Q&A

### ML / forecasting theory

**Q: Why walk-forward CV instead of k-fold?**
Standard k-fold shuffles data, which leaks future information into training for time-ordered data. Walk-forward respects temporal order — every fold trains only on data that precedes its validation window.

**Q: RMSE vs. MAE — when do they disagree?**
RMSE squares errors, so it penalizes large individual misses more heavily; MAE weighs all errors linearly. A model with one big miss and otherwise-tiny errors will look worse on RMSE than MAE relative to a model with consistent medium errors.

**Q: Why does a negative R² make sense to report?**
R² < 0 means the model did worse than predicting the training-window mean for every point — a real, valid outcome on genuinely noisy data with weak autocorrelation, not a computation error. This was observed directly on 10 of 12 models on one real dataset and reported as-is rather than hidden.

**Q: LSTM vs. GRU — what's the actual difference?**
LSTM has three gates (input, forget, output) and a separate cell state; GRU merges this into two gates (update, reset) with no separate cell state — fewer parameters, often comparable performance, faster to train.

**Q: Why does XGBoost need a detrend step but Prophet doesn't?**
Prophet models trend explicitly as one of its additive components (piecewise-linear with changepoints) — extrapolation is built in. Tree-based models split feature space into regions bounded by training data; they have no mechanism to predict outside the range of targets they've seen.

### Engineering / system design

**Q: How do you avoid one bad model blocking the whole request?**
Each model call is wrapped individually — an exception or timeout in one model is caught and recorded as a failed leaderboard entry with its error message; the other 18 continue independently in their own threads.

**Q: How is per-model timeout enforced?**
Category-based timeouts (statistical/ML/DL each get their own budget, since DL training is inherently slower) enforced via the executor's future results with a timeout, not a hard kill — a model that exceeds its timeout is marked failed and the executor is shut down without waiting for it.

**Q: Multi-target forecasting — how does one bad target column affect the others?**
It doesn't — each target column in a multi-target request runs through the full pipeline independently, and a per-target failure is caught and reported as an error entry for just that target, not a failure of the whole request.

### If they ask about the research paper

**Q: You wrote an IEEE paper about this — what dataset, and were the results real?**
A real 1,000-transaction Kaggle-style retail dataset, aggregated to a 366-day daily series by the system's own preprocessing (documented exactly: 655 duplicate timestamps summed, 21 calendar gaps filled, 3 outliers capped). All reported numbers came from actually running the live system end-to-end, not fabricated — including the honest result that most models scored a negative R² on that particular noisy dataset, and that XGBoost had the best raw RMSE but lost to KNN on the composite score.

---

## 14. Behavioral question mapping

| If asked... | Use this story |
|---|---|
| "Tell me about a challenging bug" | The 87-minute ThreadPoolExecutor hang — root cause was subtle (a context-manager's implicit blocking shutdown), not obvious from the symptom. |
| "Tell me about a time you were wrong / changed your mind" | The differencing approach for tree models — benchmarked across 10 seeds, found it made things worse, reverted before shipping. |
| "Tell me about a time you found a problem no one asked about" | The fabricated `Math.random()` historical chart data — found by reading code during an unrelated audit, not from a bug report. |
| "How do you approach trade-offs with incomplete information?" | The exogenous-feature future-value problem — no way to know future price/promo, so an explicit, conservative rule per feature type was chosen instead of guessing. |
| "How do you validate your own work?" | The composite-scoring fix — reconstructed an 18-model test case specifically to confirm the fix changed the actual ranking outcome, not just that the formula looked more correct on paper. |
| "Describe a design decision you'd defend" | Choosing SQLite-if-ever over a database now — matching infrastructure to actual, not hypothetical, requirements. |

---

*Compiled from the actual implementation and its real debugging history — not a generic template. A browsable, navigable version of this same content is also available as a published web reference.*
