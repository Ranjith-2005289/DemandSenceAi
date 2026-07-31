export default function ModelResults({ results }) {
  if (!results?.all_models?.length) return null;

  const allModels = results.all_models || [];
  const bestModelName = results.best_model?.best_model_name;

  // Sort models by score ascending
  const sortedModels = [...allModels].sort((a, b) => {
    const scoreA = Number.isFinite(a.score) ? a.score : Infinity;
    const scoreB = Number.isFinite(b.score) ? b.score : Infinity;
    return scoreA - scoreB;
  });

  return (
    <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
      <div className="mb-4 flex flex-col items-start justify-between gap-3 sm:flex-row sm:items-center">
        <h2 className="text-lg font-semibold text-slate-900">Model Comparison</h2>
        {bestModelName && (
          <span className="inline-flex items-center gap-1 rounded-full border border-green-200 bg-green-50 px-3 py-1 text-xs font-semibold text-green-800">
            <span className="text-sm">🏆</span>
            Best Model Selected
          </span>
        )}
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-slate-200 text-sm">
          <thead className="bg-slate-50">
            <tr>
              <th className="px-4 py-3 text-left font-semibold text-slate-600">Model Name</th>
              <th className="px-4 py-3 text-right font-semibold text-slate-600">Weighted Score</th>
              <th className="px-4 py-3 text-right font-semibold text-slate-600">RMSE</th>
              <th className="px-4 py-3 text-right font-semibold text-slate-600">MAE</th>
              <th className="px-4 py-3 text-right font-semibold text-slate-600">MAPE</th>
              <th className="px-4 py-3 text-right font-semibold text-slate-600">R²</th>
              <th className="px-4 py-3 text-right font-semibold text-slate-600">Time (s)</th>
              <th className="px-4 py-3 text-left font-semibold text-slate-600">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {sortedModels.map((model, index) => {
              const isBest = model.model_name === bestModelName;
              const isSuccess = model.status === "success";
              const score = Number.isFinite(model.score) ? model.score.toFixed(4) : "—";
              const rmse = Number.isFinite(model.rmse) ? model.rmse.toFixed(2) : "—";
              const mae = Number.isFinite(model.mae) ? model.mae.toFixed(2) : "—";
              const mape = Number.isFinite(model.mape) ? (model.mape * 100).toFixed(1) + "%" : "—";
              const r2 = Number.isFinite(model.r2) ? model.r2.toFixed(3) : "—";
              
              // Train + Pred Time
              const tTrain = Number.isFinite(model.train_time_sec) ? model.train_time_sec : (model.elapsed_sec || 0);
              const tPred = Number.isFinite(model.pred_time_sec) ? model.pred_time_sec : 0;
              const totalTime = (tTrain + tPred).toFixed(2);

              return (
                <tr
                  key={model.model_name}
                  className={`transition-colors ${
                    isBest
                      ? "border-l-4 border-l-green-500 bg-green-50 hover:bg-green-100"
                      : "hover:bg-slate-50"
                  }`}
                >
                  {/* Model Name */}
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      {isBest && (
                        <span className="text-lg" title="Best Model">
                          🏆
                        </span>
                      )}
                      <span
                        className={`font-semibold ${
                          isBest ? "text-green-900" : "text-slate-900"
                        }`}
                      >
                        {model.model_name}
                      </span>
                    </div>
                  </td>

                  {/* Weighted Score */}
                  <td className={`px-4 py-3 text-right font-mono font-bold ${isBest ? "text-green-700" : "text-blue-700"}`}>
                    {score}
                  </td>

                  {/* RMSE */}
                  <td className="px-4 py-3 text-right font-mono text-sm text-slate-600">
                    {rmse}
                  </td>

                  {/* MAE */}
                  <td className="px-4 py-3 text-right font-mono text-sm text-slate-600">
                    {mae}
                  </td>

                  {/* MAPE */}
                  <td className="px-4 py-3 text-right font-mono text-sm text-slate-600">
                    {mape}
                  </td>

                  {/* R² */}
                  <td className={`px-4 py-3 text-right font-mono text-sm ${isBest ? "text-green-700" : "text-slate-600"}`}>
                    {r2}
                  </td>

                  {/* Time */}
                  <td className="px-4 py-3 text-right font-mono text-sm text-slate-500">
                    {totalTime}
                  </td>

                  {/* Status */}
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      {isBest ? (
                        <span className="inline-flex items-center gap-1 rounded-full bg-green-200 px-2 py-1 text-xs font-semibold text-green-800">
                          <span>✓</span> Best
                        </span>
                      ) : isSuccess ? (
                        <span className="inline-flex items-center gap-1 rounded-full bg-blue-100 px-2 py-1 text-xs font-semibold text-blue-800">
                          <span>✓</span> Success
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 rounded-full bg-red-100 px-2 py-1 text-xs font-semibold text-red-800">
                          <span>✕</span> Failed
                        </span>
                      )}
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Stats footer */}
      {bestModelName && (
        <div className="mt-4 border-t border-slate-200 pt-3">
          <div className="grid grid-cols-2 gap-3 text-xs sm:grid-cols-3">
            <div>
              <p className="text-slate-600">Total Models</p>
              <p className="font-semibold text-slate-900">{sortedModels.length}</p>
            </div>
            <div>
              <p className="text-slate-600">Successful</p>
              <p className="font-semibold text-slate-900">
                {sortedModels.filter((m) => m.status === "success").length}
              </p>
            </div>
            <div>
              <p className="text-slate-600">Best Weighted Score</p>
              <p className="font-semibold text-green-700">
                {results.best_model?.score?.toFixed(4) || "—"}
              </p>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
