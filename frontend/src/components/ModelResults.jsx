import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Trophy, CheckCircle2, XCircle, Layers } from "lucide-react";
import { MODEL_CATEGORY, fmtNumber, fmtPct, fmtSeconds, isNum, modelLabel } from "../lib/format";

// Rows arrive from the backend already in rank order (best first): models are
// ranked by MASE on a common holdout backtest. MASE < 1 beats a seasonal-naive
// forecast.
export default function ModelResults({ results }) {
  const [showFailed, setShowFailed] = useState(false);
  const allModels = results?.all_models || [];
  if (!allModels.length) return null;

  const bestModelName = results.best_model?.best_model_name;
  const runInfo = results.model_run_info || {};
  const backtested = runInfo.evaluation === "holdout_backtest";
  const ok = allModels.filter((m) => m.status === "success");
  const failed = allModels.filter((m) => m.status !== "success");
  const ensembleMembers = results.best_model?.ensemble_members || allModels.find((m) => m.model_name === "Ensemble")?.ensemble_members;

  const rows = showFailed ? [...ok, ...failed] : ok;

  return (
    <Card className="shadow-sm">
      <CardHeader className="flex flex-row flex-wrap items-start justify-between gap-3 px-6 pt-5 pb-3">
        <div>
          <CardTitle className="text-lg font-semibold text-slate-900">Model Comparison</CardTitle>
          <p className="mt-1 text-sm text-slate-500">
            {backtested
              ? `Every model forecast the same ${runInfo.backtest_origins} held-out window(s) of ${runInfo.backtest_horizon} periods, ranked by MASE (below 1.00 beats a seasonal-naive forecast).`
              : "Series too short for a common backtest, so models are ranked by their own cross-validation error."}
          </p>
        </div>
        <div className="flex gap-2">
          <Badge variant="outline" className="border-green-300 bg-green-100 text-green-800">{ok.length} succeeded</Badge>
          {failed.length > 0 && (
            <Badge variant="outline" className="border-red-300 bg-red-100 text-red-800">{failed.length} failed</Badge>
          )}
        </div>
      </CardHeader>

      <CardContent className="px-6 pb-5">
        <div className="overflow-x-auto rounded-md border border-slate-200">
          <Table>
            <TableHeader className="bg-slate-50">
              <TableRow>
                <TableHead className="w-10 font-semibold text-slate-700">#</TableHead>
                <TableHead className="font-semibold text-slate-700">Model</TableHead>
                <TableHead className="text-right font-semibold text-slate-700" title="Mean absolute scaled error on the backtest windows">MASE</TableHead>
                <TableHead className="text-right font-semibold text-slate-700">RMSE</TableHead>
                <TableHead className="text-right font-semibold text-slate-700">MAE</TableHead>
                <TableHead className="text-right font-semibold text-slate-700">MAPE</TableHead>
                <TableHead className="text-right font-semibold text-slate-700">Time</TableHead>
                <TableHead className="text-center font-semibold text-slate-700">Status</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((model, i) => {
                const isBest = model.model_name === bestModelName;
                const isSuccess = model.status === "success";
                const time = isNum(model.train_time_sec) ? model.train_time_sec : model.elapsed_sec;
                const beatsNaive = isNum(model.mase) && model.mase < 1;

                return (
                  <TableRow key={model.model_name} className={isBest ? "bg-green-50/50 hover:bg-green-100/50" : ""}>
                    <TableCell className="font-mono text-xs text-slate-400">{isSuccess ? i + 1 : "—"}</TableCell>
                    <TableCell className="font-medium">
                      <div className="flex items-center gap-2">
                        {isBest && <Trophy className="h-4 w-4 text-green-600" />}
                        <span className={isBest ? "font-bold text-green-900" : "text-slate-900"}>{modelLabel(model.model_name)}</span>
                        <span className="text-xs text-slate-400">{MODEL_CATEGORY[model.model_name]}</span>
                      </div>
                      {!isSuccess && model.error && (
                        <p className="mt-1 max-w-md truncate text-xs text-red-600" title={model.error}>{model.error}</p>
                      )}
                    </TableCell>
                    <TableCell className={`text-right font-mono font-bold ${isBest ? "text-green-700" : beatsNaive ? "text-blue-700" : "text-slate-600"}`}>
                      {isNum(model.mase) ? model.mase.toFixed(3) : "—"}
                    </TableCell>
                    <TableCell className="text-right font-mono text-sm text-slate-600">{fmtNumber(model.rmse)}</TableCell>
                    <TableCell className="text-right font-mono text-sm text-slate-600">{fmtNumber(model.mae)}</TableCell>
                    <TableCell className="text-right font-mono text-sm text-slate-600">{fmtPct(model.mape)}</TableCell>
                    <TableCell className="text-right font-mono text-sm text-slate-500">{fmtSeconds(time)}</TableCell>
                    <TableCell className="text-center">
                      {isBest ? (
                        <Badge variant="outline" className="border-green-300 bg-green-100 text-green-800">
                          <CheckCircle2 className="mr-1 h-3 w-3" /> Best
                        </Badge>
                      ) : isSuccess ? (
                        <Badge variant="outline" className="border-blue-300 bg-blue-100 text-blue-800">
                          <CheckCircle2 className="mr-1 h-3 w-3" /> Success
                        </Badge>
                      ) : (
                        <Badge variant="outline" className="border-red-300 bg-red-100 text-red-800">
                          <XCircle className="mr-1 h-3 w-3" /> Failed
                        </Badge>
                      )}
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </div>
        {failed.length > 0 && (
          <button type="button" onClick={() => setShowFailed((v) => !v)} className="mt-3 text-xs font-medium text-slate-500 hover:text-slate-800">
            {showFailed ? "Hide" : "Show"} {failed.length} failed model{failed.length > 1 ? "s" : ""}
          </button>
        )}
      </CardContent>

      <CardFooter className="border-t border-slate-100 bg-slate-50/50 px-6 py-3 text-xs text-slate-600">
        <div className="grid w-full grid-cols-2 gap-4 sm:grid-cols-4">
          <div>
            <span className="block font-medium text-slate-500">Models run</span>
            <span className="font-semibold text-slate-900">{allModels.length}</span>
          </div>
          <div>
            <span className="block font-medium text-slate-500">Beat seasonal naive</span>
            <span className="font-semibold text-slate-900">{ok.filter((m) => isNum(m.mase) && m.mase < 1).length}</span>
          </div>
          <div>
            <span className="block font-medium text-slate-500">Best MASE</span>
            <span className="font-semibold text-green-700">{isNum(results.best_model?.mase) ? results.best_model.mase.toFixed(3) : "—"}</span>
          </div>
          {ensembleMembers?.length > 0 && (
            <div>
              <span className="flex items-center gap-1 font-medium text-slate-500"><Layers className="h-3 w-3" /> Ensemble of</span>
              <span className="font-semibold text-slate-900">{ensembleMembers.map(modelLabel).join(", ")}</span>
            </div>
          )}
        </div>
      </CardFooter>
    </Card>
  );
}
