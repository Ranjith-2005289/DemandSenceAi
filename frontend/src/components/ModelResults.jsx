import { Card, CardContent, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Trophy, CheckCircle2, XCircle } from "lucide-react";

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
    <Card className="shadow-sm">
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-lg font-semibold text-slate-900">Model Comparison</CardTitle>
        {bestModelName && (
          <Badge variant="secondary" className="bg-green-100 text-green-800 hover:bg-green-200">
            <Trophy className="mr-1 h-3 w-3" /> Best Model Selected
          </Badge>
        )}
      </CardHeader>

      <CardContent>
        <div className="rounded-md border border-slate-200">
          <Table>
            <TableHeader className="bg-slate-50">
              <TableRow>
                <TableHead className="font-semibold text-slate-700">Model Name</TableHead>
                <TableHead className="text-right font-semibold text-slate-700">Weighted Score</TableHead>
                <TableHead className="text-right font-semibold text-slate-700">RMSE</TableHead>
                <TableHead className="text-right font-semibold text-slate-700">MAE</TableHead>
                <TableHead className="text-right font-semibold text-slate-700">MAPE</TableHead>
                <TableHead className="text-right font-semibold text-slate-700">R²</TableHead>
                <TableHead className="text-right font-semibold text-slate-700">Time (s)</TableHead>
                <TableHead className="text-center font-semibold text-slate-700">Status</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {sortedModels.map((model) => {
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
                  <TableRow
                    key={model.model_name}
                    className={isBest ? "bg-green-50/50 hover:bg-green-100/50" : ""}
                  >
                    <TableCell className="font-medium">
                      <div className="flex items-center gap-2">
                        {isBest && <Trophy className="h-4 w-4 text-green-600" />}
                        <span className={isBest ? "text-green-900 font-bold" : "text-slate-900"}>
                          {model.model_name}
                        </span>
                      </div>
                    </TableCell>
                    <TableCell className={`text-right font-mono font-bold ${isBest ? "text-green-700" : "text-blue-700"}`}>
                      {score}
                    </TableCell>
                    <TableCell className="text-right font-mono text-sm text-slate-600">{rmse}</TableCell>
                    <TableCell className="text-right font-mono text-sm text-slate-600">{mae}</TableCell>
                    <TableCell className="text-right font-mono text-sm text-slate-600">{mape}</TableCell>
                    <TableCell className={`text-right font-mono text-sm ${isBest ? "text-green-700" : "text-slate-600"}`}>
                      {r2}
                    </TableCell>
                    <TableCell className="text-right font-mono text-sm text-slate-500">{totalTime}</TableCell>
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
      </CardContent>

      {bestModelName && (
        <CardFooter className="border-t border-slate-100 bg-slate-50/50 py-3 text-xs text-slate-600">
          <div className="grid w-full grid-cols-2 gap-4 sm:grid-cols-3">
            <div>
              <span className="block font-medium text-slate-500">Total Models</span>
              <span className="font-semibold text-slate-900">{sortedModels.length}</span>
            </div>
            <div>
              <span className="block font-medium text-slate-500">Successful</span>
              <span className="font-semibold text-slate-900">
                {sortedModels.filter((m) => m.status === "success").length}
              </span>
            </div>
            <div>
              <span className="block font-medium text-slate-500">Best Weighted Score</span>
              <span className="font-semibold text-green-700">
                {results.best_model?.score?.toFixed(4) || "—"}
              </span>
            </div>
          </div>
        </CardFooter>
      )}
    </Card>
  );
}
