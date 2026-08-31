import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Check, CheckCircle2, Loader2 } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

export default function StatusTracker({ status, results }) {
  const [steps, setSteps] = useState({
    upload: false,
    preprocessing: false,
    training: { arima: false, prophet: false, xgboost: false, rf: false },
    selection: false,
    forecast: false,
  });

  // Simulate step completion timing based on whether results are available
  useEffect(() => {
    if (!results) {
      setSteps({
        upload: false,
        preprocessing: false,
        training: { arima: false, prophet: false, xgboost: false, rf: false },
        selection: false,
        forecast: false,
      });
      return;
    }

    const timings = [
      { step: "upload", delay: 500 },
      { step: "preprocessing", delay: 2500 },
      { step: "training.arima", delay: 4000 },
      { step: "training.prophet", delay: 5500 },
      { step: "training.xgboost", delay: 7000 },
      { step: "training.rf", delay: 8500 },
      { step: "selection", delay: 9500 },
      { step: "forecast", delay: 10500 },
    ];

    const timeouts = timings.map(({ step, delay }) =>
      setTimeout(() => {
        setSteps((prev) => {
          if (step.startsWith("training.")) {
            const model = step.split(".")[1];
            return {
              ...prev,
              training: { ...prev.training, [model]: true },
            };
          }
          return { ...prev, [step]: true };
        });
      }, delay)
    );

    return () => timeouts.forEach((t) => clearTimeout(t));
  }, [results]);

  return (
    <Card className="shadow-sm border border-slate-200 overflow-hidden">
      {/* Custom Header */}
      <div className="px-7 pt-6 pb-5 border-b border-slate-100 bg-white">
        <p className="text-xs font-bold uppercase tracking-widest text-slate-400">Status</p>
        <p className="mt-1 text-sm font-semibold text-slate-900">Forecast Progress</p>
      </div>
      
      <CardContent className="px-7 py-6 space-y-4 bg-white">
        {/* Step 1: Upload */}
        <StepIndicator
          label="Upload Complete"
          completed={steps.upload}
          icon={steps.upload ? <Check className="h-4 w-4" /> : "1"}
        />

        {/* Step 2: Preprocessing */}
        <StepIndicator
          label="Preprocessing Data"
          completed={steps.preprocessing}
          icon={steps.preprocessing ? <Check className="h-4 w-4" /> : "2"}
          disabled={!steps.upload}
        />

        {/* Step 3: Training Models */}
        <div className={`${!steps.preprocessing ? "opacity-50" : ""}`}>
          <StepIndicator
            label="Training Models"
            completed={
              steps.training.arima &&
              steps.training.prophet &&
              steps.training.xgboost &&
              steps.training.rf
            }
            icon={
              steps.training.arima &&
              steps.training.prophet &&
              steps.training.xgboost &&
              steps.training.rf
                ? <Check className="h-4 w-4" />
                : "3"
            }
            disabled={!steps.preprocessing}
            showSubSteps
          >
            <div className="mt-3 space-y-2 pl-7">
              <ModelTrainingRow model="ARIMA" completed={steps.training.arima} />
              <ModelTrainingRow model="Prophet" completed={steps.training.prophet} />
              <ModelTrainingRow model="XGBoost" completed={steps.training.xgboost} />
              <ModelTrainingRow model="RandomForest" completed={steps.training.rf} />
            </div>
          </StepIndicator>
        </div>

        {/* Step 4: Selection */}
        <StepIndicator
          label="Selecting Best Model"
          completed={steps.selection}
          icon={steps.selection ? <Check className="h-4 w-4" /> : "4"}
          disabled={
            !steps.training.arima ||
            !steps.training.prophet ||
            !steps.training.xgboost ||
            !steps.training.rf
          }
        />

        {/* Step 5: Forecast */}
        <StepIndicator
          label="Generating Forecast"
          completed={steps.forecast}
          icon={steps.forecast ? <Check className="h-4 w-4" /> : "5"}
          disabled={!steps.selection}
        />

        {/* Final status message */}
        {steps.forecast && (
          <Alert className="mt-6 border-green-200 bg-green-50 shadow-sm">
            <CheckCircle2 className="h-4 w-4 text-green-600 mt-1" />
            <AlertTitle className="text-green-900 font-bold">Forecast Ready</AlertTitle>
            <AlertDescription className="text-green-800">
              All steps completed successfully. Review results below.
            </AlertDescription>
          </Alert>
        )}
      </CardContent>

      {/* Current status text */}
      <div className="px-7 py-5 border-t border-slate-100 bg-slate-50">
        <p className="text-xs font-medium uppercase text-slate-500">Current Status</p>
        <p className="mt-1 text-sm font-medium text-slate-700">{status || "Idle"}</p>
      </div>
    </Card>
  );
}

function StepIndicator({ label, completed, icon, disabled, showSubSteps, children }) {
  return (
    <div>
      <div className="flex items-start gap-3">
        <div
          className={`mt-0.5 flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-full border-2 font-semibold transition-all ${
            completed
              ? "border-green-500 bg-green-500 text-white"
              : disabled
              ? "border-slate-300 bg-slate-100 text-slate-400"
              : "border-blue-500 bg-blue-50 text-blue-600"
          }`}
        >
          {icon}
        </div>
        <div className="flex-1">
          <p
            className={`text-sm font-medium transition-all ${
              completed
                ? "text-green-700"
                : disabled
                ? "text-slate-400"
                : "text-slate-900"
            }`}
          >
            {label}
          </p>
          {completed && (
            <p className="mt-0.5 text-xs font-medium text-green-600">Completed</p>
          )}
        </div>
      </div>
      {showSubSteps && children}
    </div>
  );
}

function ModelTrainingRow({ model, completed }) {
  return (
    <div className="flex items-center gap-2">
      <div
        className={`h-1.5 w-1.5 rounded-full transition-all ${
          completed ? "bg-green-500" : "bg-slate-300"
        }`}
      />
      <span
        className={`text-xs font-medium transition-all ${
          completed ? "text-green-700" : "text-slate-600"
        }`}
      >
        {model}
      </span>
      {completed ? (
        <Check className="h-3 w-3 text-green-600 ml-1" />
      ) : (
        <Loader2 className="ml-auto h-3 w-3 animate-spin text-blue-400" />
      )}
    </div>
  );
}
