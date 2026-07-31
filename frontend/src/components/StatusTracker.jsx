import { useEffect, useState } from "react";

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
      // Reset if results cleared
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
    <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
      <h2 className="text-lg font-semibold text-slate-900">Forecast Progress</h2>

      <div className="mt-6 space-y-4">
        {/* Step 1: Upload */}
        <StepIndicator
          label="Upload Complete"
          completed={steps.upload}
          icon={steps.upload ? "✓" : "1"}
        />

        {/* Step 2: Preprocessing */}
        <StepIndicator
          label="Preprocessing Data"
          completed={steps.preprocessing}
          icon={steps.preprocessing ? "✓" : "2"}
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
                ? "✓"
                : "3"
            }
            disabled={!steps.preprocessing}
            showSubSteps
          >
            <div className="mt-3 space-y-2 pl-7">
              <ModelTrainingRow
                model="ARIMA"
                completed={steps.training.arima}
              />
              <ModelTrainingRow
                model="Prophet"
                completed={steps.training.prophet}
              />
              <ModelTrainingRow
                model="XGBoost"
                completed={steps.training.xgboost}
              />
              <ModelTrainingRow
                model="RandomForest"
                completed={steps.training.rf}
              />
            </div>
          </StepIndicator>
        </div>

        {/* Step 4: Selection */}
        <StepIndicator
          label="Selecting Best Model"
          completed={steps.selection}
          icon={steps.selection ? "✓" : "4"}
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
          icon={steps.forecast ? "✓" : "5"}
          disabled={!steps.selection}
        />

        {/* Final status message */}
        {steps.forecast && (
          <div className="mt-6 rounded-lg border border-green-200 bg-green-50 p-3">
            <p className="text-sm font-medium text-green-900">✓ Forecast Ready</p>
            <p className="mt-1 text-xs text-green-800">
              All steps completed successfully. Review results below.
            </p>
          </div>
        )}
      </div>

      {/* Current status text */}
      <div className="mt-4 border-t border-slate-200 pt-3">
        <p className="text-xs font-medium uppercase text-slate-600">Current Status</p>
        <p className="mt-1 text-sm text-slate-700">{status || "Idle"}</p>
      </div>
    </section>
  );
}

function StepIndicator({
  label,
  completed,
  icon,
  disabled,
  showSubSteps,
  children,
}) {
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
            <p className="mt-0.5 text-xs text-green-600">Completed</p>
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
      {completed && <span className="text-xs text-green-600">✓</span>}
      {!completed && (
        <span className="ml-auto h-2 w-8 animate-pulse rounded-full bg-blue-300" />
      )}
    </div>
  );
}
