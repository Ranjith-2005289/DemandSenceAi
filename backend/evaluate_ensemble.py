import json
import numpy as np

def calculate_mase(actual, forecast, denom):
    actual = np.array(actual)
    forecast = np.array(forecast)
    return np.mean(np.abs(actual - forecast)) / denom

def evaluate_ensemble(raw_json_path, k=3, budget=float('inf')):
    with open(raw_json_path, "r") as f:
        data = json.load(f)
        
    single_mases = []
    ensemble_mases = []
    num_models = []
    
    for series in data:
        actual = series["test_actual"]
        denom = series["mase_denom"]
        
        # Filter successful models within budget
        models = [
            m for m in series["records"] 
            if m.get("ok", False) 
            and m.get("cv_rmse") is not None
            and m.get("train_time", 0) <= budget
        ]
        
        if not models:
            continue
            
        # Sort by cv_rmse ascending
        models.sort(key=lambda x: x["cv_rmse"])
        
        # 1. Single best model
        best_model = models[0]
        single_mases.append(best_model["test_mase"])
        
        # 2. Ensemble of top K models
        top_k = models[:k]
        
        # Inverse RMSE weights
        weights = np.array([1.0 / m["cv_rmse"] if m["cv_rmse"] > 0 else 1.0 for m in top_k])
        weights /= weights.sum()
        
        # Weighted forecast
        forecasts = np.array([m["forecast"] for m in top_k])
        ensemble_forecast = np.average(forecasts, axis=0, weights=weights)
        
        # Calculate ensemble MASE
        ens_mase = calculate_mase(actual, ensemble_forecast, denom)
        ensemble_mases.append(ens_mase)
        num_models.append(len(top_k))
        
    median_single = np.median(single_mases)
    median_ensemble = np.median(ensemble_mases)
    mean_single = np.mean(single_mases)
    mean_ensemble = np.mean(ensemble_mases)
    mean_num_models = np.mean(num_models)
    
    budget_str = f"{budget}s" if budget != float('inf') else "unbounded"
    print(f"=== Results for {raw_json_path} (Top-{k} Ensemble, Budget: {budget_str}) ===")
    print(f"Single Pick:     Median MASE = {median_single:.4f} | Mean MASE = {mean_single:.4f}")
    print(f"Top-{k} Ensemble: Median MASE = {median_ensemble:.4f} | Mean MASE = {mean_ensemble:.4f}")
    print(f"Improvement (Median):  {median_single - median_ensemble:.4f}")
    print(f"Improvement (Mean):    {mean_single - mean_ensemble:.4f}")
    print(f"Average models ensembled: {mean_num_models:.2f}")
    print()

if __name__ == "__main__":
    evaluate_ensemble("results_l10.raw.json", k=3, budget=float('inf'))
    evaluate_ensemble("results_l10.raw.json", k=3, budget=2.0)
    evaluate_ensemble("results_l12.raw.json", k=3, budget=float('inf'))
    evaluate_ensemble("results_l12.raw.json", k=3, budget=1.5)
    evaluate_ensemble("results_l9_v2.raw.json", k=3, budget=float('inf'))
    evaluate_ensemble("results_l9_v2.raw.json", k=3, budget=1.5)
