import os
import mlflow
from mlflow.tracking import MlflowClient
from src.ml.quantize_model import quantize_distilbert

def find_best_model(experiment_name="banking77-distilbert-sweep", metric="final_val_accuracy"):
    """
    Find the best model from MLflow experiments.
    
    Args:
        experiment_name: Name of the MLflow experiment
        metric: Metric to use for ranking (default: final_val_accuracy)
        
    Returns:
        Dictionary with best run info and model path
    """
    print("="*80)
    print("Finding Best Model from MLflow")
    print("="*80)
    
    # Search for runs
    runs = mlflow.search_runs(experiment_names=[experiment_name])
    
    if len(runs) == 0:
        print(f"No runs found in experiment '{experiment_name}'")
        return None
    
    print(f"Found {len(runs)} runs")
    
    # Filter runs with the metric
    runs_with_metric = runs[runs[f'metrics.{metric}'].notna()]
    
    if len(runs_with_metric) == 0:
        print(f"No runs with metric '{metric}' found")
        # Try alternative metric names
        for alt_metric in ['eval_accuracy', 'metrics.eval_accuracy', 'final_test_accuracy']:
            if alt_metric in runs.columns:
                runs_with_metric = runs[runs[alt_metric].notna()]
                metric = alt_metric
                break
    
    if len(runs_with_metric) == 0:
        print("No runs with accuracy metrics found")
        return None
    
    # Find best run
    best_run = runs_with_metric.loc[runs_with_metric[f'metrics.{metric}'].idxmax()]
    
    run_id = best_run['run_id']
    best_accuracy = best_run[f'metrics.{metric}']
    
    print(f"\nBest Run:")
    print(f"  Run ID: {run_id}")
    print(f"  Run Name: {best_run.get('tags.mlflow.runName', 'Unnamed')}")
    print(f"  {metric}: {best_accuracy:.4f}")
    
    # Get model path from artifacts or parameters
    client = MlflowClient()
    artifacts = client.list_artifacts(run_id)
    
    model_path = None
    for artifact in artifacts:
        if 'model' in artifact.path.lower() or artifact.path == 'model':
            # Download artifact to get model path
            local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact.path)
            if os.path.isdir(local_path):
                model_path = local_path
                break
    
    # Alternative: check if model path is in parameters
    if model_path is None:
        original_model_path = best_run.get('params.original_model_path', None)
        if original_model_path and os.path.exists(original_model_path):
            model_path = original_model_path
    
    # Alternative: construct from output_dir parameter
    if model_path is None:
        # Try to find from run name or construct path
        run_name = best_run.get('tags.mlflow.runName', '')
        if run_name:
            # Construct path based on naming convention
            model_path = f"models/distilbert-{run_name.lower().replace('-', '_')}"
            if not os.path.exists(model_path):
                model_path = None
    
    if model_path is None:
        print("\n⚠ Could not automatically find model path")
        print("Please specify model path manually")
        return {
            "run_id": run_id,
            "accuracy": best_accuracy,
            "model_path": None
        }
    
    print(f"  Model Path: {model_path}")
    
    return {
        "run_id": run_id,
        "accuracy": best_accuracy,
        "model_path": model_path,
        "run_name": best_run.get('tags.mlflow.runName', 'Unnamed'),
        "params": {
            "num_epochs": best_run.get('params.num_epochs'),
            "batch_size": best_run.get('params.batch_size'),
            "learning_rate": best_run.get('params.learning_rate'),
        }
    }


def find_and_quantize_best(
    experiment_name="banking77-distilbert-sweep",
    metric="final_val_accuracy",
    quantized_output_dir="models/quantized/best-model"
):
    """
    Find the best model from MLflow and quantize it.
    
    Args:
        experiment_name: MLflow experiment name
        metric: Metric to use for ranking
        quantized_output_dir: Where to save quantized model
    """
    # Find best model
    best_info = find_best_model(experiment_name, metric)
    
    if best_info is None or best_info.get('model_path') is None:
        print("\n Could not find best model. Please run training first.")
        return None
    
    model_path = best_info['model_path']
    
    # Verify model exists
    if not os.path.exists(model_path):
        print(f"\n Model path does not exist: {model_path}")
        print("Please check the path or run training again.")
        return None
    
    print("\n" + "="*80)
    print("Quantizing Best Model")
    print("="*80)
    print(f"Model: {best_info['run_name']}")
    print(f"Accuracy: {best_info['accuracy']:.4f}")
    print(f"Path: {model_path}")
    
    # Quantize
    quantized_model, stats = quantize_distilbert(
        model_path=model_path,
        output_path=quantized_output_dir,
        log_to_mlflow=True,
        experiment_name=experiment_name
    )
    
    print("\n" + "="*80)
    print("Complete!")
    print("="*80)
    print(f"Best model: {best_info['run_name']}")
    print(f"Quantized model saved to: {quantized_output_dir}")
    
    return {
        "best_model": best_info,
        "quantized_stats": stats
    }


if __name__ == "__main__":
    # Option 1: Find best from sweep experiment
    find_and_quantize_best(
        experiment_name="banking77-distilbert-sweep",
        metric="final_val_accuracy"
    )
    
    # Option 2: Find best from regular experiment
    # find_and_quantize_best(
    #     experiment_name="banking77-distilbert",
    #     metric="final_val_accuracy"
    # )