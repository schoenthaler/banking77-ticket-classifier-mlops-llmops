"""
Complete pipeline: Hyperparameter sweep → Find best → Quantize best
"""
from src.ml.hyperparameter_sweep import run_hyperparameter_sweep
from src.ml.find_and_quantize_best import find_and_quantize_best

def run_full_pipeline(
    sweep_experiment="banking77-distilbert-sweep",
    data_dir="data/processed/banking77"
):
    """
    Run complete pipeline:
    1. Hyperparameter sweep
    2. Find best model
    3. Quantize best model
    """
    print("="*80)
    print("FULL PIPELINE: Sweep → Best Model → Quantization")
    print("="*80)
    
    # Step 1: Run hyperparameter sweep
    print("\n" + "="*80)
    print("STEP 1: Hyperparameter Sweep")
    print("="*80)
    results, best = run_hyperparameter_sweep(
        data_dir=data_dir,
        experiment_name=sweep_experiment
    )
    
    if best is None:
        print("\n No successful runs. Cannot proceed to quantization.")
        return None
    
    # Step 2: Find and quantize best model
    print("\n" + "="*80)
    print("STEP 2: Find and Quantize Best Model")
    print("="*80)
    
    final_result = find_and_quantize_best(
        experiment_name=sweep_experiment,
        metric="final_val_accuracy",
        quantized_output_dir="models/quantized/best-model"
    )
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"Best model: {best['config']['run_name']}")
    print(f"Validation Accuracy: {best['val_accuracy']:.4f}")
    print(f"Quantized model: models/quantized/best-model")
    
    return final_result


if __name__ == "__main__":
    run_full_pipeline()