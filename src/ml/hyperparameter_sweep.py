import os
import mlflow
from src.ml.train_mlflow import train_distilbert_mlflow

def run_hyperparameter_sweep(
    data_dir="data/processed/banking77",
    experiment_name="banking77-distilbert-sweep"
):
    """
    Run automated hyperparameter sweep with different configurations.
    All results are tracked in MLflow.
    """
    # Define hyperparameter configurations to test
    configs = [
        # Baseline
        {
            "num_epochs": 3,
            "batch_size": 16,
            "learning_rate": 2e-5,
            "run_name": "Baseline-3ep-LR2e-5-BS16"
        },
        # Different learning rates
        {
            "num_epochs": 3,
            "batch_size": 16,
            "learning_rate": 1e-5,
            "run_name": "LowLR-3ep-LR1e-5-BS16"
        },
        {
            "num_epochs": 3,
            "batch_size": 16,
            "learning_rate": 3e-5,
            "run_name": "MedLR-3ep-LR3e-5-BS16"
        },
        {
            "num_epochs": 3,
            "batch_size": 16,
            "learning_rate": 5e-5,
            "run_name": "HighLR-3ep-LR5e-5-BS16"
        },
        # Different batch sizes
        {
            "num_epochs": 3,
            "batch_size": 8,
            "learning_rate": 2e-5,
            "run_name": "SmallBS-3ep-LR2e-5-BS8"
        },
        {
            "num_epochs": 3,
            "batch_size": 32,
            "learning_rate": 2e-5,
            "run_name": "LargeBS-3ep-LR2e-5-BS32"
        },
        # More epochs
        {
            "num_epochs": 5,
            "batch_size": 16,
            "learning_rate": 2e-5,
            "run_name": "MoreEpochs-5ep-LR2e-5-BS16"
        },
        {
            "num_epochs": 5,
            "batch_size": 16,
            "learning_rate": 1e-5,
            "run_name": "MoreEpochs-5ep-LR1e-5-BS16"
        },
        # Combinations
        {
            "num_epochs": 5,
            "batch_size": 32,
            "learning_rate": 3e-5,
            "run_name": "Combo-5ep-LR3e-5-BS32"
        },
    ]
    
    print("="*80)
    print("Hyperparameter Sweep")
    print("="*80)
    print(f"Total configurations to test: {len(configs)}")
    print(f"Experiment: {experiment_name}\n")
    
    results = []
    
    for i, config in enumerate(configs, 1):
        print("\n" + "="*80)
        print(f"Configuration {i}/{len(configs)}: {config['run_name']}")
        print("="*80)
        
        # Create unique output directory for each run
        output_dir = f"models/distilbert-{config['run_name'].lower().replace('-', '_')}"
        
        try:
            # Train with this configuration
            trainer, eval_results, test_results = train_distilbert_mlflow(
                data_dir=data_dir,
                output_dir=output_dir,
                num_epochs=config["num_epochs"],
                batch_size=config["batch_size"],
                learning_rate=config["learning_rate"],
                experiment_name=experiment_name
            )
            
            # Store results
            results.append({
                "config": config,
                "output_dir": output_dir,
                "val_accuracy": eval_results.get('eval_accuracy', 0),
                "test_accuracy": test_results.get('eval_accuracy', 0),
                "run_id": mlflow.active_run().info.run_id if mlflow.active_run() else None
            })
            
            print(f"✓ Completed: {config['run_name']}")
            print(f"  Val Accuracy: {eval_results.get('eval_accuracy', 0):.4f}")
            print(f"  Test Accuracy: {test_results.get('eval_accuracy', 0):.4f}")
            
        except Exception as e:
            print(f"✗ Failed: {config['run_name']}")
            print(f"  Error: {e}")
            results.append({
                "config": config,
                "output_dir": output_dir,
                "val_accuracy": 0,
                "test_accuracy": 0,
                "error": str(e)
            })
    
    # Summary
    print("\n" + "="*80)
    print("Sweep Summary")
    print("="*80)
    
    successful = [r for r in results if r.get('val_accuracy', 0) > 0]
    if successful:
        best = max(successful, key=lambda x: x['val_accuracy'])
        print(f"\nBest Configuration: {best['config']['run_name']}")
        print(f"  Validation Accuracy: {best['val_accuracy']:.4f}")
        print(f"  Test Accuracy: {best['test_accuracy']:.4f}")
        print(f"  Model Path: {best['output_dir']}")
        if best.get('run_id'):
            print(f"  MLflow Run ID: {best['run_id']}")
    
    print(f"\nTotal runs: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(results) - len(successful)}")
    
    return results, best if successful else None


if __name__ == "__main__":
    results, best = run_hyperparameter_sweep()