import mlflow
from mlflow.tracking import MlflowClient

def rename_run(run_id, new_name):
    """
    Rename an existing MLflow run.
    
    Args:
        run_id: The ID of the run to rename
        new_name: New name for the run
    """
    client = MlflowClient()
    client.set_tag(run_id, "mlflow.runName", new_name)
    print(f"Renamed run {run_id} to '{new_name}'")

def list_runs(experiment_name="banking77-distilbert"):
    """
    List all runs in an experiment.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found")
        return
    
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    print(f"\nRuns in experiment '{experiment_name}':")
    print("-" * 80)
    for idx, run in runs.iterrows():
        run_name = run.get('tags.mlflow.runName', 'Unnamed')
        run_id = run['run_id']
        val_acc = run.get('metrics.final_val_accuracy', 'N/A')
        print(f"{idx+1}. Name: {run_name}")
        print(f"   ID: {run_id}")
        print(f"   Val Accuracy: {val_acc}")
        print()
    
    return runs

def rename_runs_interactive(experiment_name="banking77-distilbert"):
    """
    Interactive script to rename runs.
    """
    runs = list_runs(experiment_name)
    
    if runs is None or len(runs) == 0:
        print("No runs found")
        return
    
    print("\nEnter run number to rename (or 'q' to quit):")
    choice = input("> ")
    
    if choice.lower() == 'q':
        return
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(runs):
            run_id = runs.iloc[idx]['run_id']
            current_name = runs.iloc[idx].get('tags.mlflow.runName', 'Unnamed')
            print(f"\nCurrent name: {current_name}")
            new_name = input("Enter new name: ")
            rename_run(run_id, new_name)
        else:
            print("Invalid run number")
    except ValueError:
        print("Invalid input")

if __name__ == "__main__":
    # List all runs
    list_runs()
    
    rename_runs_interactive()
    
    #  rename specific run by ID
    # rename_run("your-run-id-here", "new-name-here")