import os
import mlflow
import mlflow.pytorch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

from src.ml.model_def import create_model
from src.data.label_mapping import get_label_mappings


def compute_metrics(eval_pred):
    """
    Compute metrics for evaluation.
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


class MLflowCallback(TrainerCallback):
    """Custom callback to log metrics to MLflow during training."""
    
    def __init__(self):
        super().__init__()
        self.current_epoch = 0
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log metrics at each logging step."""
        if logs:
            # Log training metrics
            if 'loss' in logs:
                mlflow.log_metric("train_loss", logs['loss'], step=state.global_step)
            if 'learning_rate' in logs:
                mlflow.log_metric("learning_rate", logs['learning_rate'], step=state.global_step)
        return control
    
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        """Log evaluation metrics."""
        if logs:
            for key, value in logs.items():
                if key.startswith('eval_'):
                    metric_name = key.replace('eval_', '')
                    mlflow.log_metric(f"val_{metric_name}", value, step=state.global_step)
        return control

def train_distilbert_mlflow(
    data_dir="data/processed/banking77",
    output_dir="models/distilbert-banking77",
    num_epochs=3,
    batch_size=16,
    learning_rate=2e-5,
    experiment_name="banking77-distilbert",
    run_name=None 
):
    """
    Train DistilBERT on Banking77 dataset with MLflow tracking.
    """
    # Set up MLflow
    mlflow.set_experiment(experiment_name)

    # Generate run name if not provided
    if run_name is None:
        run_name = f"distilbert-{num_epochs}ep-lr{learning_rate}-bs{batch_size}"
    
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params({
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "model_name": "distilbert-base-uncased",
            "max_length": 64,
            "weight_decay": 0.01
        })
        
        # Load tokenized datasets
        print("Loading tokenized datasets...")
        train_dataset = load_from_disk(os.path.join(data_dir, "train"))
        val_dataset = load_from_disk(os.path.join(data_dir, "validation"))
        test_dataset = load_from_disk(os.path.join(data_dir, "test"))
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        # Log dataset info
        mlflow.log_params({
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "test_samples": len(test_dataset)
        })
        
        # Get label mappings
        id_to_label, label_to_id = get_label_mappings()
        num_labels = len(id_to_label)
        print(f"\nNumber of classes: {num_labels}")
        mlflow.log_param("num_labels", num_labels)
        
        # Load tokenizer
        print("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        # Create model
        print("\nCreating model...")
        model = create_model(num_labels=num_labels, id_to_label=id_to_label, label_to_id=label_to_id)
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # MLflow callback
        mlflow_callback = MLflowCallback()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=100,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            save_total_limit=2,
            report_to="none"
        )
        
        # Create trainer
        print("\nInitializing trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[mlflow_callback]
        )
        
        # Train
        print("\nStarting training...")
        trainer.train()
        
        # Evaluate on validation set
        print("\nEvaluating on validation set...")
        eval_results = trainer.evaluate()
        print(f"\nValidation Accuracy: {eval_results['eval_accuracy']:.4f}")
        
        # Log final validation metrics
        for key, value in eval_results.items():
            if key.startswith('eval_'):
                metric_name = key.replace('eval_', '')
                mlflow.log_metric(f"final_val_{metric_name}", value)
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_results = trainer.evaluate(eval_dataset=test_dataset)
        print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")
        
        # Log final test metrics
        for key, value in test_results.items():
            if key.startswith('eval_'):
                metric_name = key.replace('eval_', '')
                mlflow.log_metric(f"final_test_{metric_name}", value)
        
        # Save final model
        print(f"\nSaving model to {output_dir}...")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        # Log model as artifact
        mlflow.log_artifacts(output_dir, artifact_path="model")
        
        # Log model using MLflow's PyTorch flavor (optional)
        # mlflow.pytorch.log_model(model, "pytorch_model")
        
        print("\nTraining complete!")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")
        print(f"View results: mlflow ui")
        
        return trainer, eval_results, test_results


if __name__ == "__main__":
    train_distilbert_mlflow()