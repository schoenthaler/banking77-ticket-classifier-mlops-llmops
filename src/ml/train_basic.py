import os
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score
import numpy as np

from src.ml.model_def import create_model
from src.data.label_mapping import get_label_mappings


def compute_metrics(eval_pred):
    """
    Compute accuracy metric for evaluation.
    
    Args:
        eval_pred: Tuple of (predictions, labels)
        
    Returns:
        Dictionary with accuracy score
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}


def train_distilbert(
    data_dir="data/processed/banking77",
    output_dir="models/distilbert-banking77",
    num_epochs=3,
    batch_size=16,
    learning_rate=2e-5
):
    """
    Train DistilBERT on Banking77 dataset.
    
    Args:
        data_dir: Directory containing tokenized datasets
        output_dir: Directory to save trained model
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for training
    """
    # Load tokenized datasets
    print("Loading tokenized datasets...")
    train_dataset = load_from_disk(os.path.join(data_dir, "train"))
    val_dataset = load_from_disk(os.path.join(data_dir, "validation"))
    test_dataset = load_from_disk(os.path.join(data_dir, "test"))
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Get label mappings
    id_to_label, label_to_id = get_label_mappings()
    num_labels = len(id_to_label)
    print(f"\nNumber of classes: {num_labels}")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Create model
    print("\nCreating model...")
    model = create_model(num_labels=num_labels, id_to_label=id_to_label, label_to_id=label_to_id)
    
    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
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
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    eval_results = trainer.evaluate()
    print(f"\nValidation Accuracy: {eval_results['eval_accuracy']:.4f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")
    
    # Save final model
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print("\nTraining complete!")
    return trainer, eval_results, test_results


if __name__ == "__main__":
    train_distilbert()

