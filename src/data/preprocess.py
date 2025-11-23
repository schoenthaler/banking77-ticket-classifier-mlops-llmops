import os
from transformers import AutoTokenizer
from datasets import load_from_disk

from src.data.load_data import load_banking77


def tokenize_function(examples, tokenizer, max_length=64):
    """
    Tokenize text examples.
    
    Args:
        examples: Dictionary with 'text' field
        tokenizer: Hugging Face tokenizer
        max_length: Maximum sequence length
        
    Returns:
        Tokenized examples
    """
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length
    )


def preprocess_banking77(max_length=64, output_dir="data/processed/banking77"):
    """
    Preprocess Banking77 dataset: tokenize and save to disk.
    
    Args:
        max_length: Maximum sequence length for tokenization
        output_dir: Directory to save tokenized datasets
    """
    # Load datasets
    print("Loading Banking77 dataset...")
    train, validation, test = load_banking77()
    
    print(f"Train samples: {len(train)}")
    print(f"Validation samples: {len(validation)}")
    print(f"Test samples: {len(test)}")
    
    # Load tokenizer
    print("\nLoading DistilBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Tokenize each split
    print("\nTokenizing datasets...")
    # Remove only 'text' column, keep 'label'
    columns_to_remove = [col for col in train.column_names if col != 'label']
    
    train_tokenized = train.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=columns_to_remove
    )
    
    validation_tokenized = validation.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=columns_to_remove
    )
    
    test_tokenized = test.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=columns_to_remove
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save tokenized datasets
    print(f"\nSaving tokenized datasets to {output_dir}...")
    train_tokenized.save_to_disk(os.path.join(output_dir, "train"))
    validation_tokenized.save_to_disk(os.path.join(output_dir, "validation"))
    test_tokenized.save_to_disk(os.path.join(output_dir, "test"))
    
    print("Preprocessing complete!")
    print(f"Tokenized datasets saved to: {output_dir}")
    
    return train_tokenized, validation_tokenized, test_tokenized


if __name__ == "__main__":
    preprocess_banking77()

