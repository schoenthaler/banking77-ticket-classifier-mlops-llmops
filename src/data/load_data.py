from datasets import load_dataset
from sklearn.model_selection import train_test_split


def load_banking77():
    """
    Load the banking77 dataset from Hugging Face.
    
    Returns:
        tuple: (train_dataset, validation_dataset, test_dataset)
    """
    # Load the dataset
    dataset = load_dataset("banking77")
    
    # Check if validation split exists
    if "validation" in dataset:
        train = dataset["train"]
        validation = dataset["validation"]
        test = dataset["test"]
    else:
        # Create validation split from train if it doesn't exist
        train_full = dataset["train"]
        test = dataset["test"]
        
        # Split train into train and validation (80/20 split)
        train_data = train_full.train_test_split(test_size=0.2, seed=42)
        train = train_data["train"]
        validation = train_data["test"]
    
    return train, validation, test


if __name__ == "__main__":
    # Load the datasets
    train, validation, test = load_banking77()
    
    # Print number of samples per split
    print(f"Number of samples in train: {len(train)}")
    print(f"Number of samples in validation: {len(validation)}")
    print(f"Number of samples in test: {len(test)}")
    
    # Print the first example
    print("\nFirst example from training set:")
    print(train[0])

