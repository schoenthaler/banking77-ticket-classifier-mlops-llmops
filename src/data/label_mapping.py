def get_label_mappings(dataset):
    """
    Extract label mappings from a dataset.
    
    Args:
        dataset: A Hugging Face dataset with a 'label' or 'intent' feature
        
    Returns:
        tuple: (id_to_label, label_to_id) dictionaries
    """
    # Get unique labels from the dataset
    # Banking77 uses 'intent' as the label column
    if 'intent' in dataset.features:
        labels = sorted(set(dataset['intent']))
    elif 'label' in dataset.features:
        labels = sorted(set(dataset['label']))
    else:
        raise ValueError("Dataset must have either 'intent' or 'label' feature")
    
    # Create mappings
    id_to_label = {i: label for i, label in enumerate(labels)}
    label_to_id = {label: i for i, label in enumerate(labels)}
    
    return id_to_label, label_to_id

