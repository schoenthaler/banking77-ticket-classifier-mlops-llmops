from transformers import AutoModelForSequenceClassification, AutoConfig


def create_model(num_labels, id_to_label=None, label_to_id=None):
    """
    Create a DistilBERT model for sequence classification.
    
    Args:
        num_labels: Number of classification labels (77 for Banking77)
        id_to_label: Optional mapping from label ID to label name
        label_to_id: Optional mapping from label name to label ID
        
    Returns:
        AutoModelForSequenceClassification model
    """
    model_name = "distilbert-base-uncased"
    
    # Load config and modify for classification
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = num_labels
    
    # Create model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config
    )
    
    return model

