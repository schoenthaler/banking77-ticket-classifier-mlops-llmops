"""
Clean inference interface for the quantized model.
Loads model once and provides a simple predict() function.
"""
import os
import torch
from transformers import AutoTokenizer
from typing import Dict, Optional

from src.data.label_mapping import get_label_mappings


# Module-level cache for model and tokenizer
_model = None
_tokenizer = None
_id_to_label = None
_loaded_model_path = None


def _load_model(model_path: str = "models/quantized/best-model"):
    """Load quantized model and tokenizer (cached)."""
    global _model, _tokenizer, _id_to_label, _loaded_model_path
    
    # Return cached model if already loaded for this path
    if _model is not None and _loaded_model_path == model_path:
        return _model, _tokenizer, _id_to_label
    
    # Load quantized model
    quantized_model_file = os.path.join(model_path, "quantized_model.pt")
    if not os.path.exists(quantized_model_file):
        raise FileNotFoundError(
            f"Quantized model not found at {quantized_model_file}. "
            "Please run quantization first."
        )
    
    print(f"Loading quantized model from {model_path}...")
    _model = torch.load(quantized_model_file, map_location='cpu', weights_only=False)
    _model.eval()
    _model = _model.cpu()
    
    # Load tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load label mappings
    _id_to_label, _ = get_label_mappings()
    
    _loaded_model_path = model_path
    print("[OK] Model loaded and ready for inference")
    return _model, _tokenizer, _id_to_label


def predict(text: str, model_path: Optional[str] = None) -> Dict:
    """
    Run the quantized model on a single text.
    
    Args:
        text: Input text to classify
        model_path: Optional path to quantized model (default: models/quantized/best-model)
    
    Returns:
        {
            "label": str,              # Predicted label name
            "confidence": float,       # Confidence score (0-1)
            "probs": Dict[str, float]  # All label probabilities
        }
    """
    if model_path is None:
        model_path = "models/quantized/best-model"
    
    model, tokenizer, id_to_label = _load_model(model_path)
    
    # Tokenize
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=64,
        return_tensors="pt"
    )
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_id = torch.argmax(logits, dim=-1).item()
        confidence = probabilities[0][predicted_id].item()
    
    predicted_label = id_to_label[predicted_id]
    
    # Get all probabilities
    probs = {
        id_to_label[i]: prob.item()
        for i, prob in enumerate(probabilities[0])
    }
    
    return {
        "label": predicted_label,
        "confidence": confidence,
        "probs": probs
    }


if __name__ == "__main__":
    # Test
    test_text = "How do I activate my card?"
    result = predict(test_text)
    print(f"Text: {test_text}")
    print(f"Predicted: {result['label']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"\nTop 5 predictions:")
    sorted_probs = sorted(result['probs'].items(), key=lambda x: x[1], reverse=True)
    for label, prob in sorted_probs[:5]:
        print(f"  {label}: {prob:.4f}")

