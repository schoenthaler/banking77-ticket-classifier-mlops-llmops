import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np

from src.data.label_mapping import get_label_mappings


def predict_intent(model, tokenizer, text, id_to_label):
    """
    Predict intent for a given text.
    
    Args:
        model: The model to use
        tokenizer: Tokenizer
        text: Input text
        id_to_label: Mapping from label ID to label name
        
    Returns:
        Dictionary with predictions
    """
    model.eval()
    
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
    
    # Get top 3 predictions
    top_k = 3
    top_probs, top_indices = torch.topk(probabilities[0], top_k)
    
    top_predictions = [
        {
            "intent": id_to_label[idx.item()],
            "confidence": prob.item()
        }
        for prob, idx in zip(top_probs, top_indices)
    ]
    
    return {
        "predicted_intent": predicted_label,
        "confidence": confidence,
        "top_predictions": top_predictions
    }


def compare_models(
    original_model_path="models/distilbert-banking77",
    quantized_model_path="models/quantized/distilbert-banking77",
    test_texts=None
):
    """
    Compare original and quantized models on test texts.
    
    Args:
        original_model_path: Path to original model
        quantized_model_path: Path to quantized model
        test_texts: List of test texts (optional)
    """
    if test_texts is None:
        test_texts = [
            "How do I activate my card?",
            "What is my account balance?",
            "I need to change my PIN number",
            "Can I get a refund for my purchase?",
            "How do I transfer money to another account?",
            "My card was stolen, what should I do?",
            "I forgot my passcode",
            "What are the fees for international transfers?"
        ]
    
    # Load label mappings
    id_to_label, label_to_id = get_label_mappings()
    
    # Load original model
    print("="*80)
    print("Loading Original Model")
    print("="*80)
    original_model = AutoModelForSequenceClassification.from_pretrained(original_model_path)
    original_tokenizer = AutoTokenizer.from_pretrained(original_model_path)
    print("✓ Original model loaded\n")
    
    # Load quantized model
    print("="*80)
    print("Loading Quantized Model")
    print("="*80)
    
    # Load entire quantized model object
    quantized_model_file = os.path.join(quantized_model_path, "quantized_model.pt")
    if not os.path.exists(quantized_model_file):
        raise FileNotFoundError(
            f"Quantized model not found at {quantized_model_file}. "
            "Please run quantization first."
        )
    
    try:
        quantized_model = torch.load(quantized_model_file, map_location='cpu', weights_only=False)
        quantized_model.eval()
        quantized_model = quantized_model.cpu()  # Ensure on CPU
        print("✓ Quantized model loaded")
    except Exception as e:
        print(f"Error loading quantized model: {e}")
        print("Try re-running quantization to regenerate the model file.")
        raise
    
    quantized_tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
    print("✓ Tokenizer loaded\n")
    
    # Compare predictions
    print("="*80)
    print("Comparing Predictions")
    print("="*80)
    
    agreement_count = 0
    total_count = len(test_texts)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: {text}")
        print("-" * 80)
        
        # Original model prediction
        original_pred = predict_intent(original_model, original_tokenizer, text, id_to_label)
        print(f"Original Model:")
        print(f"  Intent: {original_pred['predicted_intent']}")
        print(f"  Confidence: {original_pred['confidence']:.4f}")
        
        # Quantized model prediction
        quantized_pred = predict_intent(quantized_model, quantized_tokenizer, text, id_to_label)
        print(f"Quantized Model:")
        print(f"  Intent: {quantized_pred['predicted_intent']}")
        print(f"  Confidence: {quantized_pred['confidence']:.4f}")
        
        # Check agreement
        if original_pred['predicted_intent'] == quantized_pred['predicted_intent']:
            agreement_count += 1
            print("  ✓ Predictions match!")
        else:
            print("  ✗ Predictions differ")
            print(f"  Confidence difference: {abs(original_pred['confidence'] - quantized_pred['confidence']):.4f}")
    
    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    agreement_rate = (agreement_count / total_count) * 100
    print(f"Agreement: {agreement_count}/{total_count} ({agreement_rate:.1f}%)")
    
    if agreement_rate == 100:
        print("✓ Perfect agreement! Quantized model produces identical predictions.")
    elif agreement_rate >= 95:
        print("✓ Excellent agreement! Quantized model is highly accurate.")
    else:
        print("⚠ Some differences found. Review predictions above.")
    
    return {
        "agreement_count": agreement_count,
        "total_count": total_count,
        "agreement_rate": agreement_rate
    }


if __name__ == "__main__":
    compare_models()