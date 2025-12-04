"""
Main guardrails pipeline orchestrator.
Combines model prediction with all guardrails.
"""
from typing import Dict, Optional

from src.ml.infer import predict
from src.guardrails.urgency import check_urgency
from src.guardrails.toxicity import check_toxicity
from src.guardrails.validation import validate_label


def process_ticket(
    text: str,
    model_path: Optional[str] = None,
    use_llm: bool = False
) -> Dict:
    """
    Process a ticket through the complete pipeline:
    1) Get model prediction
    2) Run urgency, toxicity, validation guardrails
    3) Return combined result
    
    Args:
        text: Customer ticket text
        model_path: Optional path to quantized model
        use_llm: Whether to use LLM for guardrails (requires OPENAI_API_KEY)
    
    Returns:
        {
            "predicted_label": str,
            "model_confidence": float,
            "priority": "high" | "normal",
            "is_toxic": bool,
            "needs_manual_review": bool,
            "llm_notes": str,
            "urgency": Dict,
            "toxicity": Dict,
            "validation": Dict
        }
    """
    # Step 1: Get model prediction
    prediction = predict(text, model_path=model_path)
    
    predicted_label = prediction["label"]
    confidence = prediction["confidence"]
    
    # Step 2: Run guardrails
    urgency_result = check_urgency(text, use_llm=use_llm)
    toxicity_result = check_toxicity(text, use_llm=use_llm)
    validation_result = validate_label(text, predicted_label, confidence, use_llm=use_llm)
    
    # Step 3: Combine results
    priority = urgency_result["priority"]
    is_toxic = toxicity_result["toxic"]
    needs_review = validation_result["needs_manual_review"]
    
    # Determine if manual review is needed
    if is_toxic or needs_review or urgency_result["urgent"]:
        needs_review = True
    
    # Build LLM notes
    llm_notes = ""
    if use_llm:
        notes = []
        if urgency_result.get("reason"):
            notes.append(f"Urgency: {urgency_result['reason']}")
        if toxicity_result.get("reason"):
            notes.append(f"Toxicity: {toxicity_result['reason']}")
        if validation_result.get("reason"):
            notes.append(f"Validation: {validation_result['reason']}")
        llm_notes = "; ".join(notes)
    
    return {
        "predicted_label": predicted_label,
        "model_confidence": confidence,
        "priority": priority,
        "is_toxic": is_toxic,
        "needs_manual_review": needs_review,
        "llm_notes": llm_notes,
        "urgency": urgency_result,
        "toxicity": toxicity_result,
        "validation": validation_result
    }


if __name__ == "__main__":
    # Test
    test_texts = [
        "How do I activate my card?",
        "My account was hacked and money is missing!",
        "This service is terrible and I hate it",
        "What is my balance?"
    ]
    
    for text in test_texts:
        print("\n" + "="*80)
        print(f"Text: {text}")
        print("="*80)
        result = process_ticket(text)
        print(f"Predicted: {result['predicted_label']}")
        print(f"Confidence: {result['model_confidence']:.4f}")
        print(f"Priority: {result['priority']}")
        print(f"Toxic: {result['is_toxic']}")
        print(f"Needs Review: {result['needs_manual_review']}")
        if result['llm_notes']:
            print(f"Notes: {result['llm_notes']}")

