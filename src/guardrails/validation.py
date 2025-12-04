"""
Label validation guardrail.
Checks if the predicted label makes sense for the input text.
"""
import os
from typing import Dict, Optional

from src.guardrails.config import get_openai_api_key


# Low confidence threshold
LOW_CONFIDENCE_THRESHOLD = 0.5

# Very low confidence threshold
VERY_LOW_CONFIDENCE_THRESHOLD = 0.3


def validate_label(text: str, label: str, confidence: float, use_llm: bool = False) -> Dict:
    """
    Validate if the predicted label is appropriate for the text.
    
    Args:
        text: Input text
        label: Predicted label
        confidence: Model confidence score
        use_llm: Whether to use LLM (if OPENAI_API_KEY is set)
    
    Returns:
        {
            "needs_manual_review": bool,
            "reason": str
        }
    """
    needs_review = False
    reason = ""
    
    # Check confidence thresholds
    if confidence < VERY_LOW_CONFIDENCE_THRESHOLD:
        needs_review = True
        reason = f"Very low confidence ({confidence:.2f} < {VERY_LOW_CONFIDENCE_THRESHOLD})"
    elif confidence < LOW_CONFIDENCE_THRESHOLD:
        needs_review = True
        reason = f"Low confidence ({confidence:.2f} < {LOW_CONFIDENCE_THRESHOLD})"
    
    # Optional LLM validation
    if use_llm and get_openai_api_key():
        llm_result = _validate_label_llm(text, label, confidence)
        if llm_result:
            if llm_result.get("needs_manual_review", False):
                needs_review = True
                reason = f"LLM validation: {llm_result.get('reason', '')}"
    
    if not reason:
        reason = "Label validation passed"
    
    return {
        "needs_manual_review": needs_review,
        "reason": reason
    }


def _validate_label_llm(text: str, label: str, confidence: float) -> Optional[Dict]:
    """Optional LLM-based label validation."""
    try:
        import openai
        
        api_key = get_openai_api_key()
        if not api_key:
            return None
        
        client = openai.OpenAI(api_key=api_key)
        
        # Get list of valid labels for context
        from src.data.label_mapping import BANKING77_INTENTS
        valid_labels = ", ".join(BANKING77_INTENTS[:10])  # Sample for context
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a banking support classifier validator. Check if the predicted label '{label}' (confidence: {confidence:.2f}) makes sense for the customer message. Valid labels include banking intents like: {valid_labels}. Respond with JSON: {{\"needs_manual_review\": true/false, \"reason\": \"brief explanation\"}}"
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        import json
        result = json.loads(response.choices[0].message.content)
        return result
        
    except Exception as e:
        return None

