"""
Toxicity detection guardrail.
Checks if text contains toxic or inappropriate content.
"""
import os
import re
from typing import Dict, Optional

from src.guardrails.config import get_openai_api_key


# Toxic keywords (basic list - can be expanded)
TOXIC_KEYWORDS = [
    "fuck", "shit", "damn", "idiot", "stupid", "moron",
    "hate", "kill", "die", "worthless"
]

# Profanity patterns
PROFANITY_PATTERNS = [
    r'\b(f\*ck|f\*\*k|f\*\*\*)\b',
    r'\b(s\*it|s\*\*t)\b',
    r'\b(d\*mn|d\*\*n)\b'
]


def check_toxicity(text: str, use_llm: bool = False) -> Dict:
    """
    Check if text contains toxic content.
    
    Args:
        text: Input text
        use_llm: Whether to use LLM (if OPENAI_API_KEY is set)
    
    Returns:
        {
            "toxic": bool,
            "reason": str
        }
    """
    text_lower = text.lower()
    
    # Rule-based check
    toxic = False
    reason = ""
    
    # Check for toxic keywords
    for keyword in TOXIC_KEYWORDS:
        if keyword in text_lower:
            toxic = True
            reason = f"Contains toxic keyword: '{keyword}'"
            break
    
    # Check for profanity patterns
    if not toxic:
        for pattern in PROFANITY_PATTERNS:
            if re.search(pattern, text_lower):
                toxic = True
                reason = "Contains profanity"
                break
    
    # Check for excessive capitalization (potential anger)
    if not toxic and len(text) > 10:
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
        if caps_ratio > 0.5:
            toxic = True
            reason = "Excessive capitalization (potential anger)"
    
    # Optional LLM check
    if use_llm and get_openai_api_key():
        llm_result = _check_toxicity_llm(text)
        if llm_result:
            if llm_result.get("toxic", False):
                toxic = True
                reason = f"LLM detected toxicity: {llm_result.get('reason', '')}"
    
    if not reason:
        reason = "No toxicity detected"
    
    return {
        "toxic": toxic,
        "reason": reason
    }


def _check_toxicity_llm(text: str) -> Optional[Dict]:
    """Optional LLM-based toxicity check."""
    try:
        import openai
        
        api_key = get_openai_api_key()
        if not api_key:
            return None
        
        client = openai.OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a content moderation system. Determine if a customer message contains toxic, abusive, or inappropriate content. Respond with JSON: {\"toxic\": true/false, \"reason\": \"brief explanation\"}"
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

