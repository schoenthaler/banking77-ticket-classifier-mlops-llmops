"""
Urgency detection guardrail.
Checks if a ticket requires urgent attention.
"""
import os
import re
from typing import Dict, Optional

from src.guardrails.config import get_openai_api_key


# Urgency keywords
URGENT_KEYWORDS = [
    "hacked", "stolen", "fraud", "unauthorized", "missing money",
    "account locked", "card blocked", "emergency", "urgent",
    "immediately", "asap", "critical", "security breach",
    "compromised", "suspicious", "fraudulent"
]

HIGH_PRIORITY_KEYWORDS = [
    "lost card", "stolen card", "can't access", "locked out",
    "payment failed", "transfer failed", "declined", "error"
]


def check_urgency(text: str, use_llm: bool = False) -> Dict:
    """
    Check if text indicates urgent issue.
    
    Args:
        text: Input text
        use_llm: Whether to use LLM (if OPENAI_API_KEY is set)
    
    Returns:
        {
            "urgent": bool,
            "reason": str,
            "priority": "high" | "normal"
        }
    """
    text_lower = text.lower()
    
    # Rule-based check
    urgent = False
    priority = "normal"
    reason = ""
    
    # Check for urgent keywords
    for keyword in URGENT_KEYWORDS:
        if keyword in text_lower:
            urgent = True
            priority = "high"
            reason = f"Contains urgent keyword: '{keyword}'"
            break
    
    # Check for high priority keywords (if not already urgent)
    if not urgent:
        for keyword in HIGH_PRIORITY_KEYWORDS:
            if keyword in text_lower:
                priority = "high"
                reason = f"Contains high-priority keyword: '{keyword}'"
                break
    
    # Optional LLM check
    if use_llm and get_openai_api_key():
        llm_result = _check_urgency_llm(text)
        if llm_result:
            # LLM can override rule-based if it detects urgency
            if llm_result.get("urgent", False):
                urgent = True
                priority = "high"
                reason = f"LLM detected urgency: {llm_result.get('reason', '')}"
    
    if not reason:
        reason = "No urgency indicators found"
    
    return {
        "urgent": urgent,
        "priority": priority,
        "reason": reason
    }


def _check_urgency_llm(text: str) -> Optional[Dict]:
    """Optional LLM-based urgency check."""
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
                    "content": "You are a banking support classifier. Determine if a customer message requires urgent attention. Urgent issues include: security breaches, fraud, stolen cards, unauthorized transactions, account lockouts, missing money. Respond with JSON: {\"urgent\": true/false, \"reason\": \"brief explanation\"}"
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
        # Fallback to rule-based if LLM fails
        return None

