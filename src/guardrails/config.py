"""
Configuration helper for guardrails.
Loads API keys from secrets.json with fallback to environment variables.
"""
import os
import json
from typing import Optional


def get_openai_api_key() -> Optional[str]:
    """
    Get OpenAI API key from secrets.json or environment variable.
    
    Priority:
    1. secrets.json file in project root
    2. OPENAI_API_KEY environment variable
    3. None if neither is found
    
    Returns:
        API key string or None
    """
    # Try to load from secrets.json first
    secrets_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "secrets.json")
    
    if os.path.exists(secrets_path):
        try:
            with open(secrets_path, 'r') as f:
                secrets = json.load(f)
                api_key = secrets.get("openai_api_key")
                if api_key:
                    return api_key
        except (json.JSONDecodeError, KeyError, IOError) as e:
            # If file exists but can't be read/parsed, fall through to env var
            pass
    
    # Fallback to environment variable
    return os.getenv("OPENAI_API_KEY")

