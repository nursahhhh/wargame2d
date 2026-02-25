"""
LLM client wrappers - OpenRouter only.
"""

import os
import logging
from typing import Optional, Dict
import requests
import json
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Base class for LLM clients."""
    
    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key
        if not self.api_key:
            raise ValueError(f"{self.__class__.__name__} API key required")
        logger.info(f"{self.__class__.__name__}: {self.model}")
    
    @abstractmethod
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.5,
        max_tokens: int = 2500,
        response_format: Optional[Dict[str, str]] = None
    ) -> str:
        """Call LLM API (subclass implements)."""
        pass
    
    def _validate_json(self, content: str, response_format: Optional[Dict]) -> str:
        """Validate JSON response early."""
        if response_format and response_format.get("type") == "json_object":
            try:
                json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON: {e}")
                raise
        return content


class OpenRouterClient(BaseLLMClient):
    """OpenRouter client with JSON mode support (auto-instrumented by Logfire)."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "openai/gpt-4o-mini"):
        super().__init__(model, api_key or os.getenv("OPENROUTER_API_KEY"))
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
    
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.5,
        max_tokens: int = 2500,
        response_format: Optional[Dict[str, str]] = None
    ) -> str:
        """Call OpenRouter API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # "HTTP-Referer": "https://github.com/yourusername/wargame",  # Optional: For OpenRouter analytics
            # "X-Title": "WarGame Agent"  # Optional
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if response_format:
            payload["response_format"] = response_format
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return self._validate_json(content, response_format)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.error("OpenRouter rate limit hit - consider upgrading plan")
            logger.error(f"OpenRouter HTTP error: {e}")
            raise
        except Exception as e:
            logger.error(f"OpenRouter error: {e}")
            raise
