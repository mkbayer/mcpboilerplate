"""
LLM client for interfacing with local Ollama gpt-oss:20b model.
"""

import asyncio
import httpx
from typing import Dict, Optional, Any
from trend_radar.utils.logger import get_logger

logger = get_logger(__name__)


class LLMClient:
    """Client for communicating with Ollama LLM"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "gpt-oss:20b",
        timeout: int = 60
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.client = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.client:
            await self.client.aclose()
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        context: Optional[Dict] = None
    ) -> str:
        """
        Generate text using the LLM model
        
        Args:
            prompt: The main prompt text
            system_prompt: Optional system prompt for context
            temperature: Randomness in generation (0.0-1.0)
            max_tokens: Maximum tokens to generate
            context: Optional context from previous interactions
            
        Returns:
            Generated text response
        """
        if not self.client:
            raise RuntimeError("LLMClient must be used as async context manager")
        
        # Prepare the full prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"
        
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": 0.9,
                "repeat_penalty": 1.1
            }
        }
        
        if context:
            payload["context"] = context
        
        try:
            logger.debug(f"Sending request to {self.base_url}/api/generate")
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
                
        except httpx.RequestError as e:
            logger.error(f"HTTP error during LLM request: {e}")
            raise ConnectionError(f"Failed to connect to LLM service: {e}")
        except asyncio.TimeoutError:
            logger.error("LLM request timed out")
            raise TimeoutError("LLM request timed out")
        except Exception as e:
            logger.error(f"Unexpected error during LLM request: {e}")
            raise RuntimeError(f"LLM request failed: {e}")
    
    async def chat(
        self,
        messages: list[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """
        Chat-style interaction with message history
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Generation randomness
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response
        """
        # Convert chat format to single prompt
        prompt_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        
        prompt = "\n\n".join(prompt_parts)
        if not prompt.endswith("Assistant:"):
            prompt += "\n\nAssistant:"
        
        return await self.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    async def health_check(self) -> bool:
        """
        Check if the LLM service is available
        
        Returns:
            True if service is healthy, False otherwise
        """
        if not self.client:
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(10)
            )
        
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            tags = response.json()
            
            # Check if our model is available
            models = [model.get('name', '') for model in tags.get('models', [])]
            return self.model in models
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
        