"""
LLMProvider.py - A minimalistic interface for language models.

This module provides a clean, extensible interface for working with different 
language model providers (initially OpenAI and Ollama).
"""

import tiktoken
from typing import Dict, Optional
from abc import ABC, abstractmethod
import logging
import os
from datetime import datetime
from langchain_core.callbacks.base import BaseCallbackHandler

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Add this utility function at the module level
def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Count the number of tokens in a text string.
    
    Args:
        text: The text to count tokens for
        model: The model name to use for tokenization
    
    Returns:
        Number of tokens
    """
    try:
        # Map model names to encoding names used by tiktoken
        encoding_name = "cl100k_base"  # Default for newer models
        
        # Get the appropriate tokenizer
        encoding = tiktoken.get_encoding(encoding_name)
        
        # Count tokens
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception as e:
        logger.warning(f"Error counting tokens: {e}. Using approximate count.")
        # Fallback: rough approximation (4 chars per token)
        return len(text) // 4


class BaseLLMProvider(ABC):
    """Abstract base class for language model providers."""
    
    def __init__(self):
        """Initialize with a token tracker."""
        self.token_tracker = TokenUsageTracker()
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        pass

    '''
    @abstractmethod
    def generate_with_context(self, prompt: str, context: List[str], **kwargs) -> str:
        """Generate text from a prompt with additional context."""
        pass
    '''
    
    def get_token_usage(self) -> Dict:
        """Get token usage summary."""
        return self.token_tracker.get_summary()
    
    def estimate_cost(self, **kwargs) -> Dict:
        """Estimate cost based on token usage."""
        return self.token_tracker.estimate_cost(**kwargs)


class OpenAIProvider(BaseLLMProvider):
    """Provider for OpenAI language models."""
    
    def __init__(
        self, 
        model_name: str = "gpt-4o-mini", 
        temperature: float = 0.3,
        max_tokens: int = 5000,
        api_key: Optional[str] = None
    ):
        """Initialize OpenAI provider with model configuration."""
        super().__init__()  # Initialize the token tracker
        try:
            from openai import OpenAI
            
            self.model_name = model_name
            self.temperature = temperature
            self.max_tokens = max_tokens
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            
            if not self.api_key:
                logger.warning("No API key provided for OpenAI. Please set OPENAI_API_KEY environment variable.")
            
            self.client = OpenAI(api_key=self.api_key)
            logger.info(f"Initialized OpenAI provider with model: {model_name}")
            
        except ImportError:
            logger.error("OpenAI package not installed. Please install with 'pip install openai'")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI's chat completion API."""
        try:
            temperature = kwargs.get("temperature", self.temperature)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            
            # Count input tokens
            input_tokens = count_tokens(prompt, self.model_name)
            
            # Make the API call
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            result_text = response.choices[0].message.content
            
            # Get the token counts from the response if available
            if hasattr(response, 'usage'):
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
            else:
                # Estimate output tokens if not provided
                output_tokens = count_tokens(result_text, self.model_name)
            
            # Track the usage
            self.token_tracker.add_usage(
                input_tokens=input_tokens, 
                output_tokens=output_tokens,
                prompt=prompt, 
                response=result_text
            )
            
            # Log token usage
            logger.debug(f"Request used {input_tokens} input tokens and {output_tokens} output tokens")
            
            return result_text
            
        except Exception as e:
            logger.error(f"Error generating text with OpenAI: {e}")
            return f"Error: {str(e)}"
        
    '''
    def generate_with_context(self, prompt: str, context: List[str], **kwargs) -> str:
        """Generate text with context documents as part of the prompt."""
        context_text = "\n".join(context)
        combined_prompt = f"Context information:\n{context_text}\n\nQuestion: {prompt}\n\nAnswer:"
        return self.generate(combined_prompt, **kwargs)
    '''

class OllamaProvider(BaseLLMProvider):
    """Provider for Ollama language models."""
    
    def __init__(
        self, 
        model_name: str = "llama3", 
        temperature: float = 0.7,
        max_tokens: int = 500,
        host: str = "http://localhost:11434"
    ):
        """Initialize Ollama provider with model configuration."""
        super().__init__()  # Initialize the token tracker
        try:
            import requests
            
            self.model_name = model_name
            self.temperature = temperature
            self.max_tokens = max_tokens
            self.host = host
            self.requests = requests
            
            # Simple check if Ollama is running
            try:
                requests.get(f"{host}/api/tags")
                logger.info(f"Initialized Ollama provider with model: {model_name}")
            except requests.exceptions.ConnectionError:
                logger.warning(f"Could not connect to Ollama at {host}. Make sure Ollama is running.")
                
        except ImportError:
            logger.error("Requests package not installed. Please install with 'pip install requests'")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Ollama's API."""
        try:
            temperature = kwargs.get("temperature", self.temperature)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            
            # Estimate input tokens
            input_tokens = count_tokens(prompt)
            
            response = self.requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                result_text = response.json().get("response", "")
                
                # Estimate output tokens for Ollama (which doesn't return token counts)
                output_tokens = count_tokens(result_text)
                
                # Track the usage
                self.token_tracker.add_usage(
                    input_tokens=input_tokens, 
                    output_tokens=output_tokens,
                    prompt=prompt, 
                    response=result_text
                )
                
                # Log token usage
                logger.debug(f"Request used approximately {input_tokens} input tokens and {output_tokens} output tokens")
                
                return result_text
            else:
                logger.error(f"Ollama API error: {response.text}")
                return f"Error: {response.text}"
                
        except Exception as e:
            logger.error(f"Error generating text with Ollama: {e}")
            return f"Error: {str(e)}"
    
    '''
    def generate_with_context(self, prompt: str, context: List[str], **kwargs) -> str:
        """Generate text with context documents as part of the prompt."""
        context_text = "\n".join(context)
        combined_prompt = f"Context information:\n{context_text}\n\nQuestion: {prompt}\n\nAnswer:"
        return self.generate(combined_prompt, **kwargs)
    '''

def create_llm_provider(provider_type: str, **kwargs) -> BaseLLMProvider:
    """Factory function to create appropriate LLM provider."""
    if provider_type.lower() == "openai":
        return OpenAIProvider(**kwargs)
    elif provider_type.lower() == "ollama":
        return OllamaProvider(**kwargs)
    else:
        raise ValueError(f"Unsupported provider type: {provider_type}")


# Add the TokenUsageTracker class that was missing
class TokenUsageTracker:
    """Tracks token usage across multiple requests."""
    
    def __init__(self):
        """Initialize the token tracker."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.requests_count = 0
        self.history = []
    
    def add_usage(self, input_tokens: int, output_tokens: int, prompt: str = "", response: str = ""):
        """
        Record token usage from a request.
        
        Args:
            input_tokens: Number of tokens in input
            output_tokens: Number of tokens in output
            prompt: Optional truncated prompt for reference
            response: Optional truncated response for reference
        """
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.requests_count += 1
        
        # Store in history with timestamp
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'prompt': prompt[:100] + '...' if prompt and len(prompt) > 100 else prompt,
            'response': response[:100] + '...' if response and len(response) > 100 else response
        })
    
    def get_summary(self) -> Dict:
        """Get a summary of token usage."""
        return {
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
            'requests_count': self.requests_count,
            'avg_input_tokens': self.total_input_tokens // max(1, self.requests_count),
            'avg_output_tokens': self.total_output_tokens // max(1, self.requests_count)
        }
    
    def estimate_cost(self, input_price_per_1k: float = 0.01, output_price_per_1k: float = 0.03) -> Dict:
        """
        Estimate cost based on token usage.
        
        Args:
            input_price_per_1k: Price per 1000 input tokens
            output_price_per_1k: Price per 1000 output tokens
            
        Returns:
            Dictionary with cost information
        """
        input_cost = (self.total_input_tokens / 1000) * input_price_per_1k
        output_cost = (self.total_output_tokens / 1000) * output_price_per_1k
        total_cost = input_cost + output_cost
        
        return {
            'input_cost': round(input_cost, 4),
            'output_cost': round(output_cost, 4),
            'total_cost': round(total_cost, 4)
        }


class TokenUsageCallbackHandler(BaseCallbackHandler):
    """Callback handler for tracking token usage in LangChain."""
    
    def __init__(self):
        """Initialize token tracker."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.requests_count = 0
    
    def on_llm_end(self, response, **kwargs):
        """Process response when LLM finishes."""
        try:
            if hasattr(response, 'llm_output') and isinstance(response.llm_output, dict):
                token_usage = response.llm_output.get('token_usage', {})
                
                prompt_tokens = token_usage.get('prompt_tokens', 0)
                completion_tokens = token_usage.get('completion_tokens', 0)
                
                self.total_input_tokens += prompt_tokens
                self.total_output_tokens += completion_tokens
                self.requests_count += 1
                
                logger.debug(f"Request used {prompt_tokens} input tokens and {completion_tokens} output tokens")
        except Exception as e:
            logger.warning(f"Error tracking tokens: {e}")
    
    def get_summary(self):
        """Get token usage summary."""
        return {
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
            'requests_count': self.requests_count,
            'avg_input_tokens': self.total_input_tokens // max(1, self.requests_count),
            'avg_output_tokens': self.total_output_tokens // max(1, self.requests_count)
        }
    
    def estimate_cost(self, input_price_per_1k=0.01, output_price_per_1k=0.03):
        """Estimate cost based on token usage."""
        input_cost = (self.total_input_tokens / 1000) * input_price_per_1k
        output_cost = (self.total_output_tokens / 1000) * output_price_per_1k
        total_cost = input_cost + output_cost
        
        return {
            'input_cost': round(input_cost, 4),
            'output_cost': round(output_cost, 4),
            'total_cost': round(total_cost, 4)
        }