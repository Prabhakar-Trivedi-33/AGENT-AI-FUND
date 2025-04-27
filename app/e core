from typing import Dict, Any, Optional, List, Union, Protocol
from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseMessage
import logging
from abc import ABC, abstractmethod
from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate(self, messages: List[BaseMessage], temperature: float = 0.1) -> str:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    def get_streaming_model(self, callbacks: Optional[List] = None) -> Any:
        """Get a model instance that supports streaming."""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI implementation of LLM provider."""
    
    def __init__(self, model_name: str = None, api_key: str = None):
        self.model_name = model_name or settings.default_model
        self.api_key = api_key or settings.openai_api_key
        
    def _create_model(self, temperature: float = 0.1, streaming: bool = False, 
                      callbacks: Optional[List] = None) -> ChatOpenAI:
        """Create and configure a ChatOpenAI instance."""
        return ChatOpenAI(
            model=self.model_name,
            temperature=temperature,
            api_key=self.api_key,
            streaming=streaming,
            callbacks=callbacks,
            verbose=settings.app_env == "development"
        )
    
    async def generate(self, messages: List[BaseMessage], temperature: float = 0.1) -> str:
        """Generate a response from the LLM."""
        try:
            llm = self._create_model(temperature=temperature)
            response = await llm.agenerate([messages])
            return response.generations[0][0].text
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            raise
            
    def get_streaming_model(self, callbacks: Optional[List] = None) -> ChatOpenAI:
        """Get a model instance that supports streaming."""
        return self._create_model(
            temperature=settings.default_temperature, 
            streaming=True, 
            callbacks=callbacks
        )

class LLMFactory:
    """Factory for creating LLM provider instances."""
    
    @staticmethod
    def create_provider(provider_type: str = "openai", **kwargs) -> LLMProvider:
        """Create an LLM provider based on type."""
        if provider_type.lower() == "openai":
            return OpenAIProvider(**kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_type}")

# Create default LLM provider
default_llm_provider = LLMFactory.create_provider()

async def generate_response(messages: List[BaseMessage], temperature: float = None) -> str:
    """
    Generate a response from the default LLM provider.
    
    Args:
        messages: List of conversation messages
        temperature: Creativity level of the model (optional)
        
    Returns:
        str: Generated response
    """
    temp = temperature if temperature is not None else settings.default_temperature
    return await default_llm_provider.generate(messages, temperature=temp)
from pydantic import BaseSettings, Field
from typing import Optional, Dict, Any
from functools import lru_cache
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """Application configuration settings."""
    
    # API Keys
    openai_api_key: str = Field(default=os.getenv("OPENAI_API_KEY", ""))
    
    # Application Settings
    app_env: str = Field(default=os.getenv("APP_ENV", "development"))
    log_level: str = Field(default=os.getenv("LOG_LEVEL", "INFO"))
    
    # MFAPI Configuration
    mfapi_base_url: str = "https://api.mfapi.in/mf"
    mfapi_timeout: int = 30
    
    # Cache Settings
    enable_cache: bool = True
    cache_ttl: int = 3600  # 1 hour
    cache_max_size: int = 1000  # Maximum number of items in cache
    
    # LLM Settings
    default_model: str = "gpt-4-turbo"
    default_temperature: float = 0.1
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        
    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache configuration as a dictionary."""
        return {
            "enabled": self.enable_cache,
            "ttl": self.cache_ttl,
            "max_size": self.cache_max_size
        }

@lru_cache()
def get_settings() -> Settings:
    """Create and cache settings instance to avoid multiple instantiations."""
    return Settings()

# Create settings instance for direct import
settings = get_settings()
