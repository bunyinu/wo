"""
LLM API interface for making calls to various language models.

This module provides a unified interface for making API calls to different LLM providers
(xAI Grok, Mercury, etc.) using LiteLLM. It includes support for:
- Streaming responses
- Tool calls and function calling
- Retry logic with exponential backoff
- Model-specific configurations
- Comprehensive error handling and logging
"""

from typing import Union, Dict, Any, Optional, AsyncGenerator, List
import os
import json
import asyncio
from openai import OpenAIError
import litellm
from utils.logger import logger
from utils.config import config
from datetime import datetime
import traceback

# litellm.set_verbose=True
litellm.modify_params=True

# Constants
MAX_RETRIES = 3
RATE_LIMIT_DELAY = 30
RETRY_DELAY = 5

class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass

class LLMRetryError(LLMError):
    """Exception raised when retries are exhausted."""
    pass

def setup_api_keys() -> None:
    """Set up API keys from environment variables."""
    providers = ['XAI', 'MERCURY', 'GROQ', 'OPENROUTER']
    for provider in providers:
        key = getattr(config, f'{provider}_API_KEY')
        if key:
            logger.debug(f"API key set for provider: {provider}")
            
            # Set environment variables for LiteLLM
            if provider == 'XAI':
                os.environ['XAI_API_KEY'] = key
            elif provider == 'MERCURY':
                os.environ['MERCURY_API_KEY'] = key
            else:
                os.environ[f'{provider}_API_KEY'] = key
        else:
            logger.warning(f"No API key found for provider: {provider}")
    
    # Set up OpenRouter API base if not already set
    if config.OPENROUTER_API_KEY and config.OPENROUTER_API_BASE:
        os.environ['OPENROUTER_API_BASE'] = config.OPENROUTER_API_BASE
        logger.debug(f"Set OPENROUTER_API_BASE to {config.OPENROUTER_API_BASE}")
    
    # Set up AWS Bedrock credentials
    aws_access_key = config.AWS_ACCESS_KEY_ID
    aws_secret_key = config.AWS_SECRET_ACCESS_KEY
    aws_region = config.AWS_REGION_NAME
    
    if aws_access_key and aws_secret_key and aws_region:
        logger.debug(f"AWS credentials set for Bedrock in region: {aws_region}")
        # Configure LiteLLM to use AWS credentials
        os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key
        os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_key
        os.environ['AWS_REGION_NAME'] = aws_region
    else:
        logger.warning(f"Missing AWS credentials for Bedrock integration - access_key: {bool(aws_access_key)}, secret_key: {bool(aws_secret_key)}, region: {aws_region}")

async def handle_error(error: Exception, attempt: int, max_attempts: int) -> None:
    """Handle API errors with appropriate delays and logging."""
    delay = RATE_LIMIT_DELAY if isinstance(error, litellm.exceptions.RateLimitError) else RETRY_DELAY
    logger.warning(f"Error on attempt {attempt + 1}/{max_attempts}: {str(error)}")
    logger.debug(f"Waiting {delay} seconds before retry...")
    await asyncio.sleep(delay)

def prepare_params(
    messages: List[Dict[str, Any]],
    model_name: str,
    temperature: float = 0,
    max_tokens: Optional[int] = None,
    response_format: Optional[Any] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: str = "auto",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    stream: bool = False,
    top_p: Optional[float] = None,
    model_id: Optional[str] = None,
    enable_thinking: Optional[bool] = False,
    reasoning_effort: Optional[str] = 'low'
) -> Dict[str, Any]:
    """Prepare parameters for the API call."""
    params = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "response_format": response_format,
        "top_p": top_p,
        "stream": stream,
    }

    if api_key:
        params["api_key"] = api_key
    if api_base:
        params["api_base"] = api_base
    if model_id:
        params["model_id"] = model_id

    # Handle token limits
    if max_tokens is not None:
        params["max_tokens"] = max_tokens

    # Add tools if provided
    if tools:
        params.update({
            "tools": tools,
            "tool_choice": tool_choice
        })
        logger.debug(f"Added {len(tools)} tools to API parameters")
    
    # Add xAI Grok-specific parameters
    if "xai" in model_name.lower() or "grok" in model_name.lower():
        logger.debug(f"Preparing xAI Grok parameters for model: {model_name}")
        # Add any xAI Grok specific parameters here if needed
    
    # Add Mercury-specific parameters
    if "mercury" in model_name.lower():
        logger.debug(f"Preparing Mercury parameters for model: {model_name}")
        # Add any Mercury specific parameters here if needed
    
    # Add OpenRouter-specific parameters
    if model_name.startswith("openrouter/"):
        logger.debug(f"Preparing OpenRouter parameters for model: {model_name}")
        
        # Add optional site URL and app name from config
        site_url = config.OR_SITE_URL
        app_name = config.OR_APP_NAME
        if site_url or app_name:
            extra_headers = params.get("extra_headers", {})
            if site_url:
                extra_headers["HTTP-Referer"] = site_url
            if app_name:
                extra_headers["X-Title"] = app_name
            params["extra_headers"] = extra_headers
            logger.debug(f"Added OpenRouter site URL and app name to headers")

    return params

async def make_llm_api_call(
    messages: List[Dict[str, Any]],
    model_name: str,
    response_format: Optional[Any] = None,
    temperature: float = 0,
    max_tokens: Optional[int] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: str = "auto",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    stream: bool = False,
    top_p: Optional[float] = None,
    model_id: Optional[str] = None,
    enable_thinking: Optional[bool] = False,
    reasoning_effort: Optional[str] = 'low'
) -> Union[Dict[str, Any], AsyncGenerator]:
    """
    Make an API call to a language model using LiteLLM.
    
    Args:
        messages: List of message dictionaries for the conversation
        model_name: Name of the model to use (e.g., "xai/grok-latest", "mercury/mercury-coder-small")
        response_format: Desired format for the response
        temperature: Sampling temperature (0-1)
        max_tokens: Maximum tokens in the response
        tools: List of tool definitions for function calling
        tool_choice: How to select tools ("auto" or "none")
        api_key: Override default API key
        api_base: Override default API base URL
        stream: Whether to stream the response
        top_p: Top-p sampling parameter
        model_id: Optional model identifier
        enable_thinking: Whether to enable thinking
        reasoning_effort: Level of reasoning effort
        
    Returns:
        Union[Dict[str, Any], AsyncGenerator]: API response or stream
        
    Raises:
        LLMRetryError: If API call fails after retries
        LLMError: For other API-related errors
    """
    logger.debug(f"Making LLM API call to model: {model_name} (Thinking: {enable_thinking}, Effort: {reasoning_effort})")
    params = prepare_params(
        messages=messages,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
        tools=tools,
        tool_choice=tool_choice,
        api_key=api_key,
        api_base=api_base,
        stream=stream,
        top_p=top_p,
        model_id=model_id,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort
    )
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            logger.debug(f"Attempt {attempt + 1}/{MAX_RETRIES}")
            # logger.debug(f"API request parameters: {json.dumps(params, indent=2)}")
            
            response = await litellm.acompletion(**params)
            logger.debug(f"Successfully received API response from {model_name}")
            logger.debug(f"Response: {response}")
            return response
            
        except (litellm.exceptions.RateLimitError, OpenAIError, json.JSONDecodeError) as e:
            last_error = e
            await handle_error(e, attempt, MAX_RETRIES)
            
        except Exception as e:
            logger.error(f"Unexpected error during API call: {str(e)}", exc_info=True)
            raise LLMError(f"API call failed: {str(e)}")
    
    error_msg = f"Failed to make API call after {MAX_RETRIES} attempts"
    if last_error:
        error_msg += f". Last error: {str(last_error)}"
    logger.error(error_msg, exc_info=True)
    raise LLMRetryError(error_msg)

# Initialize API keys on module import
setup_api_keys()

# Test code for xAI Grok integration
async def test_xai_grok():
    """Test the xAI Grok integration with a simple query."""
    test_messages = [
        {"role": "user", "content": "Hello, can you give me a quick test response?"}
    ]
    
    try:
        print("\n--- Testing xAI Grok model ---")
        response = await make_llm_api_call(
            model_name="xai/grok-latest",
            messages=test_messages,
            temperature=0.7,
            max_tokens=100
        )
        print(f"Response: {response.choices[0].message.content}")
        print(f"Model used: {response.model}")
        
        return True
    except Exception as e:
        print(f"Error testing xAI Grok: {str(e)}")
        return False

# Test code for Mercury integration
async def test_mercury():
    """Test the Mercury integration with a simple query."""
    test_messages = [
        {"role": "user", "content": "Hello, can you give me a quick test response?"}
    ]
    
    try:
        print("\n--- Testing Mercury model ---")
        response = await make_llm_api_call(
            model_name="mercury/mercury-coder-small",
            messages=test_messages,
            temperature=0.7,
            max_tokens=100
        )
        print(f"Response: {response.choices[0].message.content}")
        print(f"Model used: {response.model}")
        
        return True
    except Exception as e:
        print(f"Error testing Mercury: {str(e)}")
        return False

if __name__ == "__main__":
    import asyncio
        
    test_success = asyncio.run(test_xai_grok())
    
    if test_success:
        print("\n✅ xAI Grok integration test completed successfully!")
    else:
        print("\n❌ xAI Grok integration test failed!")
        
    test_success = asyncio.run(test_mercury())
    
    if test_success:
        print("\n✅ Mercury integration test completed successfully!")
    else:
        print("\n❌ Mercury integration test failed!")
