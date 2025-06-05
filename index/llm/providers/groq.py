import logging
from typing import List, Optional

import backoff
from groq import AsyncGroq  # Assuming AsyncGroq for asynchronous operations

from ..llm import BaseLLMProvider, LLMResponse, Message

logger = logging.getLogger(__name__)

class GroqProvider(BaseLLMProvider):
    """
    A provider for interacting with the Groq API.
    """
    def __init__(self, model: str):
        """
        Initializes the GroqProvider.

        Args:
            model: The model name to use (e.g., "llama-3.3-70b-versatile").
        """
        super().__init__(model=model)
        # The Groq client, by default, should pick up the GROQ_API_KEY 
        # from environment variables if not explicitly passed.
        # Ref: https://console.groq.com/docs/libraries
        # client = Groq(api_key=os.environ.get("GROQ_API_KEY")) where api_key param is optional.
        self.client = AsyncGroq()

    @backoff.on_exception(
        backoff.constant,
        Exception,  # Retry on any exception. Consider refining with specific Groq API errors if known.
        max_tries=3,
        interval=0.5,
    )
    async def call(
        self,
        messages: List[Message],
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Makes an asynchronous call to the Groq API.

        Args:
            messages: A list of Message objects representing the conversation history.
            temperature: The sampling temperature to use. Groq converts 0 to 1e-8.
                         Values should ideally be > 0 and <= 2.
            max_tokens: The maximum number of tokens to generate.

        Returns:
            An LLMResponse object containing the model's response and usage data.
        
        Raises:
            ValueError: If the messages list is empty or the API response is invalid.
        """
        if not messages:
            raise ValueError("Messages list cannot be empty.")

        # Format messages to be compatible with Groq's API (OpenAI format)
        formatted_messages = [msg.to_groq_format() for msg in messages]

        if formatted_messages[0]["role"] == "system":
            # remove couple of examples from first user message because llama4 model supports only 5 images.
            # TODO: remove this once we have a model that supports more images.
            formatted_messages[1]["content"] = formatted_messages[1]["content"][0:4] + formatted_messages[1]["content"][12:]


        api_params = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature,
        }

        if max_tokens is not None:
            api_params["max_tokens"] = max_tokens

        # Groq API notes:
        # - 'N' (number of choices) must be 1 if supplied. Defaults to 1.
        # - Unsupported OpenAI fields (will result in 400 error if supplied):
        #   logprobs, logit_bias, top_logprobs, messages[].name

        response = await self.client.chat.completions.create(**api_params)

        if not response.choices or not response.choices[0].message:
            logger.error(f"Groq API response missing choices or message: {response}")
            raise ValueError("Invalid response structure from Groq API")

        content = response.choices[0].message.content
        # Handle cases where content might be None (e.g., if finish_reason indicates tool use in the future)
        if content is None:
            content = ""

        usage_data = {}
        # Attempt to extract usage data, assuming an OpenAI-compatible structure.
        # The Groq Python SDK might provide usage data in `response.usage`.
        if hasattr(response, "usage") and response.usage is not None:
            usage_data = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            }

        return LLMResponse(
            content=content,
            raw_response=response,
            usage=usage_data
        ) 