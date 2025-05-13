import logging
from typing import List, Optional

import backoff
from google import genai

from ..llm import BaseLLMProvider, LLMResponse, Message

logger = logging.getLogger(__name__)
class GeminiVertexProvider(BaseLLMProvider):
    def __init__(self, model: str, project: str = None, location: str = None):
        super().__init__(model=model)
        self.client = genai.Client(
            vertexai=True,
            project=project,
            location=location)


    @backoff.on_exception(
        backoff.constant,  # constant backoff
        Exception,     # retry on any exception
        max_tries=3,   # stop after 3 attempts
        interval=0.5,
        on_backoff=lambda details: logger.info(
            f"API error, retrying in {details['wait']:.2f} seconds... (attempt {details['tries']})"
        ),
    )
    async def call(
        self,
        messages: List[Message],
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        
        if len(messages) == 0:
            raise ValueError("Messages must be non-empty")
        
        config = {
            "temperature": temperature,
        }
        
        if messages[0].role == "system":
            system = messages[0].content[0].text
            gemini_messages = [msg.to_gemini_format() for msg in messages[1:]]

            config["system_instruction"] = {
                "text": system
            }
        else:
            gemini_messages = [msg.to_gemini_format() for msg in messages]
        
        
        if max_tokens:
            config["max_output_tokens"] = max_tokens

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=gemini_messages,
            config=config,   
        )
        
        # Extract usage information if available
        usage = {}
        if hasattr(response, "usage_metadata"):
            usage = {
                "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
                "total_tokens": getattr(response.usage_metadata, "total_token_count", 0)
            }
        
        return LLMResponse(
            content=response.text,
            raw_response=response,
            usage=usage
        ) 