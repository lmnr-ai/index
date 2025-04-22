import os
from typing import List, Optional

from google import genai

from ..llm import BaseLLMProvider, LLMResponse, Message


class GeminiProvider(BaseLLMProvider):
    def __init__(self, model: str, thinking_token_budget: int = 8192):
        super().__init__(model=model)
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.thinking_token_budget = thinking_token_budget

    async def call(
        self,
        messages: List[Message],
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        
        if len(messages) < 2 or messages[0].role != "system":
            raise ValueError("System message is required and length of messages must be at least 2")

        system = messages[0].content[0].text
        gemini_messages = [msg.to_gemini_format() for msg in messages[1:]]
        
        config = {
            "temperature": temperature,
            "thinking_config": {
                "thinking_budget": self.thinking_token_budget
            },
            "system_instruction": {
                "text": system
            }
        }
        
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