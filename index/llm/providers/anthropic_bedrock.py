import logging
import os
from typing import List, Optional

import backoff
from anthropic import AsyncAnthropicBedrock
from dotenv import load_dotenv

from ..llm import BaseLLMProvider, LLMResponse, Message

load_dotenv()

logger = logging.getLogger(__name__)


class AnthropicBedrockProvider(BaseLLMProvider):
    def __init__(self, model: str, enable_thinking: bool = True, thinking_token_budget: Optional[int] = 8192):
        super().__init__(model=model)

        self.client = AsyncAnthropicBedrock(
            aws_access_key=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            aws_region=os.getenv('AWS_REGION'),
        )
        self.enable_thinking = enable_thinking
        self.thinking_token_budget = thinking_token_budget
    @backoff.on_exception(  # noqa: F821
        backoff.constant,  # constant backoff
        Exception,     # retry on any exception
        max_tries=3,   # stop after 3 attempts
        interval=10,
    )
    async def call(
        self,
        messages: List[Message],
        temperature: float = 1,
        max_tokens: Optional[int] = 2048,
        **kwargs
    ) -> LLMResponse:
    
        messages_copy = messages.copy()

        if len(messages_copy) < 2 or messages_copy[0].role != "system":
            raise ValueError("System message is required for Anthropic Bedrock and length of messages must be at least 2")
            
        system_message = messages_copy[0]

        try:
            if self.enable_thinking:
                    
                response = await self.client.messages.create(
                    model=self.model,
                    system=system_message.to_anthropic_format(enable_cache_control=False)["content"],
                    messages=[msg.to_anthropic_format(enable_cache_control=False) for msg in messages_copy[1:]],
                    temperature=1,
                    thinking={
                        "type": "enabled",
                        "budget_tokens": self.thinking_token_budget,
                    },
                    max_tokens=max(self.thinking_token_budget + 1, max_tokens),
                    **kwargs
                )
               
                return LLMResponse(
                    content=response.content[1].text,
                    raw_response=response,
                    usage=response.usage
                )
            else:

                response = await self.client.messages.create(
                    model=self.model,
                    messages=[msg.to_anthropic_format(enable_cache_control=False) for msg in messages_copy[1:]],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system=system_message.to_anthropic_format(enable_cache_control=False)["content"],
                    **kwargs
                )
              
                return LLMResponse(
                    content=response.content[0].text,
                    raw_response=response,
                    usage=response.usage
                )
        except Exception as e:
            logger.error(f"Error calling Anthropic Bedrock: {str(e)}")
            raise e