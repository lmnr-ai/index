import logging
from typing import List, Optional

import backoff
from anthropic import AsyncAnthropic

from ..llm import BaseLLMProvider, LLMResponse, Message, ThinkingBlock
from ..providers.anthropic_bedrock import AnthropicBedrockProvider

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseLLMProvider):
    def __init__(self, model: str, enable_thinking: bool = True, thinking_token_budget: Optional[int] = 2048):
        super().__init__(model=model)
        self.client = AsyncAnthropic()
        self.thinking_token_budget = thinking_token_budget

        self.anthropic_bedrock = AnthropicBedrockProvider(model=f"us.anthropic.{model}-v1:0", enable_thinking=enable_thinking, thinking_token_budget=thinking_token_budget)

        self.enable_thinking = enable_thinking

    @backoff.on_exception(
        backoff.constant,  # constant backoff
        Exception,     # retry on any exception
        max_tries=3,   # stop after 3 attempts
        interval=10,
        on_backoff=lambda details: logger.info(
            f"API error, retrying in {details['wait']:.2f} seconds... (attempt {details['tries']})"
        )
    )
    async def call(
        self,
        messages: List[Message],
        temperature: float = -1,
        max_tokens: Optional[int] = 16000,
        **kwargs
    ) -> LLMResponse:
        # Make a copy of messages to prevent modifying the original list during retries
        messages_copy = messages.copy()
        
        if len(messages_copy) < 2 or messages_copy[0].role != "system":
            raise ValueError("System message is required for Anthropic and length of messages must be at least 2")

        system_message = messages_copy[0]

        if self.enable_thinking:

            try:
                response = await self.client.messages.create(
                    model=self.model,
                    system=system_message.to_anthropic_format()["content"],
                    messages=[msg.to_anthropic_format() for msg in messages_copy[1:]],
                    temperature=1,
                    thinking={
                        "type": "enabled",
                        "budget_tokens": self.thinking_token_budget,
                    },
                    max_tokens=max(self.thinking_token_budget + 1, max_tokens),
                    **kwargs
                )
            except Exception as e:
                logger.error(f"Error calling Anthropic: {str(e)}")
                response = await self.anthropic_bedrock.call(
                    messages_copy,
                    **kwargs
                )

            return LLMResponse(
                content=response.content[1].text,
                raw_response=response,
                usage=response.usage.model_dump(),
                thinking=ThinkingBlock(thinking=response.content[0].thinking, signature=response.content[0].signature)
            )
        else:
            response = await self.client.messages.create(
                model=self.model,
                messages=[msg.to_anthropic_format() for msg in messages_copy[1:]],
                temperature=temperature,
                max_tokens=max_tokens,
                system=system_message.to_anthropic_format()["content"],
                **kwargs
            )
     
            return LLMResponse(
                content=response.content[0].text,
                raw_response=response,
                usage=response.usage.model_dump()
            )