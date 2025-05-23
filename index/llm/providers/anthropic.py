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

        if not messages_copy:
            raise ValueError("Messages list cannot be empty.")

        conversation_messages_input: List[Message] = []

        system = []

        if messages_copy[0].role == "system":
            system = messages_copy[0].content[0].text
            conversation_messages_input = messages_copy[1:]
        else:
            conversation_messages_input = messages_copy
        
        anthropic_api_messages = [msg.to_anthropic_format() for msg in conversation_messages_input]
        
        # print(f"anthropic_api_messages: {anthropic_api_messages}")

        if self.enable_thinking:

            try:
                response = await self.client.messages.create(
                    model=self.model,
                    system=system,
                    messages=anthropic_api_messages,
                    thinking={
                        "type": "enabled",
                        "budget_tokens": self.thinking_token_budget,
                    },
                    max_tokens=max(self.thinking_token_budget + 1, max_tokens),
                    **kwargs
                )
            except Exception as e:
                logger.error(f"Error calling Anthropic: {str(e)}")
                # Fallback to anthropic_bedrock with the original messages_copy
                response = await self.anthropic_bedrock.call(
                    messages_copy, # Pass original messages_copy, bedrock provider has its own logic
                    temperature=temperature, # Pass original temperature
                    max_tokens=max_tokens,   # Pass original max_tokens
                    **kwargs
                )

            return LLMResponse(
                content=response.content[1].text,
                raw_response=response,
                usage=response.usage.model_dump(),
                thinking=ThinkingBlock(thinking=response.content[0].thinking, signature=response.content[0].signature)
            )
        else: # Not enable_thinking
            response = await self.client.messages.create(
                model=self.model,
                messages=anthropic_api_messages,
                temperature=temperature, # Use adjusted temperature
                max_tokens=max_tokens, # Use adjusted max_tokens
                system=system,
                **kwargs
            )
     
            return LLMResponse(
                content=response.content[0].text,
                raw_response=response,
                usage=response.usage.model_dump()
            )