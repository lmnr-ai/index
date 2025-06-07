import logging
from typing import List, Optional

import backoff
from anthropic import AsyncAnthropic

from ..llm import (
    BaseLLMProvider,
    LLMResponse,
    Message,
    ThinkingBlock,
    ToolCall,
    ToolDefinition,
)
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
        tools: Optional[List[ToolDefinition]] = None,
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
        
        # Convert tools to Anthropic format
        anthropic_tools = None
        if tools:
            anthropic_tools = [tool.to_anthropic_format() for tool in tools]
        
        if self.enable_thinking:

            try:
                call_params = {
                    "model": self.model,
                    "system": system,
                    "messages": anthropic_api_messages,
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": self.thinking_token_budget,
                    },
                    "max_tokens": max(self.thinking_token_budget + 1, max_tokens),
                    **kwargs
                }
                
                if anthropic_tools:
                    call_params["tools"] = anthropic_tools
                
                response = await self.client.messages.create(**call_params)
            except Exception as e:
                logger.error(f"Error calling Anthropic: {str(e)}")
                # Fallback to anthropic_bedrock with the original messages_copy
                response = await self.anthropic_bedrock.call(
                    messages_copy, # Pass original messages_copy, bedrock provider has its own logic
                    temperature=temperature, # Pass original temperature
                    max_tokens=max_tokens,   # Pass original max_tokens
                    tools=tools,  # Pass tools to bedrock provider too
                    **kwargs
                )

            # Parse tool calls from response
            tool_calls = self._extract_tool_calls(response)
            
            return LLMResponse(
                content=response.content[1].text if len(response.content) > 1 else "",
                raw_response=response,
                usage=response.usage.model_dump(),
                thinking=ThinkingBlock(thinking=response.content[0].thinking, signature=response.content[0].signature) if response.content and hasattr(response.content[0], 'thinking') else None,
                tool_calls=tool_calls
            )
        else: # Not enable_thinking
            call_params = {
                "model": self.model,
                "messages": anthropic_api_messages,
                "temperature": temperature, # Use adjusted temperature
                "max_tokens": max_tokens, # Use adjusted max_tokens
                "system": system,
                **kwargs
            }
            
            if anthropic_tools:
                call_params["tools"] = anthropic_tools
            
            response = await self.client.messages.create(**call_params)
            
            # Parse tool calls from response
            tool_calls = self._extract_tool_calls(response)
     
            return LLMResponse(
                content=response.content[0].text if response.content and hasattr(response.content[0], 'text') else "",
                raw_response=response,
                usage=response.usage.model_dump(),
                tool_calls=tool_calls
            )

    def _extract_tool_calls(self, response) -> Optional[List[ToolCall]]:
        """Extract tool calls from Anthropic response"""
        if not response.content:
            return None
            
        tool_calls = []
        for content_block in response.content:
            if hasattr(content_block, 'type') and content_block.type == 'tool_use':
                tool_calls.append(ToolCall(
                    id=content_block.id,
                    name=content_block.name,
                    parameters=content_block.input
                ))
        
        return tool_calls if tool_calls else None