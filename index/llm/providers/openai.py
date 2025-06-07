import json
from typing import List, Optional

from openai import AsyncOpenAI

from ..llm import BaseLLMProvider, LLMResponse, Message, ToolCall, ToolDefinition


class OpenAIProvider(BaseLLMProvider):
    def __init__(self, model: str, reasoning_effort: Optional[str] = "low"):
        super().__init__(model=model)
        self.client = AsyncOpenAI()
        self.reasoning_effort = reasoning_effort

    async def call(
        self,
        messages: List[Message],
        temperature: float = 1.0,
        tools: Optional[List[ToolDefinition]] = None,
    ) -> LLMResponse:

        args = {
            "temperature": temperature,
        }
    
        if self.model.startswith("o") and self.reasoning_effort:
            args["reasoning_effort"] = self.reasoning_effort
            args["temperature"] = 1

        # Convert tools to OpenAI format
        openai_tools = None
        if tools:
            openai_tools = [tool.to_openai_format() for tool in tools]

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[msg.to_openai_format() for msg in messages],
            tools=openai_tools,
            **args
        )
        
        # Extract tool calls if present
        tool_calls = None
        assistant_message = response.choices[0].message
        if assistant_message.tool_calls:
            tool_calls = []
            for tool_call in assistant_message.tool_calls:
                # Parse the arguments JSON string
                try:
                    parameters = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    parameters = {}
                
                tool_calls.append(ToolCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    parameters=parameters
                ))
        
        return LLMResponse(
            content=response.choices[0].message.content or "",
            raw_response=response,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            tool_calls=tool_calls
        ) 