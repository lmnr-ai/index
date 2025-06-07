import logging
from typing import List, Optional

import backoff
from google import genai

from ..llm import BaseLLMProvider, LLMResponse, Message, ToolCall, ToolDefinition

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
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs
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
        
        # Convert tools to Gemini format
        if tools:
            config["tools"] = [tool.to_gemini_format() for tool in tools]
        
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
        
        # Parse tool calls from response
        tool_calls = self._extract_tool_calls(response)
        
        return LLMResponse(
            content=response.text if hasattr(response, 'text') else "",
            raw_response=response,
            usage=usage,
            tool_calls=tool_calls
        )

    def _extract_tool_calls(self, response) -> Optional[List[ToolCall]]:
        """Extract tool calls from Gemini response"""
        if not hasattr(response, 'candidates') or not response.candidates:
            return None
            
        tool_calls = []
        for candidate in response.candidates:
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                for part in candidate.content.parts:
                    if hasattr(part, 'function_call'):
                        # Generate a unique ID for the tool call
                        tool_call_id = f"call_{hash(str(part.function_call))}"
                        tool_calls.append(ToolCall(
                            id=tool_call_id,
                            name=part.function_call.name,
                            parameters=dict(part.function_call.args) if hasattr(part.function_call, 'args') else {}
                        ))
        
        return tool_calls if tool_calls else None 