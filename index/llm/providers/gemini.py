import logging
import os
from typing import List, Optional

import backoff
from google import genai
from google.genai import types

from ..llm import BaseLLMProvider, LLMResponse, Message, ToolCall, ToolDefinition

logger = logging.getLogger(__name__)
class GeminiProvider(BaseLLMProvider):
    def __init__(self, model: str, thinking_token_budget: int = 8192):
        super().__init__(model=model)
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.thinking_token_budget = thinking_token_budget


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
            "thinking_config": {
                "thinking_budget": self.thinking_token_budget
            },
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
            function_declarations = [self._convert_tool_to_dict_format(tool) for tool in tools]
            config["tools"] = [types.Tool(function_declarations=function_declarations)]
        
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
        """Extract tool calls from Gemini response following official docs approach"""
        if not hasattr(response, 'candidates') or not response.candidates:
            return None
            
        tool_calls = []
        for i, candidate in enumerate(response.candidates):
            if (hasattr(candidate, 'content') and 
                hasattr(candidate.content, 'parts') and 
                candidate.content.parts):
                
                for j, part in enumerate(candidate.content.parts):
                    # Follow official docs: check for function_call attribute
                    if hasattr(part, 'function_call') and part.function_call is not None:
                        function_call = part.function_call
                        tool_call_id = f"call_{i}_{j}_{hash(str(function_call))}"
                        
                        # Extract parameters as per official docs using .args
                        parameters = {}
                        if hasattr(function_call, 'args') and function_call.args:
                            parameters = dict(function_call.args)
                        
                        tool_calls.append(ToolCall(
                            id=tool_call_id,
                            name=function_call.name,
                            parameters=parameters
                        ))
        
        return tool_calls if tool_calls else None 

    def _convert_tool_to_dict_format(self, tool: ToolDefinition) -> dict:
        """Convert a ToolDefinition to plain dictionary format as per official docs"""
        
        # Convert parameters to the expected schema format
        properties = {}
        required = []
        
        for param_name, param in tool.parameters.items():
            prop = {
                "type": param.type,  # Keep original case as per docs
                "description": param.description
            }
            if param.enum:
                prop["enum"] = param.enum
            properties[param_name] = prop
            
            if param.required:
                required.append(param_name)
        
        # Return plain dictionary format as per official docs
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        } 