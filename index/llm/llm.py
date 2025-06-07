from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

@dataclass
class MessageContent:
    """Base class for message content"""
    cache_control: Optional[bool] = None

@dataclass
class TextContent(MessageContent):
    """Text content in a message"""
    text: str = ""
    type: str = "text"

@dataclass
class ImageContent(MessageContent):
    """Image content in a message"""
    image_b64: Optional[str] = None
    image_url: Optional[str] = None
    type: str = "image"

@dataclass
class ThinkingBlock(MessageContent):
    """Thinking block in a message"""
    thinking: str = ""
    signature: str = ""
    type: str = "thinking"

@dataclass
class ToolCallBlock(MessageContent):
    """Tool call block in a message"""
    tool_call_id: str = ""
    type: str = "tool_call"

@dataclass
class ToolCall:
    """Represents a tool call from the model"""
    id: str
    name: str
    parameters: Dict[str, Any]

@dataclass
class ToolResult:
    """Represents the result of a tool execution"""
    tool_call_id: str
    content: Any
    is_error: bool = False

@dataclass
class ToolParameter:
    """Represents a parameter in a tool definition"""
    type: str
    description: str
    enum: Optional[List[str]] = None
    required: bool = True

@dataclass
class ToolDefinition:
    """Unified tool definition that can be converted to provider-specific formats"""
    name: str
    description: str
    parameters: Dict[str, ToolParameter]

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI tool format"""
        properties = {}
        required = []
        
        for param_name, param in self.parameters.items():
            prop = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                prop["enum"] = param.enum
            properties[param_name] = prop
            
            if param.required:
                required.append(param_name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
    
    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic tool format"""
        properties = {}
        required = []
        
        for param_name, param in self.parameters.items():
            prop = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                prop["enum"] = param.enum
            properties[param_name] = prop
            
            if param.required:
                required.append(param_name)
        
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    
    def to_gemini_format(self) -> Dict[str, Any]:
        """Convert to Gemini tool format"""
        properties = {}
        required = []
        
        for param_name, param in self.parameters.items():
            prop = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                prop["enum"] = param.enum
            properties[param_name] = prop
            
            if param.required:
                required.append(param_name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }

@dataclass
class Message:
    """A message in a conversation"""
    role: Union[str, MessageRole]
    content: Union[str, List[Union[TextContent, ImageContent, ThinkingBlock]]]
    name: Optional[str] = None  # For tool/function messages
    tool_call_id: Optional[str] = None  # For tool/function responses
    is_state_message: Optional[bool] = False
    tool_calls: Optional[List[ToolCall]] = None  # For assistant messages with tool calls
    tool_result: Optional[ToolResult] = None  # For user messages with tool results

    def __post_init__(self):
        # Convert role enum to string if needed
        if isinstance(self.role, MessageRole):
            self.role = self.role.value
            
        # Convert string content to TextContent if needed
        if isinstance(self.content, str):
            self.content = [TextContent(text=self.content)]
        elif isinstance(self.content, (TextContent, ImageContent)):
            self.content = [self.content]

    def to_openai_format(self) -> Dict:
        """Convert to OpenAI message format"""
        message = {"role": self.role}
        
        if self.tool_calls and self.role == "assistant":
            # Assistant message with tool calls
            if isinstance(self.content, list) and self.content:
                message["content"] = self._format_content_for_openai()
            else:
                message["content"] = None
            
            message["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": str(tool_call.parameters)
                    }
                }
                for tool_call in self.tool_calls
            ]
        elif self.tool_result and self.role == "tool":
            # Tool result message
            message["tool_call_id"] = self.tool_result.tool_call_id
            message["content"] = str(self.tool_result.content)
        else:
            # Regular message
            if isinstance(self.content, str):
                message["content"] = self.content
            elif isinstance(self.content, list):
                message["content"] = self._format_content_for_openai()

        return message
    
    def _format_content_for_openai(self) -> List[Dict]:
        """Format content blocks for OpenAI"""
        content_blocks = []
        for content_block in self.content:
            block = {}
            
            if isinstance(content_block, TextContent):
                block["type"] = "text"
                block["text"] = content_block.text
            elif isinstance(content_block, ImageContent):
                block["type"] = "image_url"
                block["image_url"] = {
                    "url": "data:image/png;base64," + content_block.image_b64
                }

            content_blocks.append(block)
        return content_blocks
    
    def to_groq_format(self) -> Dict:
        """Convert to Groq message format"""
        message = {"role": self.role}

        if isinstance(self.content, str):
            message["content"] = self.content
            
        elif isinstance(self.content, list):

            content_blocks = []

            # content of a system and assistant messages in groq can only contain text
            if self.role == "system" or self.role == "assistant":
                block = self.content[0]
                if isinstance(block, TextContent):
                    message["content"] = block.text

                return message

            for content_block in self.content:

                block = {}
                
                if isinstance(content_block, TextContent):
                    block["type"] = "text"
                    block["text"] = content_block.text
                elif isinstance(content_block, ImageContent):
                    block["type"] = "image_url"
                    block["image_url"] = {
                        "url": "data:image/png;base64," + content_block.image_b64
                    }

                content_blocks.append(block)

            message["content"] = content_blocks

        return message

    def to_anthropic_format(self, enable_cache_control: bool = True) -> Dict:
        """Convert to Anthropic message format"""
        message = {"role": self.role}

        if self.tool_calls and self.role == "assistant":
            # Assistant message with tool calls
            content_blocks = []
            
            # Add text content if any
            if isinstance(self.content, list):
                for content_block in self.content:
                    if isinstance(content_block, TextContent):
                        block = {"type": "text", "text": content_block.text}
                        if content_block.cache_control and enable_cache_control:
                            block["cache_control"] = {"type": "ephemeral"}
                        content_blocks.append(block)
            
            # Add tool use blocks
            for tool_call in self.tool_calls:
                content_blocks.append({
                    "type": "tool_use",
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "input": tool_call.parameters
                })
            
            message["content"] = content_blocks
        elif self.tool_result and self.role == "tool":
            # Tool result message - convert to user role for Anthropic
            message["role"] = "user"
            message["content"] = [
                {
                    "type": "tool_result",
                    "tool_use_id": self.tool_result.tool_call_id,
                    "content": str(self.tool_result.content),
                    "is_error": self.tool_result.is_error
                }
            ]
        else:
            # Regular message
            if isinstance(self.content, str):
                message["content"] = self.content
            elif isinstance(self.content, list):
                content_blocks = []

                for content_block in self.content:
                    block = {}

                    if isinstance(content_block, TextContent):
                        block["type"] = "text"
                        block["text"] = content_block.text
                    elif isinstance(content_block, ImageContent):
                        block["type"] = "image"
                        block["source"] = {
                            "type": "base64",
                            "media_type": "image/png",  # This should be configurable based on image type
                            "data": content_block.image_b64 if content_block.image_b64 else content_block.image_url
                        }
                    elif isinstance(content_block, ThinkingBlock):
                        block["type"] = "thinking"
                        block["thinking"] = content_block.thinking
                        block["signature"] = content_block.signature

                    if content_block.cache_control and enable_cache_control:
                        block["cache_control"] = {"type": "ephemeral"}

                    content_blocks.append(block)

                message["content"] = content_blocks
                     
        return message
    
    def to_gemini_format(self) -> Dict:
        """Convert to Gemini message format"""
        parts = []
        
        if isinstance(self.content, str):
            parts = [{"text": self.content}]
        elif isinstance(self.content, list):
            for content_block in self.content:
                if isinstance(content_block, TextContent):
                    parts.append({"text": content_block.text})
                elif isinstance(content_block, ImageContent):
                    if content_block.image_b64:
                        parts.append({"inline_data": {
                            "mime_type": "image/png",
                            "data": content_block.image_b64
                        }})
                    elif content_block.image_url:
                        parts.append({"file_data": {
                            "mime_type": "image/png",
                            "file_uri": content_block.image_url
                        }})
        
        return {
            "role": 'model' if self.role == 'assistant' else 'user',
            "parts": parts
        }
    
    def remove_cache_control(self):
        if isinstance(self.content, list):
            for content_block in self.content:
                if isinstance(content_block, TextContent):
                    content_block.cache_control = None
                elif isinstance(content_block, ImageContent):
                    content_block.cache_control = None

    def add_cache_control_to_state_message(self):

        if not self.is_state_message or not isinstance(self.content, list) or len(self.content) < 3:
            return

        if len(self.content) == 3:
            self.content[-1].cache_control = True

    def has_cache_control(self):
        
        if not isinstance(self.content, list):
            return False

        return any(content.cache_control for content in self.content)


class LLMResponse(BaseModel):
    content: str
    raw_response: Any
    usage: Dict[str, Any]
    thinking: Optional[ThinkingBlock] = None
    tool_calls: Optional[List[ToolCall]] = None


class BaseLLMProvider(ABC):
    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    async def call(
        self,
        messages: List[Message],
        temperature: float = 1,
        max_tokens: Optional[int] = None,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs
    ) -> LLMResponse:
        pass
