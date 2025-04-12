from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"  # For OpenAI function calling responses

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
class Message:
    """A message in a conversation"""
    role: Union[str, MessageRole]
    content: Union[str, List[Union[TextContent, ImageContent, ThinkingBlock]]]
    name: Optional[str] = None  # For tool/function messages
    tool_call_id: Optional[str] = None  # For tool/function responses
    is_state_message: Optional[bool] = False

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
        
        # Handle different content types
        if len(self.content) == 1 and isinstance(self.content[0], TextContent):
            message["content"] = self.content[0].text
        else:
            message["content"] = [
                {
                    "type": content.type,
                    **({"text": content.text} if isinstance(content, TextContent) else {"image_url": content.image_url})
                }
                for content in self.content
            ]
            
        if self.name:
            message["name"] = self.name
        if self.tool_call_id:
            message["tool_call_id"] = self.tool_call_id
            
        return message

    def to_anthropic_format(self, enable_cache_control: bool = True) -> Dict:
        """Convert to Anthropic message format"""
        message = {"role": self.role}

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


class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    # Add more providers as needed

class LLMModel(Enum):
    GPT4 = "gpt-4"
    GPT35 = "gpt-3.5-turbo"
    CLAUDE3_OPUS = "claude-3-opus-20240229"
    CLAUDE3_SONNET = "claude-3-sonnet-20240229"
    # Add more models as needed

class LLMResponse(BaseModel):
    content: str
    raw_response: Any
    usage: Dict[str, int]
    thinking: Optional[ThinkingBlock] = None


class BaseLLMProvider(ABC):
    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    async def call(
        self,
        messages: List[Message],
        temperature: float = 1,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        pass
