from typing import List, Optional

from openai import AsyncOpenAI

from ..llm import BaseLLMProvider, LLMResponse, Message


class OpenAIProvider(BaseLLMProvider):
    def __init__(self, model: str, system_message: Optional[str] = None):
        super().__init__(model=model)
        self.system_message = system_message
        self.client = AsyncOpenAI()
        
    def _prepare_messages(self, messages: List[Message]) -> List[Message]:
        """Prepare message list, add system message to the beginning if available"""
        if self.system_message and not any(msg.role == "system" for msg in messages):
            system_message = Message(role="system", content=self.system_message)
            return [system_message] + messages
        return messages

    async def call(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        messages = self._prepare_messages(messages)
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[msg.to_openai_format() for msg in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            raw_response=response,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        ) 