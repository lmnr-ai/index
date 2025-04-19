from typing import List, Optional

from openai import AsyncOpenAI

from ..llm import BaseLLMProvider, LLMResponse, Message


class OpenAIProvider(BaseLLMProvider):
    def __init__(self, model: str, reasoning_effort: Optional[str] = "low"):
        super().__init__(model=model)
        self.client = AsyncOpenAI()
        self.reasoning_effort = reasoning_effort

    async def call(
        self,
        messages: List[Message],
        temperature: float = 1.0,
    ) -> LLMResponse:

        args = {
            "temperature": temperature,
        }
    
        if self.model.startswith("o") and self.reasoning_effort:
            args["reasoning_effort"] = self.reasoning_effort
            args["temperature"] = 1

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[msg.to_openai_format() for msg in messages],
            **args
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