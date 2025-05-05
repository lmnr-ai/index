import json
from typing import Any, Dict

from lmnr import evaluate
from pydantic import BaseModel

from index import Agent, GeminiProvider


class CountryInfo(BaseModel):
    """Model for country information extraction"""
    country: str
    capital: str
    currency: str


async def run_agent(data: Dict[str, Any]):
    """Execute the agent with data extraction based on output_model"""
    prompt = data["prompt"]
    output_model = data.get("output_model")
    start_url = data.get("start_url")
    
    # Here we would normally initialize your LLM provider
    llm = GeminiProvider(model="gemini-2.5-pro-preview-03-25")
    
    agent = Agent(llm=llm)
    output = await agent.run(
        prompt=prompt,
        output_model=output_model,
        start_url=start_url
    )
    
    # Return the result.content from the AgentOutput
    return output.result.content


async def eval_extraction(output: Dict[str, Any], target: Dict[str, Any]):
    """Evaluate the extraction accuracy"""

    exact_match = json.dumps(output, sort_keys=True) == json.dumps(target, sort_keys=True)
    
    return exact_match

data = [
    {
        "data": {
            "prompt": "Extract information about France. For currency only use text description, such as 'Euro'.",
            "output_model": CountryInfo,
            "start_url": "https://en.wikipedia.org/wiki/France"
        },
        "target": {
            "country": "France",
            "capital": "Paris",
            "currency": "Euro"
        }
    },
    {
        "data": {
            "prompt": "Extract information about Japan. For currency only use text description, such as 'Euro'.",
            "output_model": CountryInfo,
            "start_url": "https://en.wikipedia.org/wiki/Japan"
        },
        "target": {
            "country": "Japan",
            "capital": "Tokyo",
            "currency": "Japanese yen"
        }
    },
    {
        "data": {
            "prompt": "Extract information about Brazil. For currency only use text description, such as 'Euro'.",
            "output_model": CountryInfo,
            "start_url": "https://en.wikipedia.org/wiki/Brazil"
        },
        "target": {
            "country": "Brazil",
            "capital": "Brasília",
            "currency": "Real"
        }
    },
]

evaluate(
    data=data,
    executor=run_agent,
    evaluators={"accuracy": eval_extraction},
    group_name="country_extraction",
)
