<a href="https://www.ycombinator.com/companies/laminar-ai">![Static Badge](https://img.shields.io/badge/Y%20Combinator-S24-orange)</a>
<a href="https://x.com/lmnrai">![X (formerly Twitter) Follow](https://img.shields.io/twitter/follow/lmnrai)</a>
<a href="https://discord.gg/nNFUUDAKub"> ![Static Badge](https://img.shields.io/badge/Join_Discord-464646?&logo=discord&logoColor=5865F2) </a>

# Index

Index is a state-of-the-art open-source browser agent that autonomously executes complex tasks on the web.

- [x] It is powered by Claude 3.7 Sonnet with extented thinking. More models will be supported in the future.
- [x] Index is also available as a [hosted API.](https://docs.lmnr.ai/laminar-index/introduction)
- [x] You can also try out Index via [hosted UI](https://docs.lmnr.ai/laminar-index/introduction#hosted-ui) or fully [self-host the UI](https://x.com/skull8888888888/status/1910763169489764374).
- [x] Supports advanced [browser agent observability](https://docs.lmnr.ai/laminar-index/observability) powered by [Laminar](https://lmnr.ai).

prompt: go to ycombinator.com. summarize first 3 companies in the W25 batch and make new spreadsheet in google sheets.

https://github.com/user-attachments/assets/2b46ee20-81b6-4188-92fb-4d97fe0b3d6a


## Index API

Index API is available as [hosted api](https://docs.lmnr.ai/laminar-index/introduction) on the [Laminar](https://lmnr.ai) platform. Index API manages remote browser sessions and agent infrastructure. Index API is the best way to run AI browser automation in production. To get started, [sign up](https://lmnr.ai/sign-in) and create project API key.

### Install Laminar
```bash
pip install lmnr
```

### Use Index via API
```python
from lmnr import Laminar, AsyncLaminarClient
# you can also set LMNR_PROJECT_API_KEY environment variable

# Initialize tracing
Laminar.initialize(project_api_key="your_api_key")

# Initialize the client
client = AsyncLaminarClient(api_key="your_api_key")

async def main():

    # Run a task
    response = await client.agent.run(
        prompt="Navigate to news.ycombinator.com, find a post about AI, and summarize it"
    )

    # Print the result
    print(response.result)
    
if __name__ == "__main__":
    asyncio.run(main())
```

When you call Index via API, you automatically get full browser agent observability on Laminar platform. Learn more about [Index browser observability](https://docs.lmnr.ai/laminar-index/introduction#tracing-with-laminar).

## Local Quick Start

### Install dependencies
```bash
pip install lmnr-index

# Install playwright
playwright install chromium
```

### Run the agent
```python
import asyncio
from index import Agent, AnthropicProvider

async def main():
    # Initialize the LLM provider
    llm = AnthropicProvider(
            model="claude-3-7-sonnet-20250219",
            enable_thinking=True, 
            thinking_token_budget=2048)
    
    # Create an agent with the LLM
    agent = Agent(llm=llm)
    
    # Run the agent with a task
    output = await agent.run(
        prompt="Navigate to news.ycombinator.com, find a post about AI, and summarize it"
    )
    
    # Print the result
    print(output.result)
    
if __name__ == "__main__":
    asyncio.run(main())
```

### Stream the agent's output
```python
from index import Agent, AnthropicProvider

agent = Agent(llm=AnthropicProvider(model="claude-3-7-sonnet-20250219"))    

# Stream the agent's output
async for chunk in agent.run_stream(
    prompt="Navigate to news.ycombinator.com, find a post about AI, and summarize it"):
    print(chunk)
``` 

### Enable browser agent observability

To trace Index agent's actions and record browser session you simply need to initialize Laminar tracing before running the agent.

```python
from lmnr import Laminar

Laminar.initialize(project_api_key="your_api_key")
```

Then you will get full observability on the agent's actions synced with the browser session in the Laminar platform.

<picture>
    <img src="./static/traces.png" alt="Index observability" width="800"/>
</picture>

### Run with remote CDP url
```python
import asyncio
from index import Agent, AnthropicProvider, BrowserConfig

async def main():
    # Configure browser to connect to an existing Chrome DevTools Protocol endpoint
    browser_config = BrowserConfig(
        cdp_url="<cdp_url>"
    )
    
    # Initialize the LLM provider
    llm = AnthropicProvider(model="claude-3-7-sonnet-20250219", enable_thinking=True, thinking_token_budget=2048)
    
    # Create an agent with the LLM and browser
    agent = Agent(llm=llm, browser_config=browser_config)
    
    # Run the agent with a task
    output = await agent.run(
        prompt="Navigate to news.ycombinator.com and find the top story"
    )
    
    # Print the result
    print(output.result)
    
if __name__ == "__main__":
    asyncio.run(main())
```

### Customize browser window size
```python
import asyncio
from index import Agent, AnthropicProvider, BrowserConfig

async def main():
    # Configure browser with custom viewport size
    browser_config = BrowserConfig(
        viewport_size={"width": 1200, "height": 900}
    )
    
    # Initialize the LLM provider
    llm = AnthropicProvider(model="claude-3-7-sonnet-20250219")
    
    # Create an agent with the LLM and browser
    agent = Agent(llm=llm, browser_config=browser_config)
    
    # Run the agent with a task
    output = await agent.run(
        "Navigate to a responsive website and capture how it looks in full HD resolution"
    )
    
    # Print the result
    print(output.result)
    
if __name__ == "__main__":
    asyncio.run(main())
```

---

Made with ❤️ by the [Laminar team](https://lmnr.ai)
