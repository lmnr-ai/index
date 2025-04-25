<a href="https://www.ycombinator.com/companies/laminar-ai">![Static Badge](https://img.shields.io/badge/Y%20Combinator-S24-orange)</a>
<a href="https://x.com/lmnrai">![X (formerly Twitter) Follow](https://img.shields.io/twitter/follow/lmnrai)</a>
<a href="https://discord.gg/nNFUUDAKub"> ![Static Badge](https://img.shields.io/badge/Join_Discord-464646?&logo=discord&logoColor=5865F2) </a>

# Index

Index is the SOTA open-source browser agent for autonomously executing complex tasks on the web.

- [x] Powered by reasoning LLMs with vision capabilities.
    - [x] Gemini 2.5 Pro (really fast and accurate)
    - [x] Claude 3.7 Sonnet with extended thinking (reliable and accurate)
    - [x] OpenAI o4-mini (depending on the reasoning effort, provides good balance between speed, cost and accuracy)
    - [x] Gemini 2.5 Flash (really fast, cheap, and good for less complex tasks)
- [x] `pip install lmnr-index` and use it in your project
- [x] `index run` to run the agent in the interactive CLI
- [x] Index is also available as a [serverless API.](https://docs.lmnr.ai/laminar-index/introduction)
- [x] You can also try out Index via [Chat UI](https://lmnr.ai/chat).
- [x] Supports advanced [browser agent observability](https://docs.lmnr.ai/laminar-index/observability) powered by open-source platform [Laminar](https://github.com/lmnr-ai/lmnr).

prompt: go to ycombinator.com. summarize first 3 companies in the W25 batch and make new spreadsheet in google sheets.

https://github.com/user-attachments/assets/2b46ee20-81b6-4188-92fb-4d97fe0b3d6a

## Documentation

Check out full documentation [here](https://docs.lmnr.ai/index-agent/getting-started)

## Index API

The easiest way to use Index in production is via the [serverless API](https://docs.lmnr.ai/laminar-index/introduction). Index API manages remote browser sessions, agent infrastructure and [browser observability](https://docs.lmnr.ai/laminar-index/tracing). To get started, [sign up](https://lmnr.ai/sign-in) and create project API key. Read the [docs](https://docs.lmnr.ai/laminar-index/introduction) to learn more.

### Install Laminar
```bash
pip install lmnr
```

### Use Index via API
```python
from lmnr import Laminar, LaminarClient
# you can also set LMNR_PROJECT_API_KEY environment variable

# Initialize tracing
Laminar.initialize(project_api_key="your_api_key")

# Initialize the client
client = LaminarClient(project_api_key="your_api_key")

for chunk in client.agent.run(
    stream=True,
    model_provider="gemini",
    model="gemini-2.5-pro-preview-03-25",
    prompt="Navigate to news.ycombinator.com, find a post about AI, and summarize it"
):
    print(chunk)
    
```
## Local Quick Start

### Install dependencies
```bash
pip install lmnr-index

# Install playwright
playwright install chromium
```

### Setup model API keys

Setup your model API keys in `.env` file in your project root:
```
ANTHROPIC_API_KEY=
GEMINI_API_KEY=
OPENAI_API_KEY=
```


### Run the agent with CLI

You can run Index via interactive CLI. It features:
- Browser state persistence between sessions
- Follow-up messages with support for "give human control" action
- Real-time streaming updates
- Beautiful terminal UI using Textual

You can run the agent with the following command. Remember to set API key for the selected model in the `.env` file.
```bash
index run
```

Output will look like this:

```
Loaded existing browser state
╭───────────────────── Interactive Mode ─────────────────────╮
│ Index Browser Agent Interactive Mode                       │
│ Type your message and press Enter. The agent will respond. │
│ Press Ctrl+C to exit.                                      │
╰────────────────────────────────────────────────────────────╯

Choose an LLM model:
1. Gemini 2.5 Flash
2. Claude 3.7 Sonnet
3. OpenAI o4-mini
Select model [1/2] (1): 3
Using OpenAI model: o4-mini
Loaded existing browser state

Your message: go to lmnr.ai, summarize pricing page

Agent is working...
Step 1: Opening lmnr.ai
Step 2: Opening Pricing page
Step 3: Scrolling for more pricing details
Step 4: Scrolling back up to view pricing tiers
Step 5: Provided concise summary of the three pricing tiers
```

### Running with a personal Chrome instance

You can use Index with personal Chrome browser instance instead of launching a new browser. Main advantage is that all existing logged in sessions will be available.

```bash
# Basic usage with default Chrome path
index run --local-chrome

# With custom Chrome path and debugging port
index run --local-chrome --chrome-path="/path/to/chrome" --port=9223
```

This will launch Chrome with remote debugging enabled and connect Index to it.

#### OS-specific Chrome paths

Default Chrome executable paths on different operating systems:

**macOS**:
```bash
index run --local-chrome --chrome-path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
```

**Windows**:
```bash
index run --local-chrome --chrome-path="C:\Program Files\Google\Chrome\Application\chrome.exe"
```

#### Connecting to an already running Chrome instance

If you already have Chrome running with remote debugging enabled, you can connect to it:

1. Launch Chrome with debugging enabled:
   ```bash
   # macOS
   /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222
   
   # Windows
   "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222
   ```

2. Then run Index with the same port:
   ```bash
   index run --local-chrome --port=9222
   ```

### Run the agent with code
```python
import asyncio
from index import Agent, AnthropicProvider

async def main():

    llm = AnthropicProvider(
            model="claude-3-7-sonnet-20250219",
            enable_thinking=True, 
            thinking_token_budget=2048)
    # llm = OpenAIProvider(model="o4-mini") you can also use OpenAI models

    agent = Agent(llm=llm)

    output = await agent.run(
        prompt="Navigate to news.ycombinator.com, find a post about AI, and summarize it"
    )
    
    print(output.result)
    
if __name__ == "__main__":
    asyncio.run(main())
```

### Stream the agent's output
```python
async for chunk in agent.run_stream(
    prompt="Navigate to news.ycombinator.com, find a post about AI, and summarize it"
):
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
    
    llm = AnthropicProvider(model="claude-3-7-sonnet-20250219", enable_thinking=True, thinking_token_budget=2048)
    
    agent = Agent(llm=llm, browser_config=browser_config)
    
    output = await agent.run(
        prompt="Navigate to news.ycombinator.com and find the top story"
    )
    
    print(output.result)
    
if __name__ == "__main__":
    asyncio.run(main())
```

### Run with local Chrome instance (programmatically)
```python
import asyncio
from index import Agent, AnthropicProvider, BrowserConfig

async def main():
    # Configure browser to connect to a local Chrome instance
    browser_config = BrowserConfig(
        cdp_url="http://localhost:9222"
    )
    
    llm = AnthropicProvider(model="claude-3-7-sonnet-20250219", enable_thinking=True, thinking_token_budget=2048)
    
    agent = Agent(llm=llm, browser_config=browser_config)
    
    output = await agent.run(
        prompt="Navigate to news.ycombinator.com and find the top story"
    )
    
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
    
    llm = AnthropicProvider(model="claude-3-7-sonnet-20250219")
    
    agent = Agent(llm=llm, browser_config=browser_config)
    
    output = await agent.run(
        "Navigate to a responsive website and capture how it looks in full HD resolution"
    )
    
    print(output.result)
    
if __name__ == "__main__":
    asyncio.run(main())
```

---

Made with ❤️ by the [Laminar team](https://lmnr.ai)
