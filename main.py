import asyncio
import logging
from datetime import datetime

from dotenv import load_dotenv
from lmnr import Laminar

from lmnr_index.agent.agent import Agent
from lmnr_index.llm.providers.anthropic import AnthropicProvider

BOOKING_PROMPT = """
Go to this link https://www.booking.com/searchresults.html?ss=London%2C+United+Kingdom&efdco=1&label=gen173nr-1FCAEoggI46AdIM1gEaFCIAQGYATG4AQfIAQzYAQHoAQH4AQKIAgGoAgO4AqDLtb8GwAIB0gIkZTJiOWEwYTItMWFhMy00NjExLWI2OTYtYjIxYzU5ZDViYTA02AIF4AIB&aid=304142&lang=en-us&sb=1&src_elem=sb&src=index&dest_id=-2601889&dest_type=city&checkin=2025-04-05&checkout=2025-04-06&group_adults=2&no_rooms=1&group_children=0.
Open the first hotel, summarize it, then open the second hotel and also summarize it.
"""

# BOOKING_PROMPT = """
# Go to https://www.booking.com/hotel/gb/the-bermondsey-square.en-gb.html
# What amenities in the rooms does the hotel offer?
# """


load_dotenv()
logger = logging.getLogger("lmnr")
logger.setLevel(logging.DEBUG)


# bb = Browserbase(api_key=os.environ["BROWSERBASE_API_KEY"])
# session = bb.sessions.create(project_id=os.environ["BROWSERBASE_PROJECT_ID"])

# Laminar.initialize(
#     project_api_key="8siEZ3GCnlUipp6ipHHpoTZ9HhzR3AABU28QzZeQstmZiqrlZwIywLyZTmK7gs32",
#     # project_api_key="KFyaBMX7Q6wmiaS2T5j4F84sVTHNpN94PmBECIg4xfsuqt6NiDHoEP2y6h9DtQaT",
#     # base_url="http://localhost",
#     # http_port=8000,
#     # grpc_port=8001,
# )

llm_provider = AnthropicProvider(
    model="claude-3-7-sonnet-20250219",
    enable_thinking=False,
    # thinking_token_budget=8000,
)

# browser_config = BrowserConfig(
#     cdp_url=session.connect_url
# )

# browser = Browser(browser_config)

agent = Agent(llm=llm_provider, 
            #   browser=browser
              )

async def main():
    async for chunk in agent.run_stream(
        prompt="go to https://www.lmnr.ai\nTell me about their pricing on Pro tier"
    ):
        logger.warning(f"Flushing Laminar {datetime.now()}")
        res = Laminar.flush()
        logger.warning(f"Flushed Laminar {datetime.now()}, {res}")
        if chunk.type == "final_output":
            print(chunk.content.result)
        elif chunk.type == "step":
            print(chunk.content.action_result)

asyncio.run(main())

