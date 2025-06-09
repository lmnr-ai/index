import asyncio
import base64

from dotenv import load_dotenv
load_dotenv(override=True)

from index.browser.detector import OmniparserDetector
from index.browser.browser import Browser, BrowserConfig
from index.browser.utils import put_highlight_elements_on_screenshot

endpoint = "http://xxx:8000"
detector = OmniparserDetector(endpoint)


def save(image_b64: str, name: str):
    try:
        data = base64.b64decode(image_b64)
        with open(f"{name}.png", "wb") as fp:
            fp.write(data)
        print(f"Successfully save to {name}.png")
    except Exception as e:
        print(f"Failed to save {name}.png\n{e}")


async def get_elements_by_ocr():
    browser = Browser(BrowserConfig(cdp_url="http://localhost:9222", detector=detector))
    page = await browser.get_current_page()
    await page.goto("https://en.wikipedia.org/wiki/Main_Page")

    screenshot_b64 = await browser.fast_screenshot()
    interactive_elements_data = await browser.get_interactive_elements(screenshot_b64, False)
    interactive_elements = {element.index: element for element in interactive_elements_data.elements}
    print(f"Ocr detector find {len(interactive_elements)} elements.")

    screenshot_with_highlights = put_highlight_elements_on_screenshot(
        interactive_elements, 
        screenshot_b64
    )

    save(screenshot_with_highlights, f"ocr_highlight")
    
    await browser.close()

async def get_elements_by_html():
    browser = Browser(BrowserConfig(cdp_url="http://localhost:9222", detector=None))
    page = await browser.get_current_page()
    await page.goto("https://en.wikipedia.org/wiki/Main_Page")

    screenshot_b64 = await browser.fast_screenshot()
    interactive_elements_data = await browser.get_interactive_elements(screenshot_b64, False)
    interactive_elements = {element.index: element for element in interactive_elements_data.elements}
    print(f"Html detector find {len(interactive_elements)} elements.")

    screenshot_with_highlights = put_highlight_elements_on_screenshot(
        interactive_elements, 
        screenshot_b64
    )

    save(screenshot_with_highlights, f"html_highlight")
    
    await browser.close()


async def main():
    await get_elements_by_ocr()
    await get_elements_by_html()

asyncio.run(main())