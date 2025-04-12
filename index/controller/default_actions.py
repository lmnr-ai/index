import asyncio
import json
import logging
import platform
import re

from tenacity import retry, stop_after_attempt, wait_exponential

from index.agent.models import ActionResult
from index.browser.browser import Browser

logger = logging.getLogger(__name__)

def register_default_actions(controller, output_model=None):
    """Register all default browser actions to the provided controller"""

    @controller.action('Complete task')
    async def done(text: str):
        return ActionResult(is_done=True, content=text)

    @controller.action()
    async def give_human_control(message: str, browser: Browser):
        """Give human control of the browser. Use this action when you need to use user information, such as first name, last name, email, phone number, booking information, login/password, etc. to proceed with the task. Also, if you can't solve the CAPTCHA, use this action.
        
        Args:
            message: Message to give to the human, explaining why you need human intervention.
        """
        return ActionResult(give_control=True, content=message, is_done=True)


    @controller.action()
    async def search_google(query: str, browser: Browser):
        """
        Open google search in new tab and search for the query.
        """
        page = await browser.get_current_page()
        await page.goto(f'https://www.google.com/search?q={query}&udm=14')
        await page.wait_for_load_state()
        msg = f"Searched for '{query}' in Google"
        logger.info(msg)
        return ActionResult(content=msg)

    @controller.action()
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying step after error: {retry_state.outcome.exception()}. Attempt {retry_state.attempt_number}"
        )
    )
    async def go_to_url(url: str, browser: Browser):
        """Navigate to URL in the current tab"""
        page = await browser.get_current_page()
        await page.goto(url, wait_until='domcontentloaded')
        await asyncio.sleep(1.5)
        msg = f"Navigated to {url}"
        logger.info(msg)
        return ActionResult(content=msg)

    @controller.action()
    async def go_back_to_previous_page(browser: Browser):
        """Go back to the previous page"""
        try:
            page = await browser.get_current_page()            
            await page.go_back(wait_until='domcontentloaded')
            # Wait for the page to stabilize
            await asyncio.sleep(1)
            msg = 'Navigated back to the previous page'
            logger.info(msg)
            return ActionResult(content=msg)

        except Exception as e:
            logger.debug(f'During go_back: {e}')
            return ActionResult(error=str(e))

    @controller.action()
    async def click_on_spreadsheet_cell(row: str, column: str, browser: Browser) -> ActionResult:
        """Click on a spreadsheet cell at a specific row and column. You HAVE to use this action when you need to click on a cell in a spreadsheet. DON'T try to use click_element action, it will not work.
        
        Args:
            row: Row of the cell to click on, it should be a number formatted as a string. e.g. "1"
            column: Column of the cell to click on, it should be a letter formatted as a string. e.g. "A"
        """
        page = await browser.get_current_page()
        state = browser.get_state()
        
        elements = state.interactive_elements.values()

        row_element = next((e for e in elements if e.browser_agent_id == f"row_{row}"), None)
        column_element = next((e for e in elements if e.browser_agent_id == f"column_{column}"), None)

        if not row_element or not column_element:
            return ActionResult(error='Row or column element not found - pay close attention to the row and column numbers.')

        # reseting click just in case
        await page.mouse.click(state.viewport.width / 2, state.viewport.height / 2)
        await asyncio.sleep(0.05)

        await page.mouse.click(column_element.center.x, row_element.center.y, click_count=2)
        await asyncio.sleep(0.05)

        return ActionResult(content=f'Clicked on spreadsheet cell with row {row} and column {column}')


    @controller.action()
    async def click_element(index: int, wait_after_click: bool, browser: Browser):
        """
        Click on the element with index. 

        Args:
            index: Index of the element to click on.
            wait_after_click: If True, wait for 2 second after clicking the element. Only set it to True when you think that clicking will trigger loading state, for instance navigation to new page, search, loading of a content, etc.
        """
        # clean index if it contains any non-numeric characters
        cleaned_index_str = re.sub(r'\D', '', str(index))
        if cleaned_index_str == '':
            logger.error(f'Index is not a number. Index: {index}')
            return ActionResult(error="`index` should be a valid number.")
        
        index = int(cleaned_index_str)

        state = browser.get_state()

        if index not in state.interactive_elements:
            return ActionResult(error=f"Element with index {index} does not exist - retry or use alternative actions.")

        element = state.interactive_elements[index]
        initial_pages = len(browser.context.pages) if browser.context else 0

        try:
            page = await browser.get_current_page()

            await page.mouse.click(element.center.x, element.center.y)

            msg = f'Clicked element with index {index}: <{element.tag_name}></{element.tag_name}>'

            logger.info(msg)
            if browser.context and len(browser.context.pages) > initial_pages:
                new_tab_msg = 'New tab opened - switching to it'
                msg += f' - {new_tab_msg}'
                logger.info(new_tab_msg)
                await browser.switch_to_tab(-1)
            
            if wait_after_click:
                await asyncio.sleep(2)

            return ActionResult(content=msg)
        except Exception as e:
            return ActionResult(error=str(e))
 
    @controller.action(
        description='Use this action to wait for the page to load, if you see that the content on the clean screenshot is empty or loading UI elements such as skeleton screens. This action will wait for page to load. Then you can continue with your actions.',
    )
    async def wait_for_page_to_load() -> ActionResult:
        return ActionResult(content='Waited for page to load')

    @controller.action()
    async def enter_text(text: str, press_enter: bool, browser: Browser):
        """Enter text with a keyboard. Use it AFTER you have clicked on an input element. This action will override the current text in the element.
        
        Args:
            text: Text to enter with a keyboard.
            press_enter: If True, `Enter` button will be pressed after entering the text. Use this when you think it would make sense to press `Enter` after entering the text, such as when you're submitting a form, performing a search, etc.
        """
                    
        try:
            page = await browser.get_current_page()
            # clear the element
            await page.keyboard.press("ControlOrMeta+a")
            await asyncio.sleep(0.1)
            await page.keyboard.press("Backspace")
            await asyncio.sleep(0.1)

            # input text into the element
            await page.keyboard.type(text)

            if press_enter:
                await page.keyboard.press("Enter")
                await asyncio.sleep(2)

            msg = f'Entered "{text}" on the keyboard. Make sure to double check that the text was entered to where you intended.'
            logger.info(msg)
            return ActionResult(content=msg)
        except Exception as e:
            return ActionResult(error=f'Failed to enter text. Error: {str(e)}')

    # Tab Management Actions
    @controller.action('Switch tab')
    async def switch_tab(page_id: int, browser: Browser):
        await browser.switch_to_tab(page_id)
        await asyncio.sleep(0.5)
        msg = f'Switched to tab {page_id}'
        logger.info(msg)
        return ActionResult(content=msg)

    @controller.action('Open url in new tab')
    async def open_tab(url: str, browser: Browser):
        await browser.create_new_tab(url)
        msg = f'Opened new tab with {url}'
        logger.info(msg)
        return ActionResult(content=msg)

    @controller.action(
        "Scroll entire page down. Use this action when you want to scroll entire page down to load more content. DON'T use this action if you want to scroll over a scrollable element."
    )
    async def scroll_page_down(browser: Browser):
        page = await browser.get_current_page()
        state = browser.get_state()
        # move mouse to the center of the page
        await page.mouse.move(state.viewport.width / 2, state.viewport.height / 2)
        await asyncio.sleep(0.1)
        # scroll down by one page
        await page.mouse.wheel(0, state.viewport.height * 0.8)
        return ActionResult(content="Scrolled mouse wheel down (it doesn't guarantee that something has scrolled, you need to check new state screenshot to confirm)")

    @controller.action(
        "Scroll entire page up. Use this action when you want to scroll entire page up to load more content. DON'T use this action if you want to scroll over a scrollable element."
    )
    async def scroll_page_up(browser: Browser):
        page = await browser.get_current_page()
        state = browser.get_state()
        # move mouse to the center of the page
        await page.mouse.move(state.viewport.width / 2, state.viewport.height / 2)
        await asyncio.sleep(0.1)
        # scroll up by one page
        await page.mouse.wheel(0, -state.viewport.height * 0.8)
        return ActionResult(content="Scrolled mouse wheel up (it doesn't guarantee that something has scrolled, you need to check new state screenshot to confirm)")

    @controller.action(
        "Moves mouse to the element with index `index`, located inside scrollable area of the webpage, identified by scrollbars. Then scrolls mouse wheel down."
    )
    async def scroll_down_over_element(index: int, browser: Browser):
        page = await browser.get_current_page()
        state = browser.get_state()

        if index not in state.interactive_elements:
            return ActionResult(error=f'Element index {index} does not exist - retry or use alternative actions')

        element = state.interactive_elements[index]

        await page.mouse.move(element.center.x, element.center.y)
        await asyncio.sleep(0.1)
        await page.mouse.wheel(0, state.viewport.height / 3)

        return ActionResult(content=f"Move mouse to element with index {index} and scroll mouse wheel down. (It doesn't guarantee that something has scrolled, you need to check new state screenshot to confirm)")
    
    @controller.action(
        "Moves mouse to the element with index `index`, located inside scrollable area of the webpage, identified by scrollbars. Then scrolls mouse wheel up."
    )
    async def scroll_up_over_element(index: int, browser: Browser):
        page = await browser.get_current_page()
        state = browser.get_state()

        if index not in state.interactive_elements:
            return ActionResult(error=f'Element index {index} does not exist - retry or use alternative actions')

        element = state.interactive_elements[index]

        await page.mouse.move(element.center.x, element.center.y)
        await asyncio.sleep(0.1)
        await page.mouse.wheel(0, -state.viewport.height / 3)

        return ActionResult(content=f"Move mouse to element with index {index} and scroll mouse wheel up. (It doesn't guarantee that something has scrolled, you need to check new state screenshot to confirm)")

    @controller.action(
        'Press enter key. Use this action when you need to submit a form or perform an action that requires pressing enter.'
    )
    async def press_enter(browser: Browser):
        page = await browser.get_current_page()

        await page.keyboard.press('Enter')
        return ActionResult(content='Pressed enter key')
    
    @controller.action(
        'Remove all text in the element with index.'
    )
    async def clear_text_in_element(index: int, browser: Browser):
        page = await browser.get_current_page()
        
        state = browser.get_state()

        if index not in state.interactive_elements:
            return ActionResult(error=f'Element index {index} does not exist - retry or use alternative actions')

        element = state.interactive_elements[index]

        await page.mouse.move(element.center.x, element.center.y)
        await page.mouse.click(element.center.x, element.center.y)
        await asyncio.sleep(0.1)

        if platform.system() == "Darwin":
            await page.keyboard.press('Meta+A')
        else:
            await page.keyboard.press('Control+A')
        await asyncio.sleep(0.1)
        await page.keyboard.press('Backspace')
        return ActionResult(content='Removed all text in the element with index')

    @controller.action()
    async def get_select_options(index: int, browser: Browser) -> ActionResult:
        """Get all options from a <select> element. Use this action when you need to get all options from a dropdown."""

        try:
            # Get the page and element information
            page = await browser.get_current_page()
            interactive_elements = browser.get_state().interactive_elements
            
            # Verify the element exists and is a select
            if index not in interactive_elements:
                return ActionResult(error=f"No element found with index {index}")
                
            element = interactive_elements[index]
            
            # Check if it's a select element
            if element.tag_name.lower() != 'select':
                return ActionResult(error=f"Element {index} is not a select element, it's a {element.tag_name}")
            
            # Use the unique ID to find the element
            options_data = await page.evaluate("""
            (args) => {
                // Find the select element using the unique ID
                const select = document.querySelector(`[data-browser-agent-id="${args.browserAgentId}"]`);
                if (!select) return null;
                
                // Get all options	
                return {
                    options: Array.from(select.options).map(opt => ({
                        text: opt.text,
                        value: opt.value,
                        index: opt.index
                    })),
                    id: select.id,
                    name: select.name
                };
            }
            """, {"browserAgentId": element.browser_agent_id})

            # Process options from direct approach
            formatted_options = []
            for opt in options_data['options']:
                encoded_text = json.dumps(opt['text'])
                formatted_options.append(f'{opt["index"]}: option={encoded_text}')
                
            msg = '\n'.join(formatted_options)
            msg += '\nIf you decide to use this select element, use the exact option name in select_dropdown_option'
            
            logger.info(f'Found dropdown with ID: {options_data["id"]}, Name: {options_data["name"]}')
            return ActionResult(content=msg)
            
        except Exception as e:
            logger.error(f'Failed to get dropdown options: {str(e)}')
            return ActionResult(error=f'Error getting dropdown options: {str(e)}')

    @controller.action(
        description='Select an option from a <select> element by the text (name) of the option. Use this after get_select_options and when you need to select an option from a dropdown.',
    )
    async def select_dropdown_option(
        index: int,
        option: str,
        browser: Browser,
    ) -> ActionResult:
        """Select dropdown option by the text of the option you want to select"""
        try:
            # Get the interactive element
            page = await browser.get_current_page()
            interactive_elements = browser.get_state().interactive_elements
            
            # Verify the element exists and is a select
            if index not in interactive_elements:
                return ActionResult(error=f"No element found with index {index}")
                
            element = interactive_elements[index]
            
            # Check if it's a select element
            if element.tag_name.lower() != 'select':
                return ActionResult(error=f"Element {index} is not a select element, it's a {element.tag_name}")
            
            logger.debug(f"Attempting to select '{option}' using browser_agent_id: {element.browser_agent_id}")
            
            # Use JavaScript to select the option using the unique ID
            result = await page.evaluate("""
            (args) => {
                const uniqueId = args.uniqueId;
                const optionText = args.optionText;
                
                try {
                    // Find the select element by unique ID - works across frames too
                    function findElementByUniqueId(root, id) {
                        // Check in main document first
                        let element = document.querySelector(`[data-browser-agent-id="${id}"]`);
                        if (element) return element;
                    }
                    
                    const select = findElementByUniqueId(window, uniqueId);
                    if (!select) {
                        return { 
                            success: false, 
                            error: "Select element not found with ID: " + uniqueId 
                        };
                    }
                    
                    // Find the option with matching text
                    let found = false;
                    let selectedValue = null;
                    let selectedIndex = -1;
                    
                    for (let i = 0; i < select.options.length; i++) {
                        const opt = select.options[i];
                        if (opt.text === optionText) {
                            // Select this option
                            opt.selected = true;
                            found = true;
                            selectedValue = opt.value;
                            selectedIndex = i;
                            
                            // Trigger change event
                            const event = new Event('change', { bubbles: true });
                            select.dispatchEvent(event);
                            break;
                        }
                    }
                    
                    if (found) {
                        return { 
                            success: true, 
                            value: selectedValue, 
                            index: selectedIndex 
                        };
                    } else {
                        return { 
                            success: false, 
                            error: "Option not found: " + optionText,
                            availableOptions: Array.from(select.options).map(o => o.text)
                        };
                    }
                } catch (e) {
                    return { 
                        success: false, 
                        error: e.toString() 
                    };
                }
            }
            """, {"uniqueId": element.browser_agent_id, "optionText": option})
            
            if result.get('success'):
                msg = f"Selected option '{option}' with value '{result.get('value')}' at index {result.get('index')}"
                logger.info(msg)
                return ActionResult(content=msg)
            else:
                error_msg = result.get('error', 'Unknown error')
                if 'availableOptions' in result:
                    available = result.get('availableOptions', [])
                    error_msg += f". Available options: {', '.join(available)}"
                    
                logger.error(f"Selection failed: {error_msg}")
                return ActionResult(error=error_msg)
                
        except Exception as e:
            msg = f'Selection failed: {str(e)}'
            logger.error(msg)
            return ActionResult(error=msg)
