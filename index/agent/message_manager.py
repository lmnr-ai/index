from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from index.agent.models import ActionResult, AgentLLMOutput
from index.agent.prompts import system_message
from index.agent.utils import load_demo_image_as_b64
from index.browser.models import BrowserState
from index.browser.utils import scale_b64_image
from index.llm.llm import ImageContent, Message, TextContent

logger = logging.getLogger(__name__)


class MessageManager:
	def __init__(
		self,
		action_descriptions: str,
	):
		self._messages: List[Message] = []
		self.action_descriptions = action_descriptions


	def add_system_message_and_user_prompt(self, prompt: str) -> None:

		complex_layout_highlight = load_demo_image_as_b64('complex_layout_highlight.png')
		complex_layout_small_elements = load_demo_image_as_b64('complex_layout_small_elements.png')
		still_loading = load_demo_image_as_b64('loading.png')
		still_loading_2 = load_demo_image_as_b64('loading2.png')
		scroll_over_element_example = load_demo_image_as_b64('scroll.png')
		system_msg = Message(
			role="system",
			content=[
				TextContent(text=system_message(self.action_descriptions), cache_control=True),
			],
		)

		self._messages.append(system_msg)

		self._messages.append(Message(
			role="user",
			content=[
				TextContent(text='<complex_layout_example>'),
				TextContent(text="Here's an example of a complex layout. As an example, if you want to select a 'Roster' section for Colorado Rockies. Then you need to click on element with index 121."),
				ImageContent(image_b64=complex_layout_highlight),
				TextContent(text='</complex_layout_example>'),
				TextContent(text='<small_elements_example>'),
				TextContent(text="Here's an example of small elements on the page and their functions. Element 7, represented by 'x' icon, is a 'clear text' button. Element 8 is a 'submit' button, represented by '=' icon. This clarification should help you better understand similar layouts."),
				ImageContent(image_b64=complex_layout_small_elements),
				TextContent(text='</small_elements_example>'),
				TextContent(text='<loading_pages_example>'),
				TextContent(text="Here are some examples of loading pages. If the main content on the page is empty or if there are loading elements, such as skeleton screens, page is still loading. Then, you HAVE to perform `wait_for_page_to_load` action."),
				ImageContent(image_b64=still_loading),
				ImageContent(image_b64=still_loading_2),
				TextContent(text='</loading_pages_example>'),
				TextContent(text='<scroll_over_element_example>'),
				TextContent(text="In some cases, to reveal more content, you need to scroll in scrollable areas of the webpage. Scrollable areas have VERTICAL scrollbars very clearly visible on their right side. In the screenshot below, you can clearly see a scrollbar on the right side of the list of search items. This indicates that the list is scrollable. To scroll over this area, you need to identify any element within the scrollable area and use its index with `scroll_down_over_element` action to scroll over it. In this example, approriate element is with index 15."),
				ImageContent(image_b64=scroll_over_element_example),
				TextContent(text='</scroll_over_element_example>', cache_control=True),
				TextContent(text=f"""Here is the task you need to complete:\n\n<task>\n{prompt}\n</task>

Today's date and time is: {datetime.now().strftime('%B %d, %Y, %I:%M%p')} - keep this date and time in mind when planning your actions."""),
			]
		))

	def get_messages_as_state(self) -> List[Message]:
		"""Get messages as state messages"""
		return [msg for msg in self._messages if msg.is_state_message]


	def remove_last_message(self) -> None:
		"""Remove last message from history"""
		if len(self._messages) > 1:
			self._messages.pop()

	def add_current_state_message(
		self,
		state: BrowserState,
		previous_result: ActionResult | None = None,
		user_follow_up_message: str | None = None,
	) -> None:
		"""Add browser state as a user message"""

		if state.interactive_elements:
			highlighted_elements = ''
			for element in state.interactive_elements.values():
				
				# exclude sheets elements
				if element.browser_agent_id.startswith("row_") or element.browser_agent_id.startswith("column_"):
					continue

				start_tag = f"[{element.index}]<{element.tag_name}"

				if element.input_type:
					start_tag += f" type=\"{element.input_type}\""

				start_tag += ">"
				element_text = element.text.replace('\n', ' ')
				highlighted_elements += f"{start_tag}{element_text}</{element.tag_name}>\n"
		else:
			highlighted_elements = ''

		scroll_distance_above_viewport = state.viewport.scroll_distance_above_viewport or 0
		scroll_distance_below_viewport = state.viewport.scroll_distance_below_viewport or 0

		if scroll_distance_above_viewport > 0:
			elements_text = f'{scroll_distance_above_viewport}px scroll distance above current viewport\n'
		else:
			elements_text = '[Start of page]\n'

		if highlighted_elements != '':
			elements_text += f'\nHighlighted elements:\n{highlighted_elements}'

		if scroll_distance_below_viewport > 0:
			elements_text += f'\n{scroll_distance_below_viewport}px scroll distance below current viewport\n'
		else:
			elements_text += '\n[End of page]'

		previous_action_output = ''
		if previous_result:
			previous_action_output = f'<previous_action_output>\n{previous_result.content}\n</previous_action_output>\n\n' if previous_result.content else ''

			if previous_result.error:
				previous_action_output += f'<previous_action_error>\n{previous_result.error}\n</previous_action_error>\n\n'

		if user_follow_up_message:
			user_follow_up_message = f'<user_follow_up_message>\n{user_follow_up_message}\n</user_follow_up_message>\n\n'
		else:
			user_follow_up_message = ''

		state_description = f"""{previous_action_output}{user_follow_up_message}
<viewport>
Current URL: {state.url}

Open tabs:
{state.tabs}

Current viewport information:
{elements_text}
</viewport>"""

		state_msg = Message(
			role='user',
			content=[
				TextContent(text=state_description),
				TextContent(text='<current_state_clean_screenshot>'),
				ImageContent(image_b64=state.screenshot),
				TextContent(text='</current_state_clean_screenshot>'),
				TextContent(text='<current_state>'),
				ImageContent(image_b64=state.screenshot_with_highlights),
				TextContent(text='</current_state>'),
			]
		)
	
		self._messages.append(state_msg)

	def add_message_from_model_output(self, step: int, previous_result: ActionResult | None, model_output: AgentLLMOutput, screenshot: Optional[str] = None) -> None:
		"""Add model output as AI message"""

		previous_action_output = ''

		for msg in self._messages:
			if msg.is_state_message:
				msg.content = [msg.content[0]]

		if previous_result and screenshot:
			previous_action_output = f'<action_output_{step-1}>\n{previous_result.content}\n</action_output_{step-1}>' if previous_result.content else ''

			if previous_result.error:
				previous_action_output += f'<action_error_{step-1}>\n{previous_result.error}\n</action_error_{step-1}>'

			usr_msg = Message(
				role='user',
				content=[
					TextContent(text=previous_action_output, cache_control=True),
					TextContent(text=f"<state_{step}>"),
					ImageContent(image_b64=scale_b64_image(screenshot, 0.75)),
					TextContent(text=f"</state_{step}>"),
				],
				is_state_message=True,
			)
			self._messages.append(usr_msg)

		assistant_content = [
			TextContent(text=f"""<output_{step}>
{model_output.model_dump_json(indent=2, include={"thought", "action", "summary"}).strip()}
</output_{step}>"""),
			]
		
		if model_output.thinking_block:
			assistant_content = [
				model_output.thinking_block,
			] + assistant_content
		
		msg = Message(
			role='assistant',
			content=assistant_content,
		)

		self._messages.append(msg)

	def get_messages(self) -> List[Message]:

		found_first_cache_control = False

		# clear all past cache control except the latest one
		for msg in self._messages[::-1]:

			# ignore system messages
			if msg.role == 'system':
				continue

			if found_first_cache_control:
				msg.remove_cache_control()

			if msg.has_cache_control():
				found_first_cache_control = True
			

		return self._messages
	
	def set_messages(self, messages: List[Message]) -> None:
		"""Set messages"""
		self._messages = messages
