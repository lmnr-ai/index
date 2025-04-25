from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import AsyncGenerator, Optional

from dotenv import load_dotenv
from lmnr import Laminar, LaminarSpanContext, observe, use_span
from pydantic import ValidationError

from index.agent.message_manager import MessageManager
from index.agent.models import (
	ActionResult,
	AgentLLMOutput,
	AgentOutput,
	AgentState,
	AgentStreamChunk,
	FinalOutputChunk,
	StepChunk,
	StepChunkContent,
	StepChunkError,
	TimeoutChunk,
	TimeoutChunkContent,
)
from index.browser.browser import Browser, BrowserConfig
from index.controller.controller import Controller
from index.llm.llm import BaseLLMProvider, Message

load_dotenv()
logger = logging.getLogger(__name__)

class Agent:
	def __init__(
		self,
		llm: BaseLLMProvider,
		browser_config: BrowserConfig | None = None
	):
		self.llm = llm
		self.controller = Controller()

		# Initialize browser or use the provided one
		self.browser = Browser(config=browser_config if browser_config is not None else BrowserConfig())
		
		action_descriptions = self.controller.get_action_descriptions()

		self.message_manager = MessageManager(
			action_descriptions=action_descriptions,
		)

		self.state = AgentState(
			messages=[],
		)

	async def step(self, step: int, previous_result: ActionResult | None = None, step_span_context: Optional[LaminarSpanContext] = None) -> tuple[ActionResult, str]:
		"""Execute one step of the task"""

		with Laminar.start_as_current_span(
			name="agent.step",
			parent_span_context=step_span_context,
			input={
				"step": step,
			},
		):
			state = await self.browser.update_state()

			if previous_result:
				self.message_manager.add_current_state_message(state, previous_result)

			input_messages = self.message_manager.get_messages()

			try:
				model_output = await self._generate_action(input_messages)
			except Exception as e:
				# model call failed, remove last state message from history before retrying
				self.message_manager.remove_last_message()
				raise e
			
			if previous_result:
				# we're removing the state message that we've just added because we want to append it in a different format
				self.message_manager.remove_last_message()

			self.message_manager.add_message_from_model_output(step, previous_result, model_output, state.screenshot)
			
			try:
				result: ActionResult = await self.controller.execute_action(
					model_output.action,
					self.browser
				)

				if result.is_done:
					logger.info(f'Result: {result.content}')
					self.final_output = result.content

				return result, model_output.summary
				
			except Exception as e:
				raise e


	@observe(name='agent.generate_action', ignore_input=True)
	async def _generate_action(self, input_messages: list[Message]) -> AgentLLMOutput:
		"""Get next action from LLM based on current state"""

		response = await self.llm.call(input_messages)
		
		# Extract content between <output> tags using regex, including variations like <output_32>
		pattern = r"<output(?:[^>]*)>(.*?)</output(?:[^>]*)>"
		match = re.search(pattern, response.content, re.DOTALL)
		
		json_str = ""

		if not match:
			# if we couldn't find the <output> tags, it most likely means the <output*> tag is not present in the response
			# remove closing and opening tags just in case
			closing_tag_pattern = r"</output(?:[^>]*)>"
			json_str = re.sub(closing_tag_pattern, "", response.content).strip()

			open_tag_pattern = r"<output(?:[^>]*)>"
			json_str = re.sub(open_tag_pattern, "", json_str).strip()

			json_str = json_str.replace("```json", "").replace("```", "").strip()

		else:
			# Extract just the content between the tags without any additional replacement
			json_str = match.group(1).strip()
			
		try:
			# First try to parse it directly to catch any obvious JSON issues
			try:
				json.loads(json_str)
			except json.JSONDecodeError:
				# If direct parsing fails, attempt to fix common issues
				# Remove escape characters and control characters (0x00-0x1F) that might cause problems
				json_str = json_str.replace('\\n', '\n').replace('\\r', '\r').replace('\\t', '\t')
				# Clean all control characters (0x00-0x1F) except valid JSON whitespace (\n, \r, \t)
				json_str = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', json_str)
				
			output = AgentLLMOutput.model_validate_json(json_str.strip())
			
			logger.info(f'💡 Thought: {output.thought}')
			logger.info(f'💡 Summary: {output.summary}')
			logger.info(f'🛠️ Action: {output.action.model_dump_json(exclude_unset=True)}')
			
			if response.thinking:
				output.thinking_block = response.thinking

			return output
		except ValidationError as e:
			raise ValueError(f"Could not parse response: {str(e)}\nResponse was: {json_str}")

	async def _setup_messages(self, prompt: str, agent_state: str | None = None, start_url: str | None = None):
		"""Set up messages based on state dict or initialize with system message"""
		if agent_state:
			# assuming that the structure of the state.messages is correct
			state = AgentState.model_validate_json(agent_state)
			self.message_manager.set_messages(state.messages)
			# Update browser_context to browser
			browser_state = await self.browser.update_state()
			self.message_manager.add_current_state_message(browser_state, user_follow_up_message=prompt)
		else:
			self.message_manager.add_system_message_and_user_prompt(prompt)

			if start_url:
				await self.browser.goto(start_url)
				browser_state = await self.browser.update_state()
				self.message_manager.add_current_state_message(browser_state)
				

	async def run(self, 
			   	prompt: str,
			   	max_steps: int = 100,
				agent_state: str | None = None,
			   	parent_span_context: Optional[LaminarSpanContext] = None, 		
			   	close_context: bool = True,
			   	session_id: str | None = None,
			   	return_agent_state: bool = False,
			   	return_storage_state: bool = False,
			   	start_url: str | None = None,
	) -> AgentOutput:
		"""Execute the task with maximum number of steps and return the final result
		
		Args:
			prompt: The prompt to execute the task with
			max_steps: The maximum number of steps to execute the task with. Defaults to 100.
			agent_state: Optional, the state of the agent to execute the task with
			parent_span_context: Optional, parent span context in Laminar format to execute the task with
			close_context: Whether to close the browser context after the task is executed
			session_id: Optional, Agent session id
			return_agent_state: Whether to return the agent state with the final output
			return_storage_state: Whether to return the storage state with the final output
			start_url: Optional, the URL to start the task with
		"""

		if prompt is None and agent_state is None:
			raise ValueError("Either prompt or agent_state must be provided")

		with Laminar.start_as_current_span(
			name="agent.run",
			parent_span_context=parent_span_context,
			input={
				"prompt": prompt,
				"max_steps": max_steps,
				"stream": False,
			},
		) as span:
			if session_id is not None:
				span.set_attribute("lmnr.internal.agent_session_id", session_id)
			
			await self._setup_messages(prompt, agent_state, start_url)

			step = 0
			result = None
			is_done = False

			trace_id = str(uuid.UUID(int=span.get_span_context().trace_id))

			try:
				while not is_done and step < max_steps:
					logger.info(f'📍 Step {step}')
					result, _ = await self.step(step, result)
					step += 1
					is_done = result.is_done
					
					if is_done:
						logger.info(f'✅ Task completed successfully in {step} steps')
						break
						
				if not is_done:
					logger.info('❌ Maximum number of steps reached')

			except Exception as e:
				logger.info(f'❌ Error in run: {e}')
				raise e
			finally:
				storage_state = await self.browser.get_storage_state()

				if close_context:
					# Update to close the browser directly
					await self.browser.close()

				span.set_attribute("lmnr.span.output", result.model_dump_json())

				return AgentOutput(
					agent_state=self.get_state() if return_agent_state else None,
					result=result,
					storage_state=storage_state if return_storage_state else None,
					step_count=step,
					trace_id=trace_id,
				)

	async def run_stream(self, 
						prompt: str,
						max_steps: int = 100, 
						agent_state: str | None = None,
						parent_span_context: Optional[LaminarSpanContext] = None,
						close_context: bool = True,
						timeout: Optional[int] = None,
						session_id: str | None = None,
						return_screenshots: bool = False,
						return_agent_state: bool = False,
						return_storage_state: bool = False,
						start_url: str | None = None,
						) -> AsyncGenerator[AgentStreamChunk, None]:
		"""Execute the task with maximum number of steps and stream step chunks as they happen
		
		Args:
			prompt: The prompt to execute the task with
			max_steps: The maximum number of steps to execute the task with
			agent_state: The state of the agent to execute the task with
			parent_span_context: Parent span context in Laminar format to execute the task with
			close_context: Whether to close the browser context after the task is executed
			timeout: The timeout for the task
			session_id: Agent session id
			return_screenshots: Whether to return screenshots with the step chunks
			return_agent_state: Whether to return the agent state with the final output chunk
			return_storage_state: Whether to return the storage state with the final output chunk
			start_url: Optional, the URL to start the task with
		"""
		
		# Create a span for the streaming execution
		span = Laminar.start_span(
			name="agent.run_stream",
			parent_span_context=parent_span_context,
			input={
				"prompt": prompt,
				"max_steps": max_steps,
				"stream": True,
			},
		)

		trace_id = str(uuid.UUID(int=span.get_span_context().trace_id))
		
		if session_id is not None:
			span.set_attribute("lmnr.internal.agent_session_id", session_id)
		
		with use_span(span):
			await self._setup_messages(prompt, agent_state, start_url)

		step = 0
		result = None
		is_done = False

		if timeout is not None:
			start_time = time.time()

		try:
			# Execute steps and yield results
			while not is_done and step < max_steps:
				logger.info(f'📍 Step {step}')

				with use_span(span):
					result, summary = await self.step(step, result)

				step += 1
				is_done = result.is_done

				screenshot = None
				if return_screenshots:
					state = self.browser.get_state()
					screenshot = state.screenshot

				if timeout is not None and time.time() - start_time > timeout:
					
					yield TimeoutChunk(
							content=TimeoutChunkContent(
										action_result=result, 
										summary=summary, 
										step=step, 
										agent_state=self.get_state() if return_agent_state else None, 
										screenshot=screenshot,
										trace_id=trace_id
										)
					)
					return

				yield StepChunk(
						content=StepChunkContent(
									action_result=result, 
									summary=summary, 
									trace_id=trace_id,
									screenshot=screenshot
									)
				)

				if is_done:
					logger.info(f'✅ Task completed successfully in {step} steps')
					
					storage_state = await self.browser.get_storage_state()

					# Yield the final output as a chunk
					final_output = AgentOutput(
						agent_state=self.get_state() if return_agent_state else None,
						result=result,
						storage_state=storage_state if return_storage_state else None,
						step_count=step,
						trace_id=trace_id,
					)

					span.set_attribute("lmnr.span.output", result.model_dump_json())
					yield FinalOutputChunk(content=final_output)

					break

			if not is_done:
				logger.info('❌ Maximum number of steps reached')
				yield StepChunkError(content=f'Maximum number of steps reached: {max_steps}')
			
		except Exception as e:
			logger.info(f'❌ Error in run: {e}')
			span.record_exception(e)
			
			yield StepChunkError(content=f'Error in run stream: {e}')
		finally:
			# Clean up resources		
			if close_context:
				# Update to close the browser directly
				await self.browser.close()

			span.end()
			logger.info('Stream complete, span closed')

	def get_state(self) -> AgentState:

		self.state.messages = self.message_manager.get_messages()

		return self.state
