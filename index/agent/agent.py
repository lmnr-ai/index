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

		# clean null characters from response
		response.content = re.sub(r'\x00', '', response.content)
		
		# Extract content between <output> tags using regex, including variations like <output_32>
		pattern = r"<output(?:[^>]*)>(.*?)</output(?:[^>]*)>"
		match = re.search(pattern, response.content, re.DOTALL)
		
		json_str = ""

		if not match:
			# if we couldn't find the <output> tags, it most likely means the <output*> tag is not present in the response
			# remove closing tag just in case
			closing_tag_pattern = r"</output(?:[^>]*)>$"
			json_str = re.sub(closing_tag_pattern, "", response.content).strip()
		else:
			# Extract just the content between the tags without any additional replacement
			json_str = match.group(1).strip()
			
		try:
			# First try to parse it directly to catch any obvious JSON issues
			try:
				json.loads(json_str)
			except json.JSONDecodeError:
				# If direct parsing fails, attempt to fix common issues
				# Remove any escape characters that might cause problems
				json_str = json_str.replace('\\n', '\n').replace('\\r', '\r').replace('\\t', '\t')
				
			output = AgentLLMOutput.model_validate_json(json_str.strip())
			
			logger.info(f'💡 Thought: {output.thought}')
			logger.info(f'💡 Summary: {output.summary}')
			logger.info(f'🛠️ Action: {output.action.model_dump_json(exclude_unset=True)}')
			
			if response.thinking:
				output.thinking_block = response.thinking

			return output
		except ValidationError as e:
			raise ValueError(f"Could not parse response: {str(e)}\nResponse was: {json_str}")

	async def _setup_messages(self, prompt: str | None = None, agent_state: str | None = None):
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

	async def run(self, 
			   	prompt: str | None = None,
			   	max_steps: int = 100,
				agent_state: str | None = None,
			   	parent_span_context: Optional[LaminarSpanContext] = None, 		
			   	close_context: bool = True,
			   	prev_action_result: ActionResult | None = None,
			   	session_id: str | None = None,
	) -> AgentOutput:
		"""Execute the task with maximum number of steps and return the final result
		
		Args:
			prompt: The prompt to execute the task with
			max_steps: The maximum number of steps to execute the task with
			agent_state: The state of the agent to execute the task with
			parent_span_context: The parent span context to execute the task with
			close_context: Whether to close the context after the task is executed
			prev_action_result: The previous action result to execute the task with
			session_id: The session id to execute the task with
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
			
			await self._setup_messages(prompt, agent_state)

			step = 0
			result = prev_action_result
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
			finally:
				storage_state = await self.browser.get_storage_state()

				if close_context:
					# Update to close the browser directly
					await self.browser.close()

				return AgentOutput(
					agent_state=self.get_state(),
					result=result,
					storage_state=storage_state,
					step_count=step,
					trace_id=trace_id,
				)

	async def run_stream(self, 
						prompt: str | None = None,
						max_steps: int = 100, 
						agent_state: str | None = None,
						parent_span_context: Optional[LaminarSpanContext] = None,
						close_context: bool = True,
						prev_action_result: ActionResult | None = None,
						prev_step: int | None = None,
						step_span_context: Optional[LaminarSpanContext] = None,
						timeout: Optional[int] = None,
						session_id: str | None = None,
						return_screenshots: bool = False,
						) -> AsyncGenerator[AgentStreamChunk, None]:
		"""Execute the task with maximum number of steps and stream results as they happen"""
		
		if prompt is None and agent_state is None:
			raise ValueError("Either prompt or agent_state must be provided")
		
		if prev_step is not None and (prev_action_result is None or prev_step == 0 or agent_state is None):
			raise ValueError("`prev_action_result` and `agent_state` must be provided if `prev_step` is provided")

		# Create a span for the streaming execution
		span = None
		if step_span_context is None:
			span = Laminar.start_span(
				name="agent.run_stream",
				parent_span_context=parent_span_context,
				input={
					"prompt": prompt,
					"max_steps": max_steps,
					"stream": True,
				},
			)


		if span is not None:
			trace_id = str(uuid.UUID(int=span.get_span_context().trace_id))
			
			if session_id is not None:
				span.set_attribute("lmnr.internal.agent_session_id", session_id)

		elif step_span_context is not None:
			trace_id = str(step_span_context.trace_id)
		else:
			trace_id = None
		
		with use_span(span):
			await self._setup_messages(prompt, agent_state)

		step = prev_step if prev_step is not None else 0
		result = prev_action_result
		is_done = False

		if timeout is not None:
			start_time = time.time()

		try:
			# Execute steps and yield results
			while not is_done and step < max_steps:
				logger.info(f'📍 Step {step}')

				if step_span_context is not None:
					result, summary = await self.step(step, result, step_span_context)
				else:
					with use_span(span):
						result, summary = await self.step(step, result)
				step += 1
				is_done = result.is_done

				screenshot = None
				if return_screenshots:
					state = self.browser.get_state()
					screenshot = state.screenshot

				if timeout is not None and time.time() - start_time > timeout:
					if span is not None:
						ctx = Laminar.serialize_span_context(span)
					else:
						# if span is None, it implies that we're using the step_span_context
						ctx = step_span_context.model_dump_json()

					yield TimeoutChunk(
							content=TimeoutChunkContent(
										action_result=result, 
										summary=summary, 
										step=step, 
										agent_state=self.get_state(), 
										step_parent_span_context=ctx, 
										trace_id=trace_id,
										screenshot=screenshot
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
						agent_state=self.get_state(),
						result=result,
						storage_state=storage_state,
						step_count=step,
						trace_id=trace_id,
					)

					yield FinalOutputChunk(content=final_output)

					break

			if not is_done:
				logger.info('❌ Maximum number of steps reached')
				yield StepChunkError(content=f'Maximum number of steps reached: {max_steps}')
			
		except Exception as e:
			logger.info(f'❌ Error in run: {e}')
			if span is not None:
				span.record_exception(e)
			
			yield StepChunkError(content=f'Error in run stream: {e}')
		finally:
			# Clean up resources
			try:
			
				if close_context:
					# Update to close the browser directly
					await self.browser.close()

			finally:
				if span is not None:
					span.end()
				
				logger.info('Stream complete, span closed')

	def get_state(self) -> AgentState:

		self.state.messages = self.message_manager.get_messages()

		return self.state
