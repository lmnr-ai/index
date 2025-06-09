from __future__ import annotations

import logging
import re
import time
import uuid
from typing import AsyncGenerator, Optional

from dotenv import load_dotenv
from lmnr import Laminar, LaminarSpanContext, observe, use_span
from pydantic import BaseModel

from index.agent.message_manager import MessageManager
from index.agent.models import (
	ActionModel,
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
		
		# Detect if this is a thinking model and get appropriate tool definitions
		include_thought = not self._is_thinking_model()
		self.tool_definitions = self.controller.get_tool_definitions(include_thought=include_thought)

		self.message_manager = MessageManager(
			action_descriptions=self.controller.get_action_descriptions(),  # Keep for backwards compatibility with message manager
		)

		self.state = AgentState(
			messages=[],
		)

	def _is_thinking_model(self) -> bool:
		"""Detect if the LLM is a thinking model (e.g., Anthropic with thinking enabled)"""
		# Check if it's an Anthropic provider with thinking enabled
		if hasattr(self.llm, 'enable_thinking'):
			return getattr(self.llm, 'enable_thinking', False)
		
		# For providers without thinking capability, always return False
		return False

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
				
				# Add the action result as a tool result message
				self.message_manager.add_action_result(result)

				if result.is_done:
					logger.info(f'Result: {result.content}')
					self.final_output = result.content

				return result, model_output.summary
				
			except Exception as e:
				raise e


	@observe(name='agent.generate_action', ignore_input=True)
	async def _generate_action(self, input_messages: list[Message]) -> AgentLLMOutput:
		"""Get next action from LLM using tool calls instead of JSON parsing"""

		response = await self.llm.call(input_messages, tools=self.tool_definitions)
		
		if not response.tool_calls:
			# If no tool calls, try to extract any reasoning from the text content
			thought = self._extract_thought_from_content(response.content)
			raise ValueError(f"LLM did not make any tool calls. Response content: {response.content}")
		
		# Find the action tool call (should be the main action)
		action_tool_call = None
		thought = ""
		summary = ""
		
		for tool_call in response.tool_calls:
			# All controller actions are the main tool calls
			if tool_call.name in [tool.name for tool in self.tool_definitions]:
				action_tool_call = tool_call
				break
		
		if not action_tool_call:
			raise ValueError(f"No valid action tool call found in response. Tool calls: {[tc.name for tc in response.tool_calls]}")
		
		# Extract thought and summary from the response content or parameters
		thought, summary = self._extract_thought_and_summary(response.content, action_tool_call.parameters)
		
		# Create the action model
		action = ActionModel(
			name=action_tool_call.name,
			params=action_tool_call.parameters
		)
		
		# Create AgentLLMOutput
		output = AgentLLMOutput(
			action=action,
			thought=thought,
			summary=summary
		)
		
		if response.thinking:
			output.thinking_block = response.thinking
		
		logger.info(f'💡 Summary: {output.summary}')
		logger.info(f'🛠️ Action: {output.action.model_dump_json(exclude_unset=True)}')
		
		return output

	def _extract_thought_from_content(self, content: str) -> str:
		"""Extract reasoning/thought from LLM content when no tool calls are made"""
		if not content:
			return "No reasoning provided"
		# Take first 200 chars as thought
		return content.strip()[:200] + ("..." if len(content.strip()) > 200 else "")

	def _extract_thought_and_summary(self, content: str, tool_params: dict) -> tuple[str, str]:
		"""Extract thought and summary from tool parameters (with fallbacks)"""
		
		# First priority: get from tool parameters (should be present for all tools now)
		summary = tool_params.get('summary', '')
		thought = tool_params.get('thought', '')
		
		# If summary is missing (shouldn't happen), try to extract from content or generate
		if not summary:
			summary_match = re.search(r'(?:summary|action):\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
			if summary_match:
				summary = summary_match.group(1).strip()
			else:
				# Generate summary from action name and key parameters
				action_name = tool_params.get('name', 'action')
				summary = f"Executing {action_name}"
				
				# Add key parameter info to summary
				key_params = ['url', 'text', 'query', 'element_index', 'xpath']
				param_info = []
				for key in key_params:
					if key in tool_params and tool_params[key]:
						value = str(tool_params[key])
						if len(value) > 50:
							value = value[:47] + "..."
						param_info.append(f"{key}={value}")
				
				if param_info:
					summary += f" ({', '.join(param_info[:2])})"  # Limit to 2 params to keep summary short
		
		# If thought is missing (for non-thinking models), try to extract from content or generate
		if not thought:
			thought_match = re.search(r'(?:thought|thinking|reasoning):\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
			if thought_match:
				thought = thought_match.group(1).strip()
			elif content:
				# Use content as thought, truncated
				thought = content.strip()[:200] + ("..." if len(content.strip()) > 200 else "")
			else:
				# For thinking models, we don't need thought since it's in the thinking block
				if self._is_thinking_model():
					thought = ""
				else:
					thought = "Processing next step"
		
		return thought or "", summary or "Continuing task"

	async def _setup_messages(self, 
							prompt: str, 
							agent_state: str | None = None, 
							start_url: str | None = None,
							output_model: BaseModel | str | None = None
							):
		"""Set up messages based on state dict or initialize with system message"""
		if agent_state:
			# assuming that the structure of the state.messages is correct
			state = AgentState.model_validate_json(agent_state)
			self.message_manager.set_messages(state.messages)
			# Update browser_context to browser
			browser_state = await self.browser.update_state()
			self.message_manager.add_current_state_message(browser_state, user_follow_up_message=prompt)
		else:
			self.message_manager.add_system_message_and_user_prompt(prompt, output_model)

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
			   	output_model: BaseModel | str | None = None
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
			output_model: Optional, the output model to use for the task
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
			
			await self._setup_messages(prompt, agent_state, start_url, output_model)

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
						output_model: BaseModel | str | None = None
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
			output_model: Optional, the output model to use for the task
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
			await self._setup_messages(prompt, agent_state, start_url, output_model)

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
