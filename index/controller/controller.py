import inspect
import json
import logging
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, get_type_hints

from docstring_parser import parse
from lmnr import Laminar

from index.agent.models import ActionModel, ActionResult
from index.browser.browser import Browser
from index.controller.default_actions import register_default_actions
from index.llm.llm import ToolDefinition, ToolParameter

logger = logging.getLogger(__name__)

@dataclass
class Action:
    """Represents a registered action"""
    name: str
    description: str
    function: Callable
    browser_context: bool = False


class Controller:
    """Controller for browser actions with integrated registry functionality"""
    
    def __init__(self):
        self._actions: Dict[str, Action] = {}
        # Register default actions
        register_default_actions(self)

    def action(self, description: str = None):
        """
        Decorator for registering actions
        
        Args:
            description: Optional description of what the action does.
                        If not provided, uses the function's docstring.
        """
        def decorator(func: Callable) -> Callable:

            # Use provided description or function docstring
            action_description = description
            if action_description is None:
                action_description = inspect.getdoc(func) or "No description provided"
            
            # Clean up docstring (remove indentation)
            action_description = inspect.cleandoc(action_description)

            browser_context = False
            if 'browser' in inspect.signature(func).parameters:
                browser_context = True

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await func(*args, **kwargs)

            # Register the action
            self._actions[func.__name__] = Action(
                name=func.__name__,
                description=action_description,
                function=async_wrapper,
                browser_context=browser_context,
            )
            return func

        return decorator

    async def execute_action(
        self,
        action: ActionModel,
        browser: Browser,
    ) -> ActionResult:
        """Execute an action from an ActionModel"""

        action_name = action.name
        params = action.params

        if params is not None:
            with Laminar.start_as_current_span(
                name=action_name,
                input={
                    'action': action_name,
                    'params': params,
                },
                span_type='TOOL',
            ) as span:
                
                logger.info(f'Executing action: {action_name} with params: {params}')
                action = self._actions.get(action_name)

                if action is None:
                    raise ValueError(f'Action {action_name} not found')
                
                try:

                    kwargs = params.copy() if params else {}

                    # Filter out metadata fields that shouldn't be passed to action functions
                    kwargs.pop('summary', None)
                    kwargs.pop('thought', None)

                    # Add browser to kwargs if it's provided
                    if action.browser_context and browser is not None:
                        kwargs['browser'] = browser

                    result = await action.function(**kwargs)

                    Laminar.set_span_output(result)
                    return result
                except Exception as e:
                    logger.error(f'Error executing action {action_name}: {str(e)}')
                    span.record_exception(e)
            
                    return ActionResult(error=str(e))

        else:
            raise ValueError('Params are not provided for action: {action_name}')

    def get_action_descriptions(self) -> str:
        """Return a dictionary of all registered actions and their metadata (deprecated)"""
        
        action_info = []
        
        for name, action in self._actions.items():
            sig = inspect.signature(action.function)
            type_hints = get_type_hints(action.function)
            
            # Extract parameter descriptions using docstring_parser
            param_descriptions = {}
            docstring = inspect.getdoc(action.function)
            if docstring:
                parsed_docstring = parse(docstring)
                for param in parsed_docstring.params:
                    param_descriptions[param.arg_name] = param.description
            
            # Build parameter info
            params = {}
            for param_name in sig.parameters.keys():
                if param_name == 'browser':  # Skip browser parameter in descriptions
                    continue
                    
                param_type = type_hints.get(param_name, Any).__name__
                
                params[param_name] = {
                    'type': param_type,
                    'description': param_descriptions.get(param_name, '')
                }
            
            # Use short description from docstring when available
            description = action.description
            if docstring:
                parsed_docstring = parse(docstring)
                if parsed_docstring.short_description:
                    description = parsed_docstring.short_description
            
            action_info.append(json.dumps({
                'name': name,
                'description': description,
                'parameters': params
            }, indent=2))
        
        return '\n\n'.join(action_info)

    def get_tool_definitions(self, include_thought: bool = True) -> List[ToolDefinition]:
        """Return unified tool definitions for all registered actions
        
        Args:
            include_thought: Whether to include a 'thought' field (False for thinking models)
        """
        tool_definitions = []
        
        for name, action in self._actions.items():
            sig = inspect.signature(action.function)
            type_hints = get_type_hints(action.function)
            
            # Extract parameter descriptions using docstring_parser
            param_descriptions = {}
            docstring = inspect.getdoc(action.function)
            if docstring:
                parsed_docstring = parse(docstring)
                for param in parsed_docstring.params:
                    param_descriptions[param.arg_name] = param.description
            
            # Build parameter definitions
            parameters = {}
            for param_name, param in sig.parameters.items():
                if param_name == 'browser':  # Skip browser parameter
                    continue
                
                # Convert Python type to JSON Schema type
                param_type_hint = type_hints.get(param_name, Any)
                json_type = self._python_type_to_json_type(param_type_hint)
                
                # Check if parameter is required (no default value)
                required = param.default == inspect.Parameter.empty
                
                parameters[param_name] = ToolParameter(
                    type=json_type,
                    description=param_descriptions.get(param_name, f"Parameter {param_name}"),
                    required=required
                )
            
            # Always add summary field
            parameters["summary"] = ToolParameter(
                type="string",
                description="Brief summary of what you are doing to display to the user",
                required=True
            )
            
            # Conditionally add thought field for non-thinking models
            if include_thought:
                parameters["thought"] = ToolParameter(
                    type="string", 
                    description="Your reasoning process and key points for this action",
                    required=True
                )
            
            # Use short description from docstring when available
            description = action.description
            if docstring:
                parsed_docstring = parse(docstring)
                if parsed_docstring.short_description:
                    description = parsed_docstring.short_description
            
            tool_definitions.append(ToolDefinition(
                name=name,
                description=description,
                parameters=parameters
            ))
        
        return tool_definitions

    def _python_type_to_json_type(self, python_type: type) -> str:
        """Convert Python type to JSON Schema type"""
        if python_type == str:
            return "string"
        elif python_type == int:
            return "integer"
        elif python_type == float:
            return "number"
        elif python_type == bool:
            return "boolean"
        elif python_type == list:
            return "array"
        elif python_type == dict:
            return "object"
        else:
            # Default to string for unknown types
            return "string"