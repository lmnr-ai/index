import inspect
import json
import logging
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, get_type_hints

from docstring_parser import parse
from lmnr import Laminar

from index.agent.models import ActionModel, ActionResult
from index.browser.browser import Browser
from index.controller.default_actions import register_default_actions

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
            ):
                
                logger.info(f'Executing action: {action_name} with params: {params}')
                action = self._actions.get(action_name)

                if action is None:
                    raise ValueError(f'Action {action_name} not found')
                
                try:

                    kwargs = params.copy() if params else {}

                    # Add browser to kwargs if it's provided
                    if action.browser_context and browser is not None:
                        kwargs['browser'] = browser

                    result = await action.function(**kwargs)

                    Laminar.set_span_output(result)
                    return result

                except Exception as e:
                    raise RuntimeError(f'Error executing action {action_name}: {str(e)}') from e

        else:
            raise ValueError('Params are not provided for action: {action_name}')

    def get_action_descriptions(self) -> str:
        """Return a dictionary of all registered actions and their metadata"""
        
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