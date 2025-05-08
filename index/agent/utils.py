import base64
import enum
import importlib.resources
import json
import logging
import re
from typing import Any, Dict, Type

from pydantic import BaseModel, ValidationError

from index.agent.models import AgentLLMOutput
from index.browser.utils import scale_b64_image
from index.llm.llm import BaseLLMProvider, Message

logger = logging.getLogger(__name__)

def load_demo_image_as_b64(image_name: str) -> str:
    """
    Load an image from the demo_images directory and return it as a base64 string.
    Works reliably whether the package is used directly or as a library.
    
    Args:
        image_name: Name of the image file (including extension)
        
    Returns:
        Base64 encoded string of the image
    """
    try:
        # Using importlib.resources to reliably find package data
        with importlib.resources.path('index.agent.demo_images', image_name) as img_path:
            with open(img_path, 'rb') as img_file:
                b64 = base64.b64encode(img_file.read()).decode('utf-8')
                return scale_b64_image(b64, 0.75)
    except Exception as e:
        logger.error(f"Error loading demo image {image_name}: {e}")
        raise

def pydantic_to_custom_jtd(model_class: Type[BaseModel]) -> Dict[str, Any]:
    """
    Convert a Pydantic model class to a custom JSON Typedef-like schema
    with proper array and object handling.
    """
    def python_type_to_jtd_type(annotation):
        if annotation is str:
            return {"type": "string"}
        elif annotation is int:
            return {"type": "int32"}
        elif annotation is float:
            return {"type": "float64"}
        elif annotation is bool:
            return {"type": "boolean"}
        elif isinstance(annotation, type) and issubclass(annotation, enum.Enum):
            values = [e.value for e in annotation]
            return {"type": "string", "enum": values}
        else:
            return {"type": "string"}  # fallback

    def process_model(model):
        model_schema = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        }
        
        for name, field in model.model_fields.items():
            annotation = field.annotation
            origin = getattr(annotation, "__origin__", None)
            
            if origin is list:
                inner = annotation.__args__[0]
                if isinstance(inner, type) and issubclass(inner, enum.Enum):
                    item_schema = {"type": "string", "enum": [e.value for e in inner]}
                elif hasattr(inner, "mro") and BaseModel in inner.mro():
                    item_schema = process_model(inner)
                else:
                    item_schema = python_type_to_jtd_type(inner)
                
                model_schema["properties"][name] = {
                    "type": "array",
                    "items": item_schema
                }
            elif isinstance(annotation, type) and issubclass(annotation, enum.Enum):
                model_schema["properties"][name] = {
                    "type": "string", 
                    "enum": [e.value for e in annotation]
                }
            elif hasattr(annotation, "mro") and BaseModel in annotation.mro():
                model_schema["properties"][name] = process_model(annotation)
            else:
                model_schema["properties"][name] = python_type_to_jtd_type(annotation)
            
            if field.is_required():
                model_schema["required"].append(name)
                
        return model_schema
    
    return process_model(model_class)


async def generate_proper_json(llm: BaseLLMProvider, json_str: str) -> str:

    prompt = f"""The following JSON string is malformed or has issues. Please correct it while preserving the original structure and content as much as possible.
Return ONLY the corrected JSON string, without any surrounding text, comments, or markdown. Do not add any explanations.

Problematic JSON string:
{json_str}
"""

    input_messages = [
        Message(role="user", content=prompt)
    ]

    response = await llm.call(input_messages)
    corrected_json_str = response.content.strip()
    if corrected_json_str.startswith("```json"):
        corrected_json_str = corrected_json_str[7:]
    if corrected_json_str.endswith("```"):
        corrected_json_str = corrected_json_str[:-3]
    return corrected_json_str.strip()


async def validate_json(raw_llm_response_content: str, llm: BaseLLMProvider, max_retries: int = 3) -> AgentLLMOutput:
    """
    Extracts, validates, and parses a JSON string from raw LLM output,
    attempting to fix it if necessary using retries with cleaning and LLM-based correction.
    
    Args:
        raw_llm_response_content: The raw string content from the LLM response.
        llm: The LLM provider instance for fixing JSON if needed.
        max_retries: Maximum number of attempts to parse the JSON.
        
    Returns:
        An AgentLLMOutput object.
        
    Raises:
        ValueError: If the JSON string cannot be parsed or validated after all retries.
    """
    # 1. Regex extraction from raw_llm_response_content
    pattern = r"<output(?:[^>]*)>(.*?)</output(?:[^>]*)>"
    match = re.search(pattern, raw_llm_response_content, re.DOTALL)
    
    current_json_str = ""
    if not match:
        # if we couldn't find the <output> tags, it most likely means the <output*> tag is not present in the response
        # remove closing and opening tags just in case
        closing_tag_pattern = r"</output(?:[^>]*)>"
        json_str_no_closing = re.sub(closing_tag_pattern, "", raw_llm_response_content).strip()
        open_tag_pattern = r"<output(?:[^>]*)>"
        json_str_no_tags = re.sub(open_tag_pattern, "", json_str_no_closing).strip()
        # Also remove potential markdown code blocks if not already handled by regex
        current_json_str = json_str_no_tags.replace("```json", "").replace("```", "").strip()
    else:
        current_json_str = match.group(1).strip()

    last_exception = None

    for attempt in range(max_retries):
        logger.debug(f"JSON parsing attempt {attempt + 1}/{max_retries}")
        
        # Stage 1: Try to parse the current_json_str as is
        try:
            # Remove potential markdown that might have been added by LLM fix
            temp_json_str = current_json_str
            if temp_json_str.startswith("```json"):
                temp_json_str = temp_json_str[7:]
            if temp_json_str.endswith("```"):
                temp_json_str = temp_json_str[:-3]
            temp_json_str = temp_json_str.strip()

            logger.debug(f"Attempting to parse JSON on attempt {attempt + 1}. Raw JSON: '{temp_json_str}'")
            output = AgentLLMOutput.model_validate_json(temp_json_str)
            logger.debug(f"Successfully parsed JSON on attempt {attempt + 1}.")
            return output
        except (json.JSONDecodeError, ValidationError) as e1:
            logger.warning(f"Direct JSON parsing failed on attempt {attempt + 1}: {e1}")
            last_exception = e1

            # Stage 2: Try to parse after cleaning common issues
            try:
                json_str_cleaned = current_json_str # Start with the current_json_str for cleaning
                # Removed explicit replacement of \n, \r, \t - rely on JSON parser
                # json_str_cleaned = json_str_cleaned.replace('\\\\n', '\n').replace('\\\\r', '\r').replace('\\\\t', '\t')
                # Keep control character removal
                json_str_cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', json_str_cleaned)
                
                if json_str_cleaned.startswith("```json"):
                    json_str_cleaned = json_str_cleaned[7:]
                if json_str_cleaned.endswith("```"):
                    json_str_cleaned = json_str_cleaned[:-3]
                json_str_cleaned = json_str_cleaned.strip()

                logger.debug(f"Attempting to parse cleaned JSON on attempt {attempt + 1}. Cleaned JSON: '{json_str_cleaned[:250]}...'")
                output = AgentLLMOutput.model_validate_json(json_str_cleaned)
                logger.debug(f"Successfully parsed JSON on attempt {attempt + 1} (after cleaning).")
                return output
            except (json.JSONDecodeError, ValidationError) as e2:
                logger.warning(f"Cleaned JSON parsing failed on attempt {attempt + 1}: {e2}")
                last_exception = e2 

                if attempt < max_retries - 1:
                    logger.debug(f"Attempt {attempt + 1} failed. Attempting to fix JSON with LLM.")
                    try:
                        # Pass the original problematic string (before this attempt's cleaning) to LLM
                        current_json_str = await generate_proper_json(llm, current_json_str) 
                        logger.debug(f"LLM proposed a new JSON string: '{current_json_str}'")
                    except Exception as llm_fix_exception:
                        logger.error(f"LLM call to fix JSON failed during attempt {attempt + 1}: {llm_fix_exception}")
                        # If LLM fix fails, loop continues with the previous current_json_str,
                        # and will eventually fail if parsing doesn't succeed.
                        pass 
                else:
                    logger.error(f"All {max_retries} attempts to parse JSON failed. Final attempt was with: '{current_json_str[:250]}...'")
                    break 
    
    raise ValueError(
        f"Could not parse or validate response after {max_retries} attempts. "
        f"Last error: {str(last_exception)}\\n"
        f"Final problematic JSON string after all attempts: '{current_json_str[:500]}...'"
    ) from last_exception

