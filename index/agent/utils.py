import base64
import importlib.resources
import logging
from typing import Any, Dict, Type

from pydantic import BaseModel

from index.browser.utils import scale_b64_image

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

def simplified_model_schema(model_class: Type[BaseModel]) -> Dict[str, Any]:
    """
    Generate a simplified schema that maps field names to their types.
    
    Args:
        model_class: Pydantic BaseModel class
        
    Returns:
        Dictionary mapping field names to their type representations
    """
    schema = {}
    for field_name, field_info in model_class.model_fields.items():
        # Get the field type
        if hasattr(field_info.annotation, "__origin__") and field_info.annotation.__origin__ is list:
            # Handle List[Something]
            inner_type = field_info.annotation.__args__[0]
            if hasattr(inner_type, "mro") and BaseModel in inner_type.mro():
                # Recursive handling for List[SomeModel]
                schema[field_name] = [simplified_model_schema(inner_type)]
            else:
                # Simple List[primitive]
                type_name = getattr(inner_type, "__name__", str(inner_type))
                schema[field_name] = f"List[{type_name}]"
        elif hasattr(field_info.annotation, "mro") and BaseModel in field_info.annotation.mro():
            # Recursive handling for nested models
            schema[field_name] = simplified_model_schema(field_info.annotation)
        else:
            # Simple types
            type_name = getattr(field_info.annotation, "__name__", str(field_info.annotation))
            schema[field_name] = type_name
    
    return schema