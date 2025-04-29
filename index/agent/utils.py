import base64
import enum
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