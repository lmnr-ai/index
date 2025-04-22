import base64
import importlib.resources
import logging

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