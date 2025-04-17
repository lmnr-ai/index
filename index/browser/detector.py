"""
Computer vision detector module.
"""

from abc import ABC, abstractmethod
from typing import List

from index.browser.models import InteractiveElement


class Detector(ABC):
    """Abstract interface for object detection in browser screenshots."""

    @abstractmethod
    async def detect_from_image(self, image_b64: str, detect_sheets: bool = False) -> List[InteractiveElement]:
        """
        Detect interactive elements from a base64 encoded image.
        
        Args:
            image_b64: Base64 encoded image screenshot.
            detect_sheets: Flag to indicate if specialized sheet detection should be used.
            
        Returns:
            List of detected InteractiveElement objects.
        """
        pass