"""
Computer vision detector module.
"""

from abc import ABC, abstractmethod
import base64
from importlib import resources
from io import BytesIO
import logging
from tkinter import Image
from typing import List

import requests

from index.browser.models import InteractiveElement
from playwright.async_api import Page

logger = logging.getLogger(__name__)

class Detector(ABC):
    """Abstract interface for object detection in browser screenshots."""

    @abstractmethod
    async def detect_from_image(self, image_b64: str, scale_factor: float, detect_sheets: bool = False) -> List[InteractiveElement]:
        """
        Detect interactive elements from a base64 encoded image.
        
        Args:
            image_b64: Base64 encoded image screenshot.
            scale_factor: Scale factor to scale the coordinates of screenshot to browser viewport coordinates.
            detect_sheets: Flag to indicate if specialized sheet detection should be used.
            
        Returns:
            List of detected InteractiveElement objects.
        """
        pass


class OmniparserDetector(Detector):
    CHECK_COORDINATE_INTERACTIVE = resources.read_text('index.browser', 'checkCoordinateInteractive.js')

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self.test_connect()

    def test_connect(self):
        try:
            resp = requests.get(f"{self.endpoint}/probe/")
            if resp.status_code == 200:
                logging.info("Successfully connected.")
                return True
        except Exception as e:
            logging.error(f"Failed to connect: {e}")
        return False

    async def detect_from_image(self, image_b64: str, scale_factor: float, detect_sheets: bool = False) -> List[InteractiveElement]:
        elements = []
        try:
            resp = requests.post(f"{self.endpoint}/parse/", json={"base64_image": image_b64})

            if resp.status_code == 200:
                som_image_b64 = resp.json()["som_image_base64"]
                content_list = resp.json()["parsed_content_list"]

                data = base64.b64decode(som_image_b64)
                with open(f"ocr.png", "wb") as fp:
                    fp.write(data)
                
                image_data = base64.b64decode(image_b64)
                image = Image.open(BytesIO(image_data))
                w, h = image.size
                
                for i, element in enumerate(content_list):
                    # if element["type"] != 'icon' and not element['interactivity']:
                    #     continue

                    index_id = f"cv-{i}"

                    x1, y1, x2, y2 = element["bbox"]
                    x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h
                    width = x2 - x1
                    height = y2 - y1

                    elem = InteractiveElement(
                        index=i,
                        browser_agent_id=index_id,
                        tag_name="element",
                        text="",
                        attributes={},
                        weight=1, # ocr cannot recognize the element tag, so set weight as 1
                        viewport={
                            "x": round(x1),
                            "y": round(y1),
                            "width": round(width),
                            "height": round(height),
                        },
                        page={
                            "x": round(x1),
                            "y": round(y1),
                            "width": round(width),
                            "height": round(height),
                        },
                        center={
                            "x": round(x1 + width/2),
                            "y": round(y1 + height/2)
                        },
                        input_type=None,
                        rect={
                            "left": round(x1),
                            "top": round(y1),
                            "right": round(x2),
                            "bottom": round(y2),
                            "width": round(width),
                            "height": round(height)
                        },
                        z_index=0,
                    )
                    elements.append(elem)
            else:
                logging.warn("Cannot Found elements")
        except Exception as e:
            logging.error(f"{e}")

        return elements

    async def filter_cv_elements_by_interactive(self, page: Page, cv_elements) -> List[InteractiveElement]:
        """Filter non-interactive elements through coordinate reverse mapping"""
        elements = []
        for elem in cv_elements:
            x, y = elem.center.x, elem.center.y
            try:
                result = await page.evaluate(self.CHECK_COORDINATE_INTERACTIVE, [x, y])
                if result:
                    new_elem = InteractiveElement(**result)
                    new_elem.index = elem.index
                    elements.append(new_elem)
            except Exception as e:
                logging.error(e)
        return elements
