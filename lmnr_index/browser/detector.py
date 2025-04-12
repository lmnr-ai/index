"""
Computer vision detector module that uses aioboto3 for AWS SageMaker integration.
"""

import json
import logging
from dataclasses import dataclass
from typing import List

from aioboto3.session import Session
from lmnr import observe
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from lmnr_index.browser.models import InteractiveElement

logger = logging.getLogger(__name__)

@dataclass
class CVDetection:
    """Computer vision detection result"""
    box: List[float]  # [x1, y1, x2, y2]
    class_name: str
    confidence: float


class Detector:
    """
    AWS SageMaker-based detector for computer vision element detection,
    using aioboto3 for async API calls.
    """
    
    def __init__(self, cv_endpoint_name: str, sheets_endpoint_name: str, region: str = "us-east-1"):
        """
        Initialize the detector with a SageMaker endpoint.
        
        Args:
            endpoint_name: Name of the SageMaker endpoint to use
            region: AWS region for the endpoint
        """
        self.cv_endpoint_name = cv_endpoint_name
        self.sheets_endpoint_name = sheets_endpoint_name
        self.region = region
        self.session = Session(region_name=self.region)
    
    @observe(name="detector.detect_from_image", ignore_input=True)
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=2),
        retry=retry_if_exception_type((Exception)),
        reraise=True,
    )
    async def detect_from_image(self, image_b64: str, detect_sheets: bool = False) -> List[InteractiveElement]:
        """
        Send a base64 encoded image to SageMaker for detection and return
        parsed InteractiveElement objects.
        
        Args:
            image_b64: Base64 encoded image
            
        Returns:
            List of InteractiveElement objects created from CV detections
        """
        if detect_sheets:
            return await self.call_sheets_endpoint(image_b64)
        else:
            return await self.call_cv_endpoint(image_b64)

    @observe(name="detector.call_cv_endpoint", ignore_input=True)
    async def call_cv_endpoint(self, image_b64: str) -> List[InteractiveElement]:
        """
        Send a base64 encoded image to SageMaker for detection and return
        parsed InteractiveElement objects.
        
        Args:
            image_b64: Base64 encoded image
            
        Returns:
            List of InteractiveElement objects created from CV detections
        """
        
        try:
            # Convert to bytes for SageMaker
            async with self.session.client("sagemaker-runtime") as client:
                response = await client.invoke_endpoint(
                    EndpointName=self.cv_endpoint_name,
                    ContentType="application/json",
                    Body=json.dumps({
                        "image": image_b64,
                        "conf": 0.5
                    })
                )

                # Parse response
                async with response["Body"] as stream:
                    response_body = await stream.read()

            detection_result = json.loads(response_body.decode("utf-8"))
            logger.info(f"Received detection results with {len(detection_result.get('detections', []))} detections")
            
            # Parse detections into InteractiveElement objects
            elements = []
            predictions = detection_result.get('detections', [])
            
            for i, pred in enumerate(predictions):
                # Extract bounding box
                box = pred.get('box', [0, 0, 0, 0])
                
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                
                # Create unique ID for the CV detection
                index_id = f"cv-{i}"
                
                # Create element
                element = InteractiveElement(
                    index=i,
                    browser_agent_id=index_id,
                    tag_name="element",
                    text="",
                    attributes={},
                    weight=1,
                    viewport={
                        "x": round(x1),
                        "y": round(y1),
                        "width": round(width),
                        "height": round(height)
                    },
                    page={
                        "x": round(x1),
                        "y": round(y1),
                        "width": round(width),
                        "height": round(height)
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
                    z_index=0
                )
                
                elements.append(element)
            
            logger.info(f"Created {len(elements)} interactive elements from CV detections")
            return elements 
        except Exception as e:
            logger.error(f"Error detecting from image in cv endpoint: {e}")
            return []
        
    
    async def call_sheets_endpoint(self, image_b64: str) -> List[InteractiveElement]:
        """
        Call the sheets endpoint and return the detections
        """

        logger.info("Calling sheets endpoint with image_b64")
        
        try:
            # Convert to bytes for SageMaker
            async with self.session.client("sagemaker-runtime") as client:
                response = await client.invoke_endpoint(
                    EndpointName=self.sheets_endpoint_name,
                    ContentType="application/json",
                    Body=json.dumps({
                        "image": image_b64,
                    })
                )

                # Parse response
                async with response["Body"] as stream:
                    response_body = await stream.read()

            detection_result = json.loads(response_body.decode("utf-8"))
            logger.info(f"Received detection result from SageMaker with {len(detection_result.get('detections', []))} detections")
            
            # Parse detections into InteractiveElement objects
            elements = []
            predictions = detection_result.get('detections', [])
            
            for i, pred in enumerate(predictions):
                # Extract bounding box
                box = pred.get('box', [0, 0, 0, 0])
                
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                
                # Create element
                element = InteractiveElement(
                    index=i,
                    browser_agent_id=pred.get('class_name'),
                    tag_name=pred.get('class_name'),
                    text="",
                    attributes={},
                    weight=1,
                    viewport={
                        "x": round(x1),
                        "y": round(y1),
                        "width": round(width),
                        "height": round(height)
                    },
                    page={
                        "x": round(x1),
                        "y": round(y1),
                        "width": round(width),
                        "height": round(height)
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
                    z_index=0
                )
                
                elements.append(element)
            
            logger.info(f"Created {len(elements)} interactive elements from sheets detections")
            return elements 
        except Exception as e:
            logger.error(f"Error detecting from image in sheets endpoint: {e}")
            return []