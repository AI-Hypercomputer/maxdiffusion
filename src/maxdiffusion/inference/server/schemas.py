"""Schemas for Diffusion Serving."""
from pydantic import BaseModel, Field
from typing import List, Optional, Any
from dataclasses import dataclass, field
from PIL import Image
import uuid

class InferenceRequest(BaseModel):
    """Request object for diffusion inference."""
    prompt: str
    negative_prompt: Optional[str] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None
    num_frames: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None
    # Add other params as needed (e.g. image_url for I2V)

class InferenceResponse(BaseModel):
    """Response object containing generated images and metadata."""
    images: List[Image.Image]
    
    class Config:
        arbitrary_types_allowed = True

@dataclass
class DiffusionRequestState:
    """Internal state representation of a request."""
    request_id: str
    request: InferenceRequest
    is_warmup: bool = False
    output_images: List[Any] = field(default_factory=list)
    error: Optional[str] = None
    
    @staticmethod
    def from_request(request: InferenceRequest, request_id: Optional[str] = None) -> 'DiffusionRequestState':
        if request_id is None:
            request_id = str(uuid.uuid4())
        return DiffusionRequestState(request_id=request_id, request=request)
