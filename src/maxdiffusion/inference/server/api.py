import asyncio
import base64
import io
import zmq
import zmq.asyncio
import uuid
import pickle
import uvicorn
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from typing import List, Optional
from pydantic import BaseModel

from maxdiffusion.inference.server.schemas import InferenceRequest, DiffusionRequestState

# Config
SCHEDULER_URL = "tcp://localhost:5555"

class APIResponse(BaseModel):
    images: List[str] # Base64 encoded images
    latency: float

class GlobalState:
    ctx: Optional[zmq.asyncio.Context] = None
    socket: Optional[zmq.asyncio.Socket] = None

g_state = GlobalState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    g_state.ctx = zmq.asyncio.Context()
    g_state.socket = g_state.ctx.socket(zmq.DEALER)
    identity = str(uuid.uuid4()).encode('utf-8')
    g_state.socket.setsockopt(zmq.IDENTITY, identity)
    g_state.socket.connect(SCHEDULER_URL)
    print(f"Connected to Scheduler at {SCHEDULER_URL} with identity {identity}")
    
    yield
    
    # Shutdown
    if g_state.socket:
        g_state.socket.close()
    if g_state.ctx:
        g_state.ctx.term()

app = FastAPI(lifespan=lifespan)

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

lock = asyncio.Lock()

@app.post("/generate", response_model=APIResponse)
async def generate_endpoint(request: InferenceRequest):
    req_state = DiffusionRequestState.from_request(request)
    data = pickle.dumps(req_state)
    
    async with lock:
        await g_state.socket.send_multipart([b"", data])
        reply_parts = await g_state.socket.recv_multipart()
        
    if len(reply_parts) == 2:
        _, reply_data = reply_parts
    elif len(reply_parts) == 1:
        reply_data = reply_parts[0]
    else:
        raise HTTPException(status_code=500, detail=f"Invalid ZMQ reply format")

    reply = pickle.loads(reply_data)
    
    if "error" in reply:
        raise HTTPException(status_code=500, detail=reply["error"])
        
    b64_images = [image_to_base64(img) for img in reply["images"]]
    
    return APIResponse(images=b64_images, latency=reply["latency"])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
