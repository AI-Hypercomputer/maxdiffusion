"""TPU Worker for MaxDiffusion Serving."""

import jax
from typing import List, Union
from maxdiffusion import max_logging, pyconfig
from maxdiffusion.inference.loader import InferenceLoader
from maxdiffusion.inference.runner import DiffusionRunner
from maxdiffusion.inference.server.schemas import InferenceRequest, InferenceResponse

class DiffusionTPUWorker:
    """Orchestrates model loading and request processing on TPU."""

    def __init__(self, config_args: List[str]):
        """
        Args:
            config_args: List of arguments to initialize pyconfig (e.g. ['script', 'config.yaml', '...'])
        """
        max_logging.log(f"Initializing DiffusionTPUWorker with args: {config_args}")
        pyconfig.initialize(config_args)
        self.config = pyconfig.config
        
        max_logging.log("Loading model...")
        self.loaded_model = InferenceLoader.load(self.config)
        
        max_logging.log("Initializing Runner...")
        self.runner = DiffusionRunner(self.loaded_model, self.config)
        
        # Warmup could go here

    def process_request(self, input_data: Union[InferenceRequest, List[InferenceRequest]]) -> List[InferenceResponse]:
        """
        Processes a single request or a list of requests.
        """
        requests = input_data if isinstance(input_data, list) else [input_data]
        
        # Simple Batching Logic matching the config batch size
        # Note: Runner handles global_batch_size internally based on devices * per_device.
        # We should try to fill that.
        
        # Get target batch size from config
        target_batch_size = self.config.per_device_batch_size * jax.device_count()
        
        results = []
        
        for i in range(0, len(requests), target_batch_size):
            batch = requests[i : i + target_batch_size]
            real_len = len(batch)
            
            # Pad if needed (basic padding)
            if real_len < target_batch_size:
                batch.extend([batch[-1]] * (target_batch_size - real_len))
                
            try:
                # Assuming homogenous batch for now (taking params from first request)
                req0 = batch[0]
                
                # Run Inference
                # Runner returns List[PIL.Image]
                images = self.runner.run(
                    prompt=[r.prompt for r in batch],
                    negative_prompt=[r.negative_prompt for r in batch] if req0.negative_prompt else None,
                    height=req0.height,
                    width=req0.width,
                    num_inference_steps=req0.num_inference_steps,
                    guidance_scale=req0.guidance_scale,
                    num_frames=req0.num_frames
                )
                
                # Split results
                # Each request might get multiple images/frames? 
                # Wan runner returns list of PIL images (frames flattened?).
                # If Video, it's (B, F, H, W, C). Runner logic needs verification.
                # Flux runner returns list of PIL images (B images).
                
                # Assuming 1 output per input prompt for T2I.
                # For Video, Runner returns list of frames?
                # Let's assume images is flat list of results matching batch order.
                
                # If num_frames > 1 (Wan), runner might return B objects where each object is frames?
                # Wan runner _postprocess uses imageio to save video, but returns flattened?
                # "images_np.reshape(b * f, h, w, c)" -> "pil_images = [...]"
                # So for Wan it returns (B*F) images. We need to regroup.
                
                if "wan" in self.config.model_name.lower():
                    # Regroup frames
                    num_frames = req0.num_frames or self.config.num_frames
                    batch_responses = []
                    for j in range(target_batch_size):
                        # Slice frames for this request
                        start_idx = j * num_frames
                        end_idx = (j + 1) * num_frames
                        req_frames = images[start_idx:end_idx]
                        batch_responses.append(InferenceResponse(images=req_frames))
                    results.extend(batch_responses[:real_len])
                else:
                    # Flux/SDXL (1 image per prompt usually)
                    batch_responses = [InferenceResponse(images=[img]) for img in images]
                    results.extend(batch_responses[:real_len])

            except Exception as e:
                max_logging.log(f"Error processing batch: {e}")
                # Propagate error?
                raise e
                
        return results
