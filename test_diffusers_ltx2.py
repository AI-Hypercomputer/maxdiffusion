import torch
from diffusers import LTXPipeline
from diffusers.utils import export_to_video

pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.bfloat16)
pipe.to("cuda")

prompt = "A girl is walking."
video = pipe(
    prompt=prompt,
    width=768,
    height=512,
    num_frames=21,
    num_inference_steps=20,
).frames[0]

export_to_video(video, "output_diffusers.mp4", fps=24)
print("Finished diffusers generation")
