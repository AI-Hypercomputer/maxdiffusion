import torch

latents = torch.zeros((1, 8, 126, 16))
latents_mean = torch.zeros((128,))

try:
    result = latents + latents_mean
    print("Success! Shape:", result.shape)
except Exception as e:
    print("Failed!", type(e), e)

# How about diffusers' exact code?
class MockPipeline:
    @staticmethod
    def _denormalize_audio_latents(latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor):
        return (latents * latents_std) + latents_mean
        
try:
    MockPipeline._denormalize_audio_latents(latents, torch.ones((128,)), torch.zeros((128,)))
    print("Diffusers logic succeeded?")
except Exception as e:
    print("Diffusers logic failed!", type(e), e)
