import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel


class HFGemma3TextEncoder:
  """
  A lightweight wrapper around Hugging Face's Gemma 3 model for extracting hidden states.
  This module forces execution on CPU to avoid OOM or XLA collisions when used alongside
  JAX/MaxDiffusion on TPUs.
  """

  def __init__(self, model_id: str = "google/gemma-3-12b-it", max_length: int = 8192):
    self.model_id = model_id
    self.max_length = max_length
    # Initialize the tokenizer
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    # Load the model directly to CPU in bfloat16 to save memory
    print(f"Loading {model_id} onto CPU. This may take a few moments...")
    self.model = AutoModel.from_pretrained(
        self.model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Force CPU to avoid TPU memory contention with MaxDiffusion
    )
    self.model.eval()  # Set to evaluation mode

  def encode(self, text: str | list[str]) -> np.ndarray:
    """
    Tokenizes the input text, passes it through the HF Gemma 3 model,
    and extracts ALL hidden states.

    Args:
        text: A single string or a list of strings to encode.

    Returns:
        A numpy array representing the flattened, stacked hidden states
        compatible with GemmaFeaturesExtractorProjLinear.
        Shape: (batch_size, sequence_length, 49 * 3840)
    """
    # 1. Tokenize input text
    inputs = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")

    # Ensure inputs are on the same device as the model (CPU)
    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

    # 2. Forward pass to get hidden states
    # output_hidden_states=True is the key to retrieving all 49 layers
    with torch.no_grad():
      outputs = self.model(**inputs, output_hidden_states=True)

    # 3. Extract and stack hidden states
    # outputs.hidden_states is a tuple of 49 tensors, each shaped (batch, seq_len, 3840)
    all_hidden_states = outputs.hidden_states

    # Stack them along a new dimension (dim=0 or dim=-2)
    # We want to format it so it's easy to flatten.
    # Stacked shape: (49, batch, seq_len, 3840)
    stacked_states = torch.stack(all_hidden_states, dim=0)

    # Transpose to: (batch, seq_len, 49, 3840)
    transposed_states = stacked_states.permute(1, 2, 0, 3)

    # Flatten the last two dimensions to match the Feature Extractor's expectation
    # Shape becomes: (batch, seq_len, 49 * 3840) -> (batch, seq_len, 188160)
    batch_size, seq_len, num_layers, hidden_dim = transposed_states.shape
    flattened_states = transposed_states.reshape(batch_size, seq_len, num_layers * hidden_dim)

    # 4. Convert PyTorch Tensor to NumPy Array
    # JAX/Flax can seamlessly accept and convert numpy arrays to JAX Arrays
    numpy_hidden_states = flattened_states.cpu().float().numpy()

    return numpy_hidden_states
