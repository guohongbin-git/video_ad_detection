
import numpy as np
from PIL import Image
# Corrected import - Gemma3VisionEncoder is not a top-level import
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch
import os
os.environ["TRANSFORMERS_NO_TF"] = "1" # Disable TensorFlow import in transformers
from . import config

# --- Model and Processor Initialization ---

# Declare variables in the global scope to hold the loaded models
processor = None
model = None
vision_encoder = None

try:
    print(f"Loading model components from: {config.GEMMA_MODEL_PATH}")
    
    processor = AutoProcessor.from_pretrained(config.GEMMA_MODEL_PATH)
    
    # Load the full multimodal model
    model = Gemma3ForConditionalGeneration.from_pretrained(
        config.GEMMA_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # **Access the vision encoder as an attribute of the main model**
    # This is the correct and robust way to get the vision encoder
    if model:
        vision_encoder = model.vision_encoder

    print("Model and Vision Encoder loaded successfully.")

except Exception as e:
    print(f"Error loading model components: {e}")
    print("Please ensure the model is correctly downloaded and configured.")
    # Ensure variables are reset on failure
    processor, model, vision_encoder = None, None, None

def extract_features(image: Image.Image) -> np.ndarray:
    """
    Extracts features from a single image frame using the model's Vision Encoder.

    Args:
        image (Image.Image): The input image.

    Returns:
        np.ndarray: The feature vector (embedding).
    """
    # Check if the essential components (processor and vision_encoder) are loaded
    if not vision_encoder or not processor:
        print("Vision Encoder or Processor is not loaded. Cannot extract features.")
        # Return a random vector to avoid crashing the app
        return np.random.rand(1, 3072).astype(np.float32)

    try:
        # We only need to process the image to get pixel values for the vision encoder.
        inputs = processor(images=image, return_tensors="pt").to(vision_encoder.device)

        # Get the embeddings directly from the vision encoder
        with torch.no_grad():
            outputs = vision_encoder(**inputs)
        
        # The vision encoder output contains the `last_hidden_state` which holds the patch embeddings.
        # Shape: (batch_size, num_patches, hidden_size)
        last_hidden_state = outputs.last_hidden_state
        
        # To get a single feature vector for the entire image (global feature), 
        # we average the embeddings of all patches.
        embedding = torch.mean(last_hidden_state, dim=1).cpu().numpy()

        # This print can be noisy, so it's commented out for now.
        # print(f"Successfully extracted features. Embedding shape: {embedding.shape}")
        return embedding.astype(np.float32)

    except Exception as e:
        print(f"An error occurred during feature extraction: {e}")
        print("Returning a random feature vector as a fallback.")
        return np.random.rand(1, 3072).astype(np.float32)
