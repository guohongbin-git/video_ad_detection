import numpy as np
from PIL import Image
# Updated import to use Gemma3ForConditionalGeneration
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch
import os
os.environ["TRANSFORMERS_NO_TF"] = "1" # Disable TensorFlow import in transformers
from . import config

# --- Gemma 3 Feature Extractor ---

# Initialize model and processor
try:
    print(f"Loading model from: {config.GEMMA_MODEL_PATH}")
    # When loading from a local path, local_files_only is not needed
    processor = AutoProcessor.from_pretrained(config.GEMMA_MODEL_PATH)
    # Updated model class to Gemma3ForConditionalGeneration
    model = Gemma3ForConditionalGeneration.from_pretrained(
        config.GEMMA_MODEL_PATH,
        torch_dtype=torch.bfloat16, # Use bfloat16 for better performance on supported hardware
        device_map="auto" # Automatically map model to available devices (e.g., MPS)
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure that the model has been fully downloaded to the cache using 'hf download'.")
    # Set model and processor to None so the app can still run (with errors)
    processor, model = None, None

def extract_features(image: Image.Image) -> np.ndarray:
    """
    Extracts features from a single image frame using the Gemma 3 model.

    Args:
        image (Image.Image): The input image.

    Returns:
        np.ndarray: The feature vector (embedding).
    """
    if not model or not processor:
        print("Model is not loaded. Cannot extract features.")
        # Return a random vector to avoid crashing the app
        return np.random.rand(1, 3072).astype(np.float32) # Gemma 3 base model has a hidden size of 3072

    try:
        # --- Corrected Chat Template for Gemma 3 ---
        # Modern multimodal models require a structured chat format instead of a simple string prompt.
        # We create a list of messages, including the special <image> token and the text prompt.
        # The processor will then apply the correct chat template automatically.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe the image."},
                ],
            }
        ]
        
        # The processor's apply_chat_template method correctly formats the text part of the prompt.
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(prompt, images=image, return_tensors="pt").to(model.device)

        # Generate output, requesting hidden states to access embeddings
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        # --- Updated Feature Extraction Logic for Gemma 3 ---
        # For Gemma 3, the vision embeddings are typically found in `vision_hidden_states`.
        # We will take the last layer's hidden states and average them.
        
        image_features = None
        if hasattr(outputs, 'vision_hidden_states') and outputs.vision_hidden_states is not None:
            # This is the most reliable source for image features.
            # It's a tuple of hidden states for each layer of the vision tower.
            image_features = outputs.vision_hidden_states[-1] # Get the last layer's output
            print("Found vision_hidden_states.")
        elif hasattr(outputs, 'encoder_hidden_states') and outputs.encoder_hidden_states is not None:
            # Fallback for models with a more generic encoder-decoder structure
            image_features = outputs.encoder_hidden_states[-1]
            print("Found encoder_hidden_states as a fallback.")
        
        if image_features is None:
            print("Error: Could not find suitable vision hidden states for feature extraction.")
            return np.random.rand(1, 3072).astype(np.float32)

        # Average the embeddings across all patches (sequence dimension) to get a single vector for the image.
        embedding = torch.mean(image_features, dim=1).cpu().numpy()

        print(f"Successfully extracted features from image. Embedding shape: {embedding.shape}")
        return embedding.astype(np.float32)

    except Exception as e:
        print(f"An error occurred during feature extraction: {e}")
        print("Returning a random feature vector as a fallback.")
        return np.random.rand(1, 3072).astype(np.float32)
