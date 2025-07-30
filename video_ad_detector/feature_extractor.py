
import os
# Prevent TensorFlow and JAX from being imported by Hugging Face Transformers
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_JAX"] = "1"

import numpy as np
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch
import logging # Import logging module

from . import config

# --- Logging Setup ---
log_file_path = os.path.join(config.BASE_DATA_DIR, "model_loading_error.log")
logging.basicConfig(
    level=logging.DEBUG, # Changed to DEBUG for detailed tracing
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler() # Also log to console for immediate feedback
    ]
)

# --- Model and Processor Initialization ---

processor = None
model = None
vision_encoder = None # This will hold the vision_tower

try:
    logging.info("ATTEMPTING TO LOAD MODEL COMPONENTS...")
    logging.info(f"Loading processor from: {config.GEMMA_MODEL_PATH}")
    processor = AutoProcessor.from_pretrained(config.GEMMA_MODEL_PATH)
    logging.info("Processor loaded.")

    logging.info(f"Loading full multimodal model from: {config.GEMMA_MODEL_PATH}")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        config.GEMMA_MODEL_PATH,
        
    )
    logging.info("Full multimodal model loaded.")

    if model:
        vision_encoder = model.vision_tower
    if vision_encoder is None:
        raise RuntimeError("Could not access the vision_tower from the loaded model. It might be missing or named differently.")
    logging.info("Vision Tower successfully accessed.")

    logging.info("MODEL COMPONENTS LOADED SUCCESSFULLY!")

except Exception as e:
    logging.critical(f"CRITICAL ERROR DURING MODEL LOADING: {e}", exc_info=True) # Log with full traceback
    logging.error("Please check your model path, download, and environment setup. See model_loading_error.log for details.")
    processor, model, vision_encoder = None, None, None

def extract_features(image: Image.Image) -> np.ndarray:
    """
    Extracts features from a single image frame using the model's Vision Tower.
    """
    if not vision_encoder or not processor:
        logging.warning("Vision Tower or Processor is not loaded. Cannot extract features.")
        return np.random.rand(1, 3072).astype(np.float32)

    try:
        # CRITICAL FIX: Use processor.image_processor for image-only input
        inputs = processor.image_processor(images=image, return_tensors="pt").to(vision_encoder.device)
        logging.debug(f"Inputs prepared using image_processor. Device: {vision_encoder.device}")

        with torch.no_grad():
            # The vision_encoder (vision_tower) expects pixel_values directly
            outputs = vision_encoder(pixel_values=inputs["pixel_values"])
        
        logging.debug(f"Vision encoder outputs type: {type(outputs)}")
        logging.debug(f"Vision encoder outputs: {outputs}")

        if outputs is None:
            logging.error("Vision encoder returned None output. This is unexpected.")
            raise ValueError("Vision encoder returned None output.")

        # The vision encoder output is an object (BaseModelOutputWithPooling), not a tuple.
        # We access the `last_hidden_state` attribute directly.
        last_hidden_state = outputs.last_hidden_state
        logging.debug(f"Extracted last_hidden_state. Type: {type(last_hidden_state)}")

        if last_hidden_state is None:
            logging.error("Could not retrieve last_hidden_state from the vision tower's output.")
            raise ValueError("Could not retrieve last_hidden_state from the vision tower's output.")
        
        # CRITICAL FIX: Convert to float32 before converting to numpy
        embedding = torch.mean(last_hidden_state, dim=1).cpu().float().numpy()
        logging.debug(f"Feature extraction successful. Embedding shape: {embedding.shape}")
        return embedding.astype(np.float32)

    except Exception as e:
        logging.error(f"An error occurred during feature extraction: {e}", exc_info=True) # Add exc_info=True
        logging.error("Returning a random feature vector as a fallback.")
        return np.random.rand(1, 3072).astype(np.float32)
