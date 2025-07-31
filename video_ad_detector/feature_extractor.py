import os
import numpy as np
from PIL import Image
import logging
import easyocr # Import EasyOCR
import base64
import io
import openai
import streamlit as st # Import streamlit for caching

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

# --- Component Initialization (Cached by Streamlit) ---
@st.cache_resource
def initialize_llm_client():
    logging.info("Initializing LM Studio API client...")
    try:
        client = openai.OpenAI(base_url=config.LM_STUDIO_API_BASE, api_key="lm-studio") # Add a dummy API key
        logging.info(f"LM Studio API client initialized with base URL: {config.LM_STUDIO_API_BASE}")
        return client
    except Exception as e:
        logging.critical(f"CRITICAL ERROR DURING LM STUDIO CLIENT INITIALIZATION: {e}", exc_info=True)
        return None

@st.cache_resource
def initialize_ocr_reader():
    logging.info("Initializing EasyOCR reader...")
    try:
        use_gpu = False
        try:
            import torch
            if torch.cuda.is_available():
                use_gpu = True
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                use_gpu = True
        except ImportError:
            logging.warning("PyTorch not found, EasyOCR will run on CPU.")

        ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=use_gpu)
        logging.info(f"EasyOCR reader initialized. Using GPU: {use_gpu}")
        return ocr_reader
    except Exception as e:
        logging.critical(f"CRITICAL ERROR DURING EASYOCR INITIALIZATION: {e}", exc_info=True)
        return None

# Call the cached functions to get the initialized components
client = initialize_llm_client()
ocr_reader = initialize_ocr_reader()

def pil_to_base64(image: Image.Image) -> str:
    """
    Converts a PIL Image to a Base64 encoded string.
    """
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG") # Use JPEG for efficiency
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def generate_description(image: Image.Image, perform_ocr: bool = True) -> str:
    """
    Generates a textual description of an image using the LM Studio chat completion API.
    Conditionally performs OCR internally and includes the detected text in the prompt.
    """
    if not client:
        logging.warning("LM Studio client is not loaded. Cannot generate description.")
        return "Error: Model not loaded."

    ocr_text_combined = ""
    if perform_ocr:
        if not ocr_reader:
            logging.warning("OCR Reader is not loaded. Cannot perform OCR.")
        else:
            try:
                logging.info("Performing OCR on the image...")
                image_np = np.array(image)
                if image_np.shape[2] == 4:
                    image_np = image_np[:, :, :3] # Convert RGBA to RGB

                ocr_results = ocr_reader.readtext(image_np)
                detected_texts = [res[1] for res in ocr_results]
                ocr_text_combined = " ".join(detected_texts)
                logging.info(f"OCR Detected Text: {ocr_text_combined}")
            except Exception as e:
                logging.error(f"An error occurred during OCR: {e}", exc_info=True)
                ocr_text_combined = ""

    try:
        # Convert image to Base64
        base64_image = pil_to_base64(image)

        # Construct the prompt with OCR text for the LLM
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请根据图片内容生成详细描述。"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }
        ]

        if ocr_text_combined:
            messages[0]["content"].append({
                "type": "text", 
                "text": f"作为参考，图片中识别出的文字是：“{ocr_text_combined}”。请在你的描述中自然地结合这些文字信息。"
            })

        logging.info(f"Sending chat completion request to LM Studio for model: {config.LM_STUDIO_CHAT_MODEL_NAME}")
        chat_completion = client.chat.completions.create(
            model=config.LM_STUDIO_CHAT_MODEL_NAME,
            messages=messages,
            max_tokens=200
        )
        
        description = chat_completion.choices[0].message.content
        logging.info(f"Generated Description: {description}")
        return description

    except Exception as e:
        logging.error(f"An error occurred during description generation via LM Studio API: {e}", exc_info=True)
        return "Error: Could not generate description."

def get_text_embedding(text: str) -> np.ndarray:
    """
    Generates a text embedding for the given text using the LM Studio embeddings API.
    """
    if not client:
        logging.warning("LM Studio client is not loaded. Cannot generate embedding.")
        return np.array([])
    try:
        logging.info(f"Sending embedding request to LM Studio for model: {config.LM_STUDIO_EMBEDDING_MODEL_NAME}")
        response = client.embeddings.create(
            model=config.LM_STUDIO_EMBEDDING_MODEL_NAME,
            input=[text]
        )
        embedding = np.array(response.data[0].embedding).astype(np.float32)
        logging.info(f"Text embedding generated. Shape: {embedding.shape}")
        return embedding
    except Exception as e:
        logging.error(f"An error occurred during text embedding generation via LM Studio API: {e}", exc_info=True)
        return np.array([])
