import os
import numpy as np
from PIL import Image
import logging
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

# Call the cached functions to get the initialized components
client = initialize_llm_client()

def pil_to_base64(image: Image.Image) -> str:
    """
    Converts a PIL Image to a Base64 encoded string.
    """
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG") # Use JPEG for efficiency
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def generate_description(image: Image.Image) -> str:
    """
    Generates a textual description of an image using the LM Studio chat completion API.
    """
    if not client:
        logging.warning("LM Studio client is not loaded. Cannot generate description.")
        return "Error: Model not loaded."

    try:
        # Convert image to Base64
        base64_image = pil_to_base64(image)

        # Construct the prompt for the LLM
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请详细描述图片内容。"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }
        ]

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

def calculate_semantic_similarity(desc1: str, desc2: str) -> float:
    """
    Calculates semantic similarity between two text descriptions using text embeddings.
    """
    if not desc1 or not desc2:
        return 0.0
    
    embedding1 = get_text_embedding(desc1)
    embedding2 = get_text_embedding(desc2)

    if embedding1.size == 0 or embedding2.size == 0:
        return 0.0

    # Reshape for cosine_similarity: (1, n_features)
    from sklearn.metrics.pairwise import cosine_similarity # Import here to avoid circular dependency
    return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]


def get_all_keyframes_data(video_path: str, frame_indices: list[int]) -> list[dict]:
    """
    Extracts keyframes, generates descriptions and embeddings for them.
    """
    frames_data = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return frames_data

    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            description = generate_description(frame_pil) # OCR is no longer performed
            embedding = get_text_embedding(description)
            
            frames_data.append({
                "frame_index": frame_index,
                "description": description,
                "embedding": embedding.tobytes() # Store embedding as bytes
            })
    
    cap.release()
    return frames_data
