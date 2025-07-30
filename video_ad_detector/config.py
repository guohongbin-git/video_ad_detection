
import torch

# --- Device Configuration ---
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- Model Configuration ---
# IMPORTANT: Update this path to your local Gemma 3 model directory.
GEMMA_MODEL_PATH = "./gemma-3-4b-it"


# --- Video Processing Configuration ---
KEYFRAME_EXTRACTION_INTERVAL = 1  # seconds
SIMILARITY_THRESHOLD = 0.85  # For ad matching

# --- Database Configuration ---
DATABASE_PATH = "video_ad_detector/database/metadata.db"

# --- Directory Configuration ---
MATERIALS_DIR = "video_ad_detector/materials"
RECORDED_VIDEOS_DIR = "video_ad_detector/recorded_videos"
REPORTS_DIR = "video_ad_detector/reports"
SCREENSHOTS_DIR = "video_ad_detector/reports/screenshots"

# --- Reporter Configuration ---
NUM_SCREENSHOTS = 3
