import pytest
import numpy as np
from PIL import Image
import os

# Adjust the Python path to allow importing from video_ad_detector
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from video_ad_detector import feature_extractor

def test_feature_extraction_from_dummy_image():
    """
    Tests if feature_extractor.extract_features can process a dummy image
    and return a numpy array of the expected shape.
    """
    # Create a dummy image (e.g., a black 100x100 image)
    dummy_image = Image.new('RGB', (100, 100), color = 'black')

    # Call the feature extraction function
    features = feature_extractor.extract_features(dummy_image)

    # Assertions
    assert isinstance(features, np.ndarray), "Expected output to be a numpy array"
    # The vision tower of Gemma 3 base model outputs a hidden size of 1152 after mean pooling.
    assert features.shape == (1, 1152), f"Expected shape (1, 1152), but got {features.shape}"
    assert features.dtype == np.float32, "Expected features to be float32"

    print(f"Successfully extracted features with shape: {features.shape}")
    print(f"Features dtype: {features.dtype}")
