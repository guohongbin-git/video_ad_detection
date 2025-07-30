import pytest
import os
import shutil
import numpy as np
from unittest.mock import patch, MagicMock

# Adjust sys.path for absolute imports in tests
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from video_ad_detector import config
from video_ad_detector import database
from video_ad_detector import video_processor
from video_ad_detector import ad_detector
from video_ad_detector import reporter

# --- Setup and Teardown for Tests ---
@pytest.fixture(scope="module", autouse=True)
def setup_test_environment():
    # Create test directories
    os.makedirs(config.MATERIALS_DIR, exist_ok=True)
    os.makedirs(config.RECORDED_VIDEOS_DIR, exist_ok=True)
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    os.makedirs(config.SCREENSHOTS_DIR, exist_ok=True)

    # Use a temporary database for tests
    config.DATABASE_PATH = "test_metadata.db"
    database.init_db()

    yield

    # Clean up after tests
    if os.path.exists(config.DATABASE_PATH):
        os.remove(config.DATABASE_PATH)
    shutil.rmtree(config.MATERIALS_DIR, ignore_errors=True)
    shutil.rmtree(config.RECORDED_VIDEOS_DIR, ignore_errors=True)
    shutil.rmtree(config.REPORTS_DIR, ignore_errors=True)

# --- Mocking the Feature Extractor ---
@pytest.fixture
def mock_feature_extractor():
    with patch('video_ad_detector.feature_extractor.extract_features') as mock_extract:
        # Mock a consistent feature vector for testing
        mock_extract.return_value = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        yield mock_extract

# --- Test Cases ---

def test_database_operations():
    # Clear database before test
    if os.path.exists(config.DATABASE_PATH):
        os.remove(config.DATABASE_PATH)
    database.init_db()

    filename = "test_material.mp4"
    features = np.array([[0.5, 0.6, 0.7, 0.8]], dtype=np.float32)
    database.save_material_features(filename, features)

    retrieved_features = database.get_all_material_features()
    assert len(retrieved_features) == 1
    assert retrieved_features[0][0] == filename
    assert np.array_equal(retrieved_features[0][1], features)

def test_video_processor_material(mock_feature_extractor):
    # Create a dummy video file
    dummy_video_path = os.path.join(config.MATERIALS_DIR, "dummy_material.mp4")
    with open(dummy_video_path, "w") as f:
        f.write("dummy video content")

    # Mock cv2.VideoCapture and its methods
    with patch('cv2.VideoCapture') as mock_video_capture:
        mock_cap_instance = MagicMock()
        mock_video_capture.return_value = mock_cap_instance
        mock_cap_instance.isOpened.side_effect = [True, False] # Open then close
        mock_cap_instance.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8)) # Dummy frame
        mock_cap_instance.get.return_value = 30 # Mock FPS

        features = video_processor.process_material_video(dummy_video_path)
        assert features is not None
        assert mock_feature_extractor.called # Ensure feature extractor was called

def test_ad_detector_matching():
    # Clear database before test
    if os.path.exists(config.DATABASE_PATH):
        os.remove(config.DATABASE_PATH)
    database.init_db()

    # Add a known material
    known_ad_filename = "known_ad.mp4"
    known_ad_features = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    database.save_material_features(known_ad_filename, known_ad_features)

    # Test with a matching recorded video feature
    matching_recorded_features = np.array([[0.105, 0.205, 0.305, 0.405]], dtype=np.float32) # Slightly different but high similarity
    matched_ad, similarity = ad_detector.find_matching_ad(matching_recorded_features)
    assert matched_ad == known_ad_filename
    assert similarity >= config.SIMILARITY_THRESHOLD

    # Test with a non-matching recorded video feature
    non_matching_recorded_features = np.array([[0.9, 0.8, 0.7, 0.6]], dtype=np.float32)
    matched_ad, similarity = ad_detector.find_matching_ad(non_matching_recorded_features)
    assert matched_ad is None
    assert similarity < config.SIMILARITY_THRESHOLD

def test_reporter_create_report():
    # Create a dummy video file
    dummy_recorded_video_path = os.path.join(config.RECORDED_VIDEOS_DIR, "dummy_recorded.mp4")
    with open(dummy_recorded_video_path, "w") as f:
        f.write("dummy recorded video content")

    # Mock cv2.VideoCapture and cv2.imwrite for screenshot taking
    with patch('cv2.VideoCapture') as mock_video_capture, \
         patch('cv2.imwrite') as mock_imwrite:
        mock_cap_instance = MagicMock()
        mock_video_capture.return_value = mock_cap_instance
        mock_cap_instance.get.return_value = 100 # Mock total frames
        mock_cap_instance.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8)) # Dummy frame

        matched_ad = "test_ad.mp4"
        similarity = 0.95
        reporter.create_report(dummy_recorded_video_path, matched_ad, similarity)

        # Check if PDF report was created
        report_filename = f"report_{os.path.basename(dummy_recorded_video_path)}.pdf"
        report_path = os.path.join(config.REPORTS_DIR, report_filename)
        assert os.path.exists(report_path)
        assert mock_imwrite.call_count == config.NUM_SCREENSHOTS # Ensure screenshots were attempted
