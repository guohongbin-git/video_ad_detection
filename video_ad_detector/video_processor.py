import cv2
import os
from PIL import Image
import numpy as np
from video_ad_detector import config
from video_ad_detector import feature_extractor

def get_video_duration(video_path: str) -> float:
    """
    Calculates the total duration of a video in seconds.

    Args:
        video_path (str): The path to the video file.

    Returns:
        float: The duration of the video in seconds, or 0.0 if an error occurs.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file at {video_path}")
            return 0.0
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        cap.release()

        if fps > 0:
            return frame_count / fps
        else:
            return 0.0
    except Exception as e:
        print(f"Error calculating video duration for {video_path}: {e}")
        return 0.0

def extract_frame_at_time(video_path: str, time_in_seconds: float, output_path: str) -> str | None:
    """
    Extracts a single frame from a video at a specific time and saves it as an image.

    Args:
        video_path (str): The path to the video file.
        time_in_seconds (float): The time point from which to extract the frame.
        output_path (str): The path where the output image will be saved.

    Returns:
        str | None: The path to the saved frame image, or None if an error occurs.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file at {video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            print(f"Error: Video FPS is zero for {video_path}")
            cap.release()
            return None
            
        frame_number = int(fps * time_in_seconds)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = cap.read()
        cap.release()

        if ret:
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, frame)
            if os.path.exists(output_path):
                return output_path
            else:
                print(f"Error: Failed to save frame to {output_path}")
                return None
        else:
            print(f"Error: Could not read frame at {time_in_seconds} seconds from {video_path}")
            return None
    except Exception as e:
        print(f"An error occurred during frame extraction from {video_path}: {e}")
        return None

def process_material_video(video_path: str):
    """
    Processes a material video to extract keyframes and their features.

    Args:
        video_path (str): The path to the material video.

    Returns:
        np.ndarray: The aggregated feature vector for the video.
    """
    cap = cv2.VideoCapture(video_path)
    features_list = []
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % (int(fps) * config.KEYFRAME_EXTRACTION_INTERVAL) == 0:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            features = feature_extractor.extract_features(image)
            features_list.append(features)
        
        frame_count += 1

    cap.release()

    if not features_list:
        return None

    # Aggregate features (e.g., by averaging)
    aggregated_features = np.mean(features_list, axis=0)
    return aggregated_features

def process_recorded_video(video_path: str):
    """
    Processes a recorded video to detect ad content.

    This is a placeholder for the full implementation which would include:
    1. Screen detection
    2. Screen content extraction and correction
    3. Feature extraction from the screen content
    4. Comparison with material features

    For now, it will extract features from the entire frame.
    """
    # This is a simplified version. The full implementation would be more complex.
    return process_material_video(video_path) # For now, treat it the same as a material video