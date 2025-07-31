import cv2
import os
from PIL import Image
import numpy as np
import logging # Import logging module
from video_ad_detector import config
from . import feature_extractor
from . import database

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

def process_material_video(video_path: str, progress_callback=None):
    """
    Processes a material video to extract keyframes and generate their semantic descriptions.
    These descriptions are then used by the ad_detector.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False

    video_filename = os.path.basename(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0 or fps == 0:
        print(f"Warning: No frames or zero FPS for {video_filename}")
        cap.release()
        return False

    keyframes_to_process = []
    # First frame
    keyframes_to_process.append((0, 0.0))
    # Middle frame
    if total_frames > 1:
        middle_frame_num = total_frames // 2
        middle_time = middle_frame_num / fps
        keyframes_to_process.append((middle_frame_num, middle_time))
    # Last frame
    if total_frames > 1:
        last_frame_num = total_frames - 1
        last_time = last_frame_num / fps
        keyframes_to_process.append((last_frame_num, last_time))

    print(f"Processing keyframes for material video: {video_filename}")
    descriptions = []
    for i, (frame_num, frame_time) in enumerate(keyframes_to_process):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            description = feature_extractor.generate_description(image)
            descriptions.append({"time": frame_time, "description": description})
            print(f"  - Frame at {frame_time:.2f}s description: {description[:50]}...") # Print first 50 chars
            if progress_callback:
                progress_callback((i + 1) / len(keyframes_to_process), f"Generating description for keyframe {i+1}/{len(keyframes_to_process)}")
        else:
            print(f"Warning: Could not read frame {frame_num} from {video_filename}")
    
    cap.release()

    if not descriptions:
        print(f"Error: No descriptions were generated for {video_filename}")
        return False
    
    # Save the generated descriptions to the database
    database.save_material_descriptions(video_filename, descriptions)
    print(f"Successfully processed and saved {len(descriptions)} descriptions for {video_filename}.")
    return True

def detect_video_playback_area(frame: np.ndarray) -> tuple[int, int, int, int]:
    """
    Detects the video playback area within a given frame using edge detection and contour analysis.
    It attempts to find the largest rectangular contour, assuming it corresponds to the video player.

    Args:
        frame (np.ndarray): The input video frame (H, W, C).

    Returns:
        tuple[int, int, int, int]: A tuple (x, y, w, h) representing the bounding box
                                   of the detected video playback area. If no suitable
                                   area is found, it defaults to the central 80% of the frame.
    """
    h, w, _ = frame.shape
    logging.debug(f"Input frame dimensions: W={w}, H={h}")

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and help with edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detector
    edges = cv2.Canny(blurred, 30, 100) # Adjusted thresholds
    logging.debug(f"Canny edges non-zero pixels: {np.count_nonzero(edges)}")

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logging.debug(f"Found {len(contours)} contours.")

    largest_area = 0
    best_rect = (0, 0, w, h) # Default to full frame if nothing suitable is found

    for i, contour in enumerate(contours):
        # Approximate the contour to a polygon
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # If the approximated contour has 4 vertices (a rectangle)
        if len(approx) == 4:
            x, y, current_w, current_h = cv2.boundingRect(approx)
            aspect_ratio = float(current_w) / current_h
            area = current_w * current_h
            logging.debug(f"  Contour {i}: x={x}, y={y}, w={current_w}, h={current_h}, aspect_ratio={aspect_ratio:.2f}, area={area}")

            # Filter for reasonable aspect ratios (e.g., 16:9, 4:3, or close to it)
            # and ensure it's not too small or too large (e.g., not the whole frame)
            if 0.4 < aspect_ratio < 3.0 and area > (w * h * 0.05) and area < (w * h * 0.98):
                if area > largest_area:
                    largest_area = area
                    best_rect = (x, y, current_w, current_h)
    
    # If no suitable rectangle is found, fall back to the central 80% as before
    if largest_area == 0:
        player_w = int(w * 0.8)
        player_h = int(h * 0.8)
        x = (w - player_w) // 2
        y = (h - player_h) // 2
        best_rect = (x, y, player_w, player_h)
        logging.warning("No suitable video playback area found, defaulting to central 80% of frame.")
    else:
        logging.info(f"Detected video playback area: {best_rect}")

    return best_rect

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