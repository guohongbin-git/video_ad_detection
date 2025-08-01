import cv2
import os
from PIL import Image
import numpy as np
import logging
import imagehash
from itertools import groupby
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
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orientation = "landscape" if width > height else "portrait"
    database.add_material(video_filename, orientation)

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
    # Last frame (or 25th frame from the end, as requested)
    if total_frames > 1:
        last_frame_num = max(0, total_frames - 25)
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

def detect_video_playback_area(frame: np.ndarray, expected_orientation: str = "auto") -> tuple[int, int, int, int]:
    """
    Detects the video playback area within a given frame using edge detection and contour analysis.
    It attempts to find the largest rectangular contour, assuming it corresponds to the video player.
    The detection is guided by an expected orientation (landscape, portrait, or auto).

    Args:
        frame (np.ndarray): The input video frame (H, W, C).
        expected_orientation (str): The expected orientation of the video content ("landscape", "portrait", or "auto").
                                    If "auto", it will try to find the best fit.

    Returns:
        tuple[int, int, int, int]: A tuple (x, y, w, h) representing the bounding box
                                   of the detected video playback area. If no suitable
                                   area is found, it defaults to the central 80% of the frame
                                   based on the expected orientation.
    """
    h, w, _ = frame.shape
    logging.debug(f"Input frame dimensions: W={w}, H={h}, Expected Orientation: {expected_orientation}")

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

            # Filter for reasonable aspect ratios based on expected_orientation
            is_suitable_aspect_ratio = False
            if expected_orientation == "landscape":
                if 1.3 < aspect_ratio < 2.5: # Typical landscape aspect ratios (e.g., 16:9, 4:3)
                    is_suitable_aspect_ratio = True
            elif expected_orientation == "portrait":
                if 0.4 < aspect_ratio < 0.8: # Typical portrait aspect ratios (e.g., 9:16, 3:4)
                    is_suitable_aspect_ratio = True
            else: # auto or unknown
                # Accept a wider range if orientation is not specified or auto
                if 0.4 < aspect_ratio < 2.5:
                    is_suitable_aspect_ratio = True

            # Ensure it's not too small or too large (e.g., not the whole frame)
            if is_suitable_aspect_ratio and area > (w * h * 0.05) and area < (w * h * 0.98):
                if area > largest_area:
                    largest_area = area
                    best_rect = (x, y, current_w, current_h)
    
    # If no suitable rectangle is found, fall back to a central area based on expected orientation
    if largest_area == 0:
        if expected_orientation == "portrait":
            player_h = int(h * 0.8)
            player_w = int(player_h * (9/16)) # Assume 9:16 for portrait fallback
        else: # Default to landscape fallback
            player_w = int(w * 0.8)
            player_h = int(player_w * (9/16)) # Assume 16:9 for landscape fallback
        
        x = (w - player_w) // 2
        y = (h - player_h) // 2
        best_rect = (x, y, player_w, player_h)
        logging.warning(f"No suitable video playback area found for {expected_orientation} orientation, defaulting to central 80% of frame with assumed aspect ratio.")
    else:
        logging.info(f"Detected video playback area: {best_rect} for {expected_orientation} orientation.")

    return best_rect

def cluster_frames(frames_data: list[dict], hash_size: int = 8, similarity_threshold: int = 5) -> list[dict]:
    """
    Clusters frames based on perceptual hash similarity.

    Args:
        frames_data (list[dict]): A list of dictionaries, each containing 'time' and 'image'.
        hash_size (int): The size of the perceptual hash.
        similarity_threshold (int): The maximum hash distance to be considered similar.

    Returns:
        list[dict]: A list of representative frames, one from each cluster.
    """
    for frame in frames_data:
        frame['hash'] = imagehash.phash(frame['image'], hash_size=hash_size)

    frames_data.sort(key=lambda x: str(x['hash']))

    representative_frames = []
    for hash_val, group in groupby(frames_data, key=lambda x: str(x['hash'])):
        cluster = list(group)
        # Get the first frame of the cluster as the representative
        representative_frame = cluster[0]
        # Add to the list of representatives
        representative_frames.append(representative_frame)

    return representative_frames

def get_representative_recorded_frames(recorded_video_path: str, progress_callback=None) -> list[dict]:
    """
    Processes a recorded video to extract representative frames based on perceptual hash clustering.
    """
    cap = cv2.VideoCapture(recorded_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open recorded video at {recorded_video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(fps))

    if total_frames == 0 or fps == 0:
        print(f"Warning: No frames or zero FPS for {recorded_video_path}")
        cap.release()
        return []

    frames_data = []
    frame_count = 0
    num_frames_to_process = len(range(0, total_frames, frame_interval))
    processed_frames_count = 0

    print(f"Extracting frames from {recorded_video_path}...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            current_recorded_time = frame_count / fps
            
            x, y, w, h = detect_video_playback_area(frame, expected_orientation="auto")
            cropped_frame = frame[y:y+h, x:x+w]
            
            if cropped_frame.size == 0:
                frame_count += 1
                continue

            image = Image.fromarray(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
            frames_data.append({"time": current_recorded_time, "image": image})
            
            processed_frames_count += 1
            if progress_callback:
                progress_callback(processed_frames_count / num_frames_to_process, f"Extracting frame {processed_frames_count}/{num_frames_to_process}")

        frame_count += 1
    
    cap.release()

    print(f"Clustering {len(frames_data)} extracted frames...")
    representative_frames = cluster_frames(frames_data)
    
    print(f"Generating descriptions for {len(representative_frames)} representative frames...")
    for i, frame in enumerate(representative_frames):
        frame["description"] = feature_extractor.generate_description(frame["image"])
        if progress_callback:
            progress_callback((i + 1) / len(representative_frames), f"Generating description {i+1}/{len(representative_frames)}")

    print(f"Finished extracting representative frames. Total: {len(representative_frames)}")
    return representative_frames

def process_recorded_video(video_path: str):
    """
    Processes a recorded video to detect ad content.
    """
    return process_material_video(video_path)
