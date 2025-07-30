import cv2
import numpy as np
import os
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from . import database, config, feature_extractor, video_processor

def find_matching_ad_segments(recorded_video_path: str):
    """
    Analyzes a recorded video frame by frame to find segments matching material ads.

    Args:
        recorded_video_path (str): The path to the recorded video.

    Returns:
        dict: A dictionary containing detailed report data, or None if no ad is found.
    """
    # 1. Load all material features from the database
    material_features_map = {filename: features for filename, features in database.get_all_material_features()}
    if not material_features_map:
        print("No material features found in the database.")
        return None

    # 2. Process the recorded video frame by frame
    cap = cv2.VideoCapture(recorded_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open recorded video at {recorded_video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Error: Recorded video FPS is zero.")
        cap.release()
        return None

    frame_interval = int(fps)  # Analyze one frame per second
    frame_count = 0
    detected_segments = []
    best_match_so_far = {"filename": None, "similarity": 0, "frame_number": 0}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process one frame per second
        if frame_count % frame_interval == 0:
            current_time_sec = frame_count / fps
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # 3. Extract features for the current frame
            current_features = feature_extractor.extract_features(image)

            # 4. Compare with all material features
            for material_filename, material_features in material_features_map.items():
                similarity = cosine_similarity(current_features.reshape(1, -1), material_features.reshape(1, -1))[0][0]
                
                # Track the single best match across the whole video for the report summary
                if similarity > best_match_so_far["similarity"]:
                    best_match_so_far.update({
                        "filename": material_filename,
                        "similarity": similarity,
                        "frame_number": frame_count
                    })

                # 5. If similarity exceeds threshold, record it as a potential segment start
                if similarity >= config.SIMILARITY_THRESHOLD:
                    detected_segments.append({
                        "recorded_video_time": current_time_sec,
                        "material_filename": material_filename,
                        "similarity": similarity
                    })
                    # For simplicity, we treat each match as a 1-second segment
                    # A more advanced implementation would merge consecutive matches

        frame_count += 1

    cap.release()

    if not detected_segments:
        print("No ad content detected in the video.")
        return None

    # --- Post-processing and data preparation for the report ---
    
    # For this version, we'll focus the report on the single best match found
    if best_match_so_far["filename"] is None:
        return None

    # 6. Calculate total matched duration
    # Simplified: total duration is the number of matched segments (since each is 1s)
    total_matched_duration = len(detected_segments)

    # 7. Get material video info
    best_match_filename = best_match_so_far["filename"]
    material_video_path = os.path.join(config.MATERIALS_DIR, best_match_filename)
    material_duration = video_processor.get_video_duration(material_video_path)

    # 8. Extract comparison screenshots
    comparison_screenshots = []
    # We'll take screenshots from the top 3 most similar moments
    top_segments = sorted(detected_segments, key=lambda x: x["similarity"], reverse=True)[:config.NUM_SCREENSHOTS]

    for i, segment in enumerate(top_segments):
        recorded_time = segment["recorded_video_time"]
        material_file = segment["material_filename"]
        material_path = os.path.join(config.MATERIALS_DIR, material_file)

        # This is a simplification. A robust solution needs to find the corresponding material time.
        # For now, we'll just pick a time in the material video (e.g., proportional to recorded time).
        material_time_guess = (recorded_time % material_duration) if material_duration > 0 else 0

        # Define paths for the screenshot files
        screenshot_base_name = "".join(c for c in os.path.basename(recorded_video_path) if c.isalnum()).replace(' ', '_')
        recorded_ss_path = os.path.join(config.SCREENSHOTS_DIR, f"rec_{screenshot_base_name}_{i}.jpg")
        material_ss_path = os.path.join(config.SCREENSHOTS_DIR, f"mat_{screenshot_base_name}_{i}.jpg")

        # Extract frames
        recorded_frame = video_processor.extract_frame_at_time(recorded_video_path, recorded_time, recorded_ss_path)
        material_frame = video_processor.extract_frame_at_time(material_path, material_time_guess, material_ss_path)

        if recorded_frame and material_frame:
            comparison_screenshots.append({
                "recorded_frame_path": recorded_frame,
                "material_frame_path": material_frame,
                "recorded_time": recorded_time,
                "material_time": material_time_guess
            })

    # 9. Assemble the final report data package
    report_data = {
        "recorded_video_path": recorded_video_path,
        "best_match_material_filename": best_match_filename,
        "overall_similarity_score": best_match_so_far["similarity"],
        "material_duration": material_duration,
        "total_matched_duration_in_recorded": total_matched_duration,
        "comparison_screenshots": comparison_screenshots
    }

    return report_data