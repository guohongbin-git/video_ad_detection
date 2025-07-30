
import cv2
import numpy as np
import os
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from . import database, config, feature_extractor, video_processor
import time

# --- Global cache for material features ---
MATERIAL_FRAME_FEATURES_CACHE = {}

def _load_all_material_features(filenames_to_load: list[str] | None = None):
    """
    Loads features from the database into the in-memory cache.
    If filenames_to_load is provided, only those files are loaded.
    Otherwise, all materials from the database are loaded.
    """
    MATERIAL_FRAME_FEATURES_CACHE.clear()
    print("Building material feature cache from database...")
    start_time = time.time()

    if filenames_to_load is None:
        # If no specific files are requested, load all of them.
        filenames_to_load = database.get_all_material_filenames()

    if not filenames_to_load:
        print("No material videos found in the database to load.")
        return

    for filename in filenames_to_load:
        # This is now the correct, fast way: loading pre-computed features.
        features_data = database.get_features_by_filename(filename)
        if features_data:
            # The data from DB is a list of tuples (timestamp, features_blob)
            # We need to convert it back to the desired dict format.
            MATERIAL_FRAME_FEATURES_CACHE[filename] = [
                {"time": time, "features": np.frombuffer(blob, dtype=np.float32)}
                for time, blob in features_data
            ]
            print(f"Loaded {len(features_data)} feature vectors for {filename} from database.")
        else:
            print(f"Warning: No features found in database for {filename}")

    end_time = time.time()
    print(f"Feature cache built in {end_time - start_time:.2f} seconds.")

def find_matching_ad_segments(recorded_video_path: str):
    """
    Analyzes a recorded video by comparing each of its frames against a pre-cached library
    of every frame from all material videos.
    """
    # 1. Ensure the material feature cache is loaded
    _load_all_material_features()
    if not MATERIAL_FRAME_FEATURES_CACHE:
        print("Error: Material feature cache is empty. Cannot perform detection.")
        return None

    # 2. Process the recorded video frame by frame
    cap = cv2.VideoCapture(recorded_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open recorded video at {recorded_video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(fps)) # Sample one frame per second
    frame_count = 0
    detected_matches = []

    print("Starting frame-by-frame analysis of recorded video...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # --- OPTIMIZATION: Process only one frame per second ---
        if frame_count % frame_interval == 0:
            current_recorded_time = frame_count / fps
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            current_features = feature_extractor.extract_features(image)

            # 3. Compare the current frame against every cached material frame
            for material_filename, material_frames in MATERIAL_FRAME_FEATURES_CACHE.items():
                for material_frame_data in material_frames:
                    similarity = cosine_similarity(current_features.reshape(1, -1), material_frame_data["features"].reshape(1, -1))[0][0]
                    
                    if similarity >= config.SIMILARITY_THRESHOLD:
                        detected_matches.append({
                            "recorded_time": current_recorded_time,
                            "material_filename": material_filename,
                            "material_time": material_frame_data["time"],
                            "similarity": similarity
                        })
        
        frame_count += 1

    cap.release()
    print(f"Analysis complete. Found {len(detected_matches)} potential matches.")

    if not detected_matches:
        return None

    # 4. Post-process and prepare report data
    # Sort by similarity to find the best evidence
    top_matches = sorted(detected_matches, key=lambda x: x["similarity"], reverse=True)[:config.NUM_SCREENSHOTS]

    # Determine the best overall matched material for the summary
    best_match_overall = top_matches[0]
    best_match_filename = best_match_overall["material_filename"]
    material_video_path = os.path.join(config.MATERIALS_DIR, best_match_filename)
    material_duration = video_processor.get_video_duration(material_video_path)

    # Calculate total matched duration (simplified)
    # We assume each matched frame represents 1/fps seconds of ad content.
    total_matched_duration = len(set(match["recorded_time"] for match in detected_matches)) / fps

    # 5. Extract the correctly paired screenshots for the report
    comparison_screenshots = []
    for i, match in enumerate(top_matches):
        screenshot_base_name = f"{os.path.basename(recorded_video_path)}_{i}"
        recorded_ss_path = os.path.join(config.SCREENSHOTS_DIR, f"rec_{screenshot_base_name}.jpg")
        material_ss_path = os.path.join(config.SCREENSHOTS_DIR, f"mat_{screenshot_base_name}.jpg")

        recorded_frame = video_processor.extract_frame_at_time(recorded_video_path, match["recorded_time"], recorded_ss_path)
        material_frame = video_processor.extract_frame_at_time(
            os.path.join(config.MATERIALS_DIR, match["material_filename"]),
            match["material_time"],
            material_ss_path
        )

        if recorded_frame and material_frame:
            comparison_screenshots.append({
                "recorded_frame_path": recorded_frame,
                "material_frame_path": material_frame,
                "recorded_time": match["recorded_time"],
                "material_time": match["material_time"]
            })

    # 6. Assemble the final report data package
    report_data = {
        "recorded_video_path": recorded_video_path,
        "best_match_material_filename": best_match_filename,
        "overall_similarity_score": best_match_overall["similarity"],
        "material_duration": material_duration,
        "total_matched_duration_in_recorded": total_matched_duration,
        "comparison_screenshots": comparison_screenshots
    }

    return report_data
