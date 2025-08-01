import cv2
import numpy as np
import os
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity # Now needed for embedding similarity
from . import database, config, feature_extractor, video_processor
import time
# import difflib # No longer needed for simple string similarity

# --- Global cache for material descriptions ---
MATERIAL_DESCRIPTIONS_CACHE = {}

def _load_material_descriptions(filenames_to_load: list[str] | None = None):
    """
    Loads keyframe descriptions from the database into the in-memory cache.
    """
    MATERIAL_DESCRIPTIONS_CACHE.clear()
    print("Loading material descriptions from database...")
    start_time = time.time()

    if filenames_to_load is None:
        # Change to get_all_materials to retrieve filename and orientation
        materials_data = database.get_all_materials()
    else:
        # If specific filenames are requested, we need to fetch their orientations
        # This assumes filenames_to_load only contains filenames, not full material data
        all_materials = database.get_all_materials()
        materials_data = [(f, o) for f, o in all_materials if f in filenames_to_load]

    if not materials_data:
        print("No material videos found in the database to load.")
        return

    for filename, orientation in materials_data:
        descriptions_data = database.get_descriptions_by_filename(filename)
        if descriptions_data:
            # Store orientation along with descriptions in the cache
            MATERIAL_DESCRIPTIONS_CACHE[filename] = {
                "orientation": orientation,
                "keyframes": {int(data['time'] * 1000): data for data in descriptions_data}
            }
            print(f"Loaded {len(descriptions_data)} descriptions for {filename} (Orientation: {orientation}) from database.")
        else:
            print(f"Warning: No descriptions found in database for {filename}. Consider processing it first.")

    end_time = time.time()
    print(f"Material description cache built in {end_time - start_time:.2f} seconds.")



def find_matching_ad_segments(recorded_video_path: str, recorded_frames_data: list[dict]):
    """
    Analyzes a recorded video by comparing its clustered frames' semantic descriptions
    against pre-cached keyframe descriptions from material videos.
    """
    # 1. Ensure the material description cache is loaded
    _load_material_descriptions()
    if not MATERIAL_DESCRIPTIONS_CACHE:
        print("Error: Material description cache is empty. Cannot perform detection.")
        return None

    # Initialize best matches for each material keyframe
    best_matches_per_material_keyframe = {}
    for material_filename, material_descriptions_dict in MATERIAL_DESCRIPTIONS_CACHE.items():
        best_matches_per_material_keyframe[material_filename] = {}
        for material_time, material_frame_data in material_descriptions_dict["keyframes"].items():
            best_matches_per_material_keyframe[material_filename][int(material_time * 1000)] = {
                "similarity": -1.0,
                "recorded_time": None,
                "recorded_frame_image": None,
                "recorded_frame_description": None
            }

    # 2. Use the pre-processed (clustered) recorded frames data
    if not recorded_frames_data:
        print("No representative frames extracted from recorded video. Cannot perform detection.")
        return None

    matched_recorded_times = set()

    print("Starting semantic analysis of recorded video...")
    for recorded_frame_data in recorded_frames_data:
        current_recorded_time = recorded_frame_data["time"]
        image_for_description = recorded_frame_data["image"]
        recorded_frame_description = recorded_frame_data["description"]

        # Compare with all material keyframes
        for material_filename, material_data in MATERIAL_DESCRIPTIONS_CACHE.items():
            material_descriptions_dict = material_data["keyframes"]

            for material_time, material_frame_data in material_descriptions_dict.items():
                similarity = feature_extractor.calculate_semantic_similarity(recorded_frame_description, material_frame_data["description"])
                
                if similarity >= config.PRELIMINARY_SIMILARITY_THRESHOLD and similarity > best_matches_per_material_keyframe[material_filename][int(material_time * 1000)]["similarity"]:
                    best_matches_per_material_keyframe[material_filename][int(material_time * 1000)] = {
                            "similarity": similarity,
                            "recorded_time": current_recorded_time,
                            "recorded_frame_image": image_for_description,
                            "recorded_frame_description": recorded_frame_description
                        }
                    matched_recorded_times.add(current_recorded_time)

    # Calculate total matched duration
    # This needs to be based on the original video's FPS, not just count of representative frames
    cap = cv2.VideoCapture(recorded_video_path)
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
    else:
        fps = 1 # Fallback if video cannot be opened
    total_matched_duration_in_recorded = len(matched_recorded_times) / fps

    # 3. Assemble comparison screenshots and data for report
    comparison_screenshots = []
    overall_similarity_score = 0.0
    best_match_filename = "N/A"
    material_duration = 0.0

    all_best_matches = []
    for material_filename, keyframe_matches in best_matches_per_material_keyframe.items():
        for material_time_ms, match_data in keyframe_matches.items():
            material_time = material_time_ms / 1000.0 # Convert back to seconds
            if match_data["recorded_time"] is not None: # Only include if a match was found
                material_frame_data = MATERIAL_DESCRIPTIONS_CACHE[material_filename]["keyframes"][material_time_ms]
                all_best_matches.append({
                    "recorded_time": match_data["recorded_time"],
                    "material_filename": material_filename,
                    "material_time": material_time, # Already in seconds
                    "similarity": match_data["similarity"],
                    "recorded_frame_image": match_data["recorded_frame_image"],
                    "recorded_frame_description": match_data["recorded_frame_description"],
                    "material_frame_description": material_frame_data["description"]
                })
    
    # Sort all best matches by similarity to pick top N for report, if needed
    all_best_matches_sorted = sorted(all_best_matches, key=lambda x: x["similarity"], reverse=True)

    if not all_best_matches_sorted:
        print("No ad content detected in the recorded video based on keyframe matching.")
        return None

    # Determine the best overall match for summary (highest similarity among all keyframe matches)
    best_overall_match_data = all_best_matches_sorted[0]
    overall_similarity_score = best_overall_match_data["similarity"]
    best_match_filename = best_overall_match_data["material_filename"]
    material_video_path = os.path.join(config.MATERIALS_DIR, best_match_filename)
    material_duration = video_processor.get_video_duration(material_video_path)

    # Filter comparison screenshots to only include those from the best matched material
    filtered_comparison_matches = []
    for match in all_best_matches_sorted:
        if match["material_filename"] == best_match_filename:
            filtered_comparison_matches.append(match)
    
    # Ensure we only take NUM_SCREENSHOTS (3) from the filtered list, sorted by material_time for consistency
    # This assumes material_descriptions are added in order (first, middle, last)
    comparison_screenshots_for_report = sorted(filtered_comparison_matches, key=lambda x: x["material_time"])[:config.NUM_SCREENSHOTS]

    # 5. Extract and save the correctly paired screenshots for the report
    for i, match in enumerate(comparison_screenshots_for_report):
        screenshot_base_name = f"{os.path.basename(recorded_video_path).split('.')[0]}_{i}"
        recorded_ss_path = os.path.join(config.SCREENSHOTS_DIR, f"rec_{screenshot_base_name}.jpg")
        material_ss_path = os.path.join(config.SCREENSHOTS_DIR, f"mat_{screenshot_base_name}.jpg")

        try:
            os.makedirs(config.SCREENSHOTS_DIR, exist_ok=True)
            match["recorded_frame_image"].save(recorded_ss_path)
            
            # Extract material frame on-demand
            material_video_path = os.path.join(config.MATERIALS_DIR, match["material_filename"])
            video_processor.extract_frame_at_time(material_video_path, match["material_time"], material_ss_path)
            
            comparison_screenshots.append({
                "recorded_frame_path": os.path.abspath(recorded_ss_path),
                "material_frame_path": os.path.abspath(material_ss_path),
                "recorded_time": match["recorded_time"],
                "material_time": match["material_time"],
                "recorded_frame_description": match["recorded_frame_description"],
                "material_frame_description": match["material_frame_description"]
            })
        except Exception as e:
            print(f"Error saving screenshot {i}: {e}")

    # 4. Assemble the final report data package
    report_data = {
        "recorded_video_path": recorded_video_path,
        "best_match_material_filename": best_match_filename,
        "overall_similarity_score": overall_similarity_score,
        "material_duration": material_duration,
        "total_matched_duration_in_recorded": total_matched_duration_in_recorded,
        "comparison_screenshots": comparison_screenshots # This now contains specific keyframe matches
    }

    return report_data