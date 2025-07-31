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
        filenames_to_load = database.get_all_material_filenames()

    if not filenames_to_load:
        print("No material videos found in the database to load.")
        return

    for filename in filenames_to_load:
        descriptions_data = database.get_descriptions_by_filename(filename)
        if descriptions_data:
            MATERIAL_DESCRIPTIONS_CACHE[filename] = {data['time']: data for data in descriptions_data}
            print(f"Loaded {len(descriptions_data)} descriptions for {filename} from database.")
        else:
            print(f"Warning: No descriptions found in database for {filename}. Consider processing it first.")

    end_time = time.time()
    print(f"Material description cache built in {end_time - start_time:.2f} seconds.")

def _calculate_semantic_similarity(desc1: str, desc2: str) -> float:
    """
    Calculates semantic similarity between two text descriptions using text embeddings.
    """
    if not desc1 or not desc2:
        return 0.0
    
    embedding1 = feature_extractor.get_text_embedding(desc1)
    embedding2 = feature_extractor.get_text_embedding(desc2)

    if embedding1.size == 0 or embedding2.size == 0:
        return 0.0

    # Reshape for cosine_similarity: (1, n_features)
    return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]

def find_matching_ad_segments(recorded_video_path: str):
    """
    Analyzes a recorded video by comparing its frames' semantic descriptions
    against pre-cached keyframe descriptions from material videos.
    For each material keyframe (first, middle, last), it finds the best matching
    frame in the recorded video.
    """
    # 1. Ensure the material description cache is loaded
    _load_material_descriptions()
    if not MATERIAL_DESCRIPTIONS_CACHE:
        print("Error: Material description cache is empty. Cannot perform detection.")
        return None

    # Initialize best matches for each material keyframe
    # Structure: {material_filename: {material_time: {recorded_time, similarity, recorded_frame_image, recorded_frame_description}}}
    best_matches_per_material_keyframe = {}
    for material_filename, material_descriptions_dict in MATERIAL_DESCRIPTIONS_CACHE.items():
        best_matches_per_material_keyframe[material_filename] = {}
        for material_time, material_frame_data in material_descriptions_dict.items():
            best_matches_per_material_keyframe[material_filename][material_time] = {
                "similarity": -1.0, # Initialize with a very low similarity
                "recorded_time": None,
                "recorded_frame_image": None,
                "recorded_frame_description": None
            }

    # 2. Process the recorded video frame by frame
    cap = cv2.VideoCapture(recorded_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open recorded video at {recorded_video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(fps)) # Sample one frame per second
    frame_count = 0
    total_matched_duration_in_recorded = 0.0
    matched_recorded_times = set() # To count unique matched seconds for total duration

    print("Starting semantic analysis of recorded video...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            current_recorded_time = frame_count / fps
            
            # Detect video playback area and crop the frame
            x, y, w, h = video_processor.detect_video_playback_area(frame)
            cropped_frame = frame[y:y+h, x:x+w]
            
            if cropped_frame.size == 0:
                print(f"Warning: Cropped frame is empty at time {current_recorded_time:.2f}s. Skipping.")
                frame_count += 1
                continue

            # Save cropped frame for debugging if enabled
            if config.SAVE_DEBUG_FRAMES:
                debug_frame_path = os.path.join(config.DEBUG_FRAMES_DIR, f"recorded_frame_{int(current_recorded_time * 1000)}.jpg")
                cv2.imwrite(debug_frame_path, cropped_frame)

            image_for_description = Image.fromarray(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
            
            # Stage 1: Generate preliminary description without OCR
            prelim_recorded_frame_description = feature_extractor.generate_description(image_for_description, perform_ocr=False)

            # Compare with all material keyframes for preliminary similarity
            for material_filename, material_descriptions_dict in MATERIAL_DESCRIPTIONS_CACHE.items():
                for material_time, material_frame_data in material_descriptions_dict.items():
                    prelim_similarity = _calculate_semantic_similarity(prelim_recorded_frame_description, material_frame_data["description"])
                    
                    if prelim_similarity >= config.PRELIMINARY_SIMILARITY_THRESHOLD:
                        # Stage 2: If preliminary match, generate full description with OCR for the recorded frame
                        final_recorded_frame_description = feature_extractor.generate_description(image_for_description, perform_ocr=True)
                        final_similarity = _calculate_semantic_similarity(final_recorded_frame_description, material_frame_data["description"])
                        
                        if final_similarity >= best_matches_per_material_keyframe[material_filename][material_time]["similarity"]:
                            best_matches_per_material_keyframe[material_filename][material_time] = {
                                "similarity": final_similarity,
                                "recorded_time": current_recorded_time,
                                "recorded_frame_image": image_for_description,
                                "recorded_frame_description": final_recorded_frame_description
                            }
                            matched_recorded_times.add(current_recorded_time) # Add to set for total duration
        
        frame_count += 1

    cap.release()
    
    # Calculate total matched duration
    total_matched_duration_in_recorded = len(matched_recorded_times) / fps

    # 3. Assemble comparison screenshots and data for report
    comparison_screenshots = []
    overall_similarity_score = 0.0
    best_match_filename = "N/A"
    material_duration = 0.0

    all_best_matches = []
    for material_filename, keyframe_matches in best_matches_per_material_keyframe.items():
        for material_time, match_data in keyframe_matches.items():
            if match_data["recorded_time"] is not None: # Only include if a match was found
                material_frame_data = MATERIAL_DESCRIPTIONS_CACHE[material_filename][material_time]
                all_best_matches.append({
                    "recorded_time": match_data["recorded_time"],
                    "material_filename": material_filename,
                    "material_time": material_time,
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