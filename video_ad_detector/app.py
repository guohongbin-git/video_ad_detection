import os
import argparse
from . import database
from . import video_processor
from . import ad_detector
from . import reporter
from . import config

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Local Video Ad Detector - Compares a video against a library of known ad materials.")
    parser.add_argument(
        "action", 
        choices=["add_material", "detect_ad"], 
        help="'add_material': Process a video and add it to the known ad materials database. 'detect_ad': Analyze a video to see if it contains any known ad materials."
    )
    parser.add_argument(
        "--video_path", 
        required=True, 
        help="The full path to the video file to be processed."
    )

    args = parser.parse_args()

    # Initialize the database and ensure all necessary directories exist
    database.init_db()
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    os.makedirs(config.SCREENSHOTS_DIR, exist_ok=True)

    # --- Action: Add a new video to the material database ---
    if args.action == "add_material":
        print(f"Starting: Processing material video at '{args.video_path}'")
        # Extract a representative feature vector from the material video
        features = video_processor.process_material_video(args.video_path)
        
        if features is not None:
            filename = os.path.basename(args.video_path)
            # Save the feature vector to the database, associated with the filename
            database.save_material_features(filename, features)
            print(f"Success: Material '{filename}' was processed and its features were saved to the database.")
        else:
            print(f"Error: Could not extract features from the material video at '{args.video_path}'.")

    # --- Action: Detect ads in a recorded video ---
    elif args.action == "detect_ad":
        print(f"Starting: Analyzing recorded video at '{args.video_path}' for ad content.")
        
        # The new ad_detector function handles the entire process
        report_data = ad_detector.find_matching_ad_segments(args.video_path)
        
        if report_data:
            # If ad segments were found, the function returns a data dictionary
            print(f"Ad content detected! Best match was with '{report_data['best_match_material_filename']}'.")
            print("Generating detailed PDF report...")
            
            # The new reporter function takes the data dictionary to build the PDF
            reporter.generate_report(report_data)
        else:
            # If no significant matches are found, the function returns None
            print("No matching ad content was found in the video.")

if __name__ == "__main__":
    main()