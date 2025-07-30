
import os
import argparse
from . import database
from . import video_processor
from . import ad_detector
from . import reporter
from . import config

def main():
    parser = argparse.ArgumentParser(description="Local Video Ad Detector")
    parser.add_argument("action", choices=["add_material", "detect_ad"], help="Action to perform")
    parser.add_argument("--video_path", required=True, help="Path to the video file")

    args = parser.parse_args()

    database.init_db()

    if args.action == "add_material":
        print(f"Processing material video: {args.video_path}")
        features = video_processor.process_material_video(args.video_path)
        if features is not None:
            filename = os.path.basename(args.video_path)
            database.save_material_features(filename, features)
            print(f"Material '{filename}' processed and saved.")
        else:
            print("Could not process material video.")

    elif args.action == "detect_ad":
        print(f"Processing recorded video: {args.video_path}")
        features = video_processor.process_recorded_video(args.video_path)
        if features is not None:
            matched_ad, similarity = ad_detector.find_matching_ad(features)
            if matched_ad:
                print(f"Ad detected! Matched with '{matched_ad}' (Similarity: {similarity:.2f})")
                reporter.create_report(args.video_path, matched_ad, similarity)
            else:
                print("No matching ad found.")
        else:
            print("Could not process recorded video.")

if __name__ == "__main__":
    main()
