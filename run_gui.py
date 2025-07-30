# This is a launcher script for the Streamlit GUI.
# Running this script from the project root ensures that Python's
# module system correctly recognizes 'video_ad_detector' as a package.

import sys
import os

# Add the project root to the Python path to allow absolute imports
# from the 'video_ad_detector' package.
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from video_ad_detector.gui import main

if __name__ == "__main__":
    # The main function from gui.py will be executed as if it were here,
    # but with the correct package context.
    main()
