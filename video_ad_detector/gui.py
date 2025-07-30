import streamlit as st
import os
import tempfile
import shutil
import numpy as np

# Import project modules using relative imports
from . import database
from . import video_processor
from . import ad_detector
from . import reporter
from . import config

def save_uploaded_file(uploaded_file):
    """Saves uploaded file to a temporary path and returns the path."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving temporary file: {e}")
        return None

def main():
    st.set_page_config(page_title="Video Ad Detector", layout="wide")
    st.title("📹 Video Ad Detector")

    # --- Initialize Session State ---
    if 'report_data' not in st.session_state:
        st.session_state.report_data = None
    if 'features_loaded' not in st.session_state:
        st.session_state.features_loaded = False

    # --- Initialize Database and Directories ---
    database.init_db()
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    os.makedirs(config.SCREENSHOTS_DIR, exist_ok=True)
    os.makedirs(config.MATERIALS_DIR, exist_ok=True)

    # --- Sidebar ---
    st.sidebar.title("Workflow")

    # --- 1. Load Feature Library ---
    st.sidebar.header("1. Load Feature Library")
    try:
        available_materials = database.get_all_material_filenames()
    except Exception as e:
        st.sidebar.error(f"DB Error: {e}")
        available_materials = []

    if not available_materials:
        st.sidebar.info("No materials in the database. Please add some below.")
    else:
        selected_materials = st.sidebar.multiselect(
            "Select materials to use for detection:",
            options=available_materials,
            default=available_materials
        )
        if st.sidebar.button("Load Selected Features"):
            if not selected_materials:
                st.sidebar.warning("Please select at least one material.")
            else:
                with st.spinner(f"Loading features for {len(selected_materials)} material(s)..."):
                    ad_detector._load_all_material_features(filenames_to_load=selected_materials)
                    st.session_state.features_loaded = True
                    st.sidebar.success("✅ Features loaded and ready.")

    # --- 2. Detect Ad in a Video ---
    st.sidebar.header("2. Detect Ad in Video")
    if not st.session_state.features_loaded:
        st.sidebar.info("Load a feature library before detection.")

    recorded_video_file = st.sidebar.file_uploader(
        "Upload a video to check for ads",
        type=["mp4", "mov", "avi"],
        key="recorded_uploader",
        disabled=not st.session_state.features_loaded
    )

    if st.sidebar.button("Detect Ad", disabled=not st.session_state.features_loaded or not recorded_video_file):
        st.session_state.report_data = None # Clear previous results
        st.info(f"Analyzing video: {recorded_video_file.name}. Please wait...")
        temp_video_path = save_uploaded_file(recorded_video_file)
        
        if temp_video_path:
            with st.spinner("Performing frame-by-frame analysis..."):
                st.session_state.report_data = ad_detector.find_matching_ad_segments(temp_video_path)
            os.remove(temp_video_path)

    # --- 3. Add New Ad Material (in an expander) ---
    with st.sidebar.expander("Add New Ad Material"):
        material_video_files = st.file_uploader(
            "Upload known ad videos",
            type=["mp4", "mov", "avi"],
            accept_multiple_files=True,
            key="material_uploader"
        )
        if st.button("Process and Add to Library"):
            if not material_video_files:
                st.warning("Please upload at least one file.")
            else:
                processed_count = 0
                for material_file in material_video_files:
                    st.markdown(f"---")
                    st.markdown(f"**Processing: {material_file.name}**")
                    progress_bar = st.progress(0)
                    progress_text = st.empty()

                    def update_progress(percentage, text):
                        progress_bar.progress(min(1.0, percentage))
                        progress_text.text(text)

                    temp_material_path = save_uploaded_file(material_file)
                    if temp_material_path:
                        destination_path = os.path.join(config.MATERIALS_DIR, material_file.name)
                        try:
                            shutil.copyfile(temp_material_path, destination_path)
                            # We now save features for each sampled frame, not an aggregated one.
                            video_processor.process_material_video(destination_path, progress_callback=update_progress)
                            st.success(f"Successfully processed {material_file.name}")
                            processed_count += 1
                        except Exception as e:
                            st.error(f"Error processing {material_file.name}: {e}")
                        finally:
                            os.remove(temp_material_path)
                
                if processed_count > 0:
                    st.success(f"All {processed_count} new materials added to the library.")
                    st.info("Refreshing page to update material list...")
                    st.rerun()

    # --- Main Area for Displaying Results ---
    if st.session_state.report_data:
        data = st.session_state.report_data
        st.header("Detection Results Preview")

        st.subheader("Summary")
        st.markdown(
            f"""- **Best Match Ad Material:** `{data['best_match_material_filename']}`
- **Highest Similarity Score:** `{data['overall_similarity_score']:.2f}`
- **Source Material Duration:** `{data['material_duration']:.2f} seconds`
- **Total Matched Duration in Video:** `{data['total_matched_duration_in_recorded']:.2f} seconds`"""
        )

        st.subheader("Comparative Evidence")
        for item in data["comparison_screenshots"]:
            col1, col2 = st.columns(2)
            with col1:
                st.image(item["recorded_frame_path"], caption=f"Recorded Video at {item['recorded_time']:.2f}s", use_column_width=True)
            with col2:
                st.image(item["material_frame_path"], caption=f"Material Video at {item['material_time']:.2f}s", use_column_width=True)
            st.divider()

        # --- Confirmation and PDF Generation ---
        st.header("Generate Report")
        if st.button("Confirm and Generate PDF Report"):
            with st.spinner("Generating PDF..."):
                report_filename = reporter.generate_report(data)
                if report_filename:
                    report_path = os.path.join(config.REPORTS_DIR, report_filename)
                    st.success(f"Report successfully generated!")
                    with open(report_path, "rb") as f:
                        st.download_button(
                            label="📥 Download Report",
                            data=f.read(),
                            file_name=os.path.basename(report_path),
                            mime="application/pdf"
                        )
                else:
                    st.error("Failed to generate report.")

    elif recorded_video_file and st.session_state.get('report_data', True) is None: # Check if detection was run but found nothing
         st.info("No matching ad content was found in the video.")

if __name__ == "__main__":
    main()
