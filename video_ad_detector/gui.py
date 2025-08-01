import streamlit as st
import os
import tempfile
import shutil
import numpy as np

# Import project modules using absolute imports
from video_ad_detector import database
from video_ad_detector import video_processor
from video_ad_detector import ad_detector
from video_ad_detector import reporter
from video_ad_detector import config

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
    if 'report_data_list' not in st.session_state:
        st.session_state.report_data_list = []
    if 'features_loaded' not in st.session_state:
        st.session_state.features_loaded = False

    # --- Initialize Database and Directories ---
    database.init_db()
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    os.makedirs(config.SCREENSHOTS_DIR, exist_ok=True)
    os.makedirs(config.MATERIALS_DIR, exist_ok=True)

    # --- Sidebar ---
    st.sidebar.title("Workflow")

    # --- 1. Load Description Library ---
    st.sidebar.header("1. Load Description Library")
    try:
        available_materials_data = database.get_all_materials()
        available_materials = [f for f, o in available_materials_data]
    except Exception as e:
        st.sidebar.error(f"DB Error: {e}")
        available_materials = []

    if not available_materials:
        st.sidebar.info("No materials in the database. Please add some below.")
    else:
        selected_materials = st.sidebar.multiselect(
            "Select materials to use for detection:",
            options=[f for f, o in available_materials_data],
            default=[f for f, o in available_materials_data]
        )
        if st.sidebar.button("Load Selected Descriptions"):
            if not selected_materials:
                st.sidebar.warning("Please select at least one material.")
            else:
                with st.spinner(f"Loading descriptions for {len(selected_materials)} material(s)..."):
                    ad_detector._load_material_descriptions(filenames_to_load=selected_materials)
                    st.session_state.features_loaded = True
                    st.sidebar.success("✅ Descriptions loaded and ready.")

    # --- 2. Detect Ad in Videos ---
    st.sidebar.header("2. Detect Ad in Videos")
    if not st.session_state.features_loaded:
        st.sidebar.info("Load a description library before detection.")

    recorded_video_files = st.sidebar.file_uploader(
        "Upload videos to check for ads",
        type=["mp4", "mov", "avi"],
        key="recorded_uploader",
        accept_multiple_files=True,
        disabled=not st.session_state.features_loaded
    )

    if st.sidebar.button("Detect Ads", disabled=not st.session_state.features_loaded or not recorded_video_files):
        st.session_state.report_data_list = [] # Clear previous results
        for recorded_video_file in recorded_video_files:
            st.info(f"Analyzing video: {recorded_video_file.name}. Please wait...")
            temp_video_path = None
            try:
                temp_video_path = save_uploaded_file(recorded_video_file)
                if not temp_video_path:
                    st.error(f"Failed to save uploaded video temporarily: {recorded_video_file.name}")
                    continue

                with st.spinner(f"Processing video frames for {recorded_video_file.name}..."):
                    recorded_frames_data = video_processor.get_representative_recorded_frames(temp_video_path)
                
                if recorded_frames_data:
                    with st.spinner(f"Performing analysis for {recorded_video_file.name}..."):
                        report_data = ad_detector.find_matching_ad_segments(temp_video_path, recorded_frames_data)
                        if report_data:
                            st.session_state.report_data_list.append(report_data)
                else:
                    st.error(f"No frames extracted from {recorded_video_file.name}. Cannot perform detection.")
            finally:
                if temp_video_path and os.path.exists(temp_video_path):
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
                            video_processor.process_material_video(destination_path, progress_callback=update_progress)
                            st.success(f"Successfully generated descriptions for {material_file.name}")
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
    if st.session_state.report_data_list:
        for i, data in enumerate(st.session_state.report_data_list):
            st.header(f"Detection Results for {os.path.basename(data['recorded_video_path'])}")

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
            if st.button(f"Confirm and Generate PDF Report for {os.path.basename(data['recorded_video_path'])}", key=f"pdf_{i}"):
                with st.spinner("Generating PDF..."):
                    report_filename = reporter.generate_report(data)
                    if report_filename:
                        report_path = os.path.join(config.REPORTS_DIR, report_filename)
                        st.success(f"Report successfully generated for {os.path.basename(data['recorded_video_path'])}!")
                        with open(report_path, "rb") as f:
                            st.download_button(
                                label="📥 Download Report",
                                data=f.read(),
                                file_name=os.path.basename(report_path),
                                mime="application/pdf",
                                key=f"download_{i}"
                            )
                    else:
                        st.error(f"Failed to generate report for {os.path.basename(data['recorded_video_path'])}.")

    elif recorded_video_files and not st.session_state.report_data_list:
         st.info("No matching ad content was found in any of the uploaded videos.")

if __name__ == "__main__":
    main()
