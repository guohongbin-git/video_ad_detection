
import streamlit as st
import os
import sys

# Add the project root to the Python path to resolve module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from video_ad_detector import config
from video_ad_detector import database
from video_ad_detector import video_processor
from video_ad_detector import ad_detector
from video_ad_detector import reporter

def save_uploaded_file(uploaded_file, directory):
    """Saves an uploaded file to a specific directory."""
    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def main():
    st.title("本地化视频广告投放识别系统")

    # Initialize DB
    database.init_db()

    # --- Material Video Section ---
    st.header("1. 上传广告素材视频")
    material_video = st.file_uploader("选择一个素材视频文件 (MP4, MOV, AVI)", type=["mp4", "mov", "avi"], key="material_uploader")

    if material_video is not None:
        st.write(f"已上传素材: {material_video.name}")
        if st.button("处理并入库素材"):
            with st.spinner("正在处理素材视频，提取特征中..."):
                # Save the uploaded file
                file_path = save_uploaded_file(material_video, config.MATERIALS_DIR)
                
                # Process the video and save features
                features = video_processor.process_material_video(file_path)
                if features is not None:
                    filename = os.path.basename(file_path)
                    database.save_material_features(filename, features)
                    st.success(f"素材 '{filename}' 处理成功并已存入特征库！")
                else:
                    st.error("无法处理该素材视频。")

    # --- Recorded Video Section ---
    st.header("2. 上传待检测的实拍视频")
    recorded_video = st.file_uploader("选择一个实拍视频文件 (MP4, MOV, AVI)", type=["mp4", "mov", "avi"], key="recorded_uploader")

    if recorded_video is not None:
        st.write(f"已上传待检测视频: {recorded_video.name}")
        if st.button("开始检测广告"):
            with st.spinner("正在分析视频，请稍候..."):
                # Save the uploaded file
                file_path = save_uploaded_file(recorded_video, config.RECORDED_VIDEOS_DIR)

                # Process the video to get features
                features = video_processor.process_recorded_video(file_path)
                if features is not None:
                    # Find matching ad
                    matched_ad, similarity = ad_detector.find_matching_ad(features)

                    if matched_ad:
                        st.success(f"检测到广告！匹配到素材: '{matched_ad}' (相似度: {similarity:.2f})")
                        # Generate report
                        reporter.create_report(file_path, matched_ad, similarity)
                        report_path = os.path.join(config.REPORTS_DIR, f"report_{os.path.basename(file_path)}.pdf")
                        with open(report_path, "rb") as f:
                            st.download_button("下载检测报告 (PDF)", f, file_name=f"report_{os.path.basename(file_path)}.pdf")
                    else:
                        st.warning("未检测到匹配的广告。")
                else:
                    st.error("无法处理该实拍视频。")

if __name__ == "__main__":
    main()
