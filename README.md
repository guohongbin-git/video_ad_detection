[English](#english-version) | [中文版](#中文版)

<a name="中文版"></a>

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/) [![Platform](https://img.shields.io/badge/Platform-Linux%20|%20macOS%20|%20Windows-lightgrey.svg)](https://en.wikipedia.org/wiki/Cross-platform) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/) [![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0) [![Topprism](https://img.shields.io/badge/Topprism-Data%20Intelligence-orange.svg)](https://www.topprismdata.com/)

# 视频广告检测器

这是一个使用多模态大模型（Google Gemma 3）自动检测视频中是否包含已知广告素材的工具。

## 功能特性

- **AI驱动**: 利用先进的多模态大模型 `google/gemma-3-4b-it` 理解视频帧内容。
- **特征向量化**: 将视频关键帧转换为高维特征向量，作为其内容的“指纹”。
- **精准片段检测**: 通过逐帧分析，精确识别实拍视频中包含广告内容的具体片段和时间点。
- **自动化深度报告**: 如果检测到匹配的广告，能自动生成包含详尽分析的PDF报告，其核心亮点包括：
    - **并排证据对比**: 将“实拍视频截图”与“源素材截图”并排展示，证据一目了然。
    - **精确时间戳**: 在每一组对比图下方，都清晰标注了镜头在各自视频中的时间点（如：`实拍视频: 15.3s | 源素材: 5.1s`）。
    - **时长分析**: 在报告中明确对比“源素材总时长”与“广告在实拍视频中累计出现的总时长”。
- **命令行驱动**: 提供清晰的命令行接口，方便集成与自动化。
- **Web界面 (可选)**: 同时保留了基于 Streamlit 的简单图形界面，用于快速演示。

## 工作原理

项目的工作流程分为两个阶段：

1.  **素材入库 (`add_material`)**:
    -   用户通过命令行或界面上传已知的广告“素材视频”。
    -   系统使用 `gemma-3` 模型提取视频关键帧的特征向量。
    -   所有素材的特征向量被存储在本地的 SQLite 数据库 (`metadata.db`) 中，作为比对基准。

2.  **广告检测 (`detect_ad`)**:
    -   用户通过命令行或界面提交需要检测的“实拍视频”。
    -   系统会逐帧（或按固定间隔）扫描实拍视频，并实时提取每一帧的特征向量。
    -   将每一帧的特征与数据库中所有素材的特征进行余弦相似度计算。
    -   当相似度超过预设阈值时，系统会记录下这个匹配的时间点，并累加匹配时长。
    -   检测完成后，系统会整理所有匹配数据，并调用报告模块。
    -   **报告生成**: 从相似度最高的几个匹配点中，提取实拍视频和对应源素材的帧，生成包含并排对比图、时间戳和时长分析的最终PDF报告。

## 技术栈

- **核心框架**: Python
- **AI模型**: `google/gemma-3-4b-it` (通过 `transformers` 库)
- **机器学习**: PyTorch, scikit-learn
- **视频/图像处理**: OpenCV, Pillow
- **报告生成**: ReportLab
- **Web界面**: Streamlit

## 安装与配置

1.  **克隆本项目**:
    ```bash
    git clone https://github.com/guohongbin-git/video_ad_detection.git
    cd video_ad_detection
    ```

2.  **创建并激活虚拟环境**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **安装依赖**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **下载大模型**:
    -   这是最关键的一步。您需要从 Hugging Face 下载 `gemma-3-4b-it` 模型。
    -   将下载好的模型文件夹 `gemma-3-4b-it` 放置在项目的根目录下。
    -   `config.py` 文件已配置为从此路径加载模型。

## 使用方法

### 通过命令行 (推荐)

1.  **添加广告素材**:
    ```bash
    python -m video_ad_detector.app add_material --video_path /path/to/your/ad_material.mp4
    ```

2.  **检测实拍视频**:
    ```bash
    python -m video_ad_detector.app detect_ad --video_path /path/to/your/recorded_video.mp4
    ```
    -   程序会自动进行分析。如果检测到广告，最终的PDF报告会自动生成在 `video_ad_detector/reports/` 目录下。

### 通过Web界面

1.  **激活虚拟环境**:
    ```bash
    source .venv/bin/activate
    ```

2.  **启动应用**:
    ```bash
    streamlit run run_gui.py
    ```
3.  **操作**:
    -   在打开的网页界面中，使用“上传广告素材”功能录入素材。
    -   使用“上传实拍视频进行检测”功能进行检测。结果会显示在界面上，同时PDF报告也会在后台生成。

## 项目结构

```
.
├── video_ad_detector/
│   ├── __init__.py
│   ├── ad_detector.py       # 核心检测逻辑：逐帧分析、片段匹配、数据整理
│   ├── app.py               # 命令行程序的入口点和主逻辑
│   ├── config.py            # 配置文件（模型路径、阈值等）
│   ├── database.py          # 数据库操作
│   ├── feature_extractor.py # 调用大模型提取特征向量
│   ├── gui.py               # Streamlit Web界面代码
│   ├── reporter.py          # 生成带对比图的PDF报告
│   ├── video_processor.py   # 视频处理工具（提取帧、获取时长等）
│   └── ...
└── README.md
```

---

<a name="english-version"></a>

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/) [![Platform](https://img.shields.io/badge/Platform-Linux%20|%20macOS%20|%20Windows-lightgrey.svg)](https://en.wikipedia.org/wiki/Cross-platform) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/) [![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0) [![Topprism](https://img.shields.io/badge/Topprism-Data%20Intelligence-orange.svg)](https://www.topprismdata.com/)

# Video Ad Detector

This is a tool that uses a multimodal large model (Google Gemma 3) to automatically detect whether a video contains known ad materials.

## Features

- **AI-Powered**: Utilizes the advanced multimodal large model `google/gemma-3-4b-it` to understand video frame content.
- **Feature Vectorization**: Converts video keyframes into high-dimensional feature vectors, serving as their content "fingerprint".
- **Precise Segment Detection**: Accurately identifies specific segments and timestamps within a recorded video that contain ad content through frame-by-frame analysis.
- **Automated In-depth Reporting**: If a matching ad is detected, it automatically generates a PDF report with detailed analysis, highlighting:
    - **Side-by-Side Evidence Comparison**: Displays "Recorded Video Screenshots" and "Source Material Screenshots" side-by-side for clear evidence.
    - **Precise Timestamps**: Clearly annotates the timestamp of each shot within its respective video below each pair of comparison images (e.g., `Recorded Video: 15.3s | Source Material: 5.1s`).
    - **Duration Analysis**: Explicitly compares the "Total Source Material Duration" with the "Total Accumulated Duration of the Ad in the Recorded Video" in the report.
- **Command-Line Driven**: Provides a clean command-line interface for easy integration and automation.
- **Web Interface (Optional)**: Also includes a simple graphical user interface built with Streamlit for quick demonstrations.

## How It Works

The project workflow is divided into two phases:

1.  **Material Ingestion (`add_material`)**:
    -   The user uploads known ad "material videos" via the command line or UI.
    -   The system uses the `gemma-3` model to extract feature vectors from the video's keyframes.
    -   The feature vectors of all materials are stored in a local SQLite database (`metadata.db`) as a baseline for comparison.

2.  **Ad Detection (`detect_ad`)**:
    -   The user submits a "recorded video" for detection via the command line or UI.
    -   The system scans the recorded video frame-by-frame (or at a fixed interval) and extracts a feature vector for each frame in real-time.
    -   It calculates the cosine similarity between the feature of each frame and the features of all materials in the database.
    -   When the similarity exceeds a preset threshold, the system records the timestamp of the match and accumulates the matched duration.
    -   After the analysis is complete, the system aggregates all match data and invokes the reporting module.
    -   **Report Generation**: It extracts frames from both the recorded video and the corresponding source material at the moments of highest similarity, then generates a final PDF report containing side-by-side comparisons, timestamps, and duration analysis.

## Tech Stack

- **Core Framework**: Python
- **AI Model**: `google/gemma-3-4b-it` (via `transformers` library)
- **Machine Learning**: PyTorch, scikit-learn
- **Video/Image Processing**: OpenCV, Pillow
- **Report Generation**: ReportLab
- **Web Interface**: Streamlit

## Installation and Setup

1.  **Clone this repository**:
    ```bash
    git clone https://github.com/guohongbin-git/video_ad_detection.git
    cd video_ad_detection
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the large model**:
    -   This is the most critical step. You need to download the `gemma-3-4b-it` model from Hugging Face.
    -   Place the downloaded model folder `gemma-3-4b-it` in the project's root directory.
    -   The `config.py` file is configured to load the model from this path.

## Usage

### Via Command Line (Recommended)

1.  **Add Ad Material**:
    ```bash
    python -m video_ad_detector.app add_material --video_path /path/to/your/ad_material.mp4
    ```

2.  **Detect Ads in a Recorded Video**:
    ```bash
    python -m video_ad_detector.app detect_ad --video_path /path/to/your/recorded_video.mp4
    ```
    -   The program will run the analysis automatically. If an ad is detected, the final PDF report will be generated in the `video_ad_detector/reports/` directory.

### Via Web Interface

1.  **Launch the application**:
    ```bash
    streamlit run video_ad_detector/gui.py
    ```
2.  **Instructions**:
    -   In the web interface that opens, use the "Upload Ad Material" feature to ingest materials.
    -   Use the "Upload Recorded Video for Detection" feature to perform detection. The results will be displayed on the interface, and the PDF report will be generated in the background.

## Project Structure

```
.
├── video_ad_detector/
│   ├── __init__.py
│   ├── ad_detector.py       # Core detection logic: frame-by-frame analysis, segment matching, data aggregation
│   ├── app.py               # Entry point and main logic for the command-line application
│   ├── config.py            # Configuration file (model path, thresholds, etc.)
│   ├── database.py          # Database operations
│   ├── feature_extractor.py # Invokes the large model to extract feature vectors
│   ├── gui.py               # Streamlit Web UI code
│   ├── reporter.py          # Generates PDF reports with comparison images
│   ├── video_processor.py   # Video processing utilities (frame extraction, duration retrieval, etc.)
│   └── ...
└── README.md
```