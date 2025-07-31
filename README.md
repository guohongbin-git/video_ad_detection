[English](#english-version) | [中文版](#中文版)

<a name="中文版"></a>

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/) [![Platform](https://img.shields.io/badge/Platform-Linux%20|%20macOS%20|%20Windows-lightgrey.svg)](https://en.wikipedia.org/wiki/Cross-platform) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/) [![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0) [![Topprism](https://img.shields.io/badge/Topprism-Data%20Intelligence-orange.svg)](https://www.topprismdata.com/)

# 视频广告检测器

这是一个使用多模态大模型（Google Gemma 3）自动检测视频中是否包含已知广告素材的工具。

## 功能特性

- **AI驱动**: 利用先进的多模态大模型 `google/gemma-3-4b-it` 理解视频帧内容，并结合 **OCR 技术** 识别画面文字。
- **语义理解与描述**: 将视频关键帧转换为**富有语义的文本描述**，作为其内容的“语义指纹”。
- **精准片段检测**: 通过逐帧分析，结合**视频播放区域检测**和**语义相似度比对**，精确识别实拍视频中包含广告内容的具体片段和时间点。
- **自动化深度报告**: 如果检测到匹配的广告，能自动生成包含详尽分析的PDF报告，其核心亮点包括：
    - **并排证据对比**: 将“实拍视频截图”与“源素材截图”并排展示，证据一目了然。
    - **精确时间戳**: 在每一组对比图下方，都清晰标注了镜头在各自视频中的时间点（如：`实拍视频: 15.3s | 源素材: 5.1s`）。
    - **时长分析**: 在报告中明确对比“源素材总时长”与“广告在实拍视频中累计出现的总时长”。
- **命令行驱动**: 提供清晰的命令行接口，方便集成与自动化。
- **Web界面 (可选)**: 同时保留了基于 Streamlit 的简单图形界面，用于快速演示。

## 最新更新

- **Web界面缓存功能**:
    -   现在，Web界面（`run_gui.py`）在检测视频时引入了缓存机制。
    -   当您多次对同一个录制视频进行“检查广告”操作时，如果该视频的帧数据已经被处理过，系统将直接使用缓存数据，避免重复耗时的视频帧提取和处理过程，显著提升二次检测的速度。
- **核心逻辑稳定性提升**:
    -   修复了 `ad_detector.py` 中由于 `material_time` 类型错误导致的 `ValueError`。
    -   解决了 `ad_detector.py` 中访问 `MATERIAL_DESCRIPTIONS_CACHE` 时 `KeyError` 的问题，提高了匹配逻辑的健壮性。

## 工作原理

项目的工作流程分为两个阶段：

1.  **素材入库 (`add_material`)**:
    -   用户通过命令行或界面上传已知的广告“素材视频”。
    -   系统自动从素材视频中提取**关键帧**（如第一帧、中间帧、最后一帧）。
    -   对每个关键帧，系统利用 **LM Studio** 提供的 **Gemma 3 多模态模型** 结合 **EasyOCR** 技术，生成**详细的语义描述**（包含画面内容和识别到的文字）。
    -   这些语义描述被**缓存**在内存中，作为比对基准。

2.  **广告检测 (`detect_ad`)**:
    -   用户通过命令行或界面提交需要检测的“实拍视频”。
    -   系统会逐帧（或按固定间隔）扫描实拍视频。
    -   对于每一帧，系统首先尝试**检测视频播放设备在屏幕中的区域**。
    -   然后，对检测到的视频播放区域进行裁剪，并利用 **LM Studio** 提供的 **Gemma 3 多模态模型** 结合 **EasyOCR** 技术，生成该区域的**语义描述**。
    -   将每一帧的语义描述与缓存中所有素材的语义描述进行**语义相似度计算**。此过程通过 **LM Studio** 提供的 **文本嵌入模型** 将文本描述转换为高维文本嵌入向量，然后计算这些向量的余弦相似度。
    -   当语义相似度超过预设阈值时，系统会记录下这个匹配的时间点，并累加匹配时长。
    -   检测完成后，系统会整理所有匹配数据，并调用报告模块。
    -   **报告生成**: 从语义相似度最高的几个匹配点中，提取实拍视频和对应源素材的帧，生成包含并排对比图、时间戳和时长分析的最终PDF报告。

## 技术栈

- **核心框架**: Python
- **AI模型**: `google/gemma-3-4b-it` (多模态理解，通过 **LM Studio API**)
- **文本嵌入**: `nomic-ai/nomic-embed-text-v1.5` (语义相似度计算，通过 **LM Studio API**)
- **光学字符识别 (OCR)**: EasyOCR
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

4.  **LM Studio 配置**:
    -   **安装并运行 LM Studio**: 确保您已安装 LM Studio 并正在运行。
    -   **下载并加载模型**: 在 LM Studio 中下载并加载 `google/gemma-3-4b-it` 模型。
    -   **启用 OpenAI 兼容 API**: 确保 LM Studio 的 OpenAI 兼容 API 服务已开启（通常在 `http://localhost:1234`）。
    -   **配置嵌入模型**: 如果您希望使用 LM Studio 提供的嵌入服务，请在 LM Studio 中下载并加载一个文本嵌入模型（例如 `nomic-ai/nomic-embed-text-v1.5`），并确保 `config.py` 中的 `LM_STUDIO_EMBEDDING_MODEL_NAME` 与之匹配。

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
    -   在打开的网页界面中，使用“上传广告素材”功能录入素材。系统将为素材关键帧生成语义描述。
    -   使用“上传实拍视频进行检测”功能进行检测。结果会显示在界面上，同时PDF报告也会在后台生成。

## 项目结构

```
.
├── video_ad_detector/
│   ├── __init__.py
│   ├── ad_detector.py       # 核心检测逻辑：语义分析、片段匹配、数据整理
│   ├── app.py               # 命令行程序的入口点和主逻辑
│   ├── config.py            # 配置文件（LM Studio API配置、阈值等）
│   ├── database.py          # 数据库操作 (目前主要用于管理素材文件名)
│   ├── feature_extractor.py # 调用 LM Studio API 进行多模态理解、OCR和文本嵌入
│   ├── gui.py               # Streamlit Web界面代码
│   ├── reporter.py          # 生成带对比图的PDF报告
│   ├── video_processor.py   # 视频处理工具（提取帧、视频播放区域检测等）
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