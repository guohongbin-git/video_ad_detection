[English](#english-version) | [中文版](#中文版)

<a name="中文版"></a>

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/) [![Platform](https://img.shields.io/badge/Platform-Linux%20|%20macOS%20|%20Windows-lightgrey.svg)](https://en.wikipedia.org/wiki/Cross-platform) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/) [![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0) [![Topprism](https://img.shields.io/badge/Topprism-Data%20Intelligence-orange.svg)](https://www.topprismdata.com/)

# 视频广告检测器

这是一个使用多模态大模型（Google Gemma 3N）自动检测视频中是否包含已知广告素材的工具。

## 功能特性

- **AI驱动**: 利用先进的多模态大模型 `google/gemma-3n` 理解视频帧内容。
- **高效预处理**: 在进行AI分析前，使用**感知哈希算法**对视频帧进行**图像聚类**，有效去除冗余和重复的帧，仅将代表性场景提交给AI，大幅提升处理效率。
- **语义理解与描述**: 将视频关键帧转换为**富有语义的文本描述**，作为其内容的“语义指纹”。
- **精准片段检测**: 通过对代表性帧进行**语义相似度比对**，精确识别实拍视频中包含广告内容的具体片段和时间点。
- **自动化深度报告**: 如果检测到匹配的广告，能自动生成包含详尽分析的PDF报告，其核心亮点包括：
    - **并排证据对比**: 将“实拍视频截图”与“源素材截图”并排展示，证据一目了然。
    - **精确时间戳**: 在每一组对比图下方，都清晰标注了镜头在各自视频中的时间点。
    - **时长分析**: 在报告中明确对比“源素材总时长”与“广告在实拍视频中累计出现的总时长”。
- **批量处理**: 支持一次性上传和分析多个视频。

## 工作原理

项目的工作流程分为两个阶段：

1.  **素材入库 (`add_material`)**:
    -   用户通过界面上传已知的广告“素材视频”。
    -   系统自动从素材视频中提取**关键帧**（如第一帧、中间帧、最后一帧）。
    -   对每个关键帧，系统利用 **LM Studio** 提供的 **Gemma 3N 多模态模型**生成**详细的语义描述**。
    -   这些语义描述被**存储**在本地数据库中，作为比对基准。

2.  **广告检测 (`detect_ad`)**:
    -   用户通过界面提交一个或多个需要检测的“实拍视频”。
    -   系统对每个实拍视频进行**批量抽帧**。
    -   使用**感知哈希算法 (`imagehash`)** 对所有抽取的帧进行**聚类**，为每个视觉上相似的场景组推选出一个**代表帧**。
    -   对每个**代表帧**，系统利用 **Gemma 3N 多模态模型**生成其**语义描述**。
    -   将代表帧的语义描述与数据库中所有素材的语义描述进行**语义相似度计算**（通过文本嵌入向量的余弦相似度）。
    -   当语义相似度超过预设阈值时，系统会记录下这个匹配的时间点。
    -   检测完成后，系统会整理所有匹配数据，并调用报告模块生成包含并排对比图、时间戳和时长分析的最终PDF报告。

## 技术栈

- **核心框架**: Python
- **AI模型**: `google/gemma-3n` (多模态理解，通过 **LM Studio API**)
- **文本嵌入**: `nomic-ai/nomic-embed-text-v1.5` (语义相似度计算，通过 **LM Studio API**)
- **图像聚类**: ImageHash
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
    -   **下载并加载模型**: 在 LM Studio 中下载并加载 `gemma-3n` 系列的多模态模型。
    -   **核对模型标识符**: 确保 `video_ad_detector/config.py` 文件中的 `LM_STUDIO_CHAT_MODEL_NAME` 与您在 LM Studio 中为模型设置的**模型标识符**完全一致。
    -   **启用 OpenAI 兼容 API**: 确保 LM Studio 的 OpenAI 兼容 API 服务已开启（通常在 `http://localhost:1234`）。
    -   **配置嵌入模型**: 在 LM Studio 中下载并加载一个文本嵌入模型（例如 `nomic-ai/nomic-embed-text-v1.5`），并确保 `config.py` 中的 `LM_STUDIO_EMBEDDING_MODEL_NAME` 与之匹配。

## 使用方法

1.  **激活虚拟环境**:
    ```bash
    source .venv/bin/activate
    ```

2.  **启动应用**:
    ```bash
    streamlit run run_gui.py
    ```
3.  **操作**:
    -   在打开的网页界面中，使用侧边栏的“Add New Ad Material”功能录入广告素材。
    -   使用“Detect Ad in Videos”功能上传一个或多个待检测视频。
    -   结果会直接显示在主界面上，并可以生成PDF报告。

## 项目结构

```
.
├── video_ad_detector/
│   ├── __init__.py
│   ├── ad_detector.py       # 核心检测逻辑：语义分析、片段匹配
│   ├── app.py               # 命令行程序的入口点 (当前主要使用GUI)
│   ├── config.py            # 配置文件（API、模型、阈值等）
│   ├── database.py          # 数据库操作
│   ├── feature_extractor.py # 调用AI模型进行描述生成和文本嵌入
│   ├── gui.py               # Streamlit Web界面代码
│   ├── reporter.py          # 生成PDF报告
│   ├── video_processor.py   # 视频处理：抽帧、图像聚类
│   └── ...
└── README.md
```

---

<a name="english-version"></a>

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/) [![Platform](https://img.shields.io/badge/Platform-Linux%20|%20macOS%20|%20Windows-lightgrey.svg)](https://en.wikipedia.org/wiki/Cross-platform) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/) [![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0) [![Topprism](https://img.shields.io/badge/Topprism-Data%20Intelligence-orange.svg)](https://www.topprismdata.com/)

# Video Ad Detector

This is a tool that uses a multimodal large model (Google Gemma 3N) to automatically detect whether a video contains known ad materials.

## Features

- **AI-Powered**: Utilizes the advanced multimodal large model `google/gemma-3n` to understand video frame content.
- **Efficient Pre-processing**: Employs **perceptual hashing** for **image clustering** of video frames before AI analysis. This effectively removes redundant frames, submitting only representative scenes to the AI and significantly boosting processing efficiency.
- **Semantic Understanding**: Converts video keyframes into rich **semantic descriptions**, serving as their content "fingerprint."
- **Precise Segment Detection**: Accurately identifies ad segments by performing **semantic similarity comparison** on representative frames.
- **Automated In-depth Reporting**: Generates detailed PDF reports for matched ads, featuring:
    - **Side-by-Side Evidence**: Displays recorded video screenshots next to source material screenshots.
    - **Precise Timestamps**: Annotates each comparison with exact timestamps.
    - **Duration Analysis**: Compares source material duration with the total matched duration in the recorded video.
- **Batch Processing**: Supports uploading and analyzing multiple videos at once.

## How It Works

The project workflow is divided into two phases:

1.  **Material Ingestion**:
    -   The user uploads known ad "material videos" via the UI.
    -   The system extracts **keyframes** (e.g., first, middle, last).
    -   For each keyframe, the **Gemma 3N model** (via LM Studio) generates a **detailed semantic description**.
    -   These descriptions are stored in a local database as a baseline.

2.  **Ad Detection**:
    -   The user submits one or more "recorded videos" for detection.
    -   The system performs **frame extraction** for each video.
    -   It then **clusters** all extracted frames using a perceptual hashing algorithm (`imagehash`), selecting a single **representative frame** for each visually similar group.
    -   For each **representative frame**, the **Gemma 3N model** generates a **semantic description**.
    -   The system calculates the **semantic similarity** (cosine similarity of text embeddings) between the representative frame's description and all material descriptions in the database.
    -   When similarity exceeds a preset threshold, the match is recorded.
    -   Finally, a PDF report is generated with side-by-side comparisons, timestamps, and duration analysis.

## Tech Stack

- **Core Framework**: Python
- **AI Model**: `google/gemma-3n` (multimodal understanding, via **LM Studio API**)
- **Text Embedding**: `nomic-ai/nomic-embed-text-v1.5` (semantic similarity, via **LM Studio API**)
- **Image Clustering**: ImageHash
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

4.  **LM Studio Configuration**:
    -   **Install and run LM Studio**.
    -   **Download and load a `gemma-3n` series multimodal model** in LM Studio.
    -   **Verify Model Identifier**: Ensure the `LM_STUDIO_CHAT_MODEL_NAME` in `video_ad_detector/config.py` exactly matches the **Model Identifier** you set for the model in LM Studio.
    -   **Enable OpenAI Compatible API**: Make sure the API server is running (typically at `http://localhost:1234`).
    -   **Configure Embedding Model**: Download and load a text embedding model (e.g., `nomic-ai/nomic-embed-text-v1.5`) and ensure `LM_STUDIO_EMBEDDING_MODEL_NAME` in `config.py` matches.

## Usage

1.  **Activate the virtual environment**:
    ```bash
    source .venv/bin/activate
    ```

2.  **Launch the application**:
    ```bash
    streamlit run run_gui.py
    ```
3.  **Instructions**:
    -   Use the "Add New Ad Material" section in the sidebar to ingest ad materials.
    -   Use the "Detect Ad in Videos" section to upload one or more videos for analysis.
    -   Results will be displayed in the main area, with options to generate a PDF report.

## Project Structure

```
.
├── video_ad_detector/
│   ├── __init__.py
│   ├── ad_detector.py       # Core detection logic: semantic analysis, segment matching
│   ├── app.py               # Entry point for CLI (GUI is primary)
│   ├── config.py            # Configuration (API, models, thresholds)
│   ├── database.py          # Database operations
│   ├── feature_extractor.py # AI model calls for descriptions and embeddings
│   ├── gui.py               # Streamlit Web UI code
│   ├── reporter.py          # PDF report generation
│   ├── video_processor.py   # Video processing: frame extraction, image clustering
│   └── ...
└── README.md
```
