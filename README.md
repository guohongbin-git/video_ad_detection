# 视频广告检测器

这是一个使用多模态大模型（Google Gemma 3）自动检测视频中是否包含已知广告素材的工具。

## 功能特性

- **AI驱动**: 利用先进的多模态大模型 `google/gemma-3-4b-it` 理解视频帧内容。
- **特征向量化**: 将视频关键帧转换为高维特征向量，作为其内容的“指纹”。
- **高精度匹配**: 通过计算余弦相似度，精确比对实拍视频与广告素材的特征“指紋”。
- **自动化报告**: 如果检测到匹配的广告，能自动生成包含截图证据的PDF报告。
- **Web界面**: 使用 Streamlit 构建了简单易用的图形用户界面。

## 工作原理

项目的工作流程分为两个阶段：

1.  **素材入库**:
    -   用户通过界面上传已知的广告“素材视频”。
    -   系统使用 `gemma-3` 模型提取视频关键帧的特征向量。
    -   所有素材的特征向量被存储在本地的 SQLite 数据库 (`metadata.db`) 中。

2.  **广告检测**:
    -   用户上传需要检测的“实拍视频”。
    -   系统同样提取该视频关键帧的特征向量。
    -   程序将实拍视频的特征向量与数据库中存储的所有素材向量进行余弦相似度计算。
    -   如果相似度超过预设阈值（默认为0.85），则判定为检测到广告，并生成一份详细的PDF报告。

## 技术栈

- **核心框架**: Python, Streamlit
- **AI模型**: `google/gemma-3-4b-it` (通过 `transformers` 库)
- **机器学习**: PyTorch, scikit-learn
- **视频/图像处理**: OpenCV, Pillow
- **报告生成**: ReportLab

## 安装与配置

1.  **克隆本项目**:
    ```bash
    # (本项目很快就会被上传到GitHub)
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

1.  **启动应用**:
    ```bash
    streamlit run video_ad_detector/gui.py
    ```

2.  **上传素材**:
    -   在打开的网页界面中，使用“上传广告素材”功能，选择一个或多个已知的广告视频。
    -   等待程序处理完成。这会将素材的特征录入数据库。

3.  **检测视频**:
    -   处理完所有素材后，使用“上传实拍视频进行检测”功能，选择您想要检测的视频。
    -   程序会自动进行比对，并在界面上显示结果。如果检测到广告，相关的PDF报告会自动生成在 `video_ad_detector/reports/` 目录下。

## 项目结构

```
.
├── video_ad_detector/
│   ├── __init__.py
│   ├── ad_detector.py       # 核心检测逻辑（余弦相似度计算）
│   ├── app.py               # (可能未使用)
│   ├── config.py            # 配置文件（模型路径、阈值等）
│   ├── database.py          # 数据库操作
│   ├── feature_extractor.py # 调用大模型提取特征向量
│   ├── gui.py               # Streamlit Web界面代码
│   ├── reporter.py          # 生成PDF报告
│   ├── video_processor.py   # 视频处理（提取关键帧）
│   └── database/
│       └── metadata.db      # 存储特征的数据库
└── README.md
```
