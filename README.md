# 👁️ TrueSight — Deepfake Image & Video Detection System  
**See what’s real.**

TrueSight is an AI/ML-based deepfake detection system designed to identify whether an image or video is **REAL or FAKE** using deep learning and computer vision techniques.  
The project demonstrates an **end-to-end Machine Learning + MLOps pipeline**, including training, experiment tracking, model versioning, and inference for both images and videos.

---

## 🚀 Project Overview

Deepfake media poses serious risks in digital communication, social media, journalism, and security systems.  
**TrueSight** addresses this challenge by:

- Detecting deepfake images using a CNN-based deep learning model  
- Performing frame-by-frame analysis for videos  
- Aggregating frame-level predictions into a final video-level decision  
- Tracking experiments and managing models using **MLflow**

This project is built following **industry best practices** and is suitable for **AI/ML, Computer Vision, and MLOps roles**.

---

## 🧠 Key Features

- ✅ Deepfake detection for **images and videos**
- ✅ CNN-based classification using **ResNet-18**
- ✅ Frame-level video analysis using **OpenCV**
- ✅ Optimized inference by loading the model **once per video**
- ✅ **MLflow integration** for:
  - Experiment tracking
  - Metrics logging
  - Model versioning
- ✅ Modular, scalable, and production-ready project structure
- ✅ API-ready design (FastAPI compatible)

---

## 🏗️ Project Architecture

## Project Structure

```
TrueSight/
├── README.md
├── requirements.txt
├── api/                    # (Future) API endpoints
├── data/
│   ├── processed/          # Training/validation data
│   └── videos/             # Video files for testing
├── mlartifacts/            # MLflow artifacts
├── mlruns/                 # MLflow runs
├── models/                 # Registered models
├── notebooks/              # Jupyter notebooks (future)
├── src/
│   ├── data_prep/          # Data preparation scripts
│   ├── inference/          # Prediction scripts
│   │   ├── predict.py      # Image prediction
│   │   └── video_predict.py # Video prediction
│   ├── models/             # Model training
│   │   └── train_image_model.py
│   └── utils/              # Utility functions
└── tests/                  # Unit tests
```


---

## 🧪 Model & Training Details

- **Model:** ResNet-18 (Convolutional Neural Network)
- **Framework:** PyTorch
- **Task:** Binary Classification (REAL vs FAKE)
- **Epochs:** 2 (baseline training)
- **Loss Function:** Cross-Entropy Loss
- **Optimizer:** Adam
- **Experiment Tracking:** MLflow

All training metrics, parameters, and artifacts are logged in **MLflow** to ensure reproducibility and experiment comparison.

---

## 🎥 Image & Video Inference

### 🖼️ Image Prediction
- Input: Single image
- Output: REAL / FAKE with confidence score

Example output:
{
"label": "FAKE",
"confidence": 64.57,
"real_prob": 35.43,
"fake_prob": 64.57
}


### 🎬 Video Prediction
- Extracts frames at fixed intervals
- Predicts each frame individually
- Aggregates results into a final decision

Example output:
{
"final_label": "REAL",
"fake_frame_count": 6,
"real_frame_count": 23,
"confidence": 20.69
}


---

## ⚙️ How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Swapnil-Ncode/TrueSight-A-Deepfake-Image-and-Video-Detection-System.git
cd TrueSight-A-Deepfake-Image-and-Video-Detection-System

2️⃣ Create Virtual Environment

Python 3.10 is recommended

python -m venv venv
venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Start MLflow UI
mlflow ui

Access MLflow at:
http://127.0.0.1:5000

5️⃣ Train the Model
python src/models/train_image_model.py

6️⃣ Run Image Prediction
python src/inference/predict.py

7️⃣ Run Video Prediction
python src/inference/video_predict.py

🛠️ Tech Stack

Programming Language: Python
Deep Learning: PyTorch, Torchvision
Computer Vision: OpenCV
MLOps: MLflow
Data Processing: NumPy, PIL
Version Control: Git & GitHub

