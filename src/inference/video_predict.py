import cv2
import os
import mlflow.pytorch
from predict import predict

MODEL_URI = "models:/TrueSight-Deepfake-Detection/1"

def predict_video(video_path, frame_skip=10):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"❌ Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("❌ Cannot open video file!")

    print("Loading model once...")
    model = mlflow.pytorch.load_model(MODEL_URI)
    model.eval()

    results = []
    frame_count = 0

    print("Processing video frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process every 10th frame
        if frame_count % frame_skip == 0:
            temp_path = "temp_frame.jpg"
            cv2.imwrite(temp_path, frame)

            result = predict(temp_path, preloaded_model=model)
            results.append(result["label"])

            print(f"Frame {frame_count}: {result}")

        frame_count += 1

    cap.release()
    if os.path.exists("temp_frame.jpg"):
        os.remove("temp_path.jpg")

    fake = results.count("FAKE")
    real = results.count("REAL")

    final_label = "FAKE" if fake > real else "REAL"
    confidence = round((max(fake, real) / len(results)) * 100, 2)

    return {
        "final_label": final_label,
        "fake_frame_count": fake,
        "real_frame_count": real,
        "confidence": confidence
    }


if __name__ == "__main__":
    video_path = r"D:\TrueSight\data\videos\TSV1.mp4"
    output = predict_video(video_path)
    print("\n Video Prediction Summary:")
    print(output)
