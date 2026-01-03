import torch
import mlflow
import mlflow.pytorch
from torchvision import transforms
from PIL import Image
import os

# MLflow Tracking Server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# ===== CONFIG =====
MODEL_NAME = "TrueSight-Deepfake-Detection"
MODEL_URI = f"models:/{MODEL_NAME}/1"
IMAGE_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict(image_path, preloaded_model=None):
    """
    Predict REAL/FAKE for a single image.
    
    If preloaded_model is provided, it uses that model
    (used for video optimization).
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    # Load model only if not passed externally
    if preloaded_model is None:
        print("Loading model from MLflow...")
        model = mlflow.pytorch.load_model(MODEL_URI)
    else:
        model = preloaded_model

    model.eval()

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)[0]

    real_conf = float(probs[0])
    fake_conf = float(probs[1])

    label = "FAKE" if fake_conf > real_conf else "REAL"
    confidence = round(max(fake_conf, real_conf) * 100, 2)

    return {
        "label": label,
        "confidence": confidence,
        "real_prob": round(real_conf * 100, 2),
        "fake_prob": round(fake_conf * 100, 2)
    }


if __name__ == "__main__":
    # Simple test image
    test_image = "data/processed/val/fake/TSF2.jpeg"
    result = predict(test_image)
    print("\n Prediction Result:")
    print(result)
