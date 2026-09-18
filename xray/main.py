
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
from pathlib import Path
import io
import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent

# Load trained X-ray model
densenet = tf.keras.models.load_model(
    BASE_DIR / "densenet121_feature_extractor.keras"
)

# Load NORMAL reference data
normal_center = np.load(BASE_DIR / "normal_center.npy")
normal_threshold = float(
    (BASE_DIR / "normal_threshold.txt").read_text().strip()
)

app = FastAPI(
    title="MediLens X-Ray AI API",
    version="1.0.0",
    description="AI-assisted Chest X-ray screening using DenseNet121 anomaly detection"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "MediLens X-Ray AI API",
        "status": "running"
    }


@app.post("/predict/xray")
async def predict_xray(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        image = Image.open(io.BytesIO(image_bytes)).convert("L")

        image = ImageOps.pad(
            image,
            (224, 224),
            method=Image.Resampling.LANCZOS,
            color=0
        )

        image_array = np.array(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=-1)
        image_array = np.expand_dims(image_array, axis=0)

        image_rgb = tf.image.grayscale_to_rgb(
            tf.convert_to_tensor(image_array)
        )

        features = densenet.predict(image_rgb, verbose=0)

        anomaly_distance = float(
            np.linalg.norm(features[0] - normal_center)
        )

        if anomaly_distance > normal_threshold:
            prediction = "ABNORMAL_PATTERN"
        else:
            prediction = "NORMAL_LIKE"

        return {
            "modality": "chest_xray",
            "filename": file.filename,
            "prediction": prediction,
            "anomaly_score": round(anomaly_distance, 4),
            "threshold": round(normal_threshold, 4)
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
