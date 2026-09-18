import os
import io
import json
import pickle
import numpy as np
import scipy.io
from fastapi import APIRouter, File, UploadFile, HTTPException, status
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from feature_extraction import extract_features_from_signal, FEATURE_NAMES

# Initialize APIRouter for MediLens Integration
router = APIRouter(tags=["ECG Arrhythmia Classification"])

# Paths to trained model & label mapping
MODEL_PATH = os.path.join("model", "hist_gradient_boosting.pkl")
LABEL_MAP_PATH = os.path.join("processed_data", "label_mapping.json")

# Global lazy-loaded model & metadata cache
_model = None
_label_map = None

def get_model_and_labels():
    global _model, _label_map
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Trained model not found at path '{MODEL_PATH}'.")
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)

    if _label_map is None:
        if not os.path.exists(LABEL_MAP_PATH):
            raise RuntimeError(f"Label mapping JSON not found at path '{LABEL_MAP_PATH}'.")
        with open(LABEL_MAP_PATH, "r") as f:
            _label_map = json.load(f)["id_to_class"]

    return _model, _label_map

# Pydantic Schemas for Clean MediLens Frontend Consumption
class ClassProbability(BaseModel):
    class_id: int
    class_name: str
    probability: float

class HRVMetrics(BaseModel):
    rmssd: float
    sdnn: float
    pnn50: float
    lf_hf_ratio: float

class SignalSummary(BaseModel):
    duration_seconds: float = 10.0
    sampling_rate_hz: float = 360.0
    total_samples: int = 3600
    detected_r_peaks: int

class FeatureContribution(BaseModel):
    feature_name: str
    feature_value: float
    description: str

class ECGExplainability(BaseModel):
    estimated_heart_rate_bpm: Optional[float]
    hrv_metrics: HRVMetrics
    signal_summary: SignalSummary
    top_contributing_features: List[FeatureContribution]

class ECGPredictionResponse(BaseModel):
    status: str = "success"
    filename: str
    predicted_class: str
    class_id: int
    confidence_percentage: float
    top_5_probabilities: List[ClassProbability]
    explainability: ECGExplainability
    disclaimer: str

@router.post("", response_model=ECGPredictionResponse)
@router.post("/", response_model=ECGPredictionResponse)
async def predict_ecg_endpoint(file: UploadFile = File(...)):
    """
    POST /predict/ecg Endpoint
    Accepts an uploaded MATLAB .mat file containing a 3600-sample ECG recording.
    Returns predicted arrhythmia class, confidence, top-5 probabilities, and HRV explainability metrics.
    """
    if not file.filename.endswith(".mat"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Please upload a MATLAB '.mat' file."
        )

    try:
        model, label_map = get_model_and_labels()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model or label mapping: {str(e)}"
        )

    # 1. Read file bytes into memory
    try:
        contents = await file.read()
        mat_data = scipy.io.loadmat(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse MATLAB .mat file: {str(e)}"
        )

    if 'val' not in mat_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid ECG .mat file. Missing required array key 'val'."
        )

    # 2. Extract and validate single-channel signal
    raw_signal = mat_data['val'].squeeze().astype(np.float32)
    if raw_signal.ndim != 1 or len(raw_signal) != 3600:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid ECG signal length. Expected 3600 samples, but got {len(raw_signal)}."
        )

    # 3. Apply Z-score normalization
    mean_val = float(np.mean(raw_signal))
    std_val = float(np.std(raw_signal))
    normalized_signal = (raw_signal - mean_val) / (std_val + 1e-8)

    # 4. Extract 43 domain features
    try:
        features = extract_features_from_signal(normalized_signal)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feature extraction failed: {str(e)}"
        )

    features_2d = features.reshape(1, -1)

    # 5. Model Inference & Probabilities
    pred_class_id = int(model.predict(features_2d)[0])
    probabilities = model.predict_proba(features_2d)[0]

    predicted_label = label_map[str(pred_class_id)]
    confidence = round(float(probabilities[pred_class_id]) * 100.0, 2)

    # Build Top-5 Probabilities
    top5_indices = np.argsort(probabilities)[::-1][:5]
    top_5_probs = [
        ClassProbability(
            class_id=int(cid),
            class_name=label_map[str(cid)],
            probability=round(float(probabilities[cid]) * 100.0, 2)
        )
        for cid in top5_indices
    ]

    # 6. Generate Explainability Metadata
    # Feature index mapping
    feat_dict = dict(zip(FEATURE_NAMES, features))
    mean_rr = feat_dict.get("mean_rr_interval", 0.0)
    
    # Calculate Heart Rate (BPM): 60 * 360 / mean_rr_samples
    if mean_rr > 0:
        estimated_bpm = round(float(60.0 * 360.0 / mean_rr), 1)
    else:
        estimated_bpm = None

    hrv = HRVMetrics(
        rmssd=round(float(feat_dict.get("rmssd", 0.0)), 2),
        sdnn=round(float(feat_dict.get("std_rr_interval", 0.0)), 2),
        pnn50=round(float(feat_dict.get("pnn50", 0.0)), 2),
        lf_hf_ratio=round(float(feat_dict.get("lf_hf_ratio", 0.0)), 3)
    )

    sig_summary = SignalSummary(
        detected_r_peaks=int(feat_dict.get("num_peaks", 0))
    )

    top_features_list = [
        FeatureContribution(
            feature_name="autocorr_lag50",
            feature_value=round(float(feat_dict.get("autocorr_lag50", 0.0)), 4),
            description="Lag-50 signal autocorrelation (~138ms periodic correlation)"
        ),
        FeatureContribution(
            feature_name="p95",
            feature_value=round(float(feat_dict.get("p95", 0.0)), 4),
            description="95th percentile waveform amplitude (R-peak magnitude indicator)"
        ),
        FeatureContribution(
            feature_name="mean_rr_interval",
            feature_value=round(float(feat_dict.get("mean_rr_interval", 0.0)), 2),
            description="Average inter-beat interval length in samples"
        ),
        FeatureContribution(
            feature_name="pnn50",
            feature_value=round(float(feat_dict.get("pnn50", 0.0)), 2),
            description="Percentage of successive beat interval differences > 50 samples"
        ),
        FeatureContribution(
            feature_name="rmssd",
            feature_value=round(float(feat_dict.get("rmssd", 0.0)), 2),
            description="Root mean square of successive beat differences (HRV indicator)"
        )
    ]

    explainability = ECGExplainability(
        estimated_heart_rate_bpm=estimated_bpm,
        hrv_metrics=hrv,
        signal_summary=sig_summary,
        top_contributing_features=top_features_list
    )

    disclaimer = (
        "⚠️ MEDICAL DISCLAIMER: This prediction is generated by an automated AI model "
        "for research, education, and hackathon demonstration purposes. "
        "It is NOT intended, validated, or certified for clinical diagnosis."
    )

    return ECGPredictionResponse(
        filename=file.filename,
        predicted_class=predicted_label,
        class_id=pred_class_id,
        confidence_percentage=confidence,
        top_5_probabilities=top_5_probs,
        explainability=explainability,
        disclaimer=disclaimer
    )
