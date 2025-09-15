"""
FastAPI deployment for CICIDS2017 Network Intrusion Detection Models
Enhanced with in-memory storage for responses
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import os
import warnings
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import uuid
from collections import deque


# Suppress sklearn warnings about feature names
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message="X has feature names, but")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# Initialize FastAPI app
app = FastAPI(
    title="Network Intrusion Detection API",
    description="Deploy trained ML models for network traffic classification",
    version="1.0.0"
)

# Allow frontend origin(s)
origins = [
    "http://localhost:3000",  # Next.js dev server
    "http://127.0.0.1:3000", # just in case
    # add your production URL when deployed, e.g. "https://myapp.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # or restrict to ["GET", "POST"]
    allow_headers=["*"],  # or restrict
)


# Define feature names (based on your cleaned dataset)
FEATURE_NAMES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Length of Fwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "PSH Flag Count", "ACK Flag Count", "Average Packet Size",
    "Subflow Fwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean", "Active Max", "Active Min",
    "Idle Mean", "Idle Max", "Idle Min"
]

# Target classes
TARGET_CLASSES = [
    "Normal Traffic", "DoS", "DDoS", "Port Scanning", "Brute Force", "Web Attacks", "Bots"
]

# In-memory storage for responses
# Using deque with maxlen to prevent unlimited memory growth
MAX_STORED_RESPONSES = 1000
stored_responses = deque(maxlen=MAX_STORED_RESPONSES)
latest_response = None

class NetworkFeatures(BaseModel):
    """Input model for network traffic features"""
    destination_port: float = Field(..., description="Destination Port")
    flow_duration: float = Field(..., description="Flow Duration")
    total_fwd_packets: float = Field(..., description="Total Forward Packets")
    total_length_fwd_packets: float = Field(..., description="Total Length of Forward Packets")
    fwd_packet_length_max: float = Field(..., description="Forward Packet Length Max")
    fwd_packet_length_min: float = Field(..., description="Forward Packet Length Min")
    fwd_packet_length_mean: float = Field(..., description="Forward Packet Length Mean")
    fwd_packet_length_std: float = Field(..., description="Forward Packet Length Std")
    bwd_packet_length_max: float = Field(..., description="Backward Packet Length Max")
    bwd_packet_length_min: float = Field(..., description="Backward Packet Length Min")
    bwd_packet_length_mean: float = Field(..., description="Backward Packet Length Mean")
    bwd_packet_length_std: float = Field(..., description="Backward Packet Length Std")
    flow_bytes_per_s: float = Field(..., description="Flow Bytes/s")
    flow_packets_per_s: float = Field(..., description="Flow Packets/s")
    flow_iat_mean: float = Field(..., description="Flow IAT Mean")
    flow_iat_std: float = Field(..., description="Flow IAT Std")
    flow_iat_max: float = Field(..., description="Flow IAT Max")
    flow_iat_min: float = Field(..., description="Flow IAT Min")
    fwd_iat_total: float = Field(..., description="Forward IAT Total")
    fwd_iat_mean: float = Field(..., description="Forward IAT Mean")
    fwd_iat_std: float = Field(..., description="Forward IAT Std")
    fwd_iat_max: float = Field(..., description="Forward IAT Max")
    fwd_iat_min: float = Field(..., description="Forward IAT Min")
    bwd_iat_total: float = Field(..., description="Backward IAT Total")
    bwd_iat_mean: float = Field(..., description="Backward IAT Mean")
    bwd_iat_std: float = Field(..., description="Backward IAT Std")
    bwd_iat_max: float = Field(..., description="Backward IAT Max")
    bwd_iat_min: float = Field(..., description="Backward IAT Min")
    fwd_header_length: float = Field(..., description="Forward Header Length")
    bwd_header_length: float = Field(..., description="Backward Header Length")
    fwd_packets_per_s: float = Field(..., description="Forward Packets/s")
    bwd_packets_per_s: float = Field(..., description="Backward Packets/s")
    min_packet_length: float = Field(..., description="Min Packet Length")
    max_packet_length: float = Field(..., description="Max Packet Length")
    packet_length_mean: float = Field(..., description="Packet Length Mean")
    packet_length_std: float = Field(..., description="Packet Length Std")
    packet_length_variance: float = Field(..., description="Packet Length Variance")
    fin_flag_count: float = Field(..., description="FIN Flag Count")
    psh_flag_count: float = Field(..., description="PSH Flag Count")
    ack_flag_count: float = Field(..., description="ACK Flag Count")
    average_packet_size: float = Field(..., description="Average Packet Size")
    subflow_fwd_bytes: float = Field(..., description="Subflow Forward Bytes")
    init_win_bytes_forward: float = Field(..., description="Init Win Bytes Forward")
    init_win_bytes_backward: float = Field(..., description="Init Win Bytes Backward")
    act_data_pkt_fwd: float = Field(..., description="Active Data Packets Forward")
    min_seg_size_forward: float = Field(..., description="Min Segment Size Forward")
    active_mean: float = Field(..., description="Active Mean")
    active_max: float = Field(..., description="Active Max")
    active_min: float = Field(..., description="Active Min")
    idle_mean: float = Field(..., description="Idle Mean")
    idle_max: float = Field(..., description="Idle Max")
    idle_min: float = Field(..., description="Idle Min")

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    model_name: str
    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictionResponse]
    summary: Dict[str, int]

class StoredPredictionResponse(BaseModel):
    """Enhanced response model with metadata for storage"""
    id: str
    timestamp: datetime
    input_features: NetworkFeatures
    predictions: List[PredictionResponse]
    endpoint_used: str

class PredictionHistoryResponse(BaseModel):
    """Response model for prediction history"""
    total_predictions: int
    predictions: List[StoredPredictionResponse]

# Global variables for models
models = {}
scaler = None

def load_models():
    """Load all trained models and scaler"""
    global models, scaler

    models_path = Path("../Models")

    try:
        # Load scaler
        scaler_path = models_path / "robust_scaler.joblib"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            print("✓ Scaler loaded successfully")
        else:
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")

        # Load models
        model_files = {
            "Random Forest": "random_forest.joblib",
            "XGBoost": "xgboost.joblib",
            "KNN": "knn_model.joblib"
        }

        for model_name, filename in model_files.items():
            model_path = models_path / filename
            if model_path.exists():
                models[model_name] = joblib.load(model_path)
                print(f"✓ {model_name} model loaded successfully")
            else:
                print(f"⚠ Warning: {model_name} model not found at {model_path}")

        if not models:
            raise FileNotFoundError("No models could be loaded")

    except Exception as e:
        print(f"Error loading models: {e}")
        raise

def preprocess_features(features: NetworkFeatures) -> pd.DataFrame:
    """
    Preprocess input features following the same pipeline as training
    """
    # Convert features to list in the correct order
    feature_values = [
        features.destination_port, features.flow_duration, features.total_fwd_packets,
        features.total_length_fwd_packets, features.fwd_packet_length_max, features.fwd_packet_length_min,
        features.fwd_packet_length_mean, features.fwd_packet_length_std, features.bwd_packet_length_max,
        features.bwd_packet_length_min, features.bwd_packet_length_mean, features.bwd_packet_length_std,
        features.flow_bytes_per_s, features.flow_packets_per_s, features.flow_iat_mean, features.flow_iat_std,
        features.flow_iat_max, features.flow_iat_min, features.fwd_iat_total, features.fwd_iat_mean,
        features.fwd_iat_std, features.fwd_iat_max, features.fwd_iat_min, features.bwd_iat_total,
        features.bwd_iat_mean, features.bwd_iat_std, features.bwd_iat_max, features.bwd_iat_min,
        features.fwd_header_length, features.bwd_header_length, features.fwd_packets_per_s,
        features.bwd_packets_per_s, features.min_packet_length, features.max_packet_length,
        features.packet_length_mean, features.packet_length_std, features.packet_length_variance,
        features.fin_flag_count, features.psh_flag_count, features.ack_flag_count,
        features.average_packet_size, features.subflow_fwd_bytes, features.init_win_bytes_forward,
        features.init_win_bytes_backward, features.act_data_pkt_fwd, features.min_seg_size_forward,
        features.active_mean, features.active_max, features.active_min, features.idle_mean,
        features.idle_max, features.idle_min
    ]

    # Create DataFrame with proper feature names
    features_df = pd.DataFrame([feature_values], columns=FEATURE_NAMES)

    # Handle infinite values (replace with large finite values)
    features_df = features_df.replace([np.inf, -np.inf], [1e10, -1e10])

    # Check for NaN values
    if features_df.isna().any().any():
        raise ValueError("Input contains NaN values")

    # Scale features using the loaded scaler
    if scaler is None:
        raise ValueError("Scaler not loaded")

    # Transform and return as DataFrame to preserve feature names
    scaled_array = scaler.transform(features_df)
    scaled_df = pd.DataFrame(scaled_array, columns=FEATURE_NAMES)

    return scaled_df

def get_prediction_with_probabilities(model, scaled_features, model_name: str):
    """Get prediction and probabilities from a model"""
    try:
        # Get prediction
        prediction = model.predict(scaled_features)[0]

        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(scaled_features)[0]

            # Handle XGBoost case (might return numeric labels)
            if model_name == "XGBoost" and isinstance(prediction, (int, np.integer)):
                # Map numeric prediction back to string
                label_mapping = {
                    0: 'Normal Traffic', 1: 'DoS', 2: 'DDoS', 3: 'Port Scanning',
                    4: 'Brute Force', 5: 'Web Attacks', 6: 'Bots'
                }
                prediction = label_mapping.get(prediction, f"Unknown_{prediction}")

                # Create probability dictionary
                prob_dict = {}
                for i, prob in enumerate(probabilities):
                    class_name = label_mapping.get(i, f"Unknown_{i}")
                    prob_dict[class_name] = float(prob)
            else:
                # For Random Forest and KNN
                classes = model.classes_
                prob_dict = {str(cls): float(prob) for cls, prob in zip(classes, probabilities)}

            max_prob = max(prob_dict.values())
        else:
            # If no probabilities available, create a mock confidence
            prob_dict = {str(prediction): 1.0}
            max_prob = 1.0

        return str(prediction), max_prob, prob_dict

    except Exception as e:
        raise ValueError(f"Error making prediction with {model_name}: {str(e)}")

def store_prediction_response(features: NetworkFeatures, predictions: List[PredictionResponse], endpoint: str):
    """Store prediction response in memory"""
    global latest_response

    stored_response = StoredPredictionResponse(
        id=str(uuid.uuid4()),
        timestamp=datetime.now(),
        input_features=features,
        predictions=predictions,
        endpoint_used=endpoint
    )

    # Add to deque (automatically removes oldest if at max capacity)
    stored_responses.append(stored_response)

    # Update latest response
    latest_response = stored_response

    return stored_response

# Load models on startup
load_models()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Network Intrusion Detection API",
        "available_models": list(models.keys()),
        "target_classes": TARGET_CLASSES,
        "num_features": len(FEATURE_NAMES),
        "stored_predictions": len(stored_responses)
    }

@app.get("/models")
async def list_models():
    """List all available models"""
    model_info = {}
    for name, model in models.items():
        model_info[name] = {
            "type": type(model).__name__,
            "loaded": True
        }
    return {"models": model_info}

@app.post("/predict/all", response_model=List[PredictionResponse])
async def predict_all_models(features: NetworkFeatures):
    """
    Make predictions using all available models and store the response
    """
    try:

        #print("Received packet features:", features.dict())
        # Preprocess features once
        scaled_features = preprocess_features(features)

        predictions = []
        for model_name, model in models.items():
            try:
                predicted_class, confidence, probabilities = get_prediction_with_probabilities(
                    model, scaled_features, model_name
                )

                predictions.append(PredictionResponse(
                    model_name=model_name,
                    predicted_class=predicted_class,
                    confidence=confidence,
                    all_probabilities=probabilities
                ))
            except Exception as e:
                print(f"Error with model {model_name}: {e}")
                continue

        if not predictions:
            raise HTTPException(status_code=500, detail="No models could make predictions")

        # Store the response in memory
        store_prediction_response(features, predictions, "predict/all")

        return predictions

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/predict/{model_name}", response_model=PredictionResponse)
async def predict(model_name: str, features: NetworkFeatures):
    """
    Make prediction using specified model and store the response
    """
    # Check if model exists
    if model_name not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available models: {list(models.keys())}"
        )

    try:
        # Preprocess features
        scaled_features = preprocess_features(features)

        # Get model
        model = models[model_name]

        # Make prediction
        predicted_class, confidence, probabilities = get_prediction_with_probabilities(
            model, scaled_features, model_name
        )

        prediction_response = PredictionResponse(
            model_name=model_name,
            predicted_class=predicted_class,
            confidence=confidence,
            all_probabilities=probabilities
        )

        # Store the response in memory
        store_prediction_response(features, [prediction_response], f"predict/{model_name}")

        return prediction_response

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/predict/batch/{model_name}", response_model=BatchPredictionResponse)
async def predict_batch(model_name: str, features_list: List[NetworkFeatures]):
    """
    Make batch predictions using specified model
    """
    # Check if model exists
    if model_name not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available models: {list(models.keys())}"
        )

    try:
        model = models[model_name]
        predictions = []
        summary = {}

        for features in features_list:
            # Preprocess features
            scaled_features = preprocess_features(features)

            # Make prediction
            predicted_class, confidence, probabilities = get_prediction_with_probabilities(
                model, scaled_features, model_name
            )

            prediction_response = PredictionResponse(
                model_name=model_name,
                predicted_class=predicted_class,
                confidence=confidence,
                all_probabilities=probabilities
            )

            predictions.append(prediction_response)

            # Store individual prediction (for batch, we store each one separately)
            store_prediction_response(features, [prediction_response], f"predict/batch/{model_name}")

            # Update summary
            summary[predicted_class] = summary.get(predicted_class, 0) + 1

        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# New GET endpoints for retrieving stored responses

@app.get("/predictions/latest", response_model=Optional[StoredPredictionResponse])
async def get_latest_prediction():
    """
    Get the latest prediction response
    """
    if latest_response is None:
        return None
    return latest_response

@app.get("/predictions/history", response_model=PredictionHistoryResponse)
async def get_prediction_history(limit: Optional[int] = None):
    """
    Get prediction history (latest first)
    """
    # Convert deque to list and reverse to get latest first
    all_predictions = list(reversed(stored_responses))

    # Apply limit if specified
    if limit is not None and limit > 0:
        all_predictions = all_predictions[:limit]

    return PredictionHistoryResponse(
        total_predictions=len(stored_responses),
        predictions=all_predictions
    )

@app.get("/predictions/{prediction_id}", response_model=Optional[StoredPredictionResponse])
async def get_prediction_by_id(prediction_id: str):
    """
    Get a specific prediction by ID
    """
    for prediction in stored_responses:
        if prediction.id == prediction_id:
            return prediction

    raise HTTPException(status_code=404, detail=f"Prediction with ID '{prediction_id}' not found")

@app.get("/predictions/stats")
async def get_prediction_stats():
    """
    Get statistics about stored predictions
    """
    if not stored_responses:
        return {
            "total_predictions": 0,
            "class_distribution": {},
            "model_usage": {},
            "endpoint_usage": {},
            "time_range": None
        }

    # Calculate statistics
    class_counts = {}
    model_counts = {}
    endpoint_counts = {}
    timestamps = []

    for stored_pred in stored_responses:
        timestamps.append(stored_pred.timestamp)
        endpoint_counts[stored_pred.endpoint_used] = endpoint_counts.get(stored_pred.endpoint_used, 0) + 1

        for pred in stored_pred.predictions:
            model_counts[pred.model_name] = model_counts.get(pred.model_name, 0) + 1
            class_counts[pred.predicted_class] = class_counts.get(pred.predicted_class, 0) + 1

    return {
        "total_predictions": len(stored_responses),
        "class_distribution": class_counts,
        "model_usage": model_counts,
        "endpoint_usage": endpoint_counts,
        "time_range": {
            "earliest": min(timestamps),
            "latest": max(timestamps)
        } if timestamps else None
    }

@app.delete("/predictions/clear")
async def clear_prediction_history():
    """
    Clear all stored predictions
    """
    global latest_response
    stored_responses.clear()
    latest_response = None

    return {
        "message": "Prediction history cleared successfully",
        "remaining_predictions": len(stored_responses)
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "scaler_loaded": scaler is not None,
        "available_models": list(models.keys()),
        "stored_predictions": len(stored_responses),
        "max_storage_capacity": MAX_STORED_RESPONSES
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)