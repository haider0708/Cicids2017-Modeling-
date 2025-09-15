# Network Intrusion Detection API

This FastAPI application deploys your trained machine learning models for real-time network traffic classification and intrusion detection.

## Features

- **Multiple Model Support**: Deploy Random Forest, XGBoost, and KNN models simultaneously
- **Real-time Predictions**: Get instant classifications for network traffic
- **Batch Processing**: Process multiple samples at once
- **Automatic Preprocessing**: Handles feature scaling using your trained RobustScaler
- **RESTful API**: Easy integration with existing systems
- **Interactive Documentation**: Swagger UI for testing

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_api.txt
```

### 2. Start the API Server

```bash
# Using the startup script (recommended)
./start_api.sh

# Or manually
python fastapi_deployment.py
```

### 3. Access the API

- **API Base URL**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **API Schema**: http://localhost:8000/openapi.json

## API Endpoints

### Health Check
```http
GET /health
```
Check if the API is running and models are loaded.

### List Models
```http
GET /models
```
Get information about all loaded models.

### Single Prediction
```http
POST /predict/{model_name}
```
Make a prediction using a specific model (Random Forest, XGBoost, or KNN).

### All Models Prediction
```http
POST /predict/all
```
Get predictions from all available models for comparison.

### Batch Prediction
```http
POST /predict/batch/{model_name}
```
Process multiple samples with a specific model.

## Input Format

The API expects network traffic features in JSON format. All 51 features are required:

```json
{
  "destination_port": 80.0,
  "flow_duration": 120000.0,
  "total_fwd_packets": 10.0,
  "total_length_fwd_packets": 1500.0,
  // ... (all 51 features)
  "idle_min": 100.0
}
```

### Required Features

1. **Destination Port**: Network destination port
2. **Flow Duration**: Duration of the network flow
3. **Packet Counts**: Forward/backward packet statistics
4. **Packet Lengths**: Size statistics for packets
5. **Flow Metrics**: Bytes/second, packets/second
6. **Inter-Arrival Times**: Timing between packets
7. **Header Information**: Protocol header data
8. **Flag Counts**: TCP flag statistics
9. **Window Sizes**: TCP window information
10. **Activity Metrics**: Active/idle time statistics

## Output Format

### Single Prediction Response
```json
{
  "model_name": "Random Forest",
  "predicted_class": "Normal Traffic",
  "confidence": 0.8542,
  "all_probabilities": {
    "Normal Traffic": 0.8542,
    "DoS": 0.0821,
    "DDoS": 0.0312,
    "Port Scanning": 0.0198,
    "Brute Force": 0.0089,
    "Web Attacks": 0.0032,
    "Bots": 0.0006
  }
}
```

### Attack Types

The models can classify traffic into these categories:

- **Normal Traffic**: Benign network activity
- **DoS**: Denial of Service attacks
- **DDoS**: Distributed Denial of Service attacks
- **Port Scanning**: Network reconnaissance
- **Brute Force**: Password/authentication attacks
- **Web Attacks**: SQL injection, XSS, etc.
- **Bots**: Botnet activity

## Testing

Use the provided test script to verify the API:

```bash
# Start the API server first
python fastapi_deployment.py

# Run tests in another terminal
python test_api.py
```

## Integration Examples

### Python Client
```python
import requests

# Prepare sample data
data = {
    "destination_port": 80.0,
    "flow_duration": 120000.0,
    # ... include all 51 features
}

# Make prediction
response = requests.post(
    "http://localhost:8000/predict/Random Forest",
    json=data
)

result = response.json()
print(f"Predicted class: {result['predicted_class']}")
print(f"Confidence: {result['confidence']}")
```

### curl Example
```bash
curl -X POST "http://localhost:8000/predict/Random Forest" \
     -H "Content-Type: application/json" \
     -d @sample_data.json
```

## Production Deployment

For production use, consider:

1. **Use Gunicorn**: Deploy with multiple workers
```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker fastapi_deployment:app
```

2. **Add Authentication**: Implement API keys or OAuth
3. **Rate Limiting**: Prevent API abuse
4. **Monitoring**: Add logging and metrics
5. **Load Balancing**: Use nginx or similar
6. **SSL/TLS**: Enable HTTPS

## Troubleshooting

### Model Loading Issues
- Ensure all model files exist in the `Models/` directory
- Check that models were trained with compatible scikit-learn versions
- Verify the scaler file is present

### Memory Issues
- Models require significant RAM (especially Random Forest)
- Consider loading only one model at a time for memory-constrained environments

### Performance Optimization
- Use batch predictions for multiple samples
- Consider model optimization (pruning, quantization)
- Cache predictions for repeated requests

## File Structure

```
├── fastapi_deployment.py      # Main API application
├── test_api.py               # Test scripts
├── start_api.sh              # Startup script
├── requirements_api.txt      # Python dependencies
├── Models/                   # Trained models directory
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   ├── knn_model.joblib
│   └── robust_scaler.joblib
└── README_API.md            # This documentation
```

## Support

For issues or questions:
1. Check the interactive documentation at `/docs`
2. Verify all dependencies are installed
3. Ensure models are properly trained and saved
4. Check server logs for error details
