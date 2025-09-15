# CICIDS2017 Machine Learning Project

## Overview

This project focuses on network intrusion detection using the **CICIDS2017 dataset**, implementing multiple machine learning algorithms including K-Nearest Neighbors (KNN), Random Forest, and XGBoost. The project includes complete data preprocessing pipelines, model training notebooks, and a FastAPI deployment for real-time predictions.

## Dataset Information

The **CICIDS2017 dataset** (Canadian Institute for Cybersecurity Intrusion Detection System) is a comprehensive network traffic dataset containing both benign and malicious network flows. It includes various types of cyber attacks such as:

- Brute Force attacks
- Heartbleed
- Botnet
- DoS/DDoS attacks
- Web attacks
- Infiltration attacks
- Port scans

This dataset is widely used for evaluating network-based intrusion detection systems and provides realistic network traffic data for cybersecurity research.

## Project Structure

The project contains the following components:

- **Data Processing Pipeline**: Complete preprocessing workflows for the CICIDS2017 dataset
- **Machine Learning Models**: Implementation of KNN, Random Forest, and XGBoost classifiers
- **Jupyter Notebooks**: Detailed analysis and model development notebooks
- **FastAPI Deployment**: RESTful API for model inference and predictions
- **Requirements Management**: All dependencies listed in requirements.txt

## Prerequisites

- Python 3.11.9
- Git
- Virtual environment support (venv)

## Installation and Setup

### Step 1: Install Python 3.11.9

Make sure you have Python 3.11.9 installed on your system. You can download it from [python.org](https://www.python.org/downloads/) or use a version manager like pyenv.

### Step 2: Download the CICIDS2017 Dataset

**Important**: The dataset files are not included in the repository due to their large size. You need to download them separately from Kaggle:

1. Visit the CICIDS2017 dataset on Kaggle: [https://www.kaggle.com/datasets/cicdataset/cicids2017](https://www.kaggle.com/datasets/cicdataset/cicids2017)
2. Download all the CSV files from the dataset
3. Create a folder named `datasets_original` in the project root directory
4. Copy all the downloaded CSV files into the `datasets_original` folder

The folder structure should look like this:
```
Cicids2017-Modeling-/
├── datasets_original/
│   ├── Monday-WorkingHours.pcap_ISCX.csv
│   ├── Tuesday-WorkingHours.pcap_ISCX.csv
│   ├── Wednesday-workingHours.pcap_ISCX.csv
│   ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
│   ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
│   ├── Friday-WorkingHours-Morning.pcap_ISCX.csv
│   └── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
├── FASTAPI/
└── ... (other project files)
```

### Step 3: Create and Activate Virtual Environment

```bash
# Create virtual environment
python3.11 -m venv cicids2017_env

# Activate virtual environment
# On Windows:
cicids2017_env\Scripts\activate

# On macOS/Linux:
source cicids2017_env/bin/activate
```

### Step 4: Clone the Repository

```bash
git clone https://github.com/haider0708/Cicids2017-Modeling-.git
cd Cicids2017-Modeling-
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Generate Models

**Important**: The trained models are not included in the repository. You need to train them locally by running the pipeline scripts:

1. Create the necessary folders in the project root directory:
```bash
mkdir Models
mkdir Datasets
```

2. Run the pipeline scripts in the correct order:
```bash
# First, run the data processing script
python Pipeline/processing.py

# Then, run the modeling script
python Pipeline/modeling.py
```

**Note**: 
- The `processing.py` script will process the raw data from `datasets_original` and save the cleaned/processed dataset in the `Datasets` folder
- The modeling process may take up to 15 minutes depending on your device specifications. This includes training KNN, Random Forest, and XGBoost models on the processed CICIDS2017 dataset.

The folder structure after processing and model generation should look like this:
```
Cicids2017-Modeling-/
├── datasets_original/
│   └── ... (original CSV files from Kaggle)
├── Datasets/
│   └── ... (processed dataset files)
├── Models/
│   └── ... (generated model files)
├── Pipeline/
│   ├── processing.py
│   └── modeling.py
├── FASTAPI/
└── ... (other project files)
```

### Step 7: Run the FastAPI Application

```bash
python FASTAPI/fastapi_deployment.py
```

The FastAPI server will start and be available at `http://localhost:8000` (or the specified port).

## Usage

### Accessing the API

Once the FastAPI application is running, you can:

1. **View API Documentation**: Navigate to `http://localhost:8000/docs` for interactive API documentation
2. **Make Predictions**: Send POST requests to the prediction endpoints
3. **Health Check**: Use `http://localhost:8000/health` to verify the service is running

### Jupyter Notebooks

The project includes comprehensive Jupyter notebooks for:

- **Data Exploration**: Understanding the CICIDS2017 dataset structure and characteristics
- **Data Preprocessing**: Feature engineering and data cleaning pipelines
- **Model Training**: Implementation and comparison of KNN, Random Forest, and XGBoost models
- **Model Evaluation**: Performance metrics and model comparison analysis

### Model Performance

The project implements three main algorithms:

- **K-Nearest Neighbors (KNN)**: Instance-based learning algorithm
- **Random Forest**: Ensemble method using multiple decision trees
- **XGBoost**: Gradient boosting framework optimized for performance

## API Endpoints

The FastAPI deployment provides the following endpoints:

- `GET /`: Welcome message and API information
- `POST /predict`: Make predictions on network traffic data
- `GET /model-info`: Information about the loaded models
- `GET /health`: Health check endpoint

## Project Features

- **Complete ML Pipeline**: From data preprocessing to model deployment
- **Multiple Algorithms**: Comparison of different machine learning approaches
- **RESTful API**: Easy integration with other systems
- **Interactive Documentation**: Auto-generated API docs with FastAPI
- **Scalable Architecture**: Designed for production deployment

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

Please refer to the repository for license information.

## Acknowledgments

- Canadian Institute for Cybersecurity for the CICIDS2017 dataset
- The open-source community for the machine learning libraries used

## Support

For issues and questions, please refer to the GitHub repository issues section or contact the project maintainer.

---

**Note**: Make sure to keep your virtual environment activated while working with this project. The FastAPI application will provide detailed logs and error messages to help with troubleshooting.
