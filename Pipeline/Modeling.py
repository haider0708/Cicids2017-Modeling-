import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import time
import psutil
import threading
from memory_profiler import memory_usage

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score,
                             precision_score, recall_score, f1_score)

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE


def apply_model_with_monitoring(model, X_train, y_train, cv=5):
    """
    Train model with resource monitoring (memory, time, CPU usage).

    Parameters:
        model: ML model to train
        X_train: Training features
        y_train: Training labels
        cv: Number of cross-validation folds

    Returns:
        cv_scores: Cross-validation scores
        measurements: Resource usage metrics
        trained_model: Fitted model
    """
    measurements = {}

    # CPU monitoring setup
    cpu_usage = []
    stop_flag = threading.Event()

    def monitor_cpu():
        while not stop_flag.is_set():
            cpu_usage.append(psutil.cpu_percent(interval=0.1))

    def train_model():
        model.fit(X_train, y_train)

    try:
        # Start CPU monitoring
        cpu_thread = threading.Thread(target=monitor_cpu)
        cpu_thread.start()

        # Measure memory usage and training time
        start_time = time.time()
        train_memory = max(memory_usage((train_model,)))
        training_time = time.time() - start_time

        # Stop CPU monitoring
        stop_flag.set()
        cpu_thread.join()

        # Store measurements
        measurements['Memory Usage (MB)'] = train_memory
        measurements['Training Time (s)'] = training_time
        measurements['Peak CPU Usage (%)'] = max(cpu_usage) if cpu_usage else 0
        measurements['Average CPU Usage (%)'] = sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0

        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, n_jobs=-1)

        return cv_scores, measurements, model

    except Exception as e:
        print(f"Error during model training: {e}")
        return None, None, None


def tune_hyperparameters(model, param_grid, X_train, y_train, n_iter=20, cv=3):
    """
    Perform hyperparameter tuning using RandomizedSearchCV.

    Parameters:
        model: ML model to tune
        param_grid: Parameter grid for search
        X_train: Training features
        y_train: Training labels
        n_iter: Number of parameter settings sampled
        cv: Number of cross-validation folds

    Returns:
        best_params: Best parameters found
        best_score: Best cross-validation score
    """
    # Get baseline score with default parameters
    baseline_cv = cross_val_score(model, X_train, y_train, cv=cv, n_jobs=-1)
    baseline_score = np.mean(baseline_cv)

    # Perform randomized search
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    random_search.fit(X_train, y_train)

    # Return best params only if they improve over baseline
    if random_search.best_score_ > baseline_score:
        return random_search.best_params_, random_search.best_score_
    else:
        return None, baseline_score


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate model performance and return metrics.

    Parameters:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model for identification

    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1 Score': f1_score(y_test, y_pred, average='weighted')
    }

    return metrics, y_pred


def prepare_data(file_path='cicids2017_cleaned.csv', test_size=0.3, random_state=42):
    """
    Load and prepare data for modeling.

    Parameters:
        file_path: Path to cleaned dataset
        test_size: Proportion of test set
        random_state: Random state for reproducibility

    Returns:
        Prepared datasets for training and testing
    """
    # Load data
    df = pd.read_csv(file_path)
    # Split features and target
    X = df.drop('Attack Type', axis=1)
    y = df['Attack Type']

    # Check for any infinite values
    if np.isinf(X.select_dtypes(include=[np.number])).any().any():
        print("Warning: Infinite values detected. Replacing with NaN and dropping.")
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.dropna()
        y = y.loc[X.index]

    print(f"Final dataset shape: X={X.shape}, y={y.shape}")
    print(f"Target distribution:\n{y.value_counts()}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    joblib.dump(scaler, '../Models/robust_scaler.joblib')

    # Apply undersampling to balance Normal Traffic
    print(f"Class distribution before undersampling:\n{y_train.value_counts()}")

    undersampler = RandomUnderSampler(
        sampling_strategy={'Normal Traffic': 500000},
        random_state=random_state
    )
    X_train_under, y_train_under = undersampler.fit_resample(X_train, y_train)
    X_train_scaled_under, y_train_scaled_under = undersampler.fit_resample(X_train_scaled, y_train)

    print(f"Class distribution after undersampling:\n{y_train_under.value_counts()}")

    # Apply SMOTE oversampling for minority classes
    # Check which classes are available after undersampling
    available_classes = y_train_scaled_under.unique()
    print(f"Classes available after undersampling: {available_classes}")
    print(f"Class distribution after undersampling:\n{y_train_scaled_under.value_counts()}")

    # Define desired sampling strategy, but only include classes that exist
    desired_sampling = {
        'Bots': 2000,
        'Web Attacks': 2000,
        'Brute Force': 7000,
        'Port Scanning': 70000,
        'DDoS': 90000,
        'DoS': 200000
    }

    # Filter sampling strategy to only include existing classes
    actual_sampling = {class_name: count for class_name, count in desired_sampling.items()
                       if class_name in available_classes}

    print(f"SMOTE sampling strategy: {actual_sampling}")

    if actual_sampling:  # Only apply SMOTE if there are classes to oversample
        smote = SMOTE(
            sampling_strategy=actual_sampling,
            random_state=random_state
        )
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled_under, y_train_scaled_under)
        print(f"Class distribution after SMOTE:\n{pd.Series(y_train_balanced).value_counts()}")
    else:
        # If no oversampling needed, use the undersampled data
        X_train_balanced, y_train_balanced = X_train_scaled_under, y_train_scaled_under
        print("No SMOTE applied - using undersampled data only")

    return {
        'X_train_raw': X_train_under,
        'y_train_raw': y_train_under,
        'X_train_scaled': X_train_balanced,
        'y_train_scaled': y_train_balanced,
        'X_test_raw': X_test,
        'X_test_scaled': X_test_scaled,
        'y_test': y_test
    }


def train_random_forest(data, tune_params=True):
    """
    Train Random Forest model with optional hyperparameter tuning.

    Parameters:
        data: Prepared datasets
        tune_params: Whether to perform hyperparameter tuning

    Returns:
        Trained model, metrics, and measurements
    """
    # Parameter grid for tuning
    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
    }

    # Initialize model
    rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)

    # Hyperparameter tuning
    if tune_params:
        best_params, best_score = tune_hyperparameters(
            rf_model, param_grid, data['X_train_raw'], data['y_train_raw']
        )
        if best_params:
            rf_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    else:
        # Use predefined best parameters
        best_params = {
            'n_estimators': 200,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'max_depth': None
        }
        rf_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)

    # Train model with monitoring
    cv_scores, measurements, trained_model = apply_model_with_monitoring(
        rf_model, data['X_train_raw'], data['y_train_raw']
    )

    # Evaluate model
    metrics, y_pred = evaluate_model(
        trained_model, data['X_test_raw'], data['y_test'], 'Random Forest'
    )

    # Add cross-validation score
    metrics['Cross Validation Mean'] = np.mean(cv_scores)
    metrics.update(measurements)

    # Save model
    joblib.dump(trained_model, '../Models/random_forest.joblib')

    return trained_model, metrics, y_pred


def train_xgboost(data, tune_params=True):
    """
    Train XGBoost model with optional hyperparameter tuning.

    Parameters:
        data: Prepared datasets
        tune_params: Whether to perform hyperparameter tuning

    Returns:
        Trained model, metrics, and measurements
    """
    # Create label mapping for XGBoost
    label_mapping = {
        'Normal Traffic': 0,
        'DoS': 1,
        'DDoS': 2,
        'Port Scanning': 3,
        'Brute Force': 4,
        'Web Attacks': 5,
        'Bots': 6
    }

    y_train_mapped = data['y_train_raw'].map(label_mapping)
    y_test_mapped = data['y_test'].map(label_mapping)

    # Parameter grid for tuning
    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.2, 0.3, 0.4],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'min_child_weight': [1, 5, 10],
    }

    # Initialize model
    xgb_model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(data['y_train_raw'].unique()),
        random_state=42,
        n_jobs=-1
    )

    # Hyperparameter tuning
    if tune_params:
        best_params, best_score = tune_hyperparameters(
            xgb_model, param_grid, data['X_train_raw'], y_train_mapped, n_iter=30
        )
        if best_params:
            xgb_model = xgb.XGBClassifier(
                **best_params,
                objective='multi:softmax',
                num_class=len(data['y_train_raw'].unique()),
                random_state=42,
                n_jobs=-1
            )
    else:
        # Use predefined best parameters
        best_params = {
            'subsample': 1.0,
            'n_estimators': 150,
            'min_child_weight': 1,
            'max_depth': 3,
            'learning_rate': 0.3,
            'colsample_bytree': 0.7
        }
        xgb_model = xgb.XGBClassifier(
            **best_params,
            objective='multi:softmax',
            num_class=len(data['y_train_raw'].unique()),
            random_state=42,
            n_jobs=-1
        )

    # Train model with monitoring
    cv_scores, measurements, trained_model = apply_model_with_monitoring(
        xgb_model, data['X_train_raw'], y_train_mapped
    )

    # Evaluate model
    metrics, y_pred = evaluate_model(
        trained_model, data['X_test_raw'], y_test_mapped, 'XGBoost'
    )

    # Add cross-validation score
    metrics['Cross Validation Mean'] = np.mean(cv_scores)
    metrics.update(measurements)

    # Save model
    joblib.dump(trained_model, '../Models/xgboost.joblib')

    # Convert predictions back to original labels
    reverse_mapping = {v: k for k, v in label_mapping.items()}
    y_pred_labels = [reverse_mapping[pred] for pred in y_pred]

    return trained_model, metrics, y_pred_labels


def train_knn(data, tune_params=True):
    """
    Train KNN model with optional hyperparameter tuning.

    Parameters:
        data: Prepared datasets
        tune_params: Whether to perform hyperparameter tuning

    Returns:
        Trained model, metrics, and measurements
    """
    # Parameter grid for tuning
    param_grid = {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
    }

    # Initialize model
    knn_model = KNeighborsClassifier(n_jobs=-1)

    # Hyperparameter tuning
    if tune_params:
        best_params, best_score = tune_hyperparameters(
            knn_model, param_grid, data['X_train_scaled'], data['y_train_scaled'], n_iter=6
        )
        if best_params:
            knn_model = KNeighborsClassifier(**best_params, n_jobs=-1)
    else:
        # Use predefined best parameters
        best_params = {'weights': 'distance', 'n_neighbors': 3}
        knn_model = KNeighborsClassifier(**best_params, n_jobs=-1)

    # Train model with monitoring
    cv_scores, measurements, trained_model = apply_model_with_monitoring(
        knn_model, data['X_train_scaled'], data['y_train_scaled']
    )

    # Evaluate model
    metrics, y_pred = evaluate_model(
        trained_model, data['X_test_scaled'], data['y_test'], 'KNN'
    )

    # Add cross-validation score
    metrics['Cross Validation Mean'] = np.mean(cv_scores)
    metrics.update(measurements)

    # Save model
    joblib.dump(trained_model, '../Models/knn_model.joblib')

    return trained_model, metrics, y_pred


def run_modeling_pipeline(data_file='cicids2017_cleaned.csv', tune_hyperparameters=True):
    """
    Run complete modeling pipeline with all three models.

    Parameters:
        data_file: Path to cleaned dataset
        tune_hyperparameters: Whether to perform hyperparameter tuning

    Returns:
        results_df: DataFrame with all model results
        models: Dictionary with trained models
        predictions: Dictionary with predictions
    """
    # Prepare data
    print("Preparing data...")
    data = prepare_data(data_file)

    # Train models
    print("Training Random Forest...")
    rf_model, rf_metrics, rf_pred = train_random_forest(data, tune_hyperparameters)

    print("Training XGBoost...")
    xgb_model, xgb_metrics, xgb_pred = train_xgboost(data, tune_hyperparameters)

    print("Training KNN...")
    knn_model, knn_metrics, knn_pred = train_knn(data, tune_hyperparameters)

    # Combine results
    results_df = pd.DataFrame([rf_metrics, xgb_metrics, knn_metrics])

    models = {
        'Random Forest': rf_model,
        'XGBoost': xgb_model,
        'KNN': knn_model
    }

    predictions = {
        'Random Forest': rf_pred,
        'XGBoost': xgb_pred,
        'KNN': knn_pred
    }

    print("Modeling pipeline completed!")
    return results_df, models, predictions


# Usage example
if __name__ == "__main__":
    # Run the complete pipeline
    results, trained_models, model_predictions = run_modeling_pipeline(
        data_file='../Datasets/cicids2017_cleaned.csv',
        tune_hyperparameters=False  # Set to True for hyperparameter tuning
    )

    # Display results
    print("\nModel Comparison Results:")
    print(results.round(4))

    # Save results
    results.to_csv('model_comparison_results.csv', index=False)