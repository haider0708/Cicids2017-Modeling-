import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy import stats


def load_and_merge_datasets(dataset_folder):
    """Load all CSV files from dataset folder and merge them"""
    dfs = []
    for dirname, _, filenames in os.walk(dataset_folder):
        for filename in filenames:
            if filename.endswith('.csv'):
                dfs.append(pd.read_csv(os.path.join(dirname, filename)))

    data = pd.concat(dfs, axis=0, ignore_index=True)
    return data


def clean_column_names(data):
    """Remove leading/trailing whitespace from column names"""
    col_names = {col: col.strip() for col in data.columns}
    data.rename(columns=col_names, inplace=True)
    return data


def remove_duplicates(data):
    """Remove duplicate rows"""
    data = data.drop_duplicates(keep='first')
    return data


def remove_identical_columns(data):
    """Remove columns with identical values"""
    identical_columns = {}
    columns = data.columns
    list_control = columns.copy().tolist()

    for col1 in columns:
        for col2 in columns:
            if col1 != col2:
                if data[col1].equals(data[col2]):
                    if (col1 not in identical_columns) and (col1 in list_control):
                        identical_columns[col1] = [col2]
                        list_control.remove(col2)
                    elif (col1 in identical_columns) and (col1 in list_control):
                        identical_columns[col1].append(col2)
                        list_control.remove(col2)

    for key, value in identical_columns.items():
        data.drop(columns=value, inplace=True)

    return data


def handle_infinite_values(data):
    """Replace infinite values with NaN"""
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    return data


def remove_missing_values(data):
    """Drop rows with missing values"""
    data = data.dropna()
    return data


def remove_single_value_columns(data):
    """Remove columns with only one unique value"""
    only_unique_cols = []
    for col in data.columns:
        if len(data[col].unique()) == 1:
            only_unique_cols.append(col)

    data.drop(only_unique_cols, axis=1, inplace=True)
    return data


def map_attack_types(data):
    """Map original labels to grouped attack types"""
    group_mapping = {
        'BENIGN': 'Normal Traffic',
        'DoS Hulk': 'DoS',
        'DDoS': 'DDoS',
        'PortScan': 'Port Scanning',
        'DoS GoldenEye': 'DoS',
        'FTP-Patator': 'Brute Force',
        'DoS slowloris': 'DoS',
        'DoS Slowhttptest': 'DoS',
        'SSH-Patator': 'Brute Force',
        'Bot': 'Bots',
        'Web Attack – Brute Force': 'Web Attacks',
        'Web Attack – XSS': 'Web Attacks',
        'Infiltration': 'Infiltration',
        'Web Attack – Sql Injection': 'Web Attacks',
        'Heartbleed': 'Miscellaneous'
    }

    data['Attack Type'] = data['Label'].map(group_mapping)
    data.drop(columns='Label', inplace=True)
    return data


def remove_rare_attacks(data):
    """Remove statistically irrelevant attack types"""
    data.drop(data[(data['Attack Type'] == 'Infiltration') |
                   (data['Attack Type'] == 'Miscellaneous')].index, inplace=True)
    return data


def get_feature_types(df, target_col='Attack Type'):
    """Identify numeric and categorical features"""
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()

    if target_col in numeric_features:
        numeric_features.remove(target_col)
    if target_col in categorical_features:
        categorical_features.remove(target_col)

    return numeric_features, categorical_features


def correlation_analysis(df, numeric_features, threshold=0.85):
    """Analyze correlations between numerical features"""
    corr_matrix = df[numeric_features].corr()
    high_corr = np.where(np.abs(corr_matrix) > threshold)
    high_corr = [(corr_matrix.index[x], corr_matrix.columns[y], corr_matrix.iloc[x, y])
                 for x, y in zip(*high_corr) if x != y and x < y]
    return high_corr


def remove_highly_correlated_features(data):
    """Remove features with high correlation based on predefined analysis"""
    selected_columns = ['Total Backward Packets', 'Total Length of Bwd Packets',
                        'Subflow Bwd Bytes', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size']
    data.drop(columns=selected_columns, inplace=True)
    return data


def analyze_feature_importance_rf(df, numeric_features, target_col='Attack Type'):
    """Analyze feature importance using Random Forest"""
    hyperparameters = {
        'n_estimators': 150,
        'max_depth': 30,
        'random_state': 42,
        'n_jobs': -1
    }

    X = df[numeric_features]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=hyperparameters['random_state'], stratify=y
    )

    rf = RandomForestClassifier(**hyperparameters)
    rf.fit(X_train, y_train)

    importances = rf.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': numeric_features,
        'Importance': importances
    })
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

    return feature_importance_df


def remove_low_importance_features(data):
    """Remove statistically irrelevant features based on analysis"""
    cols_to_remove = ['ECE Flag Count', 'RST Flag Count', 'Fwd URG Flags',
                      'Idle Std', 'Fwd PSH Flags', 'Active Std', 'Down/Up Ratio', 'URG Flag Count']
    # Only remove columns that exist in the dataframe
    cols_to_remove = [col for col in cols_to_remove if col in data.columns]
    data.drop(columns=cols_to_remove, inplace=True)
    data["Attack Type"] = data["Attack Type"].fillna("Web Attacks")
    return data


def process_cicids2017_data(dataset_folder, output_file='cicids2017_cleaned.csv'):
    """
    Complete data processing pipeline for CICIDS2017 dataset

    Parameters:
    -----------
    dataset_folder : str
        Path to folder containing CICIDS2017 CSV files
    output_file : str
        Name of output CSV file (optional)

    Returns:
    --------
    pandas.DataFrame
        Cleaned and processed dataset
    """

    # Step 1: Load and merge datasets
    data = load_and_merge_datasets(dataset_folder)

    # Step 2: Clean column names
    data = clean_column_names(data)

    # Step 3: Remove duplicates
    data = remove_duplicates(data)

    # Step 4: Remove identical columns
    data = remove_identical_columns(data)

    # Step 5: Handle infinite values
    data = handle_infinite_values(data)

    # Step 6: Remove missing values
    data = remove_missing_values(data)

    # Step 7: Remove single-value columns
    data = remove_single_value_columns(data)

    # Step 8: Map attack types
    data = map_attack_types(data)

    # Step 9: Remove rare attack types
    data = remove_rare_attacks(data)

    # Step 10: Remove highly correlated features
    data = remove_highly_correlated_features(data)

    # Step 11: Remove low importance features
    data = remove_low_importance_features(data)

    # Step 12: Save cleaned dataset
    if output_file:
        data.to_csv(output_file, index=False)

    return data


# Usage example:
if __name__ == "__main__":
    # Process the dataset
    dataset_folder = "Datasets_original"  # Path to your dataset folder
    cleaned_data = process_cicids2017_data(dataset_folder)

    # Optional: Get feature information
    numeric_features, categorical_features = get_feature_types(cleaned_data)
    print(f"Dataset shape: {cleaned_data.shape}")
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")