import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def apply_log_transform(df, columns):
    """Apply log transformation to specified columns in the DataFrame."""
    df[columns] = np.log1p(df[columns].clip(lower=0))
    return df


def split_features_target(data, target_col):
    """Split the dataset into features (X) and target (y)."""
    X = data.drop(columns=[target_col]).values
    y = data[target_col].values
    return X, y


def fit_scaler_and_pca(X_train, variance_threshold=0.95):
    """
    Fit a scaler and PCA on the training data.
    
    Parameters:
        X_train (ndarray): Training features.
        variance_threshold (float): Variance ratio threshold for PCA.
        
    Returns:
        scaler (StandardScaler): Fitted scaler.
        pca (PCA): Fitted PCA model.
        optimal_k (int): Optimal number of PCA components.
    """
    # Standardize the training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Fit PCA on scaled training data
    pca = PCA()
    pca.fit(X_train_scaled)

    # Determine the optimal number of components
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    optimal_k = np.argmax(cumulative_variance >= variance_threshold) + 1
    print(f"Optimal number of components (k): {optimal_k}")

    # Refit PCA with the optimal number of components
    pca = PCA(n_components=optimal_k)
    pca.fit(X_train_scaled)

    return scaler, pca


def transform_dataset(df, scaler, pca, target_col):
    """
    Transform a dataset using a fitted scaler and PCA.
    
    Parameters:
        df (pd.DataFrame): Dataset to transform.
        scaler (StandardScaler): Fitted scaler.
        pca (PCA): Fitted PCA model.
        target_col (str): Target variable column name.
        
    Returns:
        pd.DataFrame: PCA-transformed dataset with the target variable appended.
    """
    # Separate features and target
    X, y = split_features_target(df, target_col)

    # Scale and apply PCA
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)

    # Create a DataFrame for the PCA-transformed data
    pca_df = pd.DataFrame(
        X_pca,
        columns=[f'pca_{i+1}' for i in range(pca.n_components_)],
        index=df.index
    ).assign(btc_close=y)

    return pca_df


def main():
    # Load the dataset
    df = pd.read_csv("../../data/final/dataset.csv", parse_dates=["Date"], index_col="Date")

    # Columns to apply log transformation
    suitable_col = [
        'hash_rate_blockchain',  
        'btc_sma_14', 'btc_ema_14', 
        'btc_bb_high', 'btc_bb_low', 'btc_bb_mid', 'btc_bb_width', 
        'btc_atr_14', 'btc_trading_volume', 'btc_volatility_index',
        'ARK Innovation ETF', 'CBOE Volatility Index', 'Shanghai Composite Index', 
        'btc_close'
    ]

    # Apply log transformation
    df = apply_log_transform(df, suitable_col)

    # Split the dataset into train (60%), validation (20%), and test (20%)
    train_data, val_data, test_data = np.split(df, [int(0.6 * len(df)), int(0.8 * len(df))])

    # Split features and target for training
    X_train, y_train = split_features_target(train_data, 'btc_close')

    # Fit scaler and PCA on training data
    scaler, pca= fit_scaler_and_pca(X_train)

    # Save the scaler and PCA model
    joblib.dump(scaler, "../../models/scaler.pkl")
    joblib.dump(pca, "../../models/pca.pkl")

    # Transform validation, test, and future datasets
    val_pca_df = transform_dataset(val_data, scaler, pca, 'btc_close')
    test_pca_df = transform_dataset(test_data, scaler, pca, 'btc_close')

    # Example for transforming a future dataset
    future_data = pd.read_csv("../../data/final/future_dataset.csv", parse_dates=["Date"], index_col="Date")
    future_data = apply_log_transform(future_data, suitable_col)
    future_pca_df = transform_dataset(future_data, scaler, pca, 'btc_close')

    # Export PCA-transformed data to CSV
    val_pca_df.to_csv("../../data/final/val_pca_df.csv")
    test_pca_df.to_csv("../../data/final/test_pca_df.csv")
    future_pca_df.to_csv("../../data/final/future_pca_df.csv")

    print("PCA transformation completed and data exported to CSV files.")


if __name__ == "__main__":
    main()
