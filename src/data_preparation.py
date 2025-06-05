"""
data_preparation.py
Functions for loading, cleaning, and saving the Credit Card Default dataset.
"""
import pandas as pd
from ucimlrepo import fetch_ucirepo

def load_data():
    """Fetch the dataset from UCI and return features and target as DataFrames."""
    dataset = fetch_ucirepo(id=350)
    X = dataset.data.features
    y = dataset.data.targets
    return X, y

def save_processed_data(X, y, X_path, y_path):
    """Save processed features and target to CSV files."""
    X.to_csv(X_path, index=False)
    y.to_csv(y_path, index=False) 