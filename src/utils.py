"""
utils.py
Utility functions for saving figures, tables, and other helpers.
"""
import os
import joblib

def save_figure(fig, filename, folder="../outputs/figures"):
    """Save a matplotlib figure to the specified folder."""
    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, filename), bbox_inches='tight')

def save_table(df, filename, folder="../outputs/tables"):
    """Save a DataFrame as a CSV table to the specified folder."""
    os.makedirs(folder, exist_ok=True)
    df.to_csv(os.path.join(folder, filename), index=False) 

def save_model(model, filename, folder="../outputs/models"):
    """
    Save a trained model to the specified folder.
    
    Args:
        model: Trained model to save
        filename: Name of file to save model to
        folder: Directory to save model in (default: ../outputs/models)
    """
    os.makedirs(folder, exist_ok=True)
    model_path = os.path.join(folder, filename)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def load_model(filename, folder="../outputs/models"):
    """
    Load a saved model from the specified folder.
    
    Args:
        filename: Name of file containing saved model
        folder: Directory containing model file (default: ../outputs/models)
        
    Returns:
        Loaded model object
    """
    model_path = os.path.join(folder, filename)
    model = joblib.load(model_path)
    return model

def save_figures(figs, filenames, folder="../outputs/figures"):
    """
    Save multiple matplotlib figures to the specified folder.
    
    Args:
        figs: List of matplotlib figures to save
        filenames: List of filenames to save figures as
        folder: Directory to save figures in (default: ../outputs/figures)
    """
    os.makedirs(folder, exist_ok=True)
    
    if not isinstance(figs, list):
        figs = [figs]
    if not isinstance(filenames, list):
        filenames = [filenames]
        
    if len(figs) != len(filenames):
        raise ValueError("Number of figures must match number of filenames")
        
    for fig, filename in zip(figs, filenames):
        fig_path = os.path.join(folder, filename)
        fig.savefig(fig_path, bbox_inches='tight')
        print(f"Figure saved to {fig_path}")
