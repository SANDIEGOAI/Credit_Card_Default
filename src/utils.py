"""
utils.py
Utility functions for saving figures, tables, and other helpers.
"""
import os

def save_figure(fig, filename, folder="outputs/figures"):
    """Save a matplotlib figure to the specified folder."""
    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, filename), bbox_inches='tight')

def save_table(df, filename, folder="outputs/tables"):
    """Save a DataFrame as a CSV table to the specified folder."""
    os.makedirs(folder, exist_ok=True)
    df.to_csv(os.path.join(folder, filename), index=False) 