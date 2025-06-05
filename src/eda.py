"""
eda.py
Reusable functions for exploratory data analysis.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_univariate(df, column):
    """Plot histogram for a single column."""
    plt.figure(figsize=(6,4))
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.show()

def plot_bivariate(df, column, target):
    """Plot boxplot of a feature vs. target."""
    plt.figure(figsize=(6,4))
    sns.boxplot(x=target, y=column, data=df)
    plt.title(f'{column} by {target}')
    plt.show()

def plot_correlation_matrix(df):
    """Plot correlation heatmap for DataFrame."""
    plt.figure(figsize=(12,10))
    corr = df.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show() 