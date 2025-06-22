"""
eda.py
Reusable functions for exploratory data analysis.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_univariate(df, column, filename=None):
    """Plot histogram for a single column. Optionally save the figure if filename is provided."""
    # Set seaborn style for better visualization
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.2)
    sns.set_palette("husl")
    
    plt.figure(figsize=(6,4))
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')
    if filename is not None:
        from src.utils import save_figures
        save_figures(plt.gcf(), filename)
    plt.show()

def plot_bivariate(df, column, target, filename=None):
    """Plot boxplot of a feature vs. target."""
    # Set seaborn style for better visualization
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.2)
    sns.set_palette("husl")
    plt.figure(figsize=(6,4))
    sns.boxplot(x=target, y=column, data=df)
    plt.title(f'{column} by {target}')
    if filename is not None:
        from src.utils import save_figures
        save_figures(plt.gcf(), filename)
    plt.show()

def plot_correlation_matrix(df, filename=None):
    """Plot correlation heatmap for DataFrame."""
    # Set seaborn style for better visualization
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.2)
    sns.set_palette("husl")
    
    plt.figure(figsize=(12,10))
    corr = df.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm')
    plt.title('Correlation Matrix')
    if filename is not None:
        from src.utils import save_figures
        save_figures(plt.gcf(), filename)
    plt.show() 

def analyze_outliers(df, n_top_outliers=3, filename=None):
    """
    Analyze outliers in numeric columns using IQR method and visualize top outliers.
    
    Args:
        df: pandas DataFrame to analyze
        n_top_outliers: number of top outlier columns to plot (default 3)
    
    Returns:
        DataFrame with outlier statistics
    """
    # Set seaborn style for better visualization
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.2)
    sns.set_palette("husl")
    
    # Compute IQR for each numeric column
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    outlier_summary = []

    for col in numeric_cols:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        num_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_summary.append((col, Q1, Q3, IQR, lower, upper, num_outliers))

    outlier_df = pd.DataFrame(outlier_summary, columns=['Column', 'Q1', 'Q3', 'IQR', 'Lower', 'Upper', 'Num_Outliers'])
    outlier_df = outlier_df.sort_values(by='Num_Outliers', ascending=False)

    # Plot top outliers
    top_outliers = outlier_df.head(n_top_outliers)['Column']
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(top_outliers):
        plt.subplot(1, 3, i+1)
        sns.boxplot(y=df[col])
        plt.title(f'Outliers in {col}')
    plt.tight_layout()
    if filename is not None:
        from src.utils import save_figures
        save_figures(plt.gcf(), filename)
    plt.show()
    
    return outlier_df

def draw_histograms(df, variables, n_rows, n_cols, n_bins, filename=None):
    fig=plt.figure()
    for i, var_name in enumerate(variables):
        ax=fig.add_subplot(n_rows,n_cols,i+1)
        df[var_name].hist(bins=n_bins,ax=ax)
        ax.set_title(var_name)
    fig.tight_layout()  # Improves appearance a bit.
    if filename is not None:
        from src.utils import save_figures
        save_figures(plt.gcf(), filename)
    plt.show()

def set_plot_style():
    """Set consistent plot styling across visualizations"""
    plt.style.use('default')
    # Set modern style with white background
    sns.set_theme(style="whitegrid", rc={
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'grid.color': '#dddddd',
        'grid.linestyle': '--'
    })
    # Use a modern color palette
    sns.set_palette(sns.color_palette("husl", 8))
    
def create_distribution_plot(data, x_col, y_col, title, xlabel, ylabel, filename):
    """Create a styled distribution plot"""
    # Apply base styling
    set_plot_style()
    
    # Create figure
    plt.figure(figsize=(8,6), dpi=100)
    
    # Create bar plot
    ax = sns.barplot(x=x_col, y=y_col, data=data, alpha=0.8)
    
    # Style the plot
    plt.title(title, fontsize=14, pad=15, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    # Add value labels
    total = data[y_col].sum()
    for i, v in enumerate(data[y_col]):
        # Add count on top
        ax.text(i, v, f'{v:,}', 
                ha='center', va='bottom',
                fontsize=11, fontweight='bold')
        # Add percentage in middle
        percentage = (v/total) * 100
        ax.text(i, v/2, f'{percentage:.1f}%',
                ha='center', va='center',
                fontsize=11, color='white', fontweight='bold')
    
    # Enhance visual appeal
    ax.grid(True, alpha=0.3)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    return ax
