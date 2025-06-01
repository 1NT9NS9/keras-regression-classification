"""
Utility functions for the stock market neural network project.

This module contains helper functions for data visualization,
model evaluation, and other common tasks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def setup_matplotlib():
    """
    Setup matplotlib for better plots.
    """
    plt.style.use('seaborn-v0_8')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3


def create_directories():
    """
    Create necessary directories for the project.
    """
    directories = ['data', 'models', 'notebooks', 'results']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")


def plot_stock_data(data, symbol, save_path=None):
    """
    Create an interactive plot of stock data using Plotly.
    
    Args:
        data (pd.DataFrame): Stock data with OHLCV columns
        symbol (str): Stock symbol for the title
        save_path (str): Path to save the plot (optional)
    """
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price & Volume', 'Technical Indicators', 'Price Changes'),
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data['Date'],
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Volume
    fig.add_trace(
        go.Bar(
            x=data['Date'],
            y=data['Volume'],
            name='Volume',
            yaxis='y2',
            opacity=0.3
        ),
        row=1, col=1
    )
    
    # Moving averages (if available)
    if 'SMA_20' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data['Date'],
                y=data['SMA_20'],
                name='SMA 20',
                line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )
    
    if 'SMA_50' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data['Date'],
                y=data['SMA_50'],
                name='SMA 50',
                line=dict(color='red', width=1)
            ),
            row=1, col=1
        )
    
    # RSI (if available)
    if 'RSI' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data['Date'],
                y=data['RSI'],
                name='RSI',
                line=dict(color='purple')
            ),
            row=2, col=1
        )
        
        # RSI reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # Price changes (if available)
    if 'Price_change' in data.columns:
        colors = ['red' if x < 0 else 'green' for x in data['Price_change']]
        fig.add_trace(
            go.Bar(
                x=data['Date'],
                y=data['Price_change'] * 100,
                name='Price Change %',
                marker_color=colors
            ),
            row=3, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} Stock Analysis',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=800,
        showlegend=True
    )
    
    # Update y-axis for RSI
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="Price Change (%)", row=3, col=1)
    
    if save_path:
        fig.write_html(save_path)
        print(f"Plot saved to {save_path}")
    
    fig.show()


def plot_model_comparison(metrics_dict, save_path=None):
    """
    Compare multiple models using their metrics.
    
    Args:
        metrics_dict (dict): Dictionary with model names as keys and metrics as values
        save_path (str): Path to save the plot (optional)
    """
    df_metrics = pd.DataFrame(metrics_dict).T
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16)
    
    metrics = df_metrics.columns
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'lightsalmon']
    
    for i, metric in enumerate(metrics[:4]):  # Plot up to 4 metrics
        ax = axes[i//2, i%2]
        bars = ax.bar(df_metrics.index, df_metrics[metric], color=colors[i])
        ax.set_title(f'{metric} Comparison')
        ax.set_ylabel(metric)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.show()


def calculate_portfolio_metrics(returns):
    """
    Calculate portfolio performance metrics.
    
    Args:
        returns (pd.Series): Portfolio returns
        
    Returns:
        dict: Portfolio metrics
    """
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + returns.mean()) ** 252 - 1  # Assuming 252 trading days
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / volatility if volatility != 0 else 0
    
    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    metrics = {
        'Total Return': total_return,
        'Annual Return': annual_return,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }
    
    return metrics


def plot_correlation_heatmap(data, features=None, save_path=None):
    """
    Plot correlation heatmap of features.
    
    Args:
        data (pd.DataFrame): Data with features
        features (list): List of features to include (optional)
        save_path (str): Path to save the plot (optional)
    """
    if features:
        data = data[features]
    
    correlation_matrix = data.corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Correlation heatmap saved to {save_path}")
    
    plt.show()


def log_experiment(model_name, metrics, parameters, save_path='results/experiment_log.csv'):
    """
    Log experiment results to a CSV file.
    
    Args:
        model_name (str): Name of the model
        metrics (dict): Model performance metrics
        parameters (dict): Model parameters
        save_path (str): Path to save the log file
    """
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Prepare log entry
    log_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_name': model_name,
        **metrics,
        **parameters
    }
    
    # Convert to DataFrame
    df_entry = pd.DataFrame([log_entry])
    
    # Append to existing log or create new one
    if os.path.exists(save_path):
        df_existing = pd.read_csv(save_path)
        df_combined = pd.concat([df_existing, df_entry], ignore_index=True)
    else:
        df_combined = df_entry
    
    # Save to CSV
    df_combined.to_csv(save_path, index=False)
    print(f"Experiment logged to {save_path}")


def print_data_summary(data, title="Data Summary"):
    """
    Print a comprehensive summary of the dataset.
    
    Args:
        data (pd.DataFrame): Dataset to summarize
        title (str): Title for the summary
    """
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    
    print(f"Shape: {data.shape}")
    print(f"Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\nData types:")
    print(data.dtypes.value_counts())
    
    print(f"\nMissing values:")
    missing = data.isnull().sum()
    missing_pct = (missing / len(data)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct
    })
    print(missing_df[missing_df['Missing Count'] > 0])
    
    print(f"\nNumerical columns summary:")
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        print(data[numerical_cols].describe())


def save_results_summary(regression_metrics, classification_metrics, save_path='results/model_summary.txt'):
    """
    Save a summary of model results to a text file.
    
    Args:
        regression_metrics (dict): Regression model metrics
        classification_metrics (dict): Classification model metrics
        save_path (str): Path to save the summary
    """
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write("Stock Market Neural Network Models - Results Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("REGRESSION MODEL (Price Prediction)\n")
        f.write("-" * 40 + "\n")
        for metric, value in regression_metrics.items():
            f.write(f"{metric}: {value:.6f}\n")
        
        f.write("\nCLASSIFICATION MODEL (Direction Prediction)\n")
        f.write("-" * 45 + "\n")
        for metric, value in classification_metrics.items():
            f.write(f"{metric}: {value:.6f}\n")
        
        f.write("\nModel Performance Interpretation:\n")
        f.write("-" * 35 + "\n")
        f.write("Regression Model:\n")
        f.write(f"- R² Score: {'Good' if regression_metrics.get('R2', 0) > 0.7 else 'Moderate' if regression_metrics.get('R2', 0) > 0.5 else 'Poor'}\n")
        f.write(f"- MAPE: {'Good' if regression_metrics.get('MAPE', 100) < 5 else 'Moderate' if regression_metrics.get('MAPE', 100) < 10 else 'Poor'}\n")
        
        f.write("\nClassification Model:\n")
        f.write(f"- Accuracy: {'Good' if classification_metrics.get('Accuracy', 0) > 0.6 else 'Moderate' if classification_metrics.get('Accuracy', 0) > 0.4 else 'Poor'}\n")
        f.write(f"- F1-Score: {'Good' if classification_metrics.get('F1-Score', 0) > 0.6 else 'Moderate' if classification_metrics.get('F1-Score', 0) > 0.4 else 'Poor'}\n")
    
    print(f"Results summary saved to {save_path}")


# Initialize matplotlib settings when module is imported
setup_matplotlib()


# Example usage and testing
if __name__ == "__main__":
    print("Utilities module loaded successfully!")
    
    # Create project directories
    create_directories()
    
    print("All utility functions are ready to use!") 