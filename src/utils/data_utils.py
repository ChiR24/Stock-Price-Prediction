"""
Utility functions for data operations.
"""
import os
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional, Union

from src.utils.config import (
    DATA_RAW_DIR, 
    DATA_PROCESSED_DIR, 
    DATA_FEATURES_DIR,
    SEQUENCE_LENGTH,
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT
)

# Set up logging
logger = logging.getLogger(__name__)

def ensure_data_dirs():
    """
    Ensure that all data directories exist.
    """
    for directory in [DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_FEATURES_DIR]:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")

def load_stock_data(ticker: str, start_date: Optional[str] = None, 
                   end_date: Optional[str] = None, 
                   data_dir: str = DATA_RAW_DIR) -> pd.DataFrame:
    """
    Load stock data from CSV file.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date to filter data (optional)
        end_date: End date to filter data (optional)
        data_dir: Directory where data is stored
        
    Returns:
        DataFrame containing stock data
    
    Raises:
        FileNotFoundError: If the data file does not exist
        Exception: For other errors during data loading
    """
    file_path = os.path.join(data_dir, f"{ticker}.csv")
    
    try:
        logger.info(f"Loading stock data for {ticker} from {file_path}")
        df = pd.read_csv(file_path)
        
        # Convert date column to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Sort by date
        df = df.sort_values('Date')
        
        # Filter by date if provided
        if start_date:
            start_date = pd.to_datetime(start_date)
            df = df[df['Date'] >= start_date]
            
        if end_date:
            end_date = pd.to_datetime(end_date)
            df = df[df['Date'] <= end_date]
            
        logger.info(f"Successfully loaded data for {ticker}, shape: {df.shape}")
        return df
    
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"Stock data file for {ticker} not found in {data_dir}")
    
    except Exception as e:
        logger.error(f"Error loading stock data for {ticker}: {str(e)}")
        raise Exception(f"Failed to load stock data for {ticker}: {str(e)}")

def save_stock_data(df: pd.DataFrame, ticker: str, 
                   data_dir: str = DATA_PROCESSED_DIR) -> str:
    """
    Save stock data to CSV file.
    
    Args:
        df: DataFrame containing stock data
        ticker: Stock ticker symbol
        data_dir: Directory to save data
        
    Returns:
        Path to saved file
        
    Raises:
        Exception: If there's an error during saving
    """
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, f"{ticker}.csv")
    
    try:
        logger.info(f"Saving stock data for {ticker} to {file_path}")
        df.to_csv(file_path, index=False)
        logger.info(f"Successfully saved data for {ticker}, shape: {df.shape}")
        return file_path
    
    except Exception as e:
        logger.error(f"Error saving stock data for {ticker}: {str(e)}")
        raise Exception(f"Failed to save stock data for {ticker}: {str(e)}")

def create_sequences(data: np.ndarray, sequence_length: int = SEQUENCE_LENGTH) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input sequences and targets for time series prediction.
    
    Args:
        data: Input data array
        sequence_length: Length of each input sequence
        
    Returns:
        Tuple of (input sequences, target values)
    """
    X, y = [], []
    
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    
    return np.array(X), np.array(y)

def normalize_data(data: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Normalize data to have zero mean and unit variance.
    
    Args:
        data: Input data array
        
    Returns:
        Tuple of (normalized data, mean, std)
    """
    mean = np.mean(data)
    std = np.std(data)
    
    if std == 0:
        logger.warning("Standard deviation is zero, using 1 instead to avoid division by zero")
        std = 1
        
    normalized_data = (data - mean) / std
    return normalized_data, mean, std

def denormalize_data(normalized_data: np.ndarray, mean: float, std: float) -> np.ndarray:
    """
    Denormalize data that was normalized using normalize_data().
    
    Args:
        normalized_data: Normalized data array
        mean: Mean value used for normalization
        std: Standard deviation used for normalization
        
    Returns:
        Denormalized data
    """
    return normalized_data * std + mean

def normalize_dataframe(df: pd.DataFrame, columns: List[str], scaler: Optional[StandardScaler] = None) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Normalize specified columns in a DataFrame.
    
    Args:
        df: DataFrame to normalize
        columns: List of column names to normalize
        scaler: Pre-fitted scaler (optional)
        
    Returns:
        Tuple of (normalized DataFrame, fitted scaler)
    """
    df_copy = df.copy()
    
    if scaler is None:
        scaler = StandardScaler()
        df_copy[columns] = scaler.fit_transform(df_copy[columns])
    else:
        df_copy[columns] = scaler.transform(df_copy[columns])
    
    return df_copy, scaler

def train_val_test_split(data: np.ndarray, train_split: float = TRAIN_SPLIT, 
                        val_split: float = VAL_SPLIT) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training, validation, and test sets.
    
    Args:
        data: Input data array
        train_split: Percentage of data to use for training
        val_split: Percentage of data to use for validation
        
    Returns:
        Tuple of (train data, validation data, test data)
    """
    n = len(data)
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))
    
    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]
    
    return train, val, test

def plot_stock_data(df: pd.DataFrame, ticker: str, columns: List[str] = None):
    """
    Plot stock data.
    
    Args:
        df: DataFrame containing stock data
        ticker: Stock ticker symbol for plot title
        columns: List of columns to plot (defaults to ['Close'])
    """
    if columns is None:
        columns = ['Close']
    
    plt.figure(figsize=(14, 7))
    
    for column in columns:
        if column in df.columns:
            plt.plot(df['Date'], df[column], label=column)
    
    plt.title(f"{ticker} Stock Price")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    return plt.gcf()

def calculate_returns(df: pd.DataFrame, column: str = 'Close', periods: List[int] = [1, 5, 20]) -> pd.DataFrame:
    """
    Calculate returns over different periods.
    
    Args:
        df: DataFrame containing stock data
        column: Column to calculate returns for
        periods: List of periods to calculate returns for
        
    Returns:
        DataFrame with added return columns
    """
    df_copy = df.copy()
    
    for period in periods:
        df_copy[f'return_{period}d'] = df_copy[column].pct_change(periods=period)
    
    return df_copy

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add date-based features to the DataFrame.
    
    Args:
        df: DataFrame with a 'Date' column
        
    Returns:
        DataFrame with added date features
    """
    df_copy = df.copy()
    
    # Ensure Date is datetime
    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    
    # Extract date features
    df_copy['day_of_week'] = df_copy['Date'].dt.dayofweek
    df_copy['day_of_month'] = df_copy['Date'].dt.day
    df_copy['week_of_year'] = df_copy['Date'].dt.isocalendar().week
    df_copy['month'] = df_copy['Date'].dt.month
    df_copy['quarter'] = df_copy['Date'].dt.quarter
    df_copy['year'] = df_copy['Date'].dt.year
    df_copy['is_month_start'] = df_copy['Date'].dt.is_month_start.astype(int)
    df_copy['is_month_end'] = df_copy['Date'].dt.is_month_end.astype(int)
    df_copy['is_quarter_start'] = df_copy['Date'].dt.is_quarter_start.astype(int)
    df_copy['is_quarter_end'] = df_copy['Date'].dt.is_quarter_end.astype(int)
    df_copy['is_year_start'] = df_copy['Date'].dt.is_year_start.astype(int)
    df_copy['is_year_end'] = df_copy['Date'].dt.is_year_end.astype(int)
    
    return df_copy

def handle_missing_values(df: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Args:
        df: DataFrame with missing values
        method: Method to handle missing values ('ffill', 'bfill', 'interpolate', 'drop')
        
    Returns:
        DataFrame with handled missing values
    """
    df_copy = df.copy()
    
    if method == 'ffill':
        df_copy = df_copy.fillna(method='ffill')
    elif method == 'bfill':
        df_copy = df_copy.fillna(method='bfill')
    elif method == 'interpolate':
        df_copy = df_copy.interpolate(method='linear')
    elif method == 'drop':
        df_copy = df_copy.dropna()
    
    return df_copy

def detect_outliers(df: pd.DataFrame, column: str, method: str = 'zscore', threshold: float = 3.0) -> pd.Series:
    """
    Detect outliers in a DataFrame column.
    
    Args:
        df: DataFrame
        column: Column to check for outliers
        method: Method to detect outliers ('zscore' or 'iqr')
        threshold: Threshold for z-score method or IQR multiplier for IQR method
        
    Returns:
        Boolean Series where True indicates an outlier
    """
    if method == 'zscore':
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        return z_scores > threshold
    
    elif method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (df[column] < lower_bound) | (df[column] > upper_bound)
    
    else:
        raise ValueError(f"Invalid method {method}. Use 'zscore' or 'iqr'.")

def resample_data(df: pd.DataFrame, freq: str = 'W', agg_dict: Dict = None) -> pd.DataFrame:
    """
    Resample time series data to a different frequency.
    
    Args:
        df: DataFrame with a 'Date' column
        freq: Frequency string (e.g., 'D' for daily, 'W' for weekly, 'M' for monthly)
        agg_dict: Dictionary mapping columns to aggregation functions
        
    Returns:
        Resampled DataFrame
    """
    # Ensure Date is index and datetime
    df_copy = df.copy()
    if 'Date' in df_copy.columns:
        df_copy = df_copy.set_index('Date')
    
    df_copy.index = pd.to_datetime(df_copy.index)
    
    # Default aggregation dictionary if none provided
    if agg_dict is None:
        agg_dict = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }
        
        # Only include columns that exist in the DataFrame
        agg_dict = {k: v for k, v in agg_dict.items() if k in df_copy.columns}
    
    # Resample and aggregate
    resampled = df_copy.resample(freq).agg(agg_dict)
    
    # Reset index to get Date as a column again
    resampled = resampled.reset_index()
    
    return resampled 