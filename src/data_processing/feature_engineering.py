"""
Feature engineering module for stock market prediction.
This module calculates technical indicators and prepares features for model training.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Import TA library for technical indicators
import ta

# Import project modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from utils.config import (
    PROCESSED_DATA_DIR,
    HISTORICAL_WINDOW
)

# Set up logger
logger = setup_logger('feature_engineering')

class FeatureEngineering:
    """
    Feature engineering class for stock market prediction.
    """
    def __init__(self):
        """
        Initialize the FeatureEngineering class.
        """
        pass

    def calculate_technical_indicators(self, df):
        """
        Calculate technical indicators for stock data.

        Args:
            df: DataFrame with stock data (must have OHLCV columns).

        Returns:
            DataFrame with technical indicators added.
        """
        try:
            # Make a copy to avoid modifying the original DataFrame
            result_df = df.copy()

            # Ensure OHLCV columns are present and are of float type
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

            if not all(col in result_df.columns for col in required_cols):
                logger.error(f"Missing required columns. Available columns: {result_df.columns}")
                # Check if alternative names are present
                alt_cols = ['open', 'high', 'low', 'close', 'volume']
                if all(col in result_df.columns for col in alt_cols):
                    # Rename to standard
                    result_df.rename(columns={
                        'open': 'Open',
                        'high': 'High',
                        'low': 'Low',
                        'close': 'Close',
                        'volume': 'Volume'
                    }, inplace=True)
                    logger.info("Renamed columns to standard OHLCV format")
                else:
                    return result_df

            # Convert columns to float if they are not already
            for col in required_cols:
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce')

            # Calculate basic indicators using TA library

            # Moving Averages
            result_df['MA_5'] = ta.trend.sma_indicator(result_df['Close'], window=5)
            result_df['MA_10'] = ta.trend.sma_indicator(result_df['Close'], window=10)
            result_df['MA_20'] = ta.trend.sma_indicator(result_df['Close'], window=20)
            result_df['MA_50'] = ta.trend.sma_indicator(result_df['Close'], window=50)
            result_df['MA_100'] = ta.trend.sma_indicator(result_df['Close'], window=100)
            result_df['MA_200'] = ta.trend.sma_indicator(result_df['Close'], window=200)

            # Exponential Moving Averages
            result_df['EMA_5'] = ta.trend.ema_indicator(result_df['Close'], window=5)
            result_df['EMA_10'] = ta.trend.ema_indicator(result_df['Close'], window=10)
            result_df['EMA_20'] = ta.trend.ema_indicator(result_df['Close'], window=20)
            result_df['EMA_50'] = ta.trend.ema_indicator(result_df['Close'], window=50)
            result_df['EMA_100'] = ta.trend.ema_indicator(result_df['Close'], window=100)
            result_df['EMA_200'] = ta.trend.ema_indicator(result_df['Close'], window=200)

            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(result_df['Close'], window=20, window_dev=2)
            result_df['BB_upper'] = bollinger.bollinger_hband()
            result_df['BB_middle'] = bollinger.bollinger_mavg()
            result_df['BB_lower'] = bollinger.bollinger_lband()

            # MACD
            macd = ta.trend.MACD(result_df['Close'], window_slow=26, window_fast=12, window_sign=9)
            result_df['MACD'] = macd.macd()
            result_df['MACD_signal'] = macd.macd_signal()
            result_df['MACD_hist'] = macd.macd_diff()

            # RSI
            result_df['RSI_14'] = ta.momentum.RSIIndicator(result_df['Close'], window=14).rsi()

            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(
                result_df['High'], result_df['Low'], result_df['Close'], window=14, smooth_window=3
            )
            result_df['STOCH_k'] = stoch.stoch()
            result_df['STOCH_d'] = stoch.stoch_signal()

            # Average Directional Index
            adx = ta.trend.ADXIndicator(result_df['High'], result_df['Low'], result_df['Close'], window=14)
            result_df['ADX'] = adx.adx()

            # OBV (On-Balance Volume)
            result_df['OBV'] = ta.volume.OnBalanceVolumeIndicator(result_df['Close'], result_df['Volume']).on_balance_volume()

            # ATR (Average True Range)
            result_df['ATR'] = ta.volatility.AverageTrueRange(result_df['High'], result_df['Low'], result_df['Close'], window=14).average_true_range()

            # CCI (Commodity Channel Index)
            result_df['CCI'] = ta.trend.CCIIndicator(result_df['High'], result_df['Low'], result_df['Close'], window=14).cci()

            # MOM (Momentum)
            result_df['MOM_10'] = ta.momentum.ROCIndicator(result_df['Close'], window=10).roc()

            # ROC (Rate of Change)
            result_df['ROC_10'] = ta.momentum.ROCIndicator(result_df['Close'], window=10).roc()

            # Williams %R
            result_df['WILLR'] = ta.momentum.WilliamsRIndicator(result_df['High'], result_df['Low'], result_df['Close'], lbp=14).williams_r()

            # Volume Moving Averages
            result_df['VOLUME_MA_5'] = ta.trend.sma_indicator(result_df['Volume'], window=5)
            result_df['VOLUME_MA_10'] = ta.trend.sma_indicator(result_df['Volume'], window=10)
            result_df['VOLUME_MA_20'] = ta.trend.sma_indicator(result_df['Volume'], window=20)

            # Volume Oscillator
            result_df['VOLUME_OSC'] = (
                (result_df['VOLUME_MA_5'] - result_df['VOLUME_MA_20']) / result_df['VOLUME_MA_20']
            ) * 100

            # Price-Volume Relationship
            result_df['PRICE_VOLUME_RATIO'] = result_df['Close'] / result_df['Volume']

            # Calculate price changes

            # Daily Returns
            result_df['daily_return'] = result_df['Close'].pct_change()

            # Log Returns
            result_df['log_return'] = np.log(result_df['Close'] / result_df['Close'].shift(1))

            # Volatility (rolling standard deviation of returns)
            result_df['volatility_5d'] = result_df['daily_return'].rolling(window=5).std()
            result_df['volatility_10d'] = result_df['daily_return'].rolling(window=10).std()
            result_df['volatility_20d'] = result_df['daily_return'].rolling(window=20).std()

            # Calculate price momentum

            # Price Rate of Change
            result_df['price_roc_5d'] = result_df['Close'].pct_change(periods=5)
            result_df['price_roc_10d'] = result_df['Close'].pct_change(periods=10)
            result_df['price_roc_20d'] = result_df['Close'].pct_change(periods=20)

            # Calculate price relative to moving averages

            # Price/MA Ratios
            result_df['price_ma_ratio_5'] = result_df['Close'] / result_df['MA_5']
            result_df['price_ma_ratio_10'] = result_df['Close'] / result_df['MA_10']
            result_df['price_ma_ratio_20'] = result_df['Close'] / result_df['MA_20']
            result_df['price_ma_ratio_50'] = result_df['Close'] / result_df['MA_50']
            result_df['price_ma_ratio_100'] = result_df['Close'] / result_df['MA_100']
            result_df['price_ma_ratio_200'] = result_df['Close'] / result_df['MA_200']

            # MA Crossovers (1 when shorter MA crosses above longer MA, -1 when crossing below)
            result_df['ma_crossover_5_10'] = np.where(
                result_df['MA_5'] > result_df['MA_10'], 1,
                np.where(result_df['MA_5'] < result_df['MA_10'], -1, 0)
            )

            result_df['ma_crossover_10_20'] = np.where(
                result_df['MA_10'] > result_df['MA_20'], 1,
                np.where(result_df['MA_10'] < result_df['MA_20'], -1, 0)
            )

            result_df['ma_crossover_20_50'] = np.where(
                result_df['MA_20'] > result_df['MA_50'], 1,
                np.where(result_df['MA_20'] < result_df['MA_50'], -1, 0)
            )

            result_df['ma_crossover_50_200'] = np.where(
                result_df['MA_50'] > result_df['MA_200'], 1,
                np.where(result_df['MA_50'] < result_df['MA_200'], -1, 0)
            )

            # Fill NaN values with method appropriate for each feature
            numeric_cols = result_df.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                # Skip columns that shouldn't be filled (e.g., target columns)
                if 'target' in col:
                    continue

                # Forward fill most indicators
                result_df[col] = result_df[col].fillna(method='ffill')

                # If still NaN values at the beginning, backfill
                result_df[col] = result_df[col].fillna(method='bfill')

                # If still any NaN (unlikely at this point), fill with 0
                result_df[col] = result_df[col].fillna(0)

            logger.info(f"Calculated {len(result_df.columns) - len(df.columns)} technical indicators")

            return result_df

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            # Return original dataframe if an error occurred
            return df

    def create_features(self, df, target_col='Close', prediction_days=5):
        """
        Create features for training ML models.

        Args:
            df: DataFrame with processed data.
            target_col: Target column name. Defaults to 'Close'.
            prediction_days: Number of days to predict ahead. Defaults to 5.

        Returns:
            DataFrame with features suitable for ML training.
        """
        try:
            # Make a copy to avoid modifying the original DataFrame
            result_df = df.copy()

            # Sort by date to ensure correct sequence
            if 'Date' in result_df.columns:
                result_df = result_df.sort_values('Date')

            # Create target variable (future price change in percentage)
            target_price = result_df[target_col].shift(-prediction_days)
            result_df[f'target_{prediction_days}d'] = (target_price - result_df[target_col]) / result_df[target_col] * 100

            # Create lagged price features
            for lag in [1, 2, 3, 5, 10]:
                result_df[f'{target_col}_lag_{lag}'] = result_df[target_col].shift(lag)

            # Create rolling mean features
            for window in [5, 10, 20, 30]:
                result_df[f'{target_col}_rolling_mean_{window}'] = result_df[target_col].rolling(window=window).mean()

            # Create rolling std features
            for window in [5, 10, 20, 30]:
                result_df[f'{target_col}_rolling_std_{window}'] = result_df[target_col].rolling(window=window).std()

            # Create day of week, month, quarter features if Date column exists
            if 'Date' in result_df.columns:
                # Convert Date to datetime if it's not already
                if not pd.api.types.is_datetime64_any_dtype(result_df['Date']):
                    result_df['Date'] = pd.to_datetime(result_df['Date'])

                # Extract date components
                result_df['day_of_week'] = result_df['Date'].dt.dayofweek
                result_df['month'] = result_df['Date'].dt.month
                result_df['quarter'] = result_df['Date'].dt.quarter
                result_df['year'] = result_df['Date'].dt.year
                result_df['day_of_month'] = result_df['Date'].dt.day
                result_df['week_of_year'] = result_df['Date'].dt.isocalendar().week

            # Drop rows with NaN values in target or essential features
            result_df = result_df.dropna(subset=[f'target_{prediction_days}d'])

            # Fill missing values in features
            numeric_cols = result_df.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                # Skip columns that shouldn't be filled (e.g., target columns)
                if 'target' in col:
                    continue

                # Forward fill
                result_df[col] = result_df[col].fillna(method='ffill')

                # If still NaN values at the beginning, backfill
                result_df[col] = result_df[col].fillna(method='bfill')

                # If still any NaN (unlikely at this point), fill with 0
                result_df[col] = result_df[col].fillna(0)

            logger.info(f"Created feature dataframe with {len(result_df.columns)} features and {len(result_df)} rows")

            return result_df

        except Exception as e:
            logger.error(f"Error creating features: {e}")
            # Return original dataframe if an error occurred
            return df

    def add_sentiment_features(self, stock_df, sentiment_df):
        """
        Add sentiment features to the stock data DataFrame.

        Args:
            stock_df: DataFrame with stock data.
            sentiment_df: DataFrame with sentiment data.

        Returns:
            DataFrame with sentiment features.
        """
        try:
            # Check if sentiment_df is empty
            if sentiment_df.empty:
                logger.warning("Sentiment DataFrame is empty, skipping sentiment features")
                return stock_df

            # Make a copy to avoid modifying the original
            df_with_sentiment = stock_df.copy()

            # Ensure both DataFrames have a datetime index
            if not isinstance(stock_df.index, pd.DatetimeIndex):
                logger.warning("Stock DataFrame index is not DatetimeIndex, attempting conversion")
                if 'Date' in stock_df.columns:
                    df_with_sentiment = stock_df.set_index('Date')
                    df_with_sentiment.index = pd.to_datetime(df_with_sentiment.index)
                else:
                    logger.error("Cannot convert stock DataFrame index to DatetimeIndex")
                    return stock_df

            # Ensure sentiment_df has datetime columns
            if 'created_at' in sentiment_df.columns:
                sentiment_df['created_at'] = pd.to_datetime(sentiment_df['created_at'])
            elif 'timestamp' in sentiment_df.columns:
                sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'])
            else:
                logger.error("Sentiment DataFrame has no datetime column")
                return stock_df

            # Check if ticker column exists in both DataFrames
            if 'ticker' not in sentiment_df.columns:
                logger.error("Sentiment DataFrame has no ticker column")
                return stock_df

            # Group sentiment data by day and ticker
            if 'created_at' in sentiment_df.columns:
                sentiment_df['date'] = sentiment_df['created_at'].dt.date
                date_col = 'date'
            else:
                sentiment_df['date'] = sentiment_df['timestamp'].dt.date
                date_col = 'date'

            # Add ticker column to stock_df if not present
            if 'ticker' not in df_with_sentiment.columns and 'ticker' in stock_df.columns:
                df_with_sentiment['ticker'] = stock_df['ticker']

            # Calculate daily sentiment metrics
            daily_sentiment = sentiment_df.groupby([date_col, 'ticker']).agg({
                'polarity': ['mean', 'std', 'count'],
                'subjectivity': ['mean', 'std']
            })

            daily_sentiment.columns = ['_'.join(col).strip() for col in daily_sentiment.columns.values]
            daily_sentiment = daily_sentiment.reset_index()

            # Convert date to datetime for merging
            daily_sentiment[date_col] = pd.to_datetime(daily_sentiment[date_col])

            # Merge with stock data
            df_with_sentiment_reset = df_with_sentiment.reset_index()

            if 'Date' in df_with_sentiment_reset.columns:
                merge_col = 'Date'
            else:
                df_with_sentiment_reset['date'] = df_with_sentiment_reset.index
                merge_col = 'date'

            df_with_sentiment_reset[merge_col] = pd.to_datetime(df_with_sentiment_reset[merge_col])

            # Merge on date and ticker
            if 'ticker' in df_with_sentiment_reset.columns:
                merged_df = pd.merge(
                    df_with_sentiment_reset,
                    daily_sentiment,
                    left_on=[merge_col, 'ticker'],
                    right_on=[date_col, 'ticker'],
                    how='left'
                )
            else:
                # If no ticker column, assume it's for a single ticker
                ticker = sentiment_df['ticker'].iloc[0] if not sentiment_df.empty else None
                logger.warning(f"No ticker column in stock DataFrame, assuming ticker: {ticker}")

                merged_df = pd.merge(
                    df_with_sentiment_reset,
                    daily_sentiment,
                    left_on=merge_col,
                    right_on=date_col,
                    how='left'
                )

            # Fill NaN values (days with no sentiment data)
            sentiment_cols = [col for col in merged_df.columns if col.startswith(('polarity', 'subjectivity'))]
            merged_df[sentiment_cols] = merged_df[sentiment_cols].fillna(0)

            # Calculate rolling sentiment features
            for window in [3, 7, 14]:
                merged_df[f'polarity_mean_rolling_{window}d'] = merged_df['polarity_mean'].rolling(window=window).mean()
                merged_df[f'polarity_std_rolling_{window}d'] = merged_df['polarity_std'].rolling(window=window).mean()
                merged_df[f'subjectivity_mean_rolling_{window}d'] = merged_df['subjectivity_mean'].rolling(window=window).mean()

            # Set index back
            if 'Date' in merged_df.columns:
                merged_df = merged_df.set_index('Date')
            else:
                merged_df = merged_df.set_index('date')

            logger.info(f"Added {len(sentiment_cols)} sentiment features")

            return merged_df

        except Exception as e:
            logger.error(f"Error adding sentiment features: {e}")
            return stock_df

    def add_time_features(self, df):
        """
        Add time-based features to the DataFrame.

        Args:
            df: DataFrame with datetime index.

        Returns:
            DataFrame with time features.
        """
        try:
            # Ensure DataFrame has a datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                logger.warning("DataFrame index is not DatetimeIndex, attempting conversion")
                if 'Date' in df.columns:
                    df = df.set_index('Date')
                    df.index = pd.to_datetime(df.index)
                else:
                    logger.error("Cannot convert DataFrame index to DatetimeIndex")
                    return df

            # Make a copy to avoid modifying the original
            df_with_time = df.copy()

            # Extract time features
            df_with_time['day_of_week'] = df_with_time.index.dayofweek
            df_with_time['day_of_month'] = df_with_time.index.day
            df_with_time['week_of_year'] = df_with_time.index.isocalendar().week
            df_with_time['month'] = df_with_time.index.month
            df_with_time['quarter'] = df_with_time.index.quarter
            df_with_time['year'] = df_with_time.index.year
            df_with_time['is_month_start'] = df_with_time.index.is_month_start.astype(int)
            df_with_time['is_month_end'] = df_with_time.index.is_month_end.astype(int)
            df_with_time['is_quarter_start'] = df_with_time.index.is_quarter_start.astype(int)
            df_with_time['is_quarter_end'] = df_with_time.index.is_quarter_end.astype(int)
            df_with_time['is_year_start'] = df_with_time.index.is_year_start.astype(int)
            df_with_time['is_year_end'] = df_with_time.index.is_year_end.astype(int)

            # Create cyclical features
            df_with_time['sin_day_of_week'] = np.sin(2 * np.pi * df_with_time.index.dayofweek / 7)
            df_with_time['cos_day_of_week'] = np.cos(2 * np.pi * df_with_time.index.dayofweek / 7)
            df_with_time['sin_month'] = np.sin(2 * np.pi * df_with_time.index.month / 12)
            df_with_time['cos_month'] = np.cos(2 * np.pi * df_with_time.index.month / 12)

            logger.info(f"Added {len(df_with_time.columns) - len(df.columns)} time features")

            return df_with_time

        except Exception as e:
            logger.error(f"Error adding time features: {e}")
            return df

    def create_target_variable(self, df, target_column='Close', periods=1):
        """
        Create target variable for prediction.

        Args:
            df: DataFrame with stock data.
            target_column: Column to predict. Defaults to 'Close'.
            periods: Number of periods to shift for prediction. Defaults to 1 (next day).

        Returns:
            DataFrame with target variable.
        """
        try:
            # Make a copy to avoid modifying the original
            df_with_target = df.copy()

            # Create future price column
            future_column = f'future_{target_column}_{periods}d'
            df_with_target[future_column] = df_with_target[target_column].shift(-periods)

            # Create return column
            return_column = f'return_{periods}d'
            df_with_target[return_column] = df_with_target[future_column] / df_with_target[target_column] - 1

            # Create binary target (1 if price goes up, 0 if price goes down)
            binary_column = f'target_binary_{periods}d'
            df_with_target[binary_column] = (df_with_target[return_column] > 0).astype(int)

            # Create multi-class target (0: down, 1: stable, 2: up)
            multi_column = f'target_multi_{periods}d'

            def categorize_return(ret):
                if ret > 0.01:  # Up (>1%)
                    return 2
                elif ret < -0.01:  # Down (<-1%)
                    return 0
                else:  # Stable (-1% to 1%)
                    return 1

            df_with_target[multi_column] = df_with_target[return_column].apply(categorize_return)

            logger.info(f"Created target variables for {periods} day(s) ahead prediction")

            return df_with_target

        except Exception as e:
            logger.error(f"Error creating target variable: {e}")
            return df

    def prepare_prediction_data(self, df, prediction_date=None, window=None):
        """
        Prepare data for prediction on a specific date.

        Args:
            df: DataFrame with features.
            prediction_date: Date for prediction (last date in window). Defaults to last date in DataFrame.
            window: Number of days in window. Defaults to HISTORICAL_WINDOW.

        Returns:
            DataFrame with features for the prediction window.
        """
        try:
            # Copy to avoid modifying the original
            df_copy = df.copy()

            # Ensure DataFrame has a datetime index
            if not isinstance(df_copy.index, pd.DatetimeIndex):
                logger.warning("DataFrame index is not DatetimeIndex, attempting conversion")
                if 'Date' in df_copy.columns:
                    df_copy = df_copy.set_index('Date')
                    df_copy.index = pd.to_datetime(df_copy.index)
                else:
                    logger.error("Cannot convert DataFrame index to DatetimeIndex")
                    return None

            # Set defaults
            if prediction_date is None:
                prediction_date = df_copy.index.max()
            else:
                prediction_date = pd.to_datetime(prediction_date)

            if window is None:
                window = HISTORICAL_WINDOW

            # Calculate start date
            start_date = prediction_date - timedelta(days=window)

            # Filter data for window
            window_data = df_copy.loc[(df_copy.index >= start_date) & (df_copy.index <= prediction_date)]

            if window_data.empty:
                logger.error(f"No data found for prediction window {start_date} to {prediction_date}")
                return None

            logger.info(f"Prepared prediction data with {len(window_data)} rows from {start_date} to {prediction_date}")

            return window_data

        except Exception as e:
            logger.error(f"Error preparing prediction data: {e}")
            return None

    def preprocess_features(self, df, drop_na=True, fill_method='ffill'):
        """
        Preprocess features for model training.

        Args:
            df: DataFrame with features.
            drop_na: Whether to drop NaN values. Defaults to True.
            fill_method: Method to fill NaN values. Defaults to 'ffill'.

        Returns:
            Preprocessed DataFrame.
        """
        try:
            # Make a copy to avoid modifying the original
            df_processed = df.copy()

            # Fill NaN values
            if fill_method:
                df_processed = df_processed.fillna(method=fill_method)

                # For any remaining NaNs, fill with 0
                df_processed = df_processed.fillna(0)

            # Drop rows with NaN values
            if drop_na:
                original_len = len(df_processed)
                df_processed = df_processed.dropna()
                dropped = original_len - len(df_processed)
                if dropped > 0:
                    logger.info(f"Dropped {dropped} rows with NaN values")

            logger.info(f"Preprocessed features, final shape: {df_processed.shape}")

            return df_processed

        except Exception as e:
            logger.error(f"Error preprocessing features: {e}")
            return df

    def process_stock_data(self, stock_df, sentiment_df=None, add_target=True, target_periods=(1, 5)):
        """
        Process stock data by adding all features and target variables.

        Args:
            stock_df: DataFrame with stock data.
            sentiment_df: DataFrame with sentiment data. Defaults to None.
            add_target: Whether to add target variables. Defaults to True.
            target_periods: List of periods for target variables. Defaults to (1, 5).

        Returns:
            Processed DataFrame.
        """
        try:
            # Start with adding technical indicators
            processed_df = self.calculate_technical_indicators(stock_df)

            # Add time features
            processed_df = self.add_time_features(processed_df)

            # Add sentiment features if provided
            if sentiment_df is not None and not sentiment_df.empty:
                processed_df = self.add_sentiment_features(processed_df, sentiment_df)

            # Add target variables if requested
            if add_target:
                for period in target_periods:
                    processed_df = self.create_target_variable(processed_df, periods=period)

            # Save processed data
            os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

            # Get ticker from DataFrame if available
            ticker = stock_df['ticker'].iloc[0] if 'ticker' in stock_df.columns else 'combined'

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_processed_{timestamp}.csv")
            processed_df.to_csv(csv_path)
            logger.info(f"Saved processed data to {csv_path}")

            return processed_df

        except Exception as e:
            logger.error(f"Error processing stock data: {e}")
            return stock_df


def main():
    """
    Main function to test the feature engineering module.
    """
    import yfinance as yf

    # Download sample data
    ticker = 'AAPL'
    logger.info(f"Downloading sample data for {ticker}")
    stock_data = yf.download(ticker, period="1y", interval="1d")
    stock_data['ticker'] = ticker

    # Initialize feature engineering
    feature_eng = FeatureEngineering()

    # Process data
    processed_data = feature_eng.process_stock_data(stock_data)

    # Print results
    logger.info(f"Original data shape: {stock_data.shape}")
    logger.info(f"Processed data shape: {processed_data.shape}")
    logger.info(f"Added {processed_data.shape[1] - stock_data.shape[1]} new features")


if __name__ == "__main__":
    main()