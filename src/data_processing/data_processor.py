#!/usr/bin/env python
"""
Data processor module for feature engineering.
This module processes raw data and creates features for machine learning models.
"""
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Optional: Import PySpark if available
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, lag, window, mean, stddev, min, max
    from pyspark.sql.window import Window
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False

# Import project modules
from utils.logger import setup_logger
from config.settings import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    DATA_FEATURES_DIR
)

# Set up logger
logger = setup_logger('data_processor')

class DataProcessor:
    """
    Data processor class for feature engineering of stock market data.
    This class provides methods for processing raw data and creating features.
    """
    
    def __init__(self, use_spark=False):
        """
        Initialize the data processor.
        
        Args:
            use_spark (bool): Whether to use Spark for data processing.
        """
        self.use_spark = use_spark and SPARK_AVAILABLE
        
        if self.use_spark:
            try:
                self.spark = SparkSession.builder \
                    .appName("StockMarketDataProcessing") \
                    .config("spark.driver.memory", "4g") \
                    .getOrCreate()
                logger.info("Spark session initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Spark session: {e}")
                self.use_spark = False
        
        logger.info(f"Data processor initialized (use_spark={self.use_spark})")
    
    def load_stock_data(self, ticker):
        """
        Load raw stock data from file.
        
        Args:
            ticker (str): Stock ticker symbol.
            
        Returns:
            pandas.DataFrame or None: Stock data if successful, None otherwise.
        """
        file_path = os.path.join(DATA_RAW_DIR, f"{ticker}_stock_data.csv")
        
        if not os.path.exists(file_path):
            logger.error(f"Stock data file not found: {file_path}")
            return None
        
        try:
            data = pd.read_csv(file_path)
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
            
            # Sort data by date
            data.sort_index(inplace=True)
            
            logger.info(f"Loaded stock data for {ticker}: {len(data)} samples")
            return data
        
        except Exception as e:
            logger.error(f"Error loading stock data for {ticker}: {e}")
            return None
    
    def load_sentiment_data(self, ticker):
        """
        Load sentiment data from file.
        
        Args:
            ticker (str): Stock ticker symbol.
            
        Returns:
            pandas.DataFrame or None: Sentiment data if successful, None otherwise.
        """
        file_path = os.path.join(DATA_RAW_DIR, f"{ticker}_sentiment_data.csv")
        
        if not os.path.exists(file_path):
            logger.warning(f"Sentiment data file not found: {file_path}")
            return None
        
        try:
            data = pd.read_csv(file_path)
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
            
            # Sort data by date
            data.sort_index(inplace=True)
            
            logger.info(f"Loaded sentiment data for {ticker}: {len(data)} samples")
            return data
        
        except Exception as e:
            logger.error(f"Error loading sentiment data for {ticker}: {e}")
            return None
    
    def load_economic_data(self):
        """
        Load economic data from file.
        
        Returns:
            pandas.DataFrame or None: Economic data if successful, None otherwise.
        """
        file_path = os.path.join(DATA_RAW_DIR, "economic_data.csv")
        
        if not os.path.exists(file_path):
            logger.warning(f"Economic data file not found: {file_path}")
            return None
        
        try:
            data = pd.read_csv(file_path)
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
            
            # Sort data by date
            data.sort_index(inplace=True)
            
            logger.info(f"Loaded economic data: {len(data)} samples")
            return data
        
        except Exception as e:
            logger.error(f"Error loading economic data: {e}")
            return None
    
    def process_ticker_data(self, ticker, include_sentiment=True, include_economic=True):
        """
        Process data for a specific ticker.
        
        Args:
            ticker (str): Stock ticker symbol.
            include_sentiment (bool): Whether to include sentiment data.
            include_economic (bool): Whether to include economic data.
            
        Returns:
            pandas.DataFrame or None: Processed data if successful, None otherwise.
        """
        # Load stock data
        stock_data = self.load_stock_data(ticker)
        
        if stock_data is None:
            return None
        
        # Load sentiment data if requested
        sentiment_data = None
        if include_sentiment:
            sentiment_data = self.load_sentiment_data(ticker)
        
        # Load economic data if requested
        economic_data = None
        if include_economic:
            economic_data = self.load_economic_data()
        
        # Process data using Spark or Pandas
        if self.use_spark:
            return self._process_with_spark(
                stock_data=stock_data,
                sentiment_data=sentiment_data,
                economic_data=economic_data,
                ticker=ticker
            )
        else:
            return self._process_with_pandas(
                stock_data=stock_data,
                sentiment_data=sentiment_data,
                economic_data=economic_data,
                ticker=ticker
            )
    
    def _process_with_pandas(self, stock_data, sentiment_data, economic_data, ticker):
        """
        Process data using Pandas.
        
        Args:
            stock_data (pandas.DataFrame): Stock data.
            sentiment_data (pandas.DataFrame): Sentiment data.
            economic_data (pandas.DataFrame): Economic data.
            ticker (str): Stock ticker symbol.
            
        Returns:
            pandas.DataFrame: Processed data.
        """
        logger.info(f"Processing {ticker} data with Pandas")
        
        # Make a copy of the data
        data = stock_data.copy()
        
        # Calculate price and volume indicators
        self._calculate_price_indicators(data)
        self._calculate_volume_indicators(data)
        
        # Calculate technical indicators
        self._calculate_technical_indicators(data)
        
        # Merge sentiment data if available
        if sentiment_data is not None:
            data = data.join(sentiment_data, how='left')
            
            # Forward fill missing sentiment data
            sentiment_columns = sentiment_data.columns
            data[sentiment_columns] = data[sentiment_columns].fillna(method='ffill')
        
        # Merge economic data if available
        if economic_data is not None:
            data = data.join(economic_data, how='left')
            
            # Forward fill missing economic data
            economic_columns = economic_data.columns
            data[economic_columns] = data[economic_columns].fillna(method='ffill')
        
        # Remove rows with NaN values
        data_cleaned = data.dropna()
        
        # Log the data loss
        data_loss = 1 - len(data_cleaned) / len(data)
        logger.info(f"Data loss after cleaning: {data_loss:.2%}")
        
        # Save processed data
        output_file = os.path.join(DATA_PROCESSED_DIR, f"{ticker}_processed.csv")
        data_cleaned.to_csv(output_file)
        
        logger.info(f"Processed data saved to {output_file}")
        
        return data_cleaned
    
    def _process_with_spark(self, stock_data, sentiment_data, economic_data, ticker):
        """
        Process data using Spark.
        
        Args:
            stock_data (pandas.DataFrame): Stock data.
            sentiment_data (pandas.DataFrame): Sentiment data.
            economic_data (pandas.DataFrame): Economic data.
            ticker (str): Stock ticker symbol.
            
        Returns:
            pandas.DataFrame: Processed data.
        """
        logger.info(f"Processing {ticker} data with Spark")
        
        # Convert Pandas DataFrames to Spark DataFrames
        spark_stock_df = self.spark.createDataFrame(stock_data.reset_index())
        
        # Create a window for lagging operations
        window_spec = Window.orderBy("Date")
        
        # Calculate price indicators
        spark_stock_df = self._calculate_price_indicators_spark(spark_stock_df, window_spec)
        
        # Calculate volume indicators
        spark_stock_df = self._calculate_volume_indicators_spark(spark_stock_df, window_spec)
        
        # Calculate technical indicators
        spark_stock_df = self._calculate_technical_indicators_spark(spark_stock_df, window_spec)
        
        # Convert back to Pandas for merging with other data sources
        # (Spark join operations are more complex for time series data)
        data = spark_stock_df.toPandas()
        data.set_index('Date', inplace=True)
        
        # Merge sentiment data if available
        if sentiment_data is not None:
            data = data.join(sentiment_data, how='left')
            
            # Forward fill missing sentiment data
            sentiment_columns = sentiment_data.columns
            data[sentiment_columns] = data[sentiment_columns].fillna(method='ffill')
        
        # Merge economic data if available
        if economic_data is not None:
            data = data.join(economic_data, how='left')
            
            # Forward fill missing economic data
            economic_columns = economic_data.columns
            data[economic_columns] = data[economic_columns].fillna(method='ffill')
        
        # Remove rows with NaN values
        data_cleaned = data.dropna()
        
        # Log the data loss
        data_loss = 1 - len(data_cleaned) / len(data)
        logger.info(f"Data loss after cleaning: {data_loss:.2%}")
        
        # Save processed data
        output_file = os.path.join(DATA_PROCESSED_DIR, f"{ticker}_processed.csv")
        data_cleaned.to_csv(output_file)
        
        logger.info(f"Processed data saved to {output_file}")
        
        return data_cleaned
    
    def _calculate_price_indicators(self, data):
        """
        Calculate price-based indicators using Pandas.
        
        Args:
            data (pandas.DataFrame): Stock price data.
        """
        # Calculate daily returns
        data['daily_return'] = data['Close'].pct_change()
        
        # Calculate moving averages
        data['ma5'] = data['Close'].rolling(window=5).mean()
        data['ma10'] = data['Close'].rolling(window=10).mean()
        data['ma20'] = data['Close'].rolling(window=20).mean()
        data['ma50'] = data['Close'].rolling(window=50).mean()
        data['ma200'] = data['Close'].rolling(window=200).mean()
        
        # Calculate price volatility (standard deviation of returns)
        data['volatility_5d'] = data['daily_return'].rolling(window=5).std()
        data['volatility_10d'] = data['daily_return'].rolling(window=10).std()
        data['volatility_20d'] = data['daily_return'].rolling(window=20).std()
        
        # Calculate price momentum
        data['momentum_5d'] = data['Close'] / data['Close'].shift(5) - 1
        data['momentum_10d'] = data['Close'] / data['Close'].shift(10) - 1
        data['momentum_20d'] = data['Close'] / data['Close'].shift(20) - 1
        
        # Calculate price gaps
        data['gap'] = data['Open'] / data['Close'].shift(1) - 1
        
        # Calculate high-low range
        data['hl_range'] = (data['High'] - data['Low']) / data['Close']
        
        # Calculate if price is above moving averages
        data['above_ma50'] = (data['Close'] > data['ma50']).astype(int)
        data['above_ma200'] = (data['Close'] > data['ma200']).astype(int)
        
        # Calculate moving average crossovers
        data['ma_cross_5_20'] = ((data['ma5'] > data['ma20']) & 
                                 (data['ma5'].shift(1) <= data['ma20'].shift(1))).astype(int)
        data['ma_cross_10_50'] = ((data['ma10'] > data['ma50']) & 
                                  (data['ma10'].shift(1) <= data['ma50'].shift(1))).astype(int)
    
    def _calculate_volume_indicators(self, data):
        """
        Calculate volume-based indicators using Pandas.
        
        Args:
            data (pandas.DataFrame): Stock price and volume data.
        """
        # Calculate volume moving averages
        data['volume_ma5'] = data['Volume'].rolling(window=5).mean()
        data['volume_ma10'] = data['Volume'].rolling(window=10).mean()
        data['volume_ma20'] = data['Volume'].rolling(window=20).mean()
        
        # Calculate volume change
        data['volume_change'] = data['Volume'].pct_change()
        
        # Calculate volume volatility
        data['volume_volatility'] = data['volume_change'].rolling(window=10).std()
        
        # Calculate volume relative to moving average
        data['volume_ratio_ma5'] = data['Volume'] / data['volume_ma5']
        data['volume_ratio_ma20'] = data['Volume'] / data['volume_ma20']
        
        # Calculate on-balance volume (OBV)
        data['obv'] = 0
        data.loc[data['daily_return'] > 0, 'obv'] = data['Volume']
        data.loc[data['daily_return'] < 0, 'obv'] = -data['Volume']
        data['obv'] = data['obv'].cumsum()
    
    def _calculate_technical_indicators(self, data):
        """
        Calculate technical indicators using Pandas.
        
        Args:
            data (pandas.DataFrame): Stock price and volume data.
        """
        # Calculate Relative Strength Index (RSI)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate Bollinger Bands
        data['bb_middle'] = data['Close'].rolling(window=20).mean()
        data['bb_std'] = data['Close'].rolling(window=20).std()
        data['bb_upper'] = data['bb_middle'] + 2 * data['bb_std']
        data['bb_lower'] = data['bb_middle'] - 2 * data['bb_std']
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        
        # Calculate MACD
        data['ema12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['ema26'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['macd'] = data['ema12'] - data['ema26']
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        data['macd_hist'] = data['macd'] - data['macd_signal']
        
        # Calculate Average Directional Index (ADX)
        # This is a simplified version
        high_diff = data['High'].diff()
        low_diff = data['Low'].diff().abs() * -1
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        tr = pd.DataFrame([
            data['High'] - data['Low'],
            (data['High'] - data['Close'].shift(1)).abs(),
            (data['Low'] - data['Close'].shift(1)).abs()
        ]).max()
        
        atr = tr.rolling(14).mean()
        
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
        
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
        data['adx'] = dx.rolling(14).mean()
    
    def _calculate_price_indicators_spark(self, df, window_spec):
        """
        Calculate price-based indicators using Spark.
        
        Args:
            df (pyspark.sql.DataFrame): Stock price data.
            window_spec (pyspark.sql.window.Window): Window specification.
            
        Returns:
            pyspark.sql.DataFrame: DataFrame with price indicators.
        """
        # Calculate daily returns
        df = df.withColumn("daily_return", 
                          (col("Close") - lag("Close", 1).over(window_spec)) / lag("Close", 1).over(window_spec))
        
        # Calculate moving averages
        for days in [5, 10, 20, 50, 200]:
            window_avg = Window.orderBy("Date").rowsBetween(-days, 0)
            df = df.withColumn(f"ma{days}", mean("Close").over(window_avg))
        
        # Calculate price volatility
        for days in [5, 10, 20]:
            window_vol = Window.orderBy("Date").rowsBetween(-days, 0)
            df = df.withColumn(f"volatility_{days}d", stddev("daily_return").over(window_vol))
        
        # Calculate price momentum
        for days in [5, 10, 20]:
            df = df.withColumn(f"momentum_{days}d", 
                              col("Close") / lag("Close", days).over(window_spec) - 1)
        
        # Calculate price gaps
        df = df.withColumn("gap", col("Open") / lag("Close", 1).over(window_spec) - 1)
        
        # Calculate high-low range
        df = df.withColumn("hl_range", (col("High") - col("Low")) / col("Close"))
        
        # Calculate if price is above moving averages
        df = df.withColumn("above_ma50", (col("Close") > col("ma50")).cast("int"))
        df = df.withColumn("above_ma200", (col("Close") > col("ma200")).cast("int"))
        
        return df
    
    def _calculate_volume_indicators_spark(self, df, window_spec):
        """
        Calculate volume-based indicators using Spark.
        
        Args:
            df (pyspark.sql.DataFrame): Stock price and volume data.
            window_spec (pyspark.sql.window.Window): Window specification.
            
        Returns:
            pyspark.sql.DataFrame: DataFrame with volume indicators.
        """
        # Calculate volume moving averages
        for days in [5, 10, 20]:
            window_avg = Window.orderBy("Date").rowsBetween(-days, 0)
            df = df.withColumn(f"volume_ma{days}", mean("Volume").over(window_avg))
        
        # Calculate volume change
        df = df.withColumn("volume_change", 
                          (col("Volume") - lag("Volume", 1).over(window_spec)) / lag("Volume", 1).over(window_spec))
        
        # Calculate volume relative to moving average
        df = df.withColumn("volume_ratio_ma5", col("Volume") / col("volume_ma5"))
        df = df.withColumn("volume_ratio_ma20", col("Volume") / col("volume_ma20"))
        
        return df
    
    def _calculate_technical_indicators_spark(self, df, window_spec):
        """
        Calculate technical indicators using Spark.
        
        Args:
            df (pyspark.sql.DataFrame): Stock price and volume data.
            window_spec (pyspark.sql.window.Window): Window specification.
            
        Returns:
            pyspark.sql.DataFrame: DataFrame with technical indicators.
        """
        # For complex indicators like RSI, MACD, Bollinger Bands,
        # it may be more efficient to convert to pandas and back
        
        # Add placeholder columns that will be filled later
        df = df.withColumn("rsi", col("Close") * 0)
        df = df.withColumn("macd", col("Close") * 0)
        df = df.withColumn("bb_width", col("Close") * 0)
        
        return df
    
    def create_features_for_training(self, data, ticker):
        """
        Create features for training machine learning models.
        
        Args:
            data (pandas.DataFrame): Processed data.
            ticker (str): Stock ticker symbol.
            
        Returns:
            pandas.DataFrame: Features for training.
        """
        logger.info(f"Creating features for {ticker}")
        
        # Make a copy of the data
        features = data.copy()
        
        # Create target variables (future returns)
        features['target_1d'] = features['Close'].pct_change(1).shift(-1)
        features['target_5d'] = features['Close'].pct_change(5).shift(-5)
        features['target_10d'] = features['Close'].pct_change(10).shift(-10)
        features['target_20d'] = features['Close'].pct_change(20).shift(-20)
        
        # Create binary target variables (up/down)
        features['target_1d_up'] = (features['target_1d'] > 0).astype(int)
        features['target_5d_up'] = (features['target_5d'] > 0).astype(int)
        features['target_10d_up'] = (features['target_10d'] > 0).astype(int)
        features['target_20d_up'] = (features['target_20d'] > 0).astype(int)
        
        # Remove rows with NaN target values
        features = features.dropna(subset=['target_20d'])
        
        # Save features
        output_file = os.path.join(DATA_FEATURES_DIR, f"{ticker}_features.csv")
        features.to_csv(output_file)
        
        logger.info(f"Features saved to {output_file}: {len(features)} samples with {len(features.columns)} features")
        
        return features
    
    def get_feature_importance(self, model, feature_names, top_n=20):
        """
        Get feature importance from a model.
        
        Args:
            model: Trained model with feature_importances_ attribute.
            feature_names (list): List of feature names.
            top_n (int): Number of top features to return.
            
        Returns:
            pandas.DataFrame: Feature importance.
        """
        if not hasattr(model, 'feature_importances_'):
            logger.error("Model does not have feature_importances_ attribute")
            return None
        
        # Create a DataFrame with feature names and importance
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        })
        
        # Sort by importance and get top N
        importance = importance.sort_values('importance', ascending=False).head(top_n)
        
        return importance
    
    def close(self):
        """Close Spark session if it exists."""
        if hasattr(self, 'spark') and self.spark is not None:
            self.spark.stop()
            logger.info("Spark session closed")


if __name__ == "__main__":
    # Test the data processor
    processor = DataProcessor(use_spark=False)
    
    # Process data for a ticker
    ticker = "AAPL"
    processed_data = processor.process_ticker_data(ticker)
    
    if processed_data is not None:
        # Create features
        features = processor.create_features_for_training(processed_data, ticker)
        
        print(f"Features shape: {features.shape}")
        print(f"Features columns: {features.columns}")
        
    # Close spark session
    processor.close() 