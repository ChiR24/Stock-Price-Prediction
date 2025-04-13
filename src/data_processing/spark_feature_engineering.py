"""
Module for feature engineering using Spark for distributed processing.
"""
import os
import pandas as pd
import numpy as np
from pyspark.ml.feature import VectorAssembler, PCA, StandardScaler
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col, lag, when, lit, datediff, expr, udf, array
from pyspark.sql.window import Window
from pyspark.sql.types import FloatType, ArrayType

# Import project modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from utils.spark_utils import create_spark_session, stop_spark_session, spark_to_dataframe

# Set up logger
logger = setup_logger('spark_feature_engineering')

class SparkFeatureEngineering:
    """
    Class for feature engineering using Spark.
    """
    def __init__(self, spark=None):
        """
        Initialize the SparkFeatureEngineering instance.
        
        Args:
            spark: SparkSession instance. Defaults to None (create a new session).
        """
        self.spark = spark if spark else create_spark_session()
        
        if not self.spark:
            logger.error("Failed to create Spark session")
            raise Exception("Failed to create Spark session")
    
    def calculate_technical_indicators(self, spark_df, price_col='close', volume_col='volume', window_sizes=None):
        """
        Calculate technical indicators for stock prices.
        
        Args:
            spark_df: Spark DataFrame with stock price data.
            price_col: Column name for price data. Defaults to 'close'.
            volume_col: Column name for volume data. Defaults to 'volume'.
            window_sizes: Dictionary of window sizes for different indicators.
                         Defaults to None (use default window sizes).
                         
        Returns:
            Spark DataFrame with technical indicators.
        """
        try:
            # Set default window sizes if not provided
            if not window_sizes:
                window_sizes = {
                    'sma': [20, 50, 200],
                    'ema': [12, 26],
                    'rsi': 14,
                    'macd': {'fast': 12, 'slow': 26, 'signal': 9},
                    'bollinger': {'window': 20, 'std_dev': 2}
                }
            
            # Make a copy of the DataFrame
            result_df = spark_df
            
            # Create window specifications for calculations
            ticker_date_window = Window.partitionBy('ticker').orderBy('date')
            
            # Calculate Simple Moving Averages (SMA)
            for window in window_sizes['sma']:
                col_name = f'SMA_{window}'
                result_df = result_df.withColumn(
                    col_name,
                    expr(f"avg({price_col}) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN {window-1} PRECEDING AND CURRENT ROW)")
                )
                logger.info(f"Calculated {col_name}")
            
            # Calculate Exponential Moving Averages (EMA)
            for window in window_sizes['ema']:
                # EMA calculation is more complex in Spark
                # We'll use a UDF for this
                col_name = f'EMA_{window}'
                
                # First calculate SMA for the initial values
                sma_col = f'SMA_{window}_temp'
                result_df = result_df.withColumn(
                    sma_col,
                    expr(f"avg({price_col}) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN {window-1} PRECEDING AND CURRENT ROW)")
                )
                
                # Calculate EMA
                # EMA = Price(t) * k + EMA(y) * (1 – k)
                # where k = 2/(window + 1)
                k = 2.0 / (window + 1)
                
                result_df = result_df.withColumn(
                    col_name,
                    when(
                        col(sma_col).isNotNull(),
                        when(
                            lag(col_name, 1).over(ticker_date_window).isNull(),
                            col(sma_col)
                        ).otherwise(
                            col(price_col) * k + lag(col_name, 1).over(ticker_date_window) * (1 - k)
                        )
                    ).otherwise(None)
                )
                
                # Drop temporary column
                result_df = result_df.drop(sma_col)
                logger.info(f"Calculated {col_name}")
            
            # Calculate Relative Strength Index (RSI)
            window_size = window_sizes['rsi']
            
            # Calculate price changes
            result_df = result_df.withColumn(
                'price_change',
                col(price_col) - lag(col(price_col), 1).over(ticker_date_window)
            )
            
            # Calculate gains and losses
            result_df = result_df.withColumn(
                'gain',
                when(col('price_change') > 0, col('price_change')).otherwise(0)
            )
            result_df = result_df.withColumn(
                'loss',
                when(col('price_change') < 0, -col('price_change')).otherwise(0)
            )
            
            # Calculate average gains and losses
            result_df = result_df.withColumn(
                'avg_gain',
                expr(f"avg(gain) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN {window_size-1} PRECEDING AND CURRENT ROW)")
            )
            result_df = result_df.withColumn(
                'avg_loss',
                expr(f"avg(loss) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN {window_size-1} PRECEDING AND CURRENT ROW)")
            )
            
            # Calculate RS and RSI
            result_df = result_df.withColumn(
                'rs',
                when(col('avg_loss') == 0, 100).otherwise(col('avg_gain') / col('avg_loss'))
            )
            result_df = result_df.withColumn(
                'RSI',
                when(col('avg_loss') == 0, 100).otherwise(100 - (100 / (1 + col('rs'))))
            )
            
            # Drop temporary columns
            result_df = result_df.drop('price_change', 'gain', 'loss', 'avg_gain', 'avg_loss', 'rs')
            logger.info("Calculated RSI")
            
            # Calculate Bollinger Bands
            window_size = window_sizes['bollinger']['window']
            std_dev = window_sizes['bollinger']['std_dev']
            
            # Calculate SMA for Bollinger Bands
            result_df = result_df.withColumn(
                'bollinger_sma',
                expr(f"avg({price_col}) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN {window_size-1} PRECEDING AND CURRENT ROW)")
            )
            
            # Calculate standard deviation
            result_df = result_df.withColumn(
                'bollinger_std',
                expr(f"stddev({price_col}) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN {window_size-1} PRECEDING AND CURRENT ROW)")
            )
            
            # Calculate upper and lower bands
            result_df = result_df.withColumn(
                'Bollinger_Upper',
                col('bollinger_sma') + (col('bollinger_std') * std_dev)
            )
            result_df = result_df.withColumn(
                'Bollinger_Lower',
                col('bollinger_sma') - (col('bollinger_std') * std_dev)
            )
            
            # Drop temporary columns
            result_df = result_df.drop('bollinger_sma', 'bollinger_std')
            logger.info("Calculated Bollinger Bands")
            
            # Calculate MACD
            fast = window_sizes['macd']['fast']
            slow = window_sizes['macd']['slow']
            signal = window_sizes['macd']['signal']
            
            # Use already calculated EMAs
            fast_col = f'EMA_{fast}'
            slow_col = f'EMA_{slow}'
            
            # Calculate MACD line
            result_df = result_df.withColumn(
                'MACD',
                col(fast_col) - col(slow_col)
            )
            
            # Calculate MACD signal line (EMA of MACD)
            # We'll use a simple calculation here
            k = 2.0 / (signal + 1)
            
            result_df = result_df.withColumn(
                'MACD_Signal',
                when(
                    col('MACD').isNotNull(),
                    when(
                        lag('MACD_Signal', 1).over(ticker_date_window).isNull(),
                        col('MACD')
                    ).otherwise(
                        col('MACD') * k + lag('MACD_Signal', 1).over(ticker_date_window) * (1 - k)
                    )
                ).otherwise(None)
            )
            
            # Calculate MACD histogram
            result_df = result_df.withColumn(
                'MACD_Hist',
                col('MACD') - col('MACD_Signal')
            )
            
            logger.info("Calculated MACD indicators")
            
            return result_df
        
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return spark_df
    
    def combine_data_sources(self, stock_df, sentiment_df=None, economic_df=None):
        """
        Combine data from different sources (stock, sentiment, economic).
        
        Args:
            stock_df: Spark DataFrame with stock data.
            sentiment_df: Spark DataFrame with sentiment data. Defaults to None.
            economic_df: Spark DataFrame with economic data. Defaults to None.
            
        Returns:
            Combined Spark DataFrame.
        """
        try:
            # Start with stock data
            result_df = stock_df
            
            # Join with sentiment data if available
            if sentiment_df is not None:
                # Aggregate sentiment data by ticker and date
                sentiment_agg = sentiment_df.groupBy('ticker', 'date') \
                    .agg(
                        {'polarity': 'avg', 'subjectivity': 'avg'}
                    ) \
                    .withColumnRenamed('avg(polarity)', 'sentiment_polarity') \
                    .withColumnRenamed('avg(subjectivity)', 'sentiment_subjectivity')
                
                # Join with stock data
                result_df = result_df.join(
                    sentiment_agg,
                    ['ticker', 'date'],
                    'left'
                )
                
                # Fill missing values
                result_df = result_df.fillna(0, subset=['sentiment_polarity', 'sentiment_subjectivity'])
                
                logger.info("Combined stock data with sentiment data")
            
            # Join with economic data if available
            if economic_df is not None:
                # Economic data is typically not ticker-specific
                # We'll join based on date only, and pivot the economic indicators
                
                # Pivot economic data
                economic_pivot = economic_df.groupBy('date') \
                    .pivot('series_id') \
                    .agg({'value': 'first'}) \
                    .na.fill(method='forward') \
                    .na.fill(method='backward')
                
                # Rename columns to add 'economic_' prefix
                for column in economic_pivot.columns:
                    if column != 'date':
                        economic_pivot = economic_pivot.withColumnRenamed(
                            column, f'economic_{column}'
                        )
                
                # Join with result_df
                result_df = result_df.join(
                    economic_pivot,
                    ['date'],
                    'left'
                )
                
                # Fill missing values with forward fill
                economic_columns = [c for c in result_df.columns if c.startswith('economic_')]
                result_df = result_df.na.fill(method='forward', subset=economic_columns)
                
                logger.info("Combined data with economic indicators")
            
            return result_df
        
        except Exception as e:
            logger.error(f"Error combining data sources: {e}")
            return stock_df
    
    def create_features(self, spark_df, target_col='close', lag_periods=5, include_target=True):
        """
        Create features for machine learning, including lagged values.
        
        Args:
            spark_df: Spark DataFrame with combined data.
            target_col: Target column name. Defaults to 'close'.
            lag_periods: Number of periods to lag. Defaults to 5.
            include_target: Whether to include the target column. Defaults to True.
            
        Returns:
            Spark DataFrame with features.
        """
        try:
            # Make a copy of the DataFrame
            result_df = spark_df
            
            # Define window for lagging
            ticker_date_window = Window.partitionBy('ticker').orderBy('date')
            
            # Create lagged features for numeric columns
            numeric_columns = [
                c for c in result_df.columns 
                if c not in ['ticker', 'date'] and result_df.schema[c].dataType.simpleString() in ['double', 'float', 'int', 'long']
            ]
            
            # Create lag features
            for col_name in numeric_columns:
                for lag_period in range(1, lag_periods + 1):
                    lag_col_name = f"{col_name}_lag_{lag_period}"
                    result_df = result_df.withColumn(
                        lag_col_name,
                        lag(col(col_name), lag_period).over(ticker_date_window)
                    )
            
            # Create target variable (future price)
            if include_target:
                for i in range(1, lag_periods + 1):
                    result_df = result_df.withColumn(
                        f"target_{i}d",
                        lead(col(target_col), i).over(ticker_date_window)
                    )
                
                # Create target returns
                result_df = result_df.withColumn(
                    "target_return_1d",
                    (col("target_1d") - col(target_col)) / col(target_col)
                )
                
                # Create binary target (up/down)
                result_df = result_df.withColumn(
                    "target_direction",
                    when(col("target_return_1d") > 0, 1).otherwise(0)
                )
            
            # Drop rows with null values (from lagging)
            result_df = result_df.na.drop()
            
            logger.info(f"Created features with {lag_periods} lag periods")
            
            return result_df
        
        except Exception as e:
            logger.error(f"Error creating features: {e}")
            return spark_df
    
    def apply_dimensionality_reduction(self, spark_df, feature_cols=None, n_components=10):
        """
        Apply PCA dimensionality reduction.
        
        Args:
            spark_df: Spark DataFrame with features.
            feature_cols: List of feature column names. Defaults to None (use all numeric columns).
            n_components: Number of principal components. Defaults to 10.
            
        Returns:
            Spark DataFrame with PCA features.
        """
        try:
            # Make a copy of the DataFrame
            result_df = spark_df
            
            # If feature columns not provided, use all numeric columns
            if not feature_cols:
                feature_cols = [
                    c for c in result_df.columns 
                    if c not in ['ticker', 'date', 'target_1d', 'target_5d', 'target_return_1d', 'target_direction'] 
                    and result_df.schema[c].dataType.simpleString() in ['double', 'float', 'int', 'long']
                ]
            
            # Standardize features
            assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_vector")
            result_df = assembler.transform(result_df)
            
            scaler = StandardScaler(
                inputCol="features_vector", 
                outputCol="scaled_features",
                withStd=True, 
                withMean=True
            )
            scaler_model = scaler.fit(result_df)
            result_df = scaler_model.transform(result_df)
            
            # Apply PCA
            pca = PCA(
                k=n_components,
                inputCol="scaled_features",
                outputCol="pca_features"
            )
            pca_model = pca.fit(result_df)
            result_df = pca_model.transform(result_df)
            
            # Extract PCA components into individual columns
            # Convert vector to array
            result_df = result_df.withColumn("pca_array", vector_to_array("pca_features"))
            
            # Extract each component
            for i in range(n_components):
                result_df = result_df.withColumn(f"PCA_{i+1}", col("pca_array")[i])
            
            # Drop temporary columns
            result_df = result_df.drop("features_vector", "scaled_features", "pca_features", "pca_array")
            
            logger.info(f"Applied PCA dimensionality reduction with {n_components} components")
            
            return result_df
        
        except Exception as e:
            logger.error(f"Error applying dimensionality reduction: {e}")
            return spark_df
    
    def prepare_data_for_training(self, spark_df, feature_cols=None, target_col="target_direction"):
        """
        Prepare data for training ML models.
        
        Args:
            spark_df: Spark DataFrame with features.
            feature_cols: List of feature column names. Defaults to None (use all numeric columns except targets).
            target_col: Target column name. Defaults to "target_direction".
            
        Returns:
            Spark DataFrame with features vector and label.
        """
        try:
            # Make a copy of the DataFrame
            result_df = spark_df
            
            # If feature columns not provided, use all numeric columns except targets
            if not feature_cols:
                feature_cols = [
                    c for c in result_df.columns 
                    if c not in ['ticker', 'date', 'target_1d', 'target_5d', 'target_return_1d', 'target_direction', 'label'] 
                    and not c.startswith('PCA_')
                    and result_df.schema[c].dataType.simpleString() in ['double', 'float', 'int', 'long']
                ]
            
            # Assemble features into a vector
            assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
            result_df = assembler.transform(result_df)
            
            # Rename target column to "label" for ML
            result_df = result_df.withColumnRenamed(target_col, "label")
            
            logger.info(f"Prepared data for training with {len(feature_cols)} features")
            
            return result_df
        
        except Exception as e:
            logger.error(f"Error preparing data for training: {e}")
            return spark_df
    
    def close(self):
        """
        Close the Spark session.
        """
        stop_spark_session(self.spark)

def main():
    """
    Main function to demonstrate feature engineering.
    """
    try:
        # Create feature engineering instance
        spark_fe = SparkFeatureEngineering()
        
        # Load sample data
        # This is just a placeholder - in a real scenario, you'd load actual data
        data = [(
            "AAPL", "2023-01-01", 150.0, 155.0, 148.0, 152.0, 1000000.0, 152.0
        )]
        columns = ["ticker", "date", "open", "high", "low", "close", "volume", "adj_close"]
        
        # Create DataFrame
        spark_df = spark_fe.spark.createDataFrame(data, columns)
        
        # Apply feature engineering
        result_df = spark_fe.calculate_technical_indicators(spark_df)
        result_df = spark_fe.create_features(result_df)
        
        # Show results
        result_df.show()
        
        # Close Spark session
        spark_fe.close()
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")


if __name__ == "__main__":
    main() 