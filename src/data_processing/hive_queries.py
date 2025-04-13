"""
Module for executing Hive and SparkSQL queries for pattern analysis.
"""
import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr
import json

# Import project modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from utils.config import HIVE_HOST, HIVE_PORT, HIVE_USER, HIVE_PASSWORD
from utils.spark_utils import create_spark_session, stop_spark_session

# Set up logger
logger = setup_logger('hive_queries')

class HiveQueryExecutor:
    """
    Class for executing Hive queries for pattern analysis.
    """
    def __init__(self, spark=None):
        """
        Initialize the HiveQueryExecutor.
        
        Args:
            spark: SparkSession instance. Defaults to None (create a new session).
        """
        # Create a SparkSession with Hive support if not provided
        if spark is None:
            try:
                self.spark = SparkSession.builder \
                    .appName("HiveQueryExecutor") \
                    .enableHiveSupport() \
                    .getOrCreate()
                
                logger.info("Created SparkSession with Hive support")
            except Exception as e:
                logger.error(f"Failed to create SparkSession with Hive support: {e}")
                # Fall back to a regular SparkSession
                self.spark = create_spark_session()
                logger.warning("Falling back to regular SparkSession without Hive support")
        else:
            self.spark = spark
            logger.info("Using provided SparkSession")
    
    def execute_query(self, query):
        """
        Execute a Hive/SparkSQL query.
        
        Args:
            query: SQL query to execute.
            
        Returns:
            Spark DataFrame with query results.
        """
        try:
            result = self.spark.sql(query)
            logger.info(f"Executed query: {query}")
            return result
        
        except Exception as e:
            logger.error(f"Failed to execute query: {e}")
            return None
    
    def create_temp_view(self, df, view_name):
        """
        Create a temporary view for a DataFrame.
        
        Args:
            df: Spark DataFrame.
            view_name: Name for the temporary view.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            df.createOrReplaceTempView(view_name)
            logger.info(f"Created temporary view: {view_name}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to create temporary view: {e}")
            return False
    
    def create_hive_table(self, df, table_name, database=None, mode='overwrite', partition_by=None):
        """
        Create a Hive table from a DataFrame.
        
        Args:
            df: Spark DataFrame.
            table_name: Name for the Hive table.
            database: Database name. Defaults to None (use default database).
            mode: Save mode ('overwrite', 'append', 'ignore', 'error'). Defaults to 'overwrite'.
            partition_by: Column(s) to partition by. Defaults to None.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            # Create database if specified and doesn't exist
            if database:
                self.spark.sql(f"CREATE DATABASE IF NOT EXISTS {database}")
                full_table_name = f"{database}.{table_name}"
            else:
                full_table_name = table_name
            
            # Create writer
            writer = df.write.mode(mode)
            
            # Add partitioning if specified
            if partition_by:
                if isinstance(partition_by, str):
                    partition_by = [partition_by]
                writer = writer.partitionBy(*partition_by)
            
            # Save as Hive table
            writer.saveAsTable(full_table_name)
            
            logger.info(f"Created Hive table: {full_table_name}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to create Hive table: {e}")
            return False
    
    def list_tables(self, database=None):
        """
        List tables in a database.
        
        Args:
            database: Database name. Defaults to None (use default database).
            
        Returns:
            List of table names.
        """
        try:
            if database:
                tables_df = self.spark.sql(f"SHOW TABLES IN {database}")
            else:
                tables_df = self.spark.sql("SHOW TABLES")
            
            # Extract table names
            if 'tableName' in tables_df.columns:
                table_names = [row['tableName'] for row in tables_df.collect()]
            else:
                # Newer versions of Spark use 'name' instead of 'tableName'
                table_names = [row['name'] for row in tables_df.collect()]
            
            return table_names
        
        except Exception as e:
            logger.error(f"Failed to list tables: {e}")
            return []
    
    def close(self):
        """
        Close the SparkSession.
        """
        stop_spark_session(self.spark)
        logger.info("Closed SparkSession")


class StockAnalysisQueries:
    """
    Class containing pre-defined queries for stock market pattern analysis.
    """
    def __init__(self, hive_executor=None):
        """
        Initialize the StockAnalysisQueries.
        
        Args:
            hive_executor: HiveQueryExecutor instance. Defaults to None (create a new one).
        """
        self.hive_executor = hive_executor if hive_executor else HiveQueryExecutor()
    
    def find_high_volatility_stocks(self, table_name, period=30, volatility_threshold=0.02):
        """
        Find stocks with high volatility.
        
        Args:
            table_name: Name of the table containing stock data.
            period: Number of days to consider. Defaults to 30.
            volatility_threshold: Threshold for high volatility. Defaults to 0.02 (2%).
            
        Returns:
            Spark DataFrame with high volatility stocks.
        """
        query = f"""
        WITH daily_returns AS (
            SELECT
                ticker,
                date,
                close,
                (close - LAG(close) OVER (PARTITION BY ticker ORDER BY date)) / LAG(close) OVER (PARTITION BY ticker ORDER BY date) AS daily_return
            FROM {table_name}
        ),
        volatility AS (
            SELECT
                ticker,
                STDDEV(daily_return) AS volatility
            FROM daily_returns
            WHERE date >= DATE_SUB(CURRENT_DATE, {period})
            GROUP BY ticker
        )
        SELECT
            ticker,
            volatility
        FROM volatility
        WHERE volatility > {volatility_threshold}
        ORDER BY volatility DESC
        """
        
        return self.hive_executor.execute_query(query)
    
    def find_price_trends(self, table_name, period=30, trend_threshold=0.1):
        """
        Find stocks with significant price trends.
        
        Args:
            table_name: Name of the table containing stock data.
            period: Number of days to consider. Defaults to 30.
            trend_threshold: Threshold for significant trend. Defaults to 0.1 (10%).
            
        Returns:
            Spark DataFrame with trending stocks.
        """
        query = f"""
        WITH price_changes AS (
            SELECT
                ticker,
                MIN(date) AS start_date,
                MAX(date) AS end_date,
                FIRST_VALUE(close) OVER (PARTITION BY ticker ORDER BY date) AS start_price,
                LAST_VALUE(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS end_price
            FROM {table_name}
            WHERE date >= DATE_SUB(CURRENT_DATE, {period})
            GROUP BY ticker, close
        ),
        trends AS (
            SELECT
                ticker,
                start_date,
                end_date,
                start_price,
                end_price,
                (end_price - start_price) / start_price AS price_change
            FROM price_changes
        )
        SELECT
            ticker,
            start_date,
            end_date,
            start_price,
            end_price,
            price_change,
            CASE
                WHEN price_change > {trend_threshold} THEN 'Uptrend'
                WHEN price_change < -{trend_threshold} THEN 'Downtrend'
                ELSE 'Sideways'
            END AS trend_direction
        FROM trends
        WHERE ABS(price_change) > {trend_threshold}
        ORDER BY ABS(price_change) DESC
        """
        
        return self.hive_executor.execute_query(query)
    
    def find_correlation_with_sentiment(self, stock_table, sentiment_table, period=30):
        """
        Find correlation between stock prices and sentiment.
        
        Args:
            stock_table: Name of the table containing stock data.
            sentiment_table: Name of the table containing sentiment data.
            period: Number of days to consider. Defaults to 30.
            
        Returns:
            Spark DataFrame with correlation results.
        """
        query = f"""
        WITH daily_price_change AS (
            SELECT
                ticker,
                date,
                close,
                (close - LAG(close) OVER (PARTITION BY ticker ORDER BY date)) / LAG(close) OVER (PARTITION BY ticker ORDER BY date) AS price_change
            FROM {stock_table}
            WHERE date >= DATE_SUB(CURRENT_DATE, {period})
        ),
        daily_sentiment AS (
            SELECT
                ticker,
                date,
                AVG(polarity) AS avg_sentiment
            FROM {sentiment_table}
            WHERE date >= DATE_SUB(CURRENT_DATE, {period})
            GROUP BY ticker, date
        ),
        combined_data AS (
            SELECT
                p.ticker,
                p.date,
                p.price_change,
                s.avg_sentiment
            FROM daily_price_change p
            JOIN daily_sentiment s ON p.ticker = s.ticker AND p.date = s.date
        ),
        correlations AS (
            SELECT
                ticker,
                CORR(price_change, avg_sentiment) AS correlation
            FROM combined_data
            GROUP BY ticker
        )
        SELECT
            ticker,
            correlation,
            CASE
                WHEN correlation > 0.7 THEN 'Strong Positive'
                WHEN correlation BETWEEN 0.3 AND 0.7 THEN 'Positive'
                WHEN correlation BETWEEN -0.3 AND 0.3 THEN 'Weak'
                WHEN correlation BETWEEN -0.7 AND -0.3 THEN 'Negative'
                WHEN correlation < -0.7 THEN 'Strong Negative'
            END AS correlation_strength
        FROM correlations
        ORDER BY ABS(correlation) DESC
        """
        
        return self.hive_executor.execute_query(query)
    
    def find_correlation_with_economic_indicators(self, stock_table, economic_table, indicator_column, period=365):
        """
        Find correlation between stock prices and economic indicators.
        
        Args:
            stock_table: Name of the table containing stock data.
            economic_table: Name of the table containing economic data.
            indicator_column: Column name for the economic indicator.
            period: Number of days to consider. Defaults to 365 (1 year).
            
        Returns:
            Spark DataFrame with correlation results.
        """
        query = f"""
        WITH monthly_stock_returns AS (
            SELECT
                ticker,
                TRUNC(date, 'MM') AS month,
                (MAX(close) - MIN(close)) / MIN(close) AS monthly_return
            FROM {stock_table}
            WHERE date >= DATE_SUB(CURRENT_DATE, {period})
            GROUP BY ticker, TRUNC(date, 'MM')
        ),
        monthly_indicator AS (
            SELECT
                TRUNC(date, 'MM') AS month,
                AVG({indicator_column}) AS indicator_value
            FROM {economic_table}
            WHERE date >= DATE_SUB(CURRENT_DATE, {period})
            GROUP BY TRUNC(date, 'MM')
        ),
        combined_data AS (
            SELECT
                s.ticker,
                s.month,
                s.monthly_return,
                e.indicator_value
            FROM monthly_stock_returns s
            JOIN monthly_indicator e ON s.month = e.month
        ),
        correlations AS (
            SELECT
                ticker,
                CORR(monthly_return, indicator_value) AS correlation
            FROM combined_data
            GROUP BY ticker
        )
        SELECT
            ticker,
            correlation,
            CASE
                WHEN correlation > 0.7 THEN 'Strong Positive'
                WHEN correlation BETWEEN 0.3 AND 0.7 THEN 'Positive'
                WHEN correlation BETWEEN -0.3 AND 0.3 THEN 'Weak'
                WHEN correlation BETWEEN -0.7 AND -0.3 THEN 'Negative'
                WHEN correlation < -0.7 THEN 'Strong Negative'
            END AS correlation_strength
        FROM correlations
        ORDER BY ABS(correlation) DESC
        """
        
        return self.hive_executor.execute_query(query)
    
    def find_stocks_with_positive_sentiment_momentum(self, sentiment_table, period=30, window=7):
        """
        Find stocks with increasing positive sentiment momentum.
        
        Args:
            sentiment_table: Name of the table containing sentiment data.
            period: Number of days to consider. Defaults to 30.
            window: Window size for moving average. Defaults to 7 (weekly).
            
        Returns:
            Spark DataFrame with stocks showing positive sentiment momentum.
        """
        query = f"""
        WITH daily_sentiment AS (
            SELECT
                ticker,
                date,
                AVG(polarity) AS avg_sentiment
            FROM {sentiment_table}
            WHERE date >= DATE_SUB(CURRENT_DATE, {period})
            GROUP BY ticker, date
        ),
        sentiment_ma AS (
            SELECT
                ticker,
                date,
                avg_sentiment,
                AVG(avg_sentiment) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN {window-1} PRECEDING AND CURRENT ROW) AS sentiment_ma
            FROM daily_sentiment
        ),
        sentiment_momentum AS (
            SELECT
                ticker,
                date,
                sentiment_ma,
                LAG(sentiment_ma, {window}) OVER (PARTITION BY ticker ORDER BY date) AS prev_sentiment_ma,
                sentiment_ma - LAG(sentiment_ma, {window}) OVER (PARTITION BY ticker ORDER BY date) AS momentum
            FROM sentiment_ma
        )
        SELECT
            ticker,
            MAX(date) AS latest_date,
            AVG(sentiment_ma) AS avg_sentiment,
            AVG(momentum) AS avg_momentum
        FROM sentiment_momentum
        WHERE momentum > 0
        GROUP BY ticker
        ORDER BY avg_momentum DESC
        """
        
        return self.hive_executor.execute_query(query)
    
    def find_stocks_with_price_breakouts(self, table_name, period=90, breakout_threshold=0.05):
        """
        Find stocks with price breakouts.
        
        Args:
            table_name: Name of the table containing stock data.
            period: Number of days to consider. Defaults to 90.
            breakout_threshold: Threshold for breakout. Defaults to 0.05 (5%).
            
        Returns:
            Spark DataFrame with stocks showing price breakouts.
        """
        query = f"""
        WITH price_data AS (
            SELECT
                ticker,
                date,
                close,
                MAX(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN {period} PRECEDING AND 1 PRECEDING) AS prev_max,
                MIN(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN {period} PRECEDING AND 1 PRECEDING) AS prev_min
            FROM {table_name}
            WHERE date >= DATE_SUB(CURRENT_DATE, {period})
        ),
        breakouts AS (
            SELECT
                ticker,
                date,
                close,
                prev_max,
                prev_min,
                CASE
                    WHEN close > prev_max * (1 + {breakout_threshold}) THEN 'Upside Breakout'
                    WHEN close < prev_min * (1 - {breakout_threshold}) THEN 'Downside Breakout'
                    ELSE 'No Breakout'
                END AS breakout_type
            FROM price_data
        )
        SELECT
            ticker,
            date,
            close,
            prev_max,
            prev_min,
            breakout_type
        FROM breakouts
        WHERE breakout_type != 'No Breakout'
        ORDER BY date DESC
        """
        
        return self.hive_executor.execute_query(query)
    
    def close(self):
        """
        Close the HiveQueryExecutor.
        """
        if self.hive_executor:
            self.hive_executor.close()
            logger.info("Closed HiveQueryExecutor")


def main():
    """
    Main function to demonstrate Hive queries.
    """
    try:
        # Create a HiveQueryExecutor
        hive_executor = HiveQueryExecutor()
        
        # Create a StockAnalysisQueries instance
        stock_queries = StockAnalysisQueries(hive_executor)
        
        # List tables
        tables = hive_executor.list_tables()
        logger.info(f"Available tables: {tables}")
        
        # Create example DataFrames
        # Note: This is just for demonstration purposes
        # In a real scenario, you would use actual data
        
        # Create a dummy stock data DataFrame
        stock_data = hive_executor.spark.createDataFrame([
            ("AAPL", "2023-01-01", 150.0, 155.0, 148.0, 152.0, 1000000.0),
            ("AAPL", "2023-01-02", 152.0, 158.0, 151.0, 157.0, 1200000.0),
            ("MSFT", "2023-01-01", 250.0, 255.0, 248.0, 253.0, 800000.0),
            ("MSFT", "2023-01-02", 253.0, 260.0, 252.0, 258.0, 900000.0)
        ], ["ticker", "date", "open", "high", "low", "close", "volume"])
        
        # Create a dummy sentiment data DataFrame
        sentiment_data = hive_executor.spark.createDataFrame([
            ("AAPL", "2023-01-01", 0.6, 0.4, "positive"),
            ("AAPL", "2023-01-02", 0.7, 0.5, "positive"),
            ("MSFT", "2023-01-01", 0.5, 0.3, "positive"),
            ("MSFT", "2023-01-02", 0.4, 0.2, "neutral")
        ], ["ticker", "date", "polarity", "subjectivity", "sentiment"])
        
        # Create temporary views
        hive_executor.create_temp_view(stock_data, "stock_data")
        hive_executor.create_temp_view(sentiment_data, "sentiment_data")
        
        # Try a simple query
        result = hive_executor.execute_query("SELECT * FROM stock_data WHERE ticker = 'AAPL'")
        result.show()
        
        # Close resources
        stock_queries.close()
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")


if __name__ == "__main__":
    main() 