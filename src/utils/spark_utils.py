"""
Spark utility functions for the stock market prediction project.
"""
import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType, TimestampType
import json

# Import project modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from utils.config import SPARK_MASTER, SPARK_APP_NAME

# Set up logger
logger = setup_logger('spark_utils')

def create_spark_session(app_name=None, master=None, config=None):
    """
    Create a Spark session.
    
    Args:
        app_name: Spark application name. Defaults to config.SPARK_APP_NAME.
        master: Spark master URL. Defaults to config.SPARK_MASTER.
        config: Dictionary of additional configuration options. Defaults to None.
        
    Returns:
        SparkSession instance.
    """
    try:
        # Set default values
        app_name = app_name if app_name else SPARK_APP_NAME
        master = master if master else SPARK_MASTER
        
        # Start building the session
        builder = SparkSession.builder.appName(app_name).master(master)
        
        # Add additional configuration
        if config:
            for key, value in config.items():
                builder = builder.config(key, value)
        
        # Create the session
        spark = builder.getOrCreate()
        
        logger.info(f"Created Spark session: app_name={app_name}, master={master}")
        
        # Set log level
        spark.sparkContext.setLogLevel("WARN")
        
        return spark
    
    except Exception as e:
        logger.error(f"Failed to create Spark session: {e}")
        return None

def stop_spark_session(spark):
    """
    Stop a Spark session.
    
    Args:
        spark: SparkSession instance.
    """
    if spark:
        try:
            spark.stop()
            logger.info("Stopped Spark session")
        except Exception as e:
            logger.error(f"Failed to stop Spark session: {e}")

def dataframe_to_spark(spark, df, schema=None):
    """
    Convert a pandas DataFrame to a Spark DataFrame.
    
    Args:
        spark: SparkSession instance.
        df: pandas DataFrame.
        schema: Spark DataFrame schema. Defaults to None (infer schema).
        
    Returns:
        Spark DataFrame.
    """
    try:
        if schema:
            spark_df = spark.createDataFrame(df, schema=schema)
        else:
            spark_df = spark.createDataFrame(df)
        
        logger.info(f"Converted pandas DataFrame to Spark DataFrame: {df.shape} -> {spark_df.count()} rows")
        return spark_df
    
    except Exception as e:
        logger.error(f"Failed to convert pandas DataFrame to Spark DataFrame: {e}")
        return None

def spark_to_dataframe(spark_df):
    """
    Convert a Spark DataFrame to a pandas DataFrame.
    
    Args:
        spark_df: Spark DataFrame.
        
    Returns:
        pandas DataFrame.
    """
    try:
        df = spark_df.toPandas()
        logger.info(f"Converted Spark DataFrame to pandas DataFrame: {spark_df.count()} rows -> {df.shape}")
        return df
    
    except Exception as e:
        logger.error(f"Failed to convert Spark DataFrame to pandas DataFrame: {e}")
        return None

def load_dataframe_from_hdfs(spark, hdfs_path, file_format='parquet', header=True, infer_schema=True, schema=None):
    """
    Load a DataFrame from HDFS.
    
    Args:
        spark: SparkSession instance.
        hdfs_path: Path to HDFS file or directory.
        file_format: File format ('csv', 'parquet', 'json'). Defaults to 'parquet'.
        header: Whether the file has a header row (for CSV). Defaults to True.
        infer_schema: Whether to infer the schema (for CSV). Defaults to True.
        schema: Explicit schema (for CSV). Defaults to None.
        
    Returns:
        Spark DataFrame if successful, None otherwise.
    """
    try:
        # Set up read options based on file format
        reader = spark.read
        
        if file_format.lower() == 'csv':
            reader = reader.option("header", header)
            
            if infer_schema:
                reader = reader.option("inferSchema", True)
            elif schema:
                reader = reader.schema(schema)
        
        # Load the data
        spark_df = reader.format(file_format).load(hdfs_path)
        
        logger.info(f"Loaded {spark_df.count()} rows from HDFS path {hdfs_path} as {file_format}")
        return spark_df
    
    except Exception as e:
        logger.error(f"Failed to load DataFrame from HDFS: {e}")
        return None

def save_dataframe_to_hdfs(spark_df, hdfs_path, file_format='parquet', mode='overwrite', partition_by=None):
    """
    Save a Spark DataFrame to HDFS.
    
    Args:
        spark_df: Spark DataFrame.
        hdfs_path: Path to HDFS file or directory.
        file_format: File format ('csv', 'parquet', 'json'). Defaults to 'parquet'.
        mode: Save mode ('overwrite', 'append', 'ignore', 'error'). Defaults to 'overwrite'.
        partition_by: Column(s) to partition by. Defaults to None.
        
    Returns:
        True if successful, False otherwise.
    """
    try:
        writer = spark_df.write.mode(mode).format(file_format)
        
        if partition_by:
            if isinstance(partition_by, str):
                partition_by = [partition_by]
            writer = writer.partitionBy(*partition_by)
        
        writer.save(hdfs_path)
        
        logger.info(f"Saved {spark_df.count()} rows to HDFS path {hdfs_path} as {file_format}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to save DataFrame to HDFS: {e}")
        return False

def create_stock_data_schema():
    """
    Create a schema for stock data.
    
    Returns:
        Spark schema for stock data.
    """
    return StructType([
        StructField("ticker", StringType(), False),
        StructField("date", TimestampType(), False),
        StructField("open", FloatType(), True),
        StructField("high", FloatType(), True),
        StructField("low", FloatType(), True),
        StructField("close", FloatType(), True),
        StructField("volume", FloatType(), True),
        StructField("adj_close", FloatType(), True)
    ])

def create_sentiment_data_schema():
    """
    Create a schema for sentiment data.
    
    Returns:
        Spark schema for sentiment data.
    """
    return StructType([
        StructField("ticker", StringType(), False),
        StructField("date", TimestampType(), False),
        StructField("source", StringType(), True),
        StructField("text", StringType(), True),
        StructField("polarity", FloatType(), True),
        StructField("subjectivity", FloatType(), True),
        StructField("sentiment", StringType(), True)
    ])

def create_economic_data_schema():
    """
    Create a schema for economic data.
    
    Returns:
        Spark schema for economic data.
    """
    return StructType([
        StructField("series_id", StringType(), False),
        StructField("date", TimestampType(), False),
        StructField("value", FloatType(), True),
        StructField("indicator_name", StringType(), True)
    ])

def run_spark_sql(spark, spark_df, sql_query):
    """
    Run a SQL query on a Spark DataFrame.
    
    Args:
        spark: SparkSession instance.
        spark_df: Spark DataFrame.
        sql_query: SQL query string.
        
    Returns:
        Result of the SQL query as a Spark DataFrame.
    """
    try:
        # Register the DataFrame as a temp view
        temp_view_name = "temp_table"
        spark_df.createOrReplaceTempView(temp_view_name)
        
        # Run the query
        result = spark.sql(sql_query)
        
        logger.info(f"Executed SQL query: {sql_query}")
        return result
    
    except Exception as e:
        logger.error(f"Failed to run SQL query: {e}")
        return None

def load_mongodb_to_spark(spark, mongo_handler, collection_name, query=None):
    """
    Load data from MongoDB into a Spark DataFrame.
    
    Args:
        spark: SparkSession instance.
        mongo_handler: MongoDB handler instance.
        collection_name: MongoDB collection name.
        query: MongoDB query. Defaults to None (all documents).
        
    Returns:
        Spark DataFrame if successful, None otherwise.
    """
    try:
        # Load data from MongoDB as pandas DataFrame
        pandas_df = mongo_handler.load_dataframe(collection_name, query=query)
        
        if pandas_df is None or pandas_df.empty:
            logger.warning(f"No data found in MongoDB collection {collection_name}")
            return None
        
        # Convert to Spark DataFrame
        spark_df = dataframe_to_spark(spark, pandas_df)
        
        return spark_df
    
    except Exception as e:
        logger.error(f"Failed to load MongoDB data into Spark: {e}")
        return None 