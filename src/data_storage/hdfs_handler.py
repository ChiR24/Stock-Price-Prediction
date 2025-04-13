"""
Module for handling HDFS operations for the stock market prediction project.
"""
import os
import json
import pandas as pd
from hdfs import InsecureClient
from hdfs.ext.avro import AvroWriter
from pyarrow import parquet as pq
import pyarrow as pa
import io

# Import project modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from utils.config import HDFS_HOST, HDFS_PORT, HDFS_USER

# Set up logger
logger = setup_logger('hdfs_handler')

class HDFSHandler:
    """
    Class to handle HDFS operations for the stock market prediction project.
    """
    def __init__(self, host=None, port=None, user=None):
        """
        Initialize the HDFS handler.
        
        Args:
            host: HDFS host. Defaults to the one in config.
            port: HDFS port. Defaults to the one in config.
            user: HDFS user. Defaults to the one in config.
        """
        self.host = host if host else HDFS_HOST
        self.port = port if port else HDFS_PORT
        self.user = user if user else HDFS_USER
        self.client = None
        
        # Connect to HDFS
        self.connect()
    
    def connect(self):
        """
        Connect to HDFS.
        
        Returns:
            True if connection is successful, False otherwise.
        """
        try:
            hdfs_url = f"http://{self.host}:{self.port}"
            self.client = InsecureClient(hdfs_url, user=self.user)
            logger.info(f"Connected to HDFS: {hdfs_url}, user: {self.user}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to HDFS: {e}")
            return False
    
    def write_file(self, local_path, hdfs_path, overwrite=False):
        """
        Write a local file to HDFS.
        
        Args:
            local_path: Path to local file.
            hdfs_path: Path to HDFS file.
            overwrite: Whether to overwrite existing file. Defaults to False.
            
        Returns:
            True if successful, False otherwise.
        """
        if not self.client:
            logger.error("HDFS not connected")
            return False
        
        try:
            # Create parent directory if it doesn't exist
            parent_dir = os.path.dirname(hdfs_path)
            if parent_dir and not self.client.status(parent_dir, strict=False):
                self.client.makedirs(parent_dir)
            
            # Write file
            with open(local_path, 'rb') as local_file:
                self.client.write(hdfs_path, local_file, overwrite=overwrite)
            
            logger.info(f"Written local file {local_path} to HDFS path {hdfs_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to write file to HDFS: {e}")
            return False
    
    def read_file(self, hdfs_path, local_path=None):
        """
        Read a file from HDFS.
        
        Args:
            hdfs_path: Path to HDFS file.
            local_path: Path to save locally. Defaults to None (don't save locally).
            
        Returns:
            File content as bytes if successful, None otherwise.
        """
        if not self.client:
            logger.error("HDFS not connected")
            return None
        
        try:
            # Read file
            with self.client.read(hdfs_path) as hdfs_file:
                content = hdfs_file.read()
            
            # Save locally if local_path is provided
            if local_path:
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                with open(local_path, 'wb') as local_file:
                    local_file.write(content)
                logger.info(f"Read HDFS file {hdfs_path} to local path {local_path}")
            else:
                logger.info(f"Read HDFS file {hdfs_path}")
            
            return content
        except Exception as e:
            logger.error(f"Failed to read file from HDFS: {e}")
            return None
    
    def delete_file(self, hdfs_path, recursive=False):
        """
        Delete a file or directory from HDFS.
        
        Args:
            hdfs_path: Path to HDFS file or directory.
            recursive: Whether to delete recursively (for directories). Defaults to False.
            
        Returns:
            True if successful, False otherwise.
        """
        if not self.client:
            logger.error("HDFS not connected")
            return False
        
        try:
            self.client.delete(hdfs_path, recursive=recursive)
            logger.info(f"Deleted HDFS path {hdfs_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete HDFS path {hdfs_path}: {e}")
            return False
    
    def list_directory(self, hdfs_path='/'):
        """
        List the contents of a directory in HDFS.
        
        Args:
            hdfs_path: Path to HDFS directory. Defaults to root.
            
        Returns:
            List of file/directory names if successful, empty list otherwise.
        """
        if not self.client:
            logger.error("HDFS not connected")
            return []
        
        try:
            contents = self.client.list(hdfs_path)
            logger.info(f"Listed HDFS directory {hdfs_path}: {len(contents)} items")
            return contents
        except Exception as e:
            logger.error(f"Failed to list HDFS directory {hdfs_path}: {e}")
            return []
    
    def write_dataframe_csv(self, df, hdfs_path, overwrite=False):
        """
        Write a DataFrame to HDFS as CSV.
        
        Args:
            df: DataFrame to write.
            hdfs_path: Path to HDFS file.
            overwrite: Whether to overwrite existing file. Defaults to False.
            
        Returns:
            True if successful, False otherwise.
        """
        if not self.client:
            logger.error("HDFS not connected")
            return False
        
        try:
            # Convert DataFrame to CSV string
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            
            # Create parent directory if it doesn't exist
            parent_dir = os.path.dirname(hdfs_path)
            if parent_dir and not self.client.status(parent_dir, strict=False):
                self.client.makedirs(parent_dir)
            
            # Write to HDFS
            with self.client.write(hdfs_path, overwrite=overwrite) as hdfs_file:
                hdfs_file.write(csv_buffer.getvalue().encode('utf-8'))
            
            logger.info(f"Written DataFrame ({len(df)} rows) to HDFS path {hdfs_path} as CSV")
            return True
        except Exception as e:
            logger.error(f"Failed to write DataFrame to HDFS as CSV: {e}")
            return False
    
    def read_dataframe_csv(self, hdfs_path):
        """
        Read a CSV file from HDFS into a DataFrame.
        
        Args:
            hdfs_path: Path to HDFS file.
            
        Returns:
            DataFrame if successful, None otherwise.
        """
        if not self.client:
            logger.error("HDFS not connected")
            return None
        
        try:
            # Read CSV from HDFS
            with self.client.read(hdfs_path) as hdfs_file:
                csv_content = hdfs_file.read().decode('utf-8')
            
            # Convert CSV to DataFrame
            df = pd.read_csv(io.StringIO(csv_content))
            
            logger.info(f"Read CSV from HDFS path {hdfs_path} into DataFrame ({len(df)} rows)")
            return df
        except Exception as e:
            logger.error(f"Failed to read CSV from HDFS: {e}")
            return None
    
    def write_dataframe_parquet(self, df, hdfs_path, overwrite=False):
        """
        Write a DataFrame to HDFS as Parquet (more efficient for big data).
        
        Args:
            df: DataFrame to write.
            hdfs_path: Path to HDFS file.
            overwrite: Whether to overwrite existing file. Defaults to False.
            
        Returns:
            True if successful, False otherwise.
        """
        if not self.client:
            logger.error("HDFS not connected")
            return False
        
        try:
            # Convert DataFrame to PyArrow Table
            table = pa.Table.from_pandas(df)
            
            # Create parent directory if it doesn't exist
            parent_dir = os.path.dirname(hdfs_path)
            if parent_dir and not self.client.status(parent_dir, strict=False):
                self.client.makedirs(parent_dir)
            
            # Write to buffer
            buffer = io.BytesIO()
            pq.write_table(table, buffer)
            buffer.seek(0)
            
            # Write to HDFS
            with self.client.write(hdfs_path, overwrite=overwrite) as hdfs_file:
                hdfs_file.write(buffer.getvalue())
            
            logger.info(f"Written DataFrame ({len(df)} rows) to HDFS path {hdfs_path} as Parquet")
            return True
        except Exception as e:
            logger.error(f"Failed to write DataFrame to HDFS as Parquet: {e}")
            return False
    
    def read_dataframe_parquet(self, hdfs_path):
        """
        Read a Parquet file from HDFS into a DataFrame.
        
        Args:
            hdfs_path: Path to HDFS file.
            
        Returns:
            DataFrame if successful, None otherwise.
        """
        if not self.client:
            logger.error("HDFS not connected")
            return None
        
        try:
            # Read Parquet from HDFS
            with self.client.read(hdfs_path) as hdfs_file:
                buffer = io.BytesIO(hdfs_file.read())
            
            # Convert Parquet to DataFrame
            table = pq.read_table(buffer)
            df = table.to_pandas()
            
            logger.info(f"Read Parquet from HDFS path {hdfs_path} into DataFrame ({len(df)} rows)")
            return df
        except Exception as e:
            logger.error(f"Failed to read Parquet from HDFS: {e}")
            return None
    
    def export_mongodb_to_hdfs(self, mongo_handler, collection_name, hdfs_path, query=None, file_format='parquet'):
        """
        Export data from MongoDB to HDFS.
        
        Args:
            mongo_handler: MongoDB handler instance.
            collection_name: MongoDB collection name.
            hdfs_path: Path to HDFS file or directory.
            query: MongoDB query. Defaults to None (all documents).
            file_format: File format ('csv' or 'parquet'). Defaults to 'parquet'.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            # Load data from MongoDB
            df = mongo_handler.load_dataframe(collection_name, query=query)
            
            if df is None or df.empty:
                logger.warning(f"No data to export from MongoDB collection {collection_name}")
                return False
            
            # Write to HDFS
            if file_format.lower() == 'csv':
                return self.write_dataframe_csv(df, hdfs_path, overwrite=True)
            else:  # Default to Parquet
                return self.write_dataframe_parquet(df, hdfs_path, overwrite=True)
        except Exception as e:
            logger.error(f"Failed to export MongoDB data to HDFS: {e}")
            return False

def main():
    """
    Main function to demonstrate HDFS handler.
    """
    # Initialize the handler
    handler = HDFSHandler()
    
    # Check if connection was successful
    if not handler.client:
        logger.error("Failed to connect to HDFS")
        return
    
    # List root directory
    contents = handler.list_directory('/')
    logger.info(f"HDFS root directory contents: {contents}")
    
    # Create test DataFrame
    df = pd.DataFrame({
        'ticker': ['AAPL', 'MSFT', 'GOOGL'],
        'price': [150.25, 250.75, 2500.50],
        'date': ['2023-01-01', '2023-01-01', '2023-01-01']
    })
    
    # Write test DataFrame as CSV
    handler.write_dataframe_csv(df, '/stock_data/test.csv', overwrite=True)
    
    # Write test DataFrame as Parquet
    handler.write_dataframe_parquet(df, '/stock_data/test.parquet', overwrite=True)
    
    # Read back and verify
    df_csv = handler.read_dataframe_csv('/stock_data/test.csv')
    df_parquet = handler.read_dataframe_parquet('/stock_data/test.parquet')
    
    if df_csv is not None:
        logger.info(f"CSV read successful: {df_csv.shape}")
    
    if df_parquet is not None:
        logger.info(f"Parquet read successful: {df_parquet.shape}")


if __name__ == "__main__":
    main() 