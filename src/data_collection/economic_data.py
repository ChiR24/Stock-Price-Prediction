"""
Module for collecting economic data for stock market prediction.
This module provides a wrapper around EconomicDataCollector with the expected interface.
"""
import os
import pandas as pd
from datetime import datetime

# Import project modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from utils.config import RAW_DATA_DIR

# Set up logger
logger = setup_logger('economic_data')

class EconomicDataCollector:
    """
    Class for collecting economic data for stock market prediction.
    This version just logs a message that only Yahoo Finance data is being used.
    """
    def __init__(self, use_kafka=False):
        """
        Initialize the EconomicDataCollector.
        
        Args:
            use_kafka: Whether to use Kafka for data streaming. Defaults to False.
        """
        self.use_kafka = use_kafka
        logger.info("EconomicDataCollector initialized - Only Yahoo Finance stock data will be used")
    
    def collect_economic_data(self, start_date=None, end_date=None, save=True):
        """
        This method is simplified to only log a message that we're only using Yahoo Finance data.
        
        Args:
            start_date: Start date for data collection (YYYY-MM-DD). Defaults to None.
            end_date: End date for data collection (YYYY-MM-DD). Defaults to None.
            save: Whether to save the data to disk. Defaults to True.
            
        Returns:
            Empty dictionary as we're not collecting economic data.
        """
        logger.info(f"Economic data collection is skipped - Using only Yahoo Finance stock data")
        
        # Create a placeholder empty file to indicate the operation was run
        if save:
            os.makedirs(os.path.join(RAW_DATA_DIR, 'economic'), exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            info_path = os.path.join(RAW_DATA_DIR, 'economic', f"info_{timestamp}.txt")
            
            with open(info_path, 'w') as f:
                f.write("Economic data collection was skipped. Only Yahoo Finance stock data is being used.\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Date range: {start_date} to {end_date}\n")
            
            logger.info(f"Created info file at {info_path}")
        
        return {}

def main():
    """
    Main function to collect economic data.
    """
    collector = EconomicDataCollector()
    collector.collect_economic_data()

if __name__ == "__main__":
    main() 