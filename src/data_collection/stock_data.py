"""
Module for collecting stock data for stock market prediction.
This module provides a wrapper around StockDataCollector with the expected interface.
"""
import os
from datetime import datetime

# Import project modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from data_collection.stock_data_collector import StockDataCollector as BaseStockDataCollector

# Set up logger
logger = setup_logger('stock_data')

class StockDataCollector:
    """
    Class to collect stock data for stock market prediction.
    This is a wrapper around StockDataCollector from stock_data_collector with the expected interface.
    """
    def __init__(self, use_kafka=False):
        """
        Initialize the StockDataCollector.
        
        Args:
            use_kafka: Whether to use Kafka for data streaming. Defaults to False.
        """
        self.collector = BaseStockDataCollector(use_kafka=use_kafka)
        logger.info("StockDataCollector initialized")
    
    def collect_stock_data(self, ticker, start_date=None, end_date=None, save=True):
        """
        Collect stock data for the specified ticker.
        
        Args:
            ticker: Stock ticker symbol.
            start_date: Start date for data collection (YYYY-MM-DD). Defaults to None.
            end_date: End date for data collection (YYYY-MM-DD). Defaults to None.
            save: Whether to save the data to disk. Defaults to True.
            
        Returns:
            DataFrame with the stock data.
        """
        logger.info(f"Collecting stock data for {ticker} from {start_date} to {end_date}")
        
        # Call the underlying collector
        df = self.collector.collect_data(ticker, start_date, end_date, save=save)
        
        return df

def main():
    """
    Main function to collect stock data.
    """
    collector = StockDataCollector()
    collector.collect_stock_data('AAPL')

if __name__ == "__main__":
    main() 