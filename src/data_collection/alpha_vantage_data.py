"""
Module for collecting stock data for stock market prediction using Alpha Vantage API.
This module provides a wrapper around AlphaVantageDataCollector with the expected interface.
"""
import os
from datetime import datetime

# Import project modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from data_collection.alpha_vantage_data_collector import AlphaVantageDataCollector as BaseAlphaVantageDataCollector

# Set up logger
logger = setup_logger('alpha_vantage_data')

class AlphaVantageDataCollector:
    """
    Class to collect stock data for stock market prediction using Alpha Vantage API.
    This is a wrapper around AlphaVantageDataCollector with the expected interface.
    """
    def __init__(self, use_kafka=False):
        """
        Initialize the AlphaVantageDataCollector.
        
        Args:
            use_kafka: Whether to use Kafka for data streaming. Defaults to False.
        """
        self.collector = BaseAlphaVantageDataCollector(use_kafka=use_kafka)
        logger.info("AlphaVantageDataCollector initialized")
    
    def collect_stock_data(self, ticker, start_date=None, end_date=None, save=True):
        """
        Collect stock data for the specified ticker using Alpha Vantage API.
        
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
    Main function to collect stock data using Alpha Vantage API.
    """
    collector = AlphaVantageDataCollector()
    collector.collect_stock_data('AAPL')

if __name__ == "__main__":
    main() 