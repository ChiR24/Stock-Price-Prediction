"""
Module for collecting sentiment data for stock market prediction.
This module provides a wrapper around YahooSentimentCollector with the expected interface.
"""
import os
import pandas as pd
from datetime import datetime

# Import project modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from utils.config import RAW_DATA_DIR
from data_collection.yahoo_sentiment import YahooSentimentCollector

# Set up logger
logger = setup_logger('sentiment_data')

class SentimentDataCollector:
    """
    Class for collecting sentiment data for stock market prediction.
    This class uses Yahoo Finance news articles for sentiment analysis.
    """
    def __init__(self, use_kafka=False):
        """
        Initialize the SentimentDataCollector.

        Args:
            use_kafka: Whether to use Kafka for data streaming. Defaults to False.
        """
        self.use_kafka = use_kafka
        self.news_collector = YahooSentimentCollector(use_kafka=use_kafka)
        logger.info("SentimentDataCollector initialized with Yahoo Finance sentiment analysis")

    def collect_sentiment_data(self, ticker, save=True):
        """
        Collect sentiment data for a specific ticker using Yahoo Finance news articles.

        Args:
            ticker: Stock ticker symbol.
            save: Whether to save the data to disk. Defaults to True.

        Returns:
            DataFrame with sentiment data.
        """
        logger.info(f"Collecting sentiment data for {ticker} using Yahoo Finance news articles")

        # Collect sentiment data using news collector
        sentiment_df = self.news_collector.collect_sentiment_data(ticker, save=save)

        if sentiment_df.empty:
            logger.warning(f"No sentiment data found for {ticker}. Please try again later or check the ticker symbol.")
        else:
            logger.info(f"Collected {len(sentiment_df)} sentiment records for {ticker}")

        return sentiment_df

def main():
    """
    Main function to collect sentiment data.
    """
    collector = SentimentDataCollector()
    collector.collect_sentiment_data('AAPL')

if __name__ == "__main__":
    main()