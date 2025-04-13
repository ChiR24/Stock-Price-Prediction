"""
Module for collecting stock market data from Yahoo Finance API.
"""
import os
import time
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from kafka import KafkaProducer
import json

# Import project modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from utils.config import (
    RAW_DATA_DIR, 
    DEFAULT_TICKERS, 
    KAFKA_BOOTSTRAP_SERVERS, 
    KAFKA_STOCK_TOPIC
)

# Set up logger
logger = setup_logger('stock_data_collector')

class StockDataCollector:
    """
    Class to collect stock data from Yahoo Finance API.
    """
    def __init__(self, tickers=None, period="1y", interval="1d", use_kafka=False):
        """
        Initialize the StockDataCollector.
        
        Args:
            tickers: List of stock tickers to collect data for. Defaults to DEFAULT_TICKERS.
            period: The period to fetch data for (e.g., "1d", "1mo", "1y"). Defaults to "1y".
            interval: The interval between data points (e.g., "1m", "1h", "1d"). Defaults to "1d".
            use_kafka: Whether to send data to Kafka. Defaults to False.
        """
        self.tickers = tickers if tickers else DEFAULT_TICKERS
        self.period = period
        self.interval = interval
        self.use_kafka = use_kafka
        
        # Initialize Kafka producer if needed
        if self.use_kafka:
            try:
                self.producer = KafkaProducer(
                    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                    value_serializer=lambda v: json.dumps(v).encode('utf-8')
                )
                logger.info(f"Kafka producer initialized with bootstrap servers: {KAFKA_BOOTSTRAP_SERVERS}")
            except Exception as e:
                logger.error(f"Failed to initialize Kafka producer: {e}")
                self.use_kafka = False
    
    def collect_data(self, ticker, start_date=None, end_date=None, save=True):
        """
        Collect stock data for a specific ticker and date range.
        
        Args:
            ticker: Stock ticker symbol.
            start_date: Start date for data collection (YYYY-MM-DD). Defaults to None.
            end_date: End date for data collection (YYYY-MM-DD). Defaults to None.
            save: Whether to save the data to disk. Defaults to True.
            
        Returns:
            DataFrame with the stock data.
        """
        try:
            logger.info(f"Collecting data for {ticker} from {start_date} to {end_date}")
            
            # Download data
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval=self.interval,
                auto_adjust=True,
                progress=False
            )
            
            if data.empty:
                logger.warning(f"No data found for {ticker} between {start_date} and {end_date}")
                return pd.DataFrame()
            
            # Add ticker column and reset index to make date a column
            data['ticker'] = ticker
            data = data.reset_index()
            
            # Save to CSV if requested
            if save:
                os.makedirs(os.path.join(RAW_DATA_DIR, 'stocks'), exist_ok=True)
                file_suffix = f"{start_date}_{end_date}" if start_date and end_date else "data"
                csv_path = os.path.join(RAW_DATA_DIR, 'stocks', f"{ticker}_{file_suffix}.csv")
                data.to_csv(csv_path, index=False)
                logger.info(f"Saved {len(data)} records for {ticker} to {csv_path}")
            
            # Send to Kafka if enabled
            if self.use_kafka:
                records = data.to_dict('records')
                for record in records:
                    # Convert datetime to string
                    if 'Date' in record and isinstance(record['Date'], pd.Timestamp):
                        record['Date'] = record['Date'].isoformat()
                    
                    self.producer.send(KAFKA_STOCK_TOPIC, record)
                
                logger.info(f"Sent {len(records)} records for {ticker} to Kafka topic {KAFKA_STOCK_TOPIC}")
            
            return data
        
        except Exception as e:
            logger.error(f"Error collecting data for {ticker}: {e}")
            return pd.DataFrame()
    
    def collect_historical_data(self):
        """
        Collect historical stock data for all tickers.
        
        Returns:
            Dictionary of DataFrames with ticker as key and data as value.
        """
        all_data = {}
        
        for ticker in self.tickers:
            try:
                logger.info(f"Collecting historical data for {ticker}")
                data = yf.download(
                    ticker,
                    period=self.period,
                    interval=self.interval,
                    auto_adjust=True,
                    progress=False
                )
                
                if not data.empty:
                    # Add ticker column
                    data['ticker'] = ticker
                    
                    # Save to CSV
                    os.makedirs(os.path.join(RAW_DATA_DIR, 'stocks'), exist_ok=True)
                    csv_path = os.path.join(RAW_DATA_DIR, 'stocks', f"{ticker}_historical.csv")
                    data.to_csv(csv_path)
                    logger.info(f"Saved historical data for {ticker} to {csv_path}")
                    
                    all_data[ticker] = data
                else:
                    logger.warning(f"No data found for {ticker}")
                
                # Add a small delay to avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error collecting data for {ticker}: {e}")
        
        return all_data
    
    def collect_real_time_data(self):
        """
        Collect real-time (latest) stock data for all tickers.
        
        Returns:
            DataFrame with the latest data for all tickers.
        """
        latest_data = []
        
        for ticker in self.tickers:
            try:
                logger.info(f"Collecting real-time data for {ticker}")
                stock = yf.Ticker(ticker)
                data = stock.history(period="1d")
                
                if not data.empty:
                    # Add ticker column
                    data['ticker'] = ticker
                    
                    # Convert to dict for Kafka
                    latest_record = data.iloc[-1].to_dict()
                    latest_record['timestamp'] = datetime.now().isoformat()
                    
                    # Send to Kafka if enabled
                    if self.use_kafka:
                        self.producer.send(KAFKA_STOCK_TOPIC, latest_record)
                        logger.info(f"Sent {ticker} data to Kafka topic {KAFKA_STOCK_TOPIC}")
                    
                    latest_data.append(latest_record)
                else:
                    logger.warning(f"No real-time data found for {ticker}")
                
                # Add a small delay to avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error collecting real-time data for {ticker}: {e}")
        
        # Convert to DataFrame
        if latest_data:
            df = pd.DataFrame(latest_data)
            
            # Save to CSV
            os.makedirs(os.path.join(RAW_DATA_DIR, 'stocks'), exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = os.path.join(RAW_DATA_DIR, 'stocks', f"realtime_{timestamp}.csv")
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved real-time data to {csv_path}")
            
            return df
        
        return pd.DataFrame()
    
    def run_continuous_collection(self, interval_seconds=60, max_iterations=None):
        """
        Run continuous data collection at specified intervals.
        
        Args:
            interval_seconds: Time in seconds between collections. Defaults to 60.
            max_iterations: Maximum number of iterations. Defaults to None (run indefinitely).
        """
        iteration = 0
        
        logger.info(f"Starting continuous data collection every {interval_seconds} seconds")
        
        try:
            while max_iterations is None or iteration < max_iterations:
                logger.info(f"Collection iteration {iteration + 1}")
                
                # Collect real-time data
                self.collect_real_time_data()
                
                iteration += 1
                
                # Sleep until next collection
                if max_iterations is None or iteration < max_iterations:
                    logger.info(f"Sleeping for {interval_seconds} seconds")
                    time.sleep(interval_seconds)
        
        except KeyboardInterrupt:
            logger.info("Data collection stopped by user")
        
        finally:
            if self.use_kafka:
                self.producer.flush()
                logger.info("Kafka producer flushed")


def main():
    """
    Main function to run the stock data collector.
    """
    # Initialize the collector
    collector = StockDataCollector(
        tickers=DEFAULT_TICKERS,
        period="2y",
        interval="1d",
        use_kafka=False  # Set to True to use Kafka
    )
    
    # Collect historical data once
    collector.collect_historical_data()
    
    # Run continuous collection for a limited time (for testing)
    # For production, you might want to set max_iterations to None
    collector.run_continuous_collection(interval_seconds=60, max_iterations=5)


if __name__ == "__main__":
    main() 