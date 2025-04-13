"""
Module for collecting stock market data from Alpha Vantage API.
"""
import os
import time
import json
import pandas as pd
import requests
from datetime import datetime, timedelta

# Import project modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from utils.config import (
    RAW_DATA_DIR, 
    DEFAULT_TICKERS, 
    KAFKA_BOOTSTRAP_SERVERS, 
    KAFKA_STOCK_TOPIC,
    ALPHA_VANTAGE_API_KEY,
    API_RATE_LIMIT
)

# Set up logger
logger = setup_logger('alpha_vantage_data_collector')

class AlphaVantageDataCollector:
    """
    Class to collect stock data from Alpha Vantage API.
    """
    def __init__(self, tickers=None, api_key=None, interval="daily", outputsize="full", use_kafka=False):
        """
        Initialize the AlphaVantageDataCollector.
        
        Args:
            tickers: List of stock tickers to collect data for. Defaults to DEFAULT_TICKERS.
            api_key: Alpha Vantage API key. Defaults to the one in config.
            interval: The interval between data points ('daily', 'weekly', 'monthly'). Defaults to "daily".
            outputsize: Amount of data to fetch ('compact' for last 100 data points, 'full' for 20+ years). Defaults to "full".
            use_kafka: Whether to send data to Kafka. Defaults to False.
        """
        self.tickers = tickers if tickers else DEFAULT_TICKERS
        self.api_key = api_key if api_key else ALPHA_VANTAGE_API_KEY
        self.interval = interval
        self.outputsize = outputsize
        self.use_kafka = use_kafka
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limit_delay = 60 / API_RATE_LIMIT  # Time in seconds to wait between API calls
        
        # Map interval to Alpha Vantage function
        self.interval_map = {
            "daily": "TIME_SERIES_DAILY_ADJUSTED",
            "weekly": "TIME_SERIES_WEEKLY_ADJUSTED",
            "monthly": "TIME_SERIES_MONTHLY_ADJUSTED"
        }
        
        # Map interval to time series key in response
        self.time_series_map = {
            "daily": "Time Series (Daily)",
            "weekly": "Weekly Adjusted Time Series",
            "monthly": "Monthly Adjusted Time Series"
        }
        
        # Column mapping from Alpha Vantage to yfinance format
        self.column_map = {
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. adjusted close": "Adj Close",
            "6. volume": "Volume"
        }
        
        # Initialize Kafka producer if needed
        if self.use_kafka:
            try:
                from kafka import KafkaProducer
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
            
            # Check if API key is available
            if not self.api_key or self.api_key == "YOUR_API_KEY":
                logger.error("Alpha Vantage API key not found. Set ALPHA_VANTAGE_API_KEY in .env or config.py")
                return pd.DataFrame()
            
            # Build request parameters
            function = self.interval_map.get(self.interval, "TIME_SERIES_DAILY_ADJUSTED")
            params = {
                "function": function,
                "symbol": ticker,
                "outputsize": self.outputsize,
                "apikey": self.api_key
            }
            
            # Make API request
            logger.info(f"Making Alpha Vantage API request for {ticker} with function {function}")
            response = requests.get(self.base_url, params=params)
            
            # Handle API errors
            if response.status_code != 200:
                logger.error(f"Error from Alpha Vantage API: {response.status_code} - {response.text}")
                return pd.DataFrame()
            
            # Parse response
            json_data = response.json()
            
            # Check for error messages
            if "Error Message" in json_data:
                logger.error(f"Alpha Vantage API error: {json_data['Error Message']}")
                return pd.DataFrame()
            
            # Check for information messages (e.g. API limit exceeded)
            if "Information" in json_data:
                logger.warning(f"Alpha Vantage API info: {json_data['Information']}")
                # Return empty DataFrame if API limit is exceeded
                if "API call frequency" in json_data["Information"]:
                    logger.error("Alpha Vantage API limit exceeded. Waiting for rate limit to reset.")
                    return pd.DataFrame()
            
            # Get time series data
            time_series_key = self.time_series_map.get(self.interval, "Time Series (Daily)")
            if time_series_key not in json_data:
                logger.error(f"Expected data key {time_series_key} not found in API response")
                return pd.DataFrame()
            
            time_series = json_data[time_series_key]
            
            # Convert to DataFrame
            data = pd.DataFrame.from_dict(time_series, orient='index')
            
            # Rename columns to match yfinance format
            data = data.rename(columns=self.column_map)
            
            # Convert data types
            for col in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            if 'Volume' in data.columns:
                data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce').astype('int64')
            
            # Reset index to convert date string to datetime column
            data = data.reset_index()
            data = data.rename(columns={'index': 'Date'})
            data['Date'] = pd.to_datetime(data['Date'])
            
            # Sort by date in ascending order (Alpha Vantage returns most recent first)
            data = data.sort_values('Date')
            
            # Filter by date range if provided
            if start_date:
                start_date = pd.to_datetime(start_date)
                data = data[data['Date'] >= start_date]
            
            if end_date:
                end_date = pd.to_datetime(end_date)
                data = data[data['Date'] <= end_date]
            
            # Add ticker column
            data['ticker'] = ticker
            
            # Save to CSV if requested
            if save and not data.empty:
                os.makedirs(os.path.join(RAW_DATA_DIR, 'stocks'), exist_ok=True)
                file_suffix = f"{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}" if start_date and end_date else "data"
                csv_path = os.path.join(RAW_DATA_DIR, 'stocks', f"{ticker}_{file_suffix}.csv")
                data.to_csv(csv_path, index=False)
                logger.info(f"Saved {len(data)} records for {ticker} to {csv_path}")
            
            # Send to Kafka if enabled
            if self.use_kafka and not data.empty:
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
        
        finally:
            # Add a delay to respect API rate limits
            time.sleep(self.rate_limit_delay)
    
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
                data = self.collect_data(ticker, save=True)
                
                if not data.empty:
                    all_data[ticker] = data
                else:
                    logger.warning(f"No data found for {ticker}")
                
            except Exception as e:
                logger.error(f"Error collecting data for {ticker}: {e}")
        
        return all_data
    
    def collect_real_time_data(self):
        """
        Collect real-time (latest) stock data for all tickers.
        Alpha Vantage doesn't have true real-time data, so we just get the latest day's data.
        
        Returns:
            DataFrame with the latest data for all tickers.
        """
        latest_data = []
        
        for ticker in self.tickers:
            try:
                logger.info(f"Collecting latest data for {ticker}")
                
                # For real-time data, use compact outputsize to minimize API calls
                params = {
                    "function": "TIME_SERIES_DAILY_ADJUSTED",
                    "symbol": ticker,
                    "outputsize": "compact",  # Last 100 data points
                    "apikey": self.api_key
                }
                
                # Make API request
                response = requests.get(self.base_url, params=params)
                
                if response.status_code != 200:
                    logger.error(f"Error from Alpha Vantage API: {response.status_code} - {response.text}")
                    continue
                
                json_data = response.json()
                
                # Check for error messages
                if "Error Message" in json_data:
                    logger.error(f"Alpha Vantage API error: {json_data['Error Message']}")
                    continue
                
                # Get time series data
                time_series_key = "Time Series (Daily)"
                if time_series_key not in json_data:
                    logger.error(f"Expected data key {time_series_key} not found in API response")
                    continue
                
                time_series = json_data[time_series_key]
                
                # Get the latest date (first key in time_series)
                if not time_series:
                    logger.warning(f"No data found for {ticker}")
                    continue
                
                latest_date = list(time_series.keys())[0]
                latest_record = time_series[latest_date]
                
                # Convert to expected format
                formatted_record = {
                    'ticker': ticker,
                    'Date': pd.to_datetime(latest_date),
                    'Open': float(latest_record['1. open']),
                    'High': float(latest_record['2. high']),
                    'Low': float(latest_record['3. low']),
                    'Close': float(latest_record['4. close']),
                    'Adj Close': float(latest_record['5. adjusted close']),
                    'Volume': int(latest_record['6. volume']),
                    'timestamp': datetime.now().isoformat()
                }
                
                # Send to Kafka if enabled
                if self.use_kafka:
                    self.producer.send(KAFKA_STOCK_TOPIC, formatted_record)
                    logger.info(f"Sent {ticker} data to Kafka topic {KAFKA_STOCK_TOPIC}")
                
                latest_data.append(formatted_record)
                
                # Add a delay to respect API rate limits
                time.sleep(self.rate_limit_delay)
                
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
    Main function to run the Alpha Vantage data collector.
    """
    # Initialize the collector
    collector = AlphaVantageDataCollector(
        tickers=DEFAULT_TICKERS,
        interval="daily",
        outputsize="full",
        use_kafka=False  # Set to True to use Kafka
    )
    
    # Collect historical data once
    collector.collect_historical_data()
    
    # Run continuous collection for a limited time (for testing)
    # For production, you might want to set max_iterations to None
    collector.run_continuous_collection(interval_seconds=60, max_iterations=5)


if __name__ == "__main__":
    main() 