"""
Module for collecting macroeconomic data from various sources like FRED (Federal Reserve Economic Data).
"""
import os
import time
import json
import pandas as pd
import requests
from datetime import datetime, timedelta
from kafka import KafkaProducer

# Import project modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from utils.config import (
    RAW_DATA_DIR,
    FRED_API_KEY,
    MACROECONOMIC_INDICATORS,
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_ECONOMIC_TOPIC
)

# Set up logger
logger = setup_logger('economic_data_collector')

class EconomicDataCollector:
    """
    Class to collect macroeconomic data from various sources.
    """
    def __init__(self, indicators=None, use_kafka=False):
        """
        Initialize the EconomicDataCollector.
        
        Args:
            indicators: List of economic indicators to collect. Defaults to config.MACROECONOMIC_INDICATORS.
            use_kafka: Whether to send data to Kafka. Defaults to False.
        """
        self.indicators = indicators if indicators else MACROECONOMIC_INDICATORS
        self.use_kafka = use_kafka
        self.fred_api_key = FRED_API_KEY
        
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
    
    def collect_fred_data(self, series_id, start_date=None, end_date=None):
        """
        Collect economic data from FRED API.
        
        Args:
            series_id: FRED series ID.
            start_date: Start date for data collection (YYYY-MM-DD). Defaults to 5 years ago.
            end_date: End date for data collection (YYYY-MM-DD). Defaults to today.
            
        Returns:
            DataFrame with the economic data.
        """
        if not self.fred_api_key:
            logger.error("FRED API key not found. Set FRED_API_KEY in .env or config.py")
            return pd.DataFrame()
        
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
        
        try:
            # Build API URL
            url = f"https://api.stlouisfed.org/fred/series/observations"
            params = {
                'series_id': series_id,
                'api_key': self.fred_api_key,
                'file_type': 'json',
                'observation_start': start_date,
                'observation_end': end_date
            }
            
            logger.info(f"Collecting FRED data for series {series_id}")
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                observations = data.get('observations', [])
                
                if observations:
                    # Convert to DataFrame
                    df = pd.DataFrame(observations)
                    
                    # Convert date strings to datetime
                    df['date'] = pd.to_datetime(df['date'])
                    
                    # Convert value strings to float, handling missing values
                    df['value'] = pd.to_numeric(df['value'], errors='coerce')
                    
                    # Add series_id column
                    df['series_id'] = series_id
                    
                    # Add indicator name if available
                    indicator_name = next((item['name'] for item in self.indicators if item['series_id'] == series_id), None)
                    if indicator_name:
                        df['indicator_name'] = indicator_name
                    
                    # Save to CSV
                    os.makedirs(os.path.join(RAW_DATA_DIR, 'economic'), exist_ok=True)
                    csv_path = os.path.join(RAW_DATA_DIR, 'economic', f"{series_id}_data.csv")
                    df.to_csv(csv_path, index=False)
                    logger.info(f"Saved {len(df)} observations for {series_id} to {csv_path}")
                    
                    return df
                else:
                    logger.warning(f"No observations found for series {series_id}")
            else:
                logger.error(f"Failed to collect data for series {series_id}: {response.status_code} - {response.text}")
        
        except Exception as e:
            logger.error(f"Error collecting FRED data for series {series_id}: {e}")
        
        return pd.DataFrame()
    
    def collect_all_indicators(self):
        """
        Collect data for all configured economic indicators.
        
        Returns:
            Dictionary with indicator series_id as key and DataFrame as value.
        """
        all_data = {}
        
        for indicator in self.indicators:
            series_id = indicator['series_id']
            df = self.collect_fred_data(series_id)
            
            if not df.empty:
                all_data[series_id] = df
                
                # Send to Kafka if enabled
                if self.use_kafka:
                    # Convert to list of dictionaries for Kafka
                    records = df.to_dict('records')
                    for record in records:
                        # Convert datetime to string
                        if isinstance(record.get('date'), pd.Timestamp):
                            record['date'] = record['date'].isoformat()
                        
                        self.producer.send(KAFKA_ECONOMIC_TOPIC, record)
                    
                    logger.info(f"Sent {len(records)} records for {series_id} to Kafka topic {KAFKA_ECONOMIC_TOPIC}")
            
            # Add a small delay to avoid rate limiting
            time.sleep(1)
        
        return all_data
    
    def run_continuous_collection(self, interval_hours=24, max_iterations=None):
        """
        Run continuous data collection at specified intervals.
        
        Args:
            interval_hours: Time in hours between collections. Defaults to 24.
            max_iterations: Maximum number of iterations. Defaults to None (run indefinitely).
        """
        iteration = 0
        interval_seconds = interval_hours * 60 * 60
        
        logger.info(f"Starting continuous economic data collection every {interval_hours} hours")
        
        try:
            while max_iterations is None or iteration < max_iterations:
                logger.info(f"Economic data collection iteration {iteration + 1}")
                
                # Collect all indicators
                self.collect_all_indicators()
                
                iteration += 1
                
                # Sleep until next collection
                if max_iterations is None or iteration < max_iterations:
                    logger.info(f"Sleeping for {interval_hours} hours")
                    time.sleep(interval_seconds)
        
        except KeyboardInterrupt:
            logger.info("Economic data collection stopped by user")
        
        finally:
            if self.use_kafka:
                self.producer.flush()
                logger.info("Kafka producer flushed")

def main():
    """
    Main function to run the economic data collector.
    """
    # Initialize the collector
    collector = EconomicDataCollector(
        indicators=MACROECONOMIC_INDICATORS,
        use_kafka=False  # Set to True to use Kafka
    )
    
    # Collect all indicators once
    collector.collect_all_indicators()
    
    # For testing, uncomment to run continuous collection
    # collector.run_continuous_collection(interval_hours=24, max_iterations=5)


if __name__ == "__main__":
    main() 