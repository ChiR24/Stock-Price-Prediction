#!/usr/bin/env python
"""
Script to run the real-time stock market prediction system.
This script starts the data collection and prediction components.
"""
import os
import sys

# Add the virtual environment's site-packages to the Python path
venv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.venv', 'Lib', 'site-packages')
if os.path.exists(venv_path):
    sys.path.insert(0, venv_path)

import argparse
import time
import threading
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from utils.logger import setup_logger
from utils.config import DEFAULT_TICKERS
from data_collection.stock_data import StockDataCollector
from data_collection.sentiment_data import SentimentDataCollector
from data_collection.economic_data import EconomicDataCollector
from real_time.real_time_predictor import RealTimePredictor

# Set up logger
logger = setup_logger('real_time_system')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Real-time Stock Market Prediction System')

    # Data collection parameters
    parser.add_argument('--tickers', nargs='+', default=DEFAULT_TICKERS,
                        help='List of stock tickers to analyze')
    parser.add_argument('--data-source', choices=['yahoo', 'alpha_vantage'], default='yahoo',
                        help='Source for stock data (yahoo or alpha_vantage)')
    parser.add_argument('--use-kafka', action='store_true',
                        help='Use Kafka for data streaming')

    # Prediction parameters
    parser.add_argument('--model-type', choices=['lstm', 'ensemble', 'all'], default='all',
                        help='Type of model to use for prediction')
    parser.add_argument('--prediction-horizon', type=int, default=5,
                        help='Number of days ahead to predict')
    parser.add_argument('--use-mongodb', action='store_true',
                        help='Store predictions in MongoDB')

    # System parameters
    parser.add_argument('--collection-interval', type=int, default=60,
                        help='Interval in seconds between data collections')
    parser.add_argument('--run-time', type=int, default=None,
                        help='Time in minutes to run the system (None for indefinite)')

    return parser.parse_args()

def run_stock_collector(args, stop_event):
    """Run the stock data collector."""
    try:
        logger.info("Starting stock data collector")

        # Initialize collector
        collector = StockDataCollector(use_kafka=args.use_kafka)

        # Collect initial historical data
        collector.collect_stock_data(
            ticker=args.tickers[0],  # Start with first ticker
            save=True
        )

        # Run continuous collection
        iteration = 0

        while not stop_event.is_set():
            logger.info(f"Stock collection iteration {iteration + 1}")

            # Collect real-time data for all tickers
            for ticker in args.tickers:
                try:
                    collector.collect_stock_data(ticker=ticker, save=True)
                    logger.info(f"Collected stock data for {ticker}")
                except Exception as e:
                    logger.error(f"Error collecting stock data for {ticker}: {e}")

            iteration += 1

            # Sleep until next collection
            if not stop_event.is_set():
                logger.info(f"Sleeping for {args.collection_interval} seconds")
                # Sleep in small increments to check stop_event more frequently
                for _ in range(args.collection_interval):
                    if stop_event.is_set():
                        break
                    time.sleep(1)

        logger.info("Stock data collector stopped")

    except Exception as e:
        logger.error(f"Error in stock collector thread: {e}")

def run_sentiment_collector(args, stop_event):
    """Run the sentiment data collector."""
    try:
        logger.info("Starting sentiment data collector")

        # Initialize collector
        collector = SentimentDataCollector(use_kafka=args.use_kafka)

        # Run continuous collection
        iteration = 0

        while not stop_event.is_set():
            logger.info(f"Sentiment collection iteration {iteration + 1}")

            # Collect sentiment data for all tickers
            for ticker in args.tickers:
                try:
                    collector.collect_sentiment_data(ticker=ticker, save=True)
                    logger.info(f"Collected sentiment data for {ticker}")
                except Exception as e:
                    logger.error(f"Error collecting sentiment data for {ticker}: {e}")

            iteration += 1

            # Sleep until next collection (longer interval for sentiment)
            if not stop_event.is_set():
                sentiment_interval = args.collection_interval * 5  # 5x longer than stock data
                logger.info(f"Sleeping for {sentiment_interval} seconds")
                # Sleep in small increments to check stop_event more frequently
                for _ in range(sentiment_interval):
                    if stop_event.is_set():
                        break
                    time.sleep(1)

        logger.info("Sentiment data collector stopped")

    except Exception as e:
        logger.error(f"Error in sentiment collector thread: {e}")

def run_economic_collector(args, stop_event):
    """Run the economic data collector."""
    try:
        logger.info("Starting economic data collector")

        # Initialize collector
        collector = EconomicDataCollector(use_kafka=args.use_kafka)

        # Run continuous collection
        iteration = 0

        while not stop_event.is_set():
            logger.info(f"Economic collection iteration {iteration + 1}")

            # Collect economic data
            try:
                collector.collect_economic_data(save=True)
                logger.info("Collected economic data")
            except Exception as e:
                logger.error(f"Error collecting economic data: {e}")

            iteration += 1

            # Sleep until next collection (much longer interval for economic data)
            if not stop_event.is_set():
                economic_interval = args.collection_interval * 60  # 60x longer than stock data
                logger.info(f"Sleeping for {economic_interval} seconds")
                # Sleep in small increments to check stop_event more frequently
                for _ in range(economic_interval):
                    if stop_event.is_set():
                        break
                    time.sleep(1)

        logger.info("Economic data collector stopped")

    except Exception as e:
        logger.error(f"Error in economic collector thread: {e}")

def run_predictor(args, stop_event):
    """Run the real-time predictor."""
    try:
        logger.info("Starting real-time predictor")

        # Initialize predictor
        predictor = RealTimePredictor(
            tickers=args.tickers,
            model_type=args.model_type,
            prediction_horizon=args.prediction_horizon,
            use_mongodb=args.use_mongodb
        )

        # Start predictor
        if predictor.start():
            logger.info("Real-time predictor started")

            # Keep running until stop event is set
            while not stop_event.is_set():
                time.sleep(1)

            # Stop predictor
            predictor.stop()
            logger.info("Real-time predictor stopped")
        else:
            logger.error("Failed to start real-time predictor")

    except Exception as e:
        logger.error(f"Error in predictor thread: {e}")

def main():
    """Main function to run the real-time prediction system."""
    # Parse command line arguments
    args = parse_arguments()

    # Print configuration
    logger.info("Configuration:")
    logger.info(f"  Tickers: {args.tickers}")
    logger.info(f"  Data source: {args.data_source}")
    logger.info(f"  Use Kafka: {args.use_kafka}")
    logger.info(f"  Model type: {args.model_type}")
    logger.info(f"  Prediction horizon: {args.prediction_horizon}")
    logger.info(f"  Use MongoDB: {args.use_mongodb}")
    logger.info(f"  Collection interval: {args.collection_interval} seconds")
    logger.info(f"  Run time: {args.run_time if args.run_time else 'indefinite'} minutes")

    # Create stop event
    stop_event = threading.Event()

    # Create threads
    threads = []

    # Stock collector thread
    stock_thread = threading.Thread(
        target=run_stock_collector,
        args=(args, stop_event),
        daemon=True
    )
    threads.append(stock_thread)

    # Sentiment collector thread
    sentiment_thread = threading.Thread(
        target=run_sentiment_collector,
        args=(args, stop_event),
        daemon=True
    )
    threads.append(sentiment_thread)

    # Economic collector thread
    economic_thread = threading.Thread(
        target=run_economic_collector,
        args=(args, stop_event),
        daemon=True
    )
    threads.append(economic_thread)

    # Predictor thread
    if args.use_kafka:
        predictor_thread = threading.Thread(
            target=run_predictor,
            args=(args, stop_event),
            daemon=True
        )
        threads.append(predictor_thread)

    # Start threads
    for thread in threads:
        thread.start()

    try:
        # Run for specified time or indefinitely
        if args.run_time:
            logger.info(f"Running for {args.run_time} minutes")
            time.sleep(args.run_time * 60)
            stop_event.set()
        else:
            logger.info("Running indefinitely (press Ctrl+C to stop)")
            while True:
                time.sleep(1)

    except KeyboardInterrupt:
        logger.info("System interrupted by user")
        stop_event.set()

    # Wait for threads to finish
    for thread in threads:
        thread.join(timeout=5.0)

    logger.info("Real-time prediction system stopped")

if __name__ == "__main__":
    main()
