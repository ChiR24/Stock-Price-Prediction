#!/usr/bin/env python
"""
Main script for stock market prediction.
This module integrates data collection, processing, and model training.
"""
import os
import sys

# Add the virtual environment's site-packages to the Python path
venv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.venv', 'Lib', 'site-packages')
if os.path.exists(venv_path):
    sys.path.insert(0, venv_path)

import argparse
import logging
import pandas as pd
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from data_collection.stock_data import StockDataCollector
from data_collection.alpha_vantage_data import AlphaVantageDataCollector
from data_collection.sentiment_data import SentimentDataCollector
from data_collection.economic_data import EconomicDataCollector
from data_processing.data_processor import DataProcessor
from ml_models.lstm_model import LSTMModel
from ml_models.ensemble_model import EnsembleModel
from utils.logger import setup_logger
from config.settings import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    DATA_FEATURES_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    DEFAULT_TICKERS,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE
)

# Set up logger
logger = setup_logger('main')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Stock Market Prediction System')

    # Data collection parameters
    parser.add_argument('--tickers', nargs='+', default=DEFAULT_TICKERS,
                        help='List of stock tickers to analyze')
    parser.add_argument('--start-date', default=DEFAULT_START_DATE,
                        help='Start date for data collection (YYYY-MM-DD)')
    parser.add_argument('--end-date', default=DEFAULT_END_DATE,
                        help='End date for data collection (YYYY-MM-DD)')
    parser.add_argument('--data-source', choices=['yahoo', 'alpha_vantage'], default='yahoo',
                        help='Source for stock data (yahoo or alpha_vantage)')

    # Data processing parameters
    parser.add_argument('--use-spark', action='store_true',
                        help='Use Spark for data processing')
    parser.add_argument('--skip-data-collection', action='store_true',
                        help='Skip data collection, use existing data')
    parser.add_argument('--skip-data-processing', action='store_true',
                        help='Skip data processing, use existing features')

    # Model parameters
    parser.add_argument('--model-type', choices=['lstm', 'ensemble', 'all'], default='lstm',
                        help='Type of model to train')
    parser.add_argument('--sequence-length', type=int, default=10,
                        help='Sequence length for time series models')
    parser.add_argument('--target-column', default='target_5d',
                        help='Target column for prediction')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip model training, use existing models')

    return parser.parse_args()

def collect_data(args):
    """Collect data from various sources."""
    logger.info("Starting data collection")

    # Create directories if they don't exist
    os.makedirs(DATA_RAW_DIR, exist_ok=True)

    # Collect stock data
    if args.data_source == 'yahoo':
        logger.info("Using Yahoo Finance as data source")
        stock_collector = StockDataCollector()
    else:
        logger.info("Using Alpha Vantage as data source")
        stock_collector = AlphaVantageDataCollector()

    for ticker in args.tickers:
        try:
            stock_collector.collect_stock_data(
                ticker=ticker,
                start_date=args.start_date,
                end_date=args.end_date,
                save=True
            )
            logger.info(f"Collected stock data for {ticker}")
        except Exception as e:
            logger.error(f"Error collecting stock data for {ticker}: {e}")

    # Collect sentiment data
    try:
        sentiment_collector = SentimentDataCollector()
        for ticker in args.tickers:
            sentiment_collector.collect_sentiment_data(
                ticker=ticker,
                start_date=args.start_date,
                end_date=args.end_date,
                save=True
            )
            logger.info(f"Collected sentiment data for {ticker}")
    except Exception as e:
        logger.error(f"Error collecting sentiment data: {e}")

    # Collect economic data
    try:
        economic_collector = EconomicDataCollector()
        economic_collector.collect_economic_data(
            start_date=args.start_date,
            end_date=args.end_date,
            save=True
        )
        logger.info("Collected economic data")
    except Exception as e:
        logger.error(f"Error collecting economic data: {e}")

    logger.info("Data collection completed")

def process_data(args):
    """Process data and create features for model training."""
    logger.info("Starting data processing")

    # Create directories if they don't exist
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    os.makedirs(DATA_FEATURES_DIR, exist_ok=True)

    # Initialize data processor
    processor = DataProcessor(use_spark=args.use_spark)

    # Process data for each ticker
    for ticker in args.tickers:
        try:
            # Process ticker data
            processed_data = processor.process_ticker_data(
                ticker=ticker,
                include_sentiment=True,
                include_economic=True
            )

            if processed_data is not None:
                # Create features for training
                features = processor.create_features_for_training(
                    data=processed_data,
                    ticker=ticker
                )

                logger.info(f"Processed data and created features for {ticker}")
            else:
                logger.error(f"Failed to process data for {ticker}")

        except Exception as e:
            logger.error(f"Error processing data for {ticker}: {e}")

    logger.info("Data processing completed")

def train_models(args):
    """Train and evaluate prediction models."""
    logger.info("Starting model training")

    # Create directories if they don't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Train models for each ticker
    for ticker in args.tickers:
        try:
            # Load features
            features_file = os.path.join(DATA_FEATURES_DIR, f"{ticker}_features.csv")

            if not os.path.exists(features_file):
                logger.error(f"Features file not found for {ticker}")
                continue

            features_data = pd.read_csv(features_file)
            logger.info(f"Loaded features for {ticker}: {len(features_data)} samples")

            # Check if target column exists
            if args.target_column not in features_data.columns:
                logger.error(f"Target column '{args.target_column}' not found in features for {ticker}")
                continue

            # Train LSTM model
            if args.model_type in ['lstm', 'all']:
                lstm_model = LSTMModel(
                    sequence_length=args.sequence_length,
                    lstm_units=64,
                    dropout_rate=0.2,
                    learning_rate=0.001,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    patience=10,
                    model_name=f"lstm_{ticker}"
                )

                lstm_results = lstm_model.train_and_evaluate(
                    data=features_data,
                    target_col=args.target_column,
                    test_size=0.2,
                    val_size=0.2
                )

                logger.info(f"LSTM model training completed for {ticker}")
                logger.info(f"LSTM model evaluation: {lstm_results['eval_results']}")

            # Train Ensemble model
            if args.model_type in ['ensemble', 'all']:
                # Implement ensemble model training here
                pass

        except Exception as e:
            logger.error(f"Error training models for {ticker}: {e}")

    logger.info("Model training completed")

def main():
    """Main function to run the stock market prediction system."""
    # Parse command line arguments
    args = parse_arguments()

    # Print configuration
    logger.info("Configuration:")
    logger.info(f"  Tickers: {args.tickers}")
    logger.info(f"  Date range: {args.start_date} to {args.end_date}")
    logger.info(f"  Data source: {args.data_source}")
    logger.info(f"  Model type: {args.model_type}")
    logger.info(f"  Use Spark: {args.use_spark}")

    # Create necessary directories
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    os.makedirs(DATA_FEATURES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Data collection
    if not args.skip_data_collection:
        collect_data(args)
    else:
        logger.info("Skipping data collection")

    # Data processing
    if not args.skip_data_processing:
        process_data(args)
    else:
        logger.info("Skipping data processing")

    # Model training
    if not args.skip_training:
        train_models(args)
    else:
        logger.info("Skipping model training")

    logger.info("Stock market prediction system execution completed")

if __name__ == "__main__":
    main()