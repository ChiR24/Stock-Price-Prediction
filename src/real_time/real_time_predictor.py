"""
Module for real-time stock price prediction using trained models.
This module consumes data from Kafka, processes it, and makes predictions.
"""
import os
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime
from kafka import KafkaConsumer
from pymongo import MongoClient

# Import project modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from utils.config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_STOCK_TOPIC,
    KAFKA_SENTIMENT_TOPIC,
    KAFKA_ECONOMIC_TOPIC,
    MONGODB_URI,
    MONGODB_DB,
    MODELS_DIR
)
from data_processing.feature_engineering import FeatureEngineering
from ml_models.lstm_model import LSTMModel
from ml_models.ensemble_model import EnsembleModel
from real_time.kafka_handler import KafkaHandler

# Set up logger
logger = setup_logger('real_time_predictor')

class RealTimePredictor:
    """
    Class for real-time stock price prediction.
    """
    def __init__(self, tickers=None, model_type='lstm', prediction_horizon=5, use_mongodb=True):
        """
        Initialize the RealTimePredictor.
        
        Args:
            tickers: List of stock tickers to predict. Defaults to None (load all available models).
            model_type: Type of model to use ('lstm', 'ensemble', 'all'). Defaults to 'lstm'.
            prediction_horizon: Number of days ahead to predict. Defaults to 5.
            use_mongodb: Whether to store predictions in MongoDB. Defaults to True.
        """
        self.tickers = tickers
        self.model_type = model_type
        self.prediction_horizon = prediction_horizon
        self.use_mongodb = use_mongodb
        
        # Initialize components
        self.kafka_handler = KafkaHandler(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)
        self.feature_engineering = FeatureEngineering()
        
        # Initialize MongoDB client if needed
        if self.use_mongodb:
            try:
                self.mongo_client = MongoClient(MONGODB_URI)
                self.mongo_db = self.mongo_client[MONGODB_DB]
                self.predictions_collection = self.mongo_db['stock_predictions']
                logger.info(f"Connected to MongoDB: {MONGODB_URI}, database: {MONGODB_DB}")
            except Exception as e:
                logger.error(f"Failed to connect to MongoDB: {e}")
                self.use_mongodb = False
        
        # Load models
        self.models = self._load_models()
        
        # Initialize data storage for real-time processing
        self.stock_data = {}
        self.sentiment_data = {}
        self.economic_data = {}
        
        logger.info(f"RealTimePredictor initialized with model_type={model_type}, prediction_horizon={prediction_horizon}")
    
    def _load_models(self):
        """
        Load trained models for each ticker.
        
        Returns:
            Dictionary of models with ticker as key.
        """
        models = {}
        
        # Get list of tickers from model directory if not provided
        if not self.tickers:
            try:
                model_files = os.listdir(MODELS_DIR)
                self.tickers = list(set([f.split('_')[1] for f in model_files if f.startswith('lstm_') or f.startswith('ensemble_')]))
                logger.info(f"Found models for tickers: {self.tickers}")
            except Exception as e:
                logger.error(f"Failed to get tickers from model directory: {e}")
                return models
        
        # Load models for each ticker
        for ticker in self.tickers:
            ticker_models = {}
            
            # Load LSTM model if requested
            if self.model_type in ['lstm', 'all']:
                try:
                    lstm_model = LSTMModel(
                        ticker=ticker,
                        sequence_length=60,
                        prediction_horizon=self.prediction_horizon
                    )
                    
                    # Try to load the model
                    if lstm_model.load_model():
                        ticker_models['lstm'] = lstm_model
                        logger.info(f"Loaded LSTM model for {ticker}")
                    else:
                        logger.warning(f"Failed to load LSTM model for {ticker}")
                except Exception as e:
                    logger.error(f"Error loading LSTM model for {ticker}: {e}")
            
            # Load Ensemble model if requested
            if self.model_type in ['ensemble', 'all']:
                try:
                    ensemble_model = EnsembleModel(
                        ticker=ticker,
                        prediction_horizon=self.prediction_horizon
                    )
                    
                    # Try to load the model
                    if ensemble_model.load_model():
                        ticker_models['ensemble'] = ensemble_model
                        logger.info(f"Loaded Ensemble model for {ticker}")
                    else:
                        logger.warning(f"Failed to load Ensemble model for {ticker}")
                except Exception as e:
                    logger.error(f"Error loading Ensemble model for {ticker}: {e}")
            
            # Add ticker models to the main dictionary
            if ticker_models:
                models[ticker] = ticker_models
        
        logger.info(f"Loaded models for {len(models)} tickers")
        return models
    
    def _process_stock_message(self, message):
        """
        Process a stock data message from Kafka.
        
        Args:
            message: Message from Kafka.
        """
        try:
            # Extract ticker
            ticker = message.get('ticker')
            
            if not ticker:
                logger.warning("Received stock message without ticker")
                return
            
            # Initialize ticker data if not exists
            if ticker not in self.stock_data:
                self.stock_data[ticker] = []
            
            # Add message to stock data
            self.stock_data[ticker].append(message)
            
            # Keep only the latest 200 data points
            if len(self.stock_data[ticker]) > 200:
                self.stock_data[ticker] = self.stock_data[ticker][-200:]
            
            logger.debug(f"Processed stock message for {ticker}")
            
            # Make prediction if we have enough data
            if len(self.stock_data[ticker]) >= 60:  # Minimum sequence length for LSTM
                self._make_prediction(ticker)
        
        except Exception as e:
            logger.error(f"Error processing stock message: {e}")
    
    def _process_sentiment_message(self, message):
        """
        Process a sentiment data message from Kafka.
        
        Args:
            message: Message from Kafka.
        """
        try:
            # Extract ticker
            ticker = message.get('ticker')
            
            if not ticker:
                logger.warning("Received sentiment message without ticker")
                return
            
            # Initialize ticker data if not exists
            if ticker not in self.sentiment_data:
                self.sentiment_data[ticker] = []
            
            # Add message to sentiment data
            self.sentiment_data[ticker].append(message)
            
            # Keep only the latest 1000 data points
            if len(self.sentiment_data[ticker]) > 1000:
                self.sentiment_data[ticker] = self.sentiment_data[ticker][-1000:]
            
            logger.debug(f"Processed sentiment message for {ticker}")
        
        except Exception as e:
            logger.error(f"Error processing sentiment message: {e}")
    
    def _process_economic_message(self, message):
        """
        Process an economic data message from Kafka.
        
        Args:
            message: Message from Kafka.
        """
        try:
            # Extract series ID
            series_id = message.get('series_id')
            
            if not series_id:
                logger.warning("Received economic message without series_id")
                return
            
            # Initialize series data if not exists
            if series_id not in self.economic_data:
                self.economic_data[series_id] = []
            
            # Add message to economic data
            self.economic_data[series_id].append(message)
            
            # Keep only the latest 1000 data points
            if len(self.economic_data[series_id]) > 1000:
                self.economic_data[series_id] = self.economic_data[series_id][-1000:]
            
            logger.debug(f"Processed economic message for {series_id}")
        
        except Exception as e:
            logger.error(f"Error processing economic message: {e}")
    
    def _prepare_data_for_prediction(self, ticker):
        """
        Prepare data for prediction.
        
        Args:
            ticker: Stock ticker symbol.
            
        Returns:
            DataFrame with prepared data.
        """
        try:
            # Check if we have stock data for this ticker
            if ticker not in self.stock_data or not self.stock_data[ticker]:
                logger.warning(f"No stock data available for {ticker}")
                return None
            
            # Convert stock data to DataFrame
            stock_df = pd.DataFrame(self.stock_data[ticker])
            
            # Convert date strings to datetime
            if 'Date' in stock_df.columns:
                stock_df['Date'] = pd.to_datetime(stock_df['Date'])
                stock_df.set_index('Date', inplace=True)
            elif 'timestamp' in stock_df.columns:
                stock_df['Date'] = pd.to_datetime(stock_df['timestamp'])
                stock_df.set_index('Date', inplace=True)
            
            # Sort by date
            stock_df.sort_index(inplace=True)
            
            # Process data using feature engineering
            processed_df = self.feature_engineering.calculate_technical_indicators(stock_df)
            
            # Add sentiment features if available
            if ticker in self.sentiment_data and self.sentiment_data[ticker]:
                sentiment_df = pd.DataFrame(self.sentiment_data[ticker])
                
                # Process sentiment data
                if 'created_at' in sentiment_df.columns:
                    sentiment_df['Date'] = pd.to_datetime(sentiment_df['created_at'])
                    sentiment_df.set_index('Date', inplace=True)
                
                # Aggregate sentiment by date
                if 'polarity' in sentiment_df.columns:
                    daily_sentiment = sentiment_df.groupby(pd.Grouper(freq='D')).agg({
                        'polarity': 'mean',
                        'subjectivity': 'mean'
                    })
                    
                    # Add sentiment to processed data
                    processed_df = self.feature_engineering.add_sentiment_features(processed_df, daily_sentiment)
            
            # Add economic features if available
            # This is more complex and would require mapping economic indicators to dates
            
            logger.info(f"Prepared data for {ticker} prediction: {processed_df.shape}")
            return processed_df
        
        except Exception as e:
            logger.error(f"Error preparing data for prediction: {e}")
            return None
    
    def _make_prediction(self, ticker):
        """
        Make prediction for a ticker.
        
        Args:
            ticker: Stock ticker symbol.
        """
        try:
            # Check if we have models for this ticker
            if ticker not in self.models or not self.models[ticker]:
                logger.warning(f"No models available for {ticker}")
                return
            
            # Prepare data
            data = self._prepare_data_for_prediction(ticker)
            
            if data is None or data.empty:
                logger.warning(f"No data available for {ticker} prediction")
                return
            
            # Make predictions with each model
            predictions = {}
            
            for model_name, model in self.models[ticker].items():
                try:
                    # Make prediction
                    if model_name == 'lstm':
                        future_df = model.predict_future(data, steps=self.prediction_horizon)
                    else:  # ensemble
                        future_df = model.predict_future(data, steps=self.prediction_horizon)
                    
                    if future_df is not None:
                        predictions[model_name] = future_df
                        logger.info(f"Made {model_name} prediction for {ticker}")
                
                except Exception as e:
                    logger.error(f"Error making {model_name} prediction for {ticker}: {e}")
            
            # Store predictions in MongoDB if enabled
            if self.use_mongodb and predictions:
                self._store_predictions(ticker, predictions)
            
            return predictions
        
        except Exception as e:
            logger.error(f"Error making prediction for {ticker}: {e}")
            return None
    
    def _store_predictions(self, ticker, predictions):
        """
        Store predictions in MongoDB.
        
        Args:
            ticker: Stock ticker symbol.
            predictions: Dictionary of predictions by model.
        """
        try:
            # Create document
            document = {
                'ticker': ticker,
                'timestamp': datetime.now().isoformat(),
                'prediction_horizon': self.prediction_horizon,
                'predictions': {}
            }
            
            # Add predictions for each model
            for model_name, pred_df in predictions.items():
                # Convert DataFrame to dictionary
                pred_dict = {}
                
                for date, row in pred_df.iterrows():
                    date_str = date.strftime('%Y-%m-%d')
                    pred_dict[date_str] = row.to_dict()
                
                document['predictions'][model_name] = pred_dict
            
            # Insert into MongoDB
            self.predictions_collection.insert_one(document)
            
            logger.info(f"Stored predictions for {ticker} in MongoDB")
        
        except Exception as e:
            logger.error(f"Error storing predictions in MongoDB: {e}")
    
    def _message_handler(self, topic, message):
        """
        Handle messages from Kafka.
        
        Args:
            topic: Kafka topic.
            message: Message from Kafka.
        """
        if topic == KAFKA_STOCK_TOPIC:
            self._process_stock_message(message)
        elif topic == KAFKA_SENTIMENT_TOPIC:
            self._process_sentiment_message(message)
        elif topic == KAFKA_ECONOMIC_TOPIC:
            self._process_economic_message(message)
        else:
            logger.warning(f"Received message from unknown topic: {topic}")
    
    def start(self):
        """
        Start consuming messages from Kafka and making predictions.
        """
        try:
            # Create Kafka consumer
            topics = [KAFKA_STOCK_TOPIC, KAFKA_SENTIMENT_TOPIC, KAFKA_ECONOMIC_TOPIC]
            
            if self.kafka_handler.create_consumer(topics, group_id="predictor-group"):
                # Start consuming messages
                self.kafka_handler.consume_messages(
                    topics,
                    self._message_handler,
                    group_id="predictor-group"
                )
                
                logger.info(f"Started consuming messages from topics: {topics}")
                return True
            else:
                logger.error("Failed to create Kafka consumer")
                return False
        
        except Exception as e:
            logger.error(f"Error starting real-time predictor: {e}")
            return False
    
    def stop(self):
        """
        Stop consuming messages from Kafka.
        """
        try:
            # Stop Kafka consumer
            self.kafka_handler.stop_consuming()
            
            # Close MongoDB connection if enabled
            if self.use_mongodb and hasattr(self, 'mongo_client'):
                self.mongo_client.close()
                logger.info("Closed MongoDB connection")
            
            logger.info("Stopped real-time predictor")
            return True
        
        except Exception as e:
            logger.error(f"Error stopping real-time predictor: {e}")
            return False


def main():
    """
    Main function to run the real-time predictor.
    """
    try:
        # Initialize predictor
        predictor = RealTimePredictor(
            model_type='all',
            prediction_horizon=5,
            use_mongodb=True
        )
        
        # Start predictor
        if predictor.start():
            logger.info("Real-time predictor started")
            
            # Keep running until interrupted
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Real-time predictor interrupted by user")
            
            # Stop predictor
            predictor.stop()
        else:
            logger.error("Failed to start real-time predictor")
    
    except Exception as e:
        logger.error(f"Error in main function: {e}")


if __name__ == "__main__":
    main()
