"""
Configuration settings for the stock market prediction application.
"""
import os
import sys

# Add the virtual environment's site-packages to the Python path
venv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.venv', 'Lib', 'site-packages')
if os.path.exists(venv_path):
    sys.path.insert(0, venv_path)

import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
DATA_RAW_DIR = os.path.join(DATA_DIR, 'raw')
DATA_PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
DATA_FEATURES_DIR = os.path.join(DATA_DIR, 'features')

# Model directories
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
MODEL_CHECKPOINTS_DIR = os.path.join(MODEL_DIR, 'checkpoints')
MODEL_ARTIFACTS_DIR = os.path.join(MODEL_DIR, 'artifacts')

# Logs directory
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')

# Ensure critical directories exist
for directory in [DATA_DIR, DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_FEATURES_DIR,
                  MODEL_DIR, MODEL_CHECKPOINTS_DIR, MODEL_ARTIFACTS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Data collection settings
DEFAULT_START_DATE = "2010-01-01"
DEFAULT_END_DATE = "2023-12-31"
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]

# Data processing settings
SEQUENCE_LENGTH = 60  # Number of days to use for prediction
TRAIN_SPLIT = 0.8     # Percentage of data to use for training
VAL_SPLIT = 0.1       # Percentage of data to use for validation
TEST_SPLIT = 0.1      # Percentage of data to use for testing

# Model training settings
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10

# Feature settings
PRICE_FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']
TECHNICAL_INDICATORS = ['SMA_20', 'EMA_20', 'RSI', 'MACD', 'Bollinger_Upper', 'Bollinger_Lower']
TARGET_COLUMN = 'Close'

# API settings
API_RATE_LIMIT = 5  # Requests per minute
API_TIMEOUT = 30    # Seconds

# Default tickers to analyze
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT"]

# API keys (should be moved to environment variables in production)
ALPHA_VANTAGE_API_KEY = "YOUR_API_KEY"  # Replace with your Alpha Vantage API key
YAHOO_FINANCE_API_KEY = "YOUR_API_KEY"  # Replace with your Yahoo Finance API key (if applicable)
FRED_API_KEY = os.getenv('FRED_API_KEY', 'YOUR_FRED_API_KEY')  # Replace with your FRED API key

# Logging configuration
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = os.path.join(LOGS_DIR, "app.log")

# Data collection settings
DEFAULT_INTERVAL = "1d"  # daily data

# Feature engineering settings
TECHNICAL_INDICATORS = {
    "SMA": [20, 50, 200],  # Simple Moving Average periods
    "EMA": [12, 26],  # Exponential Moving Average periods
    "RSI": [14],  # Relative Strength Index periods
    "MACD": {"fast": 12, "slow": 26, "signal": 9},  # Moving Average Convergence Divergence
    "Bollinger": {"window": 20, "std_dev": 2},  # Bollinger Bands
}

# Model training settings
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_SEQUENCE_LENGTH = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 50
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_DROPOUT_RATE = 0.2

# Prediction settings
DEFAULT_PREDICTION_DAYS = 5  # Default number of days to predict ahead

# Web application settings
DASH_DEBUG = True
DEFAULT_PORT = 8050
DEFAULT_HOST = "127.0.0.1"

# Project base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Data directories
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(DATA_DIR, 'models')
MODELS_DIR = MODEL_DIR  # Alias for compatibility

# Create directories if they don't exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# API keys and credentials
FRED_API_KEY = os.getenv('FRED_API_KEY', '')
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')

# Social media keywords for each ticker
SOCIAL_MEDIA_KEYWORDS = {
    'AAPL': ['Apple', 'AAPL', 'iPhone', 'iPad', 'Tim Cook'],
    'MSFT': ['Microsoft', 'MSFT', 'Windows', 'Azure', 'Satya Nadella'],
    'GOOG': ['Google', 'GOOG', 'GOOGL', 'Alphabet', 'Sundar Pichai'],
    'AMZN': ['Amazon', 'AMZN', 'AWS', 'Jeff Bezos', 'Andy Jassy'],
    'META': ['Meta', 'Facebook', 'META', 'Instagram', 'Mark Zuckerberg'],
    'TSLA': ['Tesla', 'TSLA', 'Elon Musk', 'Electric Vehicle'],
    'NVDA': ['NVIDIA', 'NVDA', 'GPU', 'Jensen Huang'],
    'JPM': ['JPMorgan', 'JPM', 'Jamie Dimon'],
    'V': ['Visa', 'V', 'Payment'],
    'JNJ': ['Johnson & Johnson', 'JNJ', 'Pharmaceutical'],
    'WMT': ['Walmart', 'WMT', 'Retail'],
    'PG': ['Procter & Gamble', 'PG', 'Consumer Goods'],
    'MA': ['Mastercard', 'MA', 'Payment'],
    'UNH': ['UnitedHealth', 'UNH', 'Healthcare'],
    'HD': ['Home Depot', 'HD', 'Retail'],
    'BAC': ['Bank of America', 'BAC', 'Banking'],
    'XOM': ['Exxon Mobil', 'XOM', 'Oil', 'Energy'],
    'PFE': ['Pfizer', 'PFE', 'Pharmaceutical'],
    'CSCO': ['Cisco', 'CSCO', 'Networking'],
    'VZ': ['Verizon', 'VZ', 'Telecommunications']
}

# Database configuration
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
MONGODB_DB = os.getenv('MONGODB_DB', 'stock_prediction')

# Spark configuration
SPARK_MASTER = os.getenv('SPARK_MASTER', 'local[*]')
SPARK_APP_NAME = 'StockMarketPrediction'

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
KAFKA_STOCK_TOPIC = 'stock-data'
KAFKA_SENTIMENT_TOPIC = 'sentiment-data'
KAFKA_ECONOMIC_TOPIC = 'economic-data'

# Web application settings
FLASK_PORT = int(os.getenv('FLASK_PORT', 5000))
FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'

# Model settings
PREDICTION_HORIZON = 5  # Days to predict ahead
HISTORICAL_WINDOW = 30  # Days of history to use for prediction
MODEL_TYPE = os.getenv('MODEL_TYPE', 'lstm')  # 'lstm', 'xgboost', etc.

# Social media keywords
SOCIAL_MEDIA_KEYWORDS = {
    'AAPL': ['Apple', 'iPhone', 'MacBook', 'Tim Cook', '$AAPL'],
    'MSFT': ['Microsoft', 'Windows', 'Azure', 'Satya Nadella', '$MSFT'],
    'GOOG': ['Google', 'Alphabet', 'Android', 'Sundar Pichai', '$GOOG'],
    'AMZN': ['Amazon', 'AWS', 'Jeff Bezos', 'Andy Jassy', '$AMZN'],
    'META': ['Meta', 'Facebook', 'Instagram', 'Mark Zuckerberg', '$META']
}

# Macroeconomic indicators
MACROECONOMIC_INDICATORS = [
    {'series_id': 'GDP', 'name': 'Gross Domestic Product', 'frequency': 'Quarterly'},
    {'series_id': 'UNRATE', 'name': 'Unemployment Rate', 'frequency': 'Monthly'},
    {'series_id': 'CPIAUCSL', 'name': 'Consumer Price Index', 'frequency': 'Monthly'},
    {'series_id': 'FEDFUNDS', 'name': 'Federal Funds Rate', 'frequency': 'Monthly'},
    {'series_id': 'INDPRO', 'name': 'Industrial Production Index', 'frequency': 'Monthly'},
    {'series_id': 'HOUST', 'name': 'Housing Starts', 'frequency': 'Monthly'},
    {'series_id': 'PCE', 'name': 'Personal Consumption Expenditures', 'frequency': 'Monthly'},
    {'series_id': 'RETAIL', 'name': 'Retail Sales', 'frequency': 'Monthly'},
    {'series_id': 'M2', 'name': 'M2 Money Stock', 'frequency': 'Weekly'}
]

# HDFS configuration
HDFS_HOST = os.getenv('HDFS_HOST', 'localhost')
HDFS_PORT = int(os.getenv('HDFS_PORT', 9000))
HDFS_USER = os.getenv('HDFS_USER', 'hdfs')

# Hive configuration
HIVE_HOST = os.getenv('HIVE_HOST', 'localhost')
HIVE_PORT = int(os.getenv('HIVE_PORT', 10000))
HIVE_USER = os.getenv('HIVE_USER', 'hive')
HIVE_PASSWORD = os.getenv('HIVE_PASSWORD', '')