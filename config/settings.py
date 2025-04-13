"""
Configuration settings for the stock market prediction application.
This file imports all settings from src.utils.config for backwards compatibility.
"""
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all settings from utils.config
from src.utils.config import *

# Module-specific settings for config.settings
MODELS_DIR = MODEL_DIR  # Ensure MODELS_DIR is defined
RESULTS_DIR = os.path.join(MODEL_DIR, 'results')  # Define RESULTS_DIR

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# Define data directories with new paths if needed
DATA_RAW_DIR = RAW_DATA_DIR
DATA_PROCESSED_DIR = PROCESSED_DATA_DIR
DATA_FEATURES_DIR = os.path.join(DATA_DIR, 'features')
os.makedirs(DATA_FEATURES_DIR, exist_ok=True) 