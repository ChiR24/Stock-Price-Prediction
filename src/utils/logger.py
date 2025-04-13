"""
Logging utility for the stock market prediction project.
"""
import logging
import os
import sys
from datetime import datetime

# Create logs directory
logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Generate log filename based on current date
log_filename = os.path.join(logs_dir, f'stock_prediction_{datetime.now().strftime("%Y%m%d")}.log')

# Configure logging
def setup_logger(name):
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: The name of the logger

    Returns:
        A configured logger instance
    """
    # Create a custom logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Check if logger already has handlers to avoid duplicates
    if not logger.handlers:
        # Create handlers
        file_handler = logging.FileHandler(log_filename)
        console_handler = logging.StreamHandler(sys.stdout)
        
        # Set level
        file_handler.setLevel(logging.DEBUG)
        console_handler.setLevel(logging.INFO)
        
        # Create formatters
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Add formatters to handlers
        file_handler.setFormatter(file_format)
        console_handler.setFormatter(console_format)
        
        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

# Create a default logger for importing
logger = setup_logger('stock_prediction') 