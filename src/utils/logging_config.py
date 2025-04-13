"""
Logging configuration for the stock market prediction application.
"""
import os
import logging
from logging.handlers import RotatingFileHandler
from src.utils.config import LOG_LEVEL, LOG_FORMAT, LOG_FILE, LOG_DIR

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

def setup_logger(name, log_file=LOG_FILE, level=LOG_LEVEL, format_str=LOG_FORMAT):
    """
    Set up a logger with the specified name, log file, level, and format.
    
    Args:
        name (str): Name of the logger
        log_file (str): Path to the log file
        level (int): Logging level
        format_str (str): Log format string
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create handlers
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    console_handler = logging.StreamHandler()
    
    # Create formatter and add it to handlers
    formatter = logging.Formatter(format_str)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger 