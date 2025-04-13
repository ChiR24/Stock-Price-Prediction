"""
Script to compare stock data from Yahoo Finance and Alpha Vantage.
This demonstrates how to use both data collectors and compares the results.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import project modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from data_collection.stock_data import StockDataCollector
from data_collection.alpha_vantage_data import AlphaVantageDataCollector
from utils.config import RAW_DATA_DIR

# Set up logger
logger = setup_logger('compare_data_sources')

def collect_data_from_both_sources(ticker, start_date=None, end_date=None):
    """
    Collect stock data from both Yahoo Finance and Alpha Vantage.
    
    Args:
        ticker: Stock ticker symbol.
        start_date: Start date for data collection (YYYY-MM-DD). Defaults to 30 days ago.
        end_date: End date for data collection (YYYY-MM-DD). Defaults to today.
        
    Returns:
        Tuple of DataFrames (yahoo_data, alpha_vantage_data).
    """
    # Set default dates if not provided
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if not start_date:
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    logger.info(f"Collecting data for {ticker} from {start_date} to {end_date}")
    
    # Create collectors
    yahoo_collector = StockDataCollector()
    alpha_vantage_collector = AlphaVantageDataCollector()
    
    # Collect data from Yahoo Finance
    yahoo_data = yahoo_collector.collect_stock_data(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        save=True
    )
    
    # Collect data from Alpha Vantage
    alpha_vantage_data = alpha_vantage_collector.collect_stock_data(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        save=True
    )
    
    logger.info(f"Yahoo Finance data: {len(yahoo_data)} records")
    logger.info(f"Alpha Vantage data: {len(alpha_vantage_data)} records")
    
    return yahoo_data, alpha_vantage_data

def compare_data(yahoo_data, alpha_vantage_data, ticker):
    """
    Compare data from Yahoo Finance and Alpha Vantage.
    
    Args:
        yahoo_data: DataFrame with Yahoo Finance data.
        alpha_vantage_data: DataFrame with Alpha Vantage data.
        ticker: Stock ticker symbol.
        
    Returns:
        DataFrame with comparison results.
    """
    logger.info("Comparing data from both sources")
    
    # Check if both DataFrames have data
    if yahoo_data.empty or alpha_vantage_data.empty:
        logger.warning("One or both data sources returned empty data")
        return pd.DataFrame()
    
    # Ensure both DataFrames have 'Date' as datetime
    yahoo_data['Date'] = pd.to_datetime(yahoo_data['Date'])
    alpha_vantage_data['Date'] = pd.to_datetime(alpha_vantage_data['Date'])
    
    # Set Date as index for both DataFrames
    yahoo_data.set_index('Date', inplace=True)
    alpha_vantage_data.set_index('Date', inplace=True)
    
    # Get overlapping dates
    common_dates = yahoo_data.index.intersection(alpha_vantage_data.index)
    
    if len(common_dates) == 0:
        logger.warning("No overlapping dates between the two data sources")
        return pd.DataFrame()
    
    # Filter data to common dates
    yahoo_filtered = yahoo_data.loc[common_dates]
    alpha_vantage_filtered = alpha_vantage_data.loc[common_dates]
    
    # Create comparison DataFrame
    comparison = pd.DataFrame(index=common_dates)
    
    # Add close prices
    comparison['Yahoo_Close'] = yahoo_filtered['Close']
    comparison['AlphaVantage_Close'] = alpha_vantage_filtered['Close']
    
    # Calculate differences and stats
    comparison['Diff'] = comparison['Yahoo_Close'] - comparison['AlphaVantage_Close']
    comparison['Diff_Pct'] = (comparison['Diff'] / comparison['Yahoo_Close']) * 100
    
    # Calculate statistics
    stats = {
        'mean_diff': comparison['Diff'].mean(),
        'mean_diff_pct': comparison['Diff_Pct'].mean(),
        'max_diff': comparison['Diff'].max(),
        'min_diff': comparison['Diff'].min(),
        'std_diff': comparison['Diff'].std(),
        'correlation': comparison['Yahoo_Close'].corr(comparison['AlphaVantage_Close'])
    }
    
    logger.info(f"Comparison statistics: {stats}")
    
    return comparison, stats

def plot_comparison(comparison, ticker, save_path=None):
    """
    Plot comparison of data from Yahoo Finance and Alpha Vantage.
    
    Args:
        comparison: DataFrame with comparison data.
        ticker: Stock ticker symbol.
        save_path: Path to save the plot. Defaults to None (don't save).
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot close prices
    comparison[['Yahoo_Close', 'AlphaVantage_Close']].plot(ax=ax1)
    ax1.set_title(f'{ticker} Close Price Comparison')
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot difference
    comparison['Diff'].plot(ax=ax2, color='red')
    ax2.set_title('Price Difference (Yahoo - Alpha Vantage)')
    ax2.set_ylabel('Difference')
    ax2.grid(True, alpha=0.3)
    
    # Add zero line
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot if save_path is provided
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved comparison plot to {save_path}")
    
    plt.show()

def main():
    """
    Main function to compare stock data from Yahoo Finance and Alpha Vantage.
    """
    # Ticker to compare
    ticker = 'AAPL'
    
    # Date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    # Collect data
    yahoo_data, alpha_vantage_data = collect_data_from_both_sources(ticker, start_date, end_date)
    
    # Compare data
    if not yahoo_data.empty and not alpha_vantage_data.empty:
        comparison, stats = compare_data(yahoo_data, alpha_vantage_data, ticker)
        
        # Save results
        os.makedirs(os.path.join(RAW_DATA_DIR, 'comparisons'), exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save comparison data
        comparison_path = os.path.join(RAW_DATA_DIR, 'comparisons', f"{ticker}_comparison_{timestamp}.csv")
        comparison.to_csv(comparison_path)
        logger.info(f"Saved comparison data to {comparison_path}")
        
        # Plot comparison
        plot_path = os.path.join(RAW_DATA_DIR, 'comparisons', f"{ticker}_comparison_plot_{timestamp}.png")
        plot_comparison(comparison, ticker, plot_path)
    else:
        logger.warning("Could not compare data due to empty datasets")

if __name__ == "__main__":
    main() 