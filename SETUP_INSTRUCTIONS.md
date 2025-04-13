# Big Data-Driven Stock Market Prediction - Setup Instructions

This document provides detailed instructions for setting up and running the Big Data-Driven Stock Market Prediction system.

## Prerequisites

- Python 3.8 or higher
- MongoDB (optional, can use in-memory simulation)
- Kafka (optional, can use in-memory simulation)
- HDFS (optional, can use local file system)
- Hive (optional, can use local file system)

## Installation

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a virtual environment**:
   ```
   python -m venv .venv
   ```

3. **Activate the virtual environment**:
   - Windows:
     ```
     .venv\Scripts\activate
     ```
   - Linux/Mac:
     ```
     source .venv/bin/activate
     ```

4. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

5. **Configure environment variables**:
   - Copy the `.env.example` file to `.env`
   - Edit the `.env` file to set your API keys and configuration
   - Important API keys to set:
     - `ALPHA_VANTAGE_API_KEY`: Get from [Alpha Vantage](https://www.alphavantage.co/) (optional)
     - `FRED_API_KEY`: Get from [FRED](https://fred.stlouisfed.org/docs/api/api_key.html) (optional)
   - Note: Yahoo Finance data is fetched without requiring an API key

## Running the System

### Option 1: Run the entire system

To run all components of the system (data collection, real-time prediction, and dashboard):

```
python run_system.py --all
```

You can also run specific components:

```
python run_system.py --data-collection --tickers AAPL MSFT GOOG
python run_system.py --real-time-prediction
python run_system.py --dashboard
```

### Option 2: Run components individually

1. **Start data collection**:
   ```
   python src/main.py --tickers AAPL MSFT GOOG --data-source yahoo
   ```

2. **Start real-time prediction**:
   ```
   python src/run_real_time_prediction.py --tickers AAPL MSFT GOOG
   ```

3. **Launch the web dashboard**:
   ```
   python src/visualization/app.py
   ```

## Component Details

### Data Collection

The data collection component fetches real-time stock data from Yahoo Finance (no API key required) or Alpha Vantage (API key required), real-time sentiment data from Yahoo Finance news articles, and economic data from FRED. The data is stored in MongoDB and/or HDFS. No mock or simulated data is used - the system only works with real-time data.

### Real-Time Prediction

The real-time prediction component consumes data from Kafka, processes it using feature engineering, and makes predictions using trained models. The predictions are stored in MongoDB.

### Visualization Dashboard

The visualization dashboard displays stock data, technical indicators, and predictions. It also provides tools for analyzing the data and comparing different models.

## Troubleshooting

### MongoDB Connection Issues

If you encounter issues connecting to MongoDB, the system will use an in-memory simulation. You can check the logs for messages like:

```
Failed to connect to MongoDB: <error>. Using in-memory simulation instead.
```

### Kafka Connection Issues

If you encounter issues connecting to Kafka, the system will use an in-memory simulation. You can check the logs for messages like:

```
Failed to create Kafka producer/consumer: <error>. Using simulator instead.
```

### Missing API Keys

If you haven't set up API keys, some data sources may not be available. You can check the logs for messages like:

```
API key not found for <service>. Using default data source instead.
```

## Additional Resources

- [Project Documentation](./docs/README.md)
- [API Reference](./docs/API.md)
- [Architecture Overview](./docs/ARCHITECTURE.md)
