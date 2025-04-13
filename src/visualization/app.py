"""
Web application for visualizing stock market predictions.
"""
import os
import sys

# Add the virtual environment's site-packages to the Python path
venv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.venv', 'Lib', 'site-packages')
if os.path.exists(venv_path):
    sys.path.insert(0, venv_path)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, callback, Output, Input, State, dash_table
import dash_bootstrap_components as dbc
import yfinance as yf

# Import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from utils.config import DEFAULT_TICKERS, PROCESSED_DATA_DIR, MODEL_DIR, MONGODB_URI, MONGODB_DB
from data_collection.stock_data_collector import StockDataCollector
from data_processing.feature_engineering import FeatureEngineering
from ml_models.lstm_model import LSTMModel
from ml_models.model_benchmark import ModelBenchmark
from ml_models.spark_ml_models import SparkRandomForestModel, SparkGradientBoostedTreesModel, SparkLinearModel

# MongoDB for real-time predictions
from pymongo import MongoClient

# Set up logger
logger = setup_logger('visualization_app')

# Initialize services
stock_collector = StockDataCollector()
feature_engineering = FeatureEngineering()

# Initialize MongoDB client for real-time predictions
try:
    mongo_client = MongoClient(MONGODB_URI)
    mongo_db = mongo_client[MONGODB_DB]
    predictions_collection = mongo_db['stock_predictions']
    logger.info(f"Connected to MongoDB: {MONGODB_URI}, database: {MONGODB_DB}")
    mongodb_available = True
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    mongodb_available = False

# Global variables
PREDICTION_DAYS = 5
PAGE_SIZE = 20

# Initialize the Dash app with increased timeout
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    update_title='Loading...',
)

# Configure longer callback timeout
app.server.config['TIMEOUT'] = 300  # 5 minutes in seconds

# Define the app layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Stock Market Prediction Dashboard", className="text-center mb-4"),
            html.P("A data-driven stock market prediction tool using big data and machine learning.", className="text-center"),
        ], width=12)
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Select Stock"),
                dbc.CardBody([
                    dcc.Dropdown(
                        id="ticker-dropdown",
                        options=[{"label": ticker, "value": ticker} for ticker in DEFAULT_TICKERS],
                        value=DEFAULT_TICKERS[0],
                        clearable=False,
                    ),
                    dbc.Button("Update Data", id="update-button", color="primary", className="mt-2"),
                    dbc.Spinner(html.Div(id="update-status"), color="primary"),
                ])
            ], className="mb-4"),

            dbc.Card([
                dbc.CardHeader("Date Range"),
                dbc.CardBody([
                    dcc.DatePickerRange(
                        id="date-range",
                        min_date_allowed=datetime.now() - timedelta(days=365*2),
                        max_date_allowed=datetime.now(),
                        start_date=datetime.now() - timedelta(days=180),
                        end_date=datetime.now(),
                        display_format="YYYY-MM-DD",
                    ),
                ])
            ], className="mb-4"),

            dbc.Card([
                dbc.CardHeader("Prediction Settings"),
                dbc.CardBody([
                    html.Label("Prediction Horizon (Days)"),
                    dcc.Slider(
                        id="prediction-horizon-slider",
                        min=1,
                        max=30,
                        step=1,
                        value=PREDICTION_DAYS,
                        marks={i: str(i) for i in range(0, 31, 5)},
                    ),
                    dbc.Button("Run Prediction", id="predict-button", color="success", className="mt-3"),
                    dbc.Spinner(html.Div(id="prediction-status"), color="success"),
                ])
            ], className="mb-4"),

            dbc.Card([
                dbc.CardHeader("Technical Indicators"),
                dbc.CardBody([
                    dcc.Dropdown(
                        id="indicator-dropdown",
                        options=[
                            {"label": "Simple Moving Average (SMA)", "value": "sma"},
                            {"label": "Exponential Moving Average (EMA)", "value": "ema"},
                            {"label": "Relative Strength Index (RSI)", "value": "rsi"},
                            {"label": "Bollinger Bands", "value": "bollinger"},
                            {"label": "MACD", "value": "macd"},
                        ],
                        value=["sma", "rsi"],
                        multi=True,
                    ),
                ])
            ], className="mb-4"),
        ], width=3),

        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label="Price Chart", tab_id="tab-price", children=[
                    dcc.Graph(id="price-chart", style={"height": "70vh"}),
                ]),
                dbc.Tab(label="Technical Indicators", tab_id="tab-indicators", children=[
                    dcc.Graph(id="indicator-chart", style={"height": "70vh"}),
                ]),
                dbc.Tab(label="Prediction Results", tab_id="tab-prediction", children=[
                    dcc.Graph(id="prediction-chart", style={"height": "70vh"}),
                ]),
                dbc.Tab(label="Real-Time Predictions", tab_id="tab-realtime", children=[
                    dbc.Card([
                        dbc.CardHeader("Real-Time Predictions"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.H5("Latest Real-Time Predictions"),
                                    html.P("Predictions are updated in real-time from the streaming data pipeline."),
                                    dbc.Button("Refresh", id="refresh-realtime-button", color="primary", className="mb-3"),
                                    dbc.Spinner(html.Div(id="realtime-status"), color="primary"),
                                ], width=12),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dcc.Graph(id="realtime-prediction-chart", style={"height": "50vh"}),
                                ], width=12),
                            ]),
                        ]),
                    ]),
                ]),
                dbc.Tab(label="Performance Metrics", tab_id="tab-performance", children=[
                    html.Div(id="performance-metrics"),
                ]),
                dbc.Tab(label="Model Comparison", tab_id="tab-model-comparison", children=[
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Model Selection"),
                                dbc.CardBody([
                                    dcc.Checklist(
                                        id="model-checklist",
                                        options=[
                                            {"label": "LSTM", "value": "lstm"},
                                            {"label": "Random Forest", "value": "rf"},
                                            {"label": "Gradient Boosted Trees", "value": "gbt"},
                                            {"label": "Linear Model", "value": "linear"},
                                        ],
                                        value=["lstm", "rf"],
                                        inline=True,
                                    ),
                                    dbc.Button("Run Comparison", id="compare-button", color="primary", className="mt-2"),
                                    dbc.Spinner(html.Div(id="comparison-status"), color="primary"),
                                ])
                            ], className="mb-4"),
                        ], width=12),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Performance Metrics"),
                                dbc.CardBody([
                                    html.Div(id="metrics-comparison-table"),
                                ])
                            ], className="mb-4"),
                        ], width=6),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Prediction Accuracy"),
                                dbc.CardBody([
                                    dcc.Graph(id="accuracy-comparison-chart"),
                                ])
                            ], className="mb-4"),
                        ], width=6),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Feature Importance"),
                                dbc.CardBody([
                                    dcc.Graph(id="feature-importance-chart"),
                                ])
                            ], className="mb-4"),
                        ], width=12),
                    ]),
                ]),
            ], id="tabs", active_tab="tab-price"),
        ], width=9),
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Recent Data"),
                dbc.CardBody([
                    html.Div(id="recent-data-table"),
                ]),
            ]),
        ], width=12),
    ], className="mt-4"),

    dbc.Row([
        dbc.Col([
            html.Footer([
                html.P("Stock Market Prediction Dashboard - Built with Dash and Plotly"),
                html.P("Data source: Yahoo Finance"),
            ], className="text-center text-muted mt-4 mb-4"),
        ], width=12),
    ]),
], fluid=True)


@app.callback(
    Output("update-status", "children"),
    Input("update-button", "n_clicks"),
    State("ticker-dropdown", "value"),
    prevent_initial_call=True,
)
def update_data(n_clicks, ticker):
    """
    Update stock data for the selected ticker.
    """
    if n_clicks is None:
        return ""

    try:
        # Collect historical data
        logger.info(f"Collecting historical data for {ticker}")
        # Create a collector specifically for this ticker
        ticker_collector = StockDataCollector(tickers=[ticker], period="2y")
        stock_data = ticker_collector.collect_historical_data()

        if ticker in stock_data and not stock_data[ticker].empty:
            return html.Div(f"Data updated successfully for {ticker}", style={"color": "green"})
        else:
            return html.Div(f"No data found for {ticker}", style={"color": "orange"})
    except Exception as e:
        logger.error(f"Error updating data: {e}")
        return html.Div(f"Error updating data: {str(e)}", style={"color": "red"})


@app.callback(
    Output("price-chart", "figure"),
    [
        Input("ticker-dropdown", "value"),
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
        Input("indicator-dropdown", "value"),
    ],
)
def update_price_chart(ticker, start_date, end_date, indicators):
    """
    Update the price chart with selected ticker and date range.
    """
    try:
        # Download data using yfinance
        stock_data = yf.download(ticker, start=start_date, end=end_date)

        if stock_data.empty:
            return create_empty_figure("No data available for the selected period")

        # Create figure
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.1, row_heights=[0.7, 0.3],
                           subplot_titles=["Price", "Volume"])

        # Add candlestick trace
        fig.add_trace(
            go.Candlestick(
                x=stock_data.index,
                open=stock_data["Open"],
                high=stock_data["High"],
                low=stock_data["Low"],
                close=stock_data["Close"],
                name="Price",
            ),
            row=1, col=1
        )

        # Add technical indicators
        if "sma" in indicators:
            for period in [20, 50, 200]:
                stock_data[f"SMA_{period}"] = stock_data["Close"].rolling(window=period).mean()
                fig.add_trace(
                    go.Scatter(
                        x=stock_data.index,
                        y=stock_data[f"SMA_{period}"],
                        name=f"SMA {period}",
                        line=dict(width=1),
                    ),
                    row=1, col=1
                )

        if "ema" in indicators:
            for period in [12, 26]:
                stock_data[f"EMA_{period}"] = stock_data["Close"].ewm(span=period, adjust=False).mean()
                fig.add_trace(
                    go.Scatter(
                        x=stock_data.index,
                        y=stock_data[f"EMA_{period}"],
                        name=f"EMA {period}",
                        line=dict(width=1, dash="dash"),
                    ),
                    row=1, col=1
                )

        if "bollinger" in indicators:
            period = 20
            std_dev = 2
            stock_data["SMA_20"] = stock_data["Close"].rolling(window=period).mean()
            stock_data["STD_20"] = stock_data["Close"].rolling(window=period).std()
            stock_data["Upper_Band"] = stock_data["SMA_20"] + (stock_data["STD_20"] * std_dev)
            stock_data["Lower_Band"] = stock_data["SMA_20"] - (stock_data["STD_20"] * std_dev)

            fig.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=stock_data["Upper_Band"],
                    name="Upper Band",
                    line=dict(width=1, dash="dot"),
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=stock_data["Lower_Band"],
                    name="Lower Band",
                    line=dict(width=1, dash="dot"),
                    fill='tonexty',
                    fillcolor='rgba(173, 216, 230, 0.2)',
                ),
                row=1, col=1
            )

        # Add volume trace
        fig.add_trace(
            go.Bar(
                x=stock_data.index,
                y=stock_data["Volume"],
                name="Volume",
                marker_color="rgba(100, 100, 255, 0.5)",
            ),
            row=2, col=1
        )

        # Update layout
        fig.update_layout(
            title=f"{ticker} Stock Price",
            yaxis_title="Price",
            yaxis2_title="Volume",
            xaxis_rangeslider_visible=False,
            height=600,
        )

        return fig

    except Exception as e:
        logger.error(f"Error updating price chart: {e}")
        return create_empty_figure(f"Error: {str(e)}")


@app.callback(
    Output("indicator-chart", "figure"),
    [
        Input("ticker-dropdown", "value"),
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
        Input("indicator-dropdown", "value"),
    ],
)
def update_indicator_chart(ticker, start_date, end_date, indicators):
    """
    Update the technical indicator chart with selected ticker and date range.
    """
    try:
        # Download data using yfinance
        stock_data = yf.download(ticker, start=start_date, end=end_date)

        if stock_data.empty:
            return create_empty_figure("No data available for the selected period")

        # Create a subplot for each indicator
        num_indicators = sum([
            "rsi" in indicators,
            "macd" in indicators,
        ])

        if num_indicators == 0:
            return create_empty_figure("No indicators selected for this chart. Please select RSI or MACD.")

        fig = make_subplots(
            rows=num_indicators + 1,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6] + [0.4 / num_indicators] * num_indicators,
            subplot_titles=["Price"] + [ind.upper() for ind in indicators if ind in ["rsi", "macd"]],
        )

        # Add price trace to first subplot
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data["Close"],
                name="Close Price",
                line=dict(color="blue"),
            ),
            row=1, col=1
        )

        current_row = 2

        # Calculate and add RSI
        if "rsi" in indicators:
            delta = stock_data["Close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            fig.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=rsi,
                    name="RSI",
                    line=dict(color="purple"),
                ),
                row=current_row, col=1
            )

            # Add RSI overbought/oversold lines
            fig.add_shape(
                type="line",
                x0=stock_data.index[0],
                y0=70,
                x1=stock_data.index[-1],
                y1=70,
                line=dict(color="red", width=1, dash="dash"),
                row=current_row,
                col=1,
            )

            fig.add_shape(
                type="line",
                x0=stock_data.index[0],
                y0=30,
                x1=stock_data.index[-1],
                y1=30,
                line=dict(color="green", width=1, dash="dash"),
                row=current_row,
                col=1,
            )

            current_row += 1

        # Calculate and add MACD
        if "macd" in indicators:
            ema12 = stock_data["Close"].ewm(span=12, adjust=False).mean()
            ema26 = stock_data["Close"].ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_histogram = macd_line - signal_line

            fig.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=macd_line,
                    name="MACD Line",
                    line=dict(color="blue"),
                ),
                row=current_row, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=signal_line,
                    name="Signal Line",
                    line=dict(color="red"),
                ),
                row=current_row, col=1
            )

            fig.add_trace(
                go.Bar(
                    x=stock_data.index,
                    y=macd_histogram,
                    name="MACD Histogram",
                    marker_color=np.where(macd_histogram > 0, "green", "red"),
                ),
                row=current_row, col=1
            )

            # Add zero line
            fig.add_shape(
                type="line",
                x0=stock_data.index[0],
                y0=0,
                x1=stock_data.index[-1],
                y1=0,
                line=dict(color="gray", width=1, dash="dash"),
                row=current_row,
                col=1,
            )

            current_row += 1

        # Update layout
        fig.update_layout(
            title=f"{ticker} Technical Indicators",
            xaxis_rangeslider_visible=False,
            height=800,
        )

        return fig

    except Exception as e:
        logger.error(f"Error updating indicator chart: {e}")
        return create_empty_figure(f"Error: {str(e)}")


@app.callback(
    [
        Output("prediction-status", "children"),
        Output("prediction-chart", "figure"),
        Output("performance-metrics", "children"),
    ],
    Input("predict-button", "n_clicks"),
    [
        State("ticker-dropdown", "value"),
        State("date-range", "start_date"),
        State("date-range", "end_date"),
        State("prediction-horizon-slider", "value"),
    ],
    prevent_initial_call=True,
)
def run_prediction(n_clicks, ticker, start_date, end_date, prediction_horizon):
    """
    Run prediction for the selected ticker and display results.
    """
    if n_clicks is None:
        return "", create_empty_figure("No prediction results available"), ""

    try:
        # Download data
        stock_data = yf.download(ticker, start=start_date, end=end_date)

        if stock_data.empty:
            return html.Div("No data available for the selected period", style={"color": "red"}), \
                   create_empty_figure("No data available for the selected period"), \
                   ""

        # Add ticker column
        stock_data["ticker"] = ticker

        # Process data
        processed_data = feature_engineering.process_stock_data(
            stock_data,
            add_target=True,
            target_periods=(prediction_horizon,)
        )

        # Prepare data for model
        target_col = f"future_Close_{prediction_horizon}d"
        X = processed_data.drop(["ticker", target_col, f"return_{prediction_horizon}d",
                              f"target_binary_{prediction_horizon}d",
                              f"target_multi_{prediction_horizon}d"], axis=1)
        y = processed_data[target_col]

        # Handle NaN values
        X = X.fillna(method="ffill").fillna(0)

        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

        # Initialize model
        model = LSTMModel(ticker=ticker, sequence_length=10, prediction_horizon=prediction_horizon)

        # Create model directory if it doesn't exist
        os.makedirs(MODEL_DIR, exist_ok=True)

        try:
            # Check if model already exists
            model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl") and ticker in f]

            if model_files:
                # Load the latest model
                latest_model = sorted(model_files)[-1]
                model_path = os.path.join(MODEL_DIR, latest_model)
                model.load_model(model_path)
                logger.info(f"Loaded existing model from {model_path}")
            else:
                # Build and train a new model
                model.build_model()
                model.train(
                    X_train, y_train,
                    X_val=X_test, y_val=y_test
                )
                model.save_model()
        except Exception as e:
            logger.warning(f"Error loading/training model: {e}. Training new model.")
            # Build and train a new model
            model.build_model()
            model.train(
                X_train, y_train,
                X_val=X_test, y_val=y_test
            )
            model.save_model()

        # Make predictions
        predictions = model.predict(X_test)

        # Create prediction chart
        fig = go.Figure()

        # Add actual prices
        fig.add_trace(
            go.Scatter(
                x=y_test.index,
                y=y_test.values,
                name="Actual",
                line=dict(color="blue"),
            )
        )

        # Add predictions
        fig.add_trace(
            go.Scatter(
                x=y_test.index,
                y=predictions,
                name="Predicted",
                line=dict(color="red"),
            )
        )

        # Update layout
        fig.update_layout(
            title=f"{ticker} Price Prediction (Horizon: {prediction_horizon} days)",
            xaxis_title="Date",
            yaxis_title="Price",
            height=600,
        )

        # Calculate performance metrics
        mse = np.mean((y_test.values - predictions)**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test.values - predictions))

        # Create metrics display
        metrics_display = dbc.Card([
            dbc.CardHeader("Model Performance Metrics"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("MSE"),
                            dbc.CardBody(f"{mse:.4f}"),
                        ], className="text-center"),
                    ], width=4),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("RMSE"),
                            dbc.CardBody(f"{rmse:.4f}"),
                        ], className="text-center"),
                    ], width=4),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("MAE"),
                            dbc.CardBody(f"{mae:.4f}"),
                        ], className="text-center"),
                    ], width=4),
                ]),
                html.Hr(),
                html.H5("Prediction Accuracy"),
                dcc.Graph(
                    figure=px.scatter(
                        x=y_test.values,
                        y=predictions,
                        labels={"x": "Actual", "y": "Predicted"},
                        title="Actual vs. Predicted Prices",
                    )
                ),
            ]),
        ])

        return html.Div("Prediction completed successfully", style={"color": "green"}), fig, metrics_display

    except Exception as e:
        logger.error(f"Error running prediction: {e}")
        return html.Div(f"Error running prediction: {str(e)}", style={"color": "red"}), \
               create_empty_figure(f"Error: {str(e)}"), \
               ""


@app.callback(
    Output("recent-data-table", "children"),
    [
        Input("ticker-dropdown", "value"),
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
    ],
)
def update_recent_data_table(ticker, start_date, end_date):
    """
    Update the recent data table with selected ticker and date range.
    """
    try:
        # Download data using yfinance
        stock_data = yf.download(ticker, start=start_date, end=end_date)

        if stock_data.empty:
            return html.Div("No data available for the selected period")

        # Get the most recent data
        recent_data = stock_data.tail(PAGE_SIZE).reset_index()

        # Format the data
        recent_data["Date"] = recent_data["Date"].dt.strftime("%Y-%m-%d")
        recent_data["Open"] = recent_data["Open"].round(2).astype(str)
        recent_data["High"] = recent_data["High"].round(2).astype(str)
        recent_data["Low"] = recent_data["Low"].round(2).astype(str)
        recent_data["Close"] = recent_data["Close"].round(2).astype(str)
        recent_data["Adj Close"] = recent_data["Adj Close"].round(2).astype(str)
        recent_data["Volume"] = recent_data["Volume"].astype(str)

        # Create table
        table = dbc.Table.from_dataframe(
            recent_data,
            striped=True,
            bordered=True,
            hover=True,
            responsive=True,
        )

        return table

    except Exception as e:
        logger.error(f"Error updating recent data table: {e}")
        return html.Div(f"Error: {str(e)}")


def create_empty_figure(message):
    """
    Create an empty figure with a message.
    """
    return {
        "data": [],
        "layout": {
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "annotations": [
                {
                    "text": message,
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 20}
                }
            ]
        }
    }


@app.callback(
    [
        Output("comparison-status", "children"),
        Output("metrics-comparison-table", "children"),
        Output("accuracy-comparison-chart", "figure"),
        Output("feature-importance-chart", "figure")
    ],
    [Input("compare-button", "n_clicks")],
    [State("model-checklist", "value")],
    prevent_initial_call=True
)
def run_model_comparison(n_clicks, selected_models):
    if n_clicks is None:
        return "Select models and click 'Run Comparison'", None, {}, {}

    # Initialize benchmark
    benchmark = ModelBenchmark(output_dir='output/benchmarks')

    # Add selected models
    if "lstm" in selected_models:
        benchmark.add_model(LSTMModel(name="LSTM", epochs=10), "LSTM")
    if "rf" in selected_models:
        benchmark.add_model(SparkRandomForestModel(name="RandomForest"), "Random Forest")
    if "gbt" in selected_models:
        benchmark.add_model(SparkGradientBoostedTreesModel(name="GBT"), "Gradient Boosted Trees")
    if "linear" in selected_models:
        benchmark.add_model(SparkLinearModel(name="Linear"), "Linear Model")

    # Load example data from CSV (for demo purposes)
    try:
        data = pd.read_csv('data/processed/sample_features.csv')
        if data.empty:
            return "Error: No data available", None, {}, {}
    except:
        # Generate sample data for demonstration
        np.random.seed(42)
        n_samples = 1000
        features = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
        X = pd.DataFrame(np.random.randn(n_samples, len(features)), columns=features)
        y = (0.3*X['feature1'] - 0.5*X['feature2'] + 0.7*X['feature3'] + np.random.randn(n_samples)*0.2) > 0
        data = X.copy()
        data['target'] = y

    # Split data for training and testing
    train_data = data.sample(frac=0.8, random_state=42)
    test_data = data.drop(train_data.index)

    # Run benchmark
    try:
        benchmark.train_models(train_data, 'target')
        benchmark.evaluate_models(test_data, 'target')
        benchmark.calculate_metrics()
        results = benchmark.get_results()
        benchmark.close()
    except Exception as e:
        return f"Error during benchmark: {str(e)}", None, {}, {}

    # Create metrics table
    metrics_df = pd.DataFrame(results)
    metrics_table = dash_table.DataTable(
        id='metrics-table',
        columns=[{"name": col, "id": col} for col in metrics_df.columns],
        data=metrics_df.to_dict('records'),
        style_cell={'textAlign': 'left', 'padding': '10px'},
        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
    )

    # Create accuracy comparison chart
    models = list(results.index)
    accuracy_fig = go.Figure()
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        if metric in results.columns:
            accuracy_fig.add_trace(go.Bar(
                x=models,
                y=results[metric],
                name=metric.capitalize()
            ))

    accuracy_fig.update_layout(
        barmode='group',
        title='Model Performance Metrics',
        xaxis_title='Model',
        yaxis_title='Score',
        legend_title='Metric',
        template='plotly_white'
    )

    # Create feature importance chart
    importance_df = pd.DataFrame()
    for model_name, model_obj in benchmark.models.items():
        if hasattr(model_obj, 'feature_importance_') and model_obj.feature_importance_ is not None:
            model_importance = pd.DataFrame({
                'Feature': data.drop('target', axis=1).columns,
                'Importance': model_obj.feature_importance_,
                'Model': model_name
            })
            importance_df = pd.concat([importance_df, model_importance])

    if not importance_df.empty:
        importance_fig = px.bar(
            importance_df,
            x='Feature',
            y='Importance',
            color='Model',
            barmode='group',
            title='Feature Importance by Model'
        )
    else:
        importance_fig = go.Figure()
        importance_fig.update_layout(
            title='Feature Importance Not Available',
            annotations=[dict(
                text='Feature importance data not available for selected models',
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5
            )]
        )

    return "Comparison completed", metrics_table, accuracy_fig, importance_fig


@app.callback(
    [
        Output("realtime-status", "children"),
        Output("realtime-prediction-chart", "figure"),
    ],
    [
        Input("refresh-realtime-button", "n_clicks"),
        Input("ticker-dropdown", "value"),
    ],
    prevent_initial_call=True,
)
def update_realtime_predictions(n_clicks, ticker):
    """
    Update real-time predictions from MongoDB.
    """
    if not mongodb_available:
        return (
            html.Div("MongoDB connection not available. Real-time predictions cannot be displayed.", style={"color": "red"}),
            create_empty_figure("MongoDB connection not available")
        )

    try:
        # Get the latest prediction for the ticker
        latest_prediction = predictions_collection.find_one(
            {"ticker": ticker},
            sort=[("timestamp", -1)]
        )

        if not latest_prediction:
            return (
                html.Div(f"No real-time predictions found for {ticker}", style={"color": "orange"}),
                create_empty_figure(f"No real-time predictions found for {ticker}")
            )

        # Create figure
        fig = go.Figure()

        # Get current date for reference
        current_date = datetime.now().date()

        # Add predictions for each model
        for model_name, predictions in latest_prediction["predictions"].items():
            dates = []
            values = []

            # Sort predictions by date
            for date_str, pred in sorted(predictions.items()):
                dates.append(date_str)
                # Get the predicted value (could be different key based on model)
                if "Predicted_Close" in pred:
                    values.append(pred["Predicted_Close"])
                elif "Predicted_Value" in pred:
                    values.append(pred["Predicted_Value"])
                else:
                    # Try to get the first numeric value
                    for key, value in pred.items():
                        if isinstance(value, (int, float)):
                            values.append(value)
                            break

            # Add trace
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=values,
                    name=f"{model_name.upper()} Prediction",
                    mode="lines+markers"
                )
            )

        # Update layout
        fig.update_layout(
            title=f"Real-Time Predictions for {ticker} (Last updated: {latest_prediction['timestamp']})",
            xaxis_title="Date",
            yaxis_title="Predicted Price",
            legend_title="Model",
            height=500,
        )

        return (
            html.Div(f"Real-time predictions loaded successfully for {ticker}", style={"color": "green"}),
            fig
        )

    except Exception as e:
        logger.error(f"Error updating real-time predictions: {e}")
        return (
            html.Div(f"Error updating real-time predictions: {str(e)}", style={"color": "red"}),
            create_empty_figure(f"Error: {str(e)}")
        )


def main():
    """
    Main function to run the application.
    """
    app.run(
        debug=True,
        dev_tools_hot_reload=False,  # Disable hot reload to reduce server load
        host='127.0.0.1',
        port=8050,
        threaded=True  # Enable threading for better performance
    )


if __name__ == "__main__":
    main()