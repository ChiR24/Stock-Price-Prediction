#!/usr/bin/env python
"""
ARIMA model for stock market prediction.
This module implements an AutoRegressive Integrated Moving Average model for time series forecasting.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import pickle

# Import project modules
from ml_models.model_base import ModelBase
from utils.logger import setup_logger
from config.settings import MODELS_DIR

# Set up logger
logger = setup_logger('arima_model')

class ARIMAModel(ModelBase):
    """
    ARIMA model for stock market prediction.
    This class implements an AutoRegressive Integrated Moving Average model for time series forecasting.
    """
    
    def __init__(self, ticker, order=(5, 1, 0), prediction_horizon=1, seasonal_order=None):
        """
        Initialize the ARIMA model.
        
        Args:
            ticker (str): Stock ticker symbol.
            order (tuple): ARIMA order (p, d, q) parameters.
            prediction_horizon (int): Number of days ahead to predict.
            seasonal_order (tuple): Seasonal order parameters (P, D, Q, S) for SARIMA models.
        """
        super().__init__(ticker, 'arima')
        
        self.order = order
        self.seasonal_order = seasonal_order
        self.prediction_horizon = prediction_horizon
        
        # Define model file paths
        self.model_file = os.path.join(
            MODELS_DIR, 
            f"{ticker}_arima_{prediction_horizon}d.pkl"
        )
        
        # Initialize model
        self.model = None
        self.model_fit = None
        
        logger.info(f"ARIMA model initialized for {ticker} - "
                    f"Order: {order}, "
                    f"Seasonal Order: {seasonal_order}, "
                    f"Prediction Horizon: {prediction_horizon}")
    
    def prepare_data(self, data, target_column='Close'):
        """
        Prepare data for ARIMA model training and prediction.
        
        Args:
            data (pandas.DataFrame): Input data.
            target_column (str): Target column name.
            
        Returns:
            tuple: (train_data, test_data) split into train and test sets.
        """
        logger.info(f"Preparing data for ARIMA model - Target: {target_column}")
        
        # Make a copy of the data
        df = data.copy()
        
        # Ensure target column exists
        if target_column not in df.columns:
            logger.error(f"Target column '{target_column}' not found in data")
            return None
        
        # Extract target series
        target_series = df[target_column].copy()
        
        # Split data into train and test sets (80%, 20%)
        train_size = int(len(target_series) * 0.8)
        
        train_data = target_series.iloc[:train_size]
        test_data = target_series.iloc[train_size:]
        
        logger.info(f"Data split - Train: {len(train_data)}, Test: {len(test_data)}")
        
        # Check for stationarity
        self._check_stationarity(train_data)
        
        return train_data, test_data
    
    def _check_stationarity(self, series):
        """
        Check if the time series is stationary using Augmented Dickey-Fuller test.
        
        Args:
            series (pandas.Series): Time series to check.
        
        Returns:
            bool: True if stationary, False otherwise.
        """
        result = adfuller(series.dropna())
        
        logger.info(f"ADF Statistic: {result[0]:.6f}")
        logger.info(f"p-value: {result[1]:.6f}")
        
        for key, value in result[4].items():
            logger.info(f"Critical Value ({key}): {value:.6f}")
        
        if result[1] <= 0.05:
            logger.info("Series is stationary (reject H0)")
            return True
        else:
            logger.warning("Series is not stationary (fail to reject H0)")
            return False
    
    def build_model(self):
        """
        Build the ARIMA model.
        
        Returns:
            statsmodels.tsa.arima.model.ARIMA: Built ARIMA model.
        """
        logger.info(f"Building ARIMA model with order {self.order}")
        
        if self.seasonal_order:
            # If seasonal parameters are provided, use SARIMA
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            self.model = SARIMAX(
                np.zeros(10),  # Dummy data, will be replaced in fit
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            logger.info(f"SARIMA model built with seasonal order {self.seasonal_order}")
        else:
            # Use regular ARIMA
            self.model = ARIMA(
                np.zeros(10),  # Dummy data, will be replaced in fit
                order=self.order
            )
            logger.info("ARIMA model built")
        
        return self.model
    
    def train(self, train_data):
        """
        Train the ARIMA model.
        
        Args:
            train_data (pandas.Series): Training data.
            
        Returns:
            statsmodels.tsa.arima.model.ARIMAResults: Fitted model.
        """
        logger.info(f"Starting ARIMA model training for {self.ticker}")
        
        # Build model if not already built
        if self.model is None:
            self.build_model()
        
        try:
            # Fit the model
            if self.seasonal_order:
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                model = SARIMAX(
                    train_data,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            else:
                model = ARIMA(train_data, order=self.order)
            
            self.model_fit = model.fit()
            
            # Log model summary
            logger.info(f"ARIMA model trained successfully:")
            logger.info(f"AIC: {self.model_fit.aic:.4f}")
            logger.info(f"BIC: {self.model_fit.bic:.4f}")
            
            # Save the model
            self.save()
            
            return self.model_fit
        
        except Exception as e:
            logger.error(f"Error training ARIMA model: {e}")
            return None
    
    def predict(self, n_steps, exog=None):
        """
        Make predictions using the trained model.
        
        Args:
            n_steps (int): Number of steps to forecast.
            exog (array): Exogenous variables for prediction (if used in model).
            
        Returns:
            pandas.Series: Predicted values.
        """
        if self.model_fit is None:
            logger.error("Model not trained. Cannot make predictions.")
            return None
        
        try:
            # Get forecast
            forecast = self.model_fit.forecast(steps=n_steps, exog=exog)
            
            logger.info(f"Made {n_steps}-step forecast")
            
            return forecast
        
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return None
    
    def evaluate(self, test_data, dynamic=False, plot_results=True):
        """
        Evaluate the trained model.
        
        Args:
            test_data (pandas.Series): Test data.
            dynamic (bool): Whether to use dynamic forecasting.
            plot_results (bool): Whether to plot the results.
            
        Returns:
            dict: Evaluation metrics.
        """
        if self.model_fit is None:
            logger.error("Model not trained. Cannot evaluate.")
            return None
        
        try:
            # Get predictions
            predictions = self.model_fit.get_forecast(steps=len(test_data))
            predicted_mean = predictions.predicted_mean
            
            # Calculate metrics
            metrics = {}
            
            metrics['mse'] = mean_squared_error(test_data, predicted_mean)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(test_data, predicted_mean)
            
            # R² can be negative for predictions, indicating poor fit
            metrics['r2'] = r2_score(test_data, predicted_mean)
            
            logger.info(f"Evaluation metrics - MSE: {metrics['mse']:.4f}, "
                        f"RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, "
                        f"R²: {metrics['r2']:.4f}")
            
            # Calculate financial metrics
            fin_metrics = self.calculate_financial_metrics(test_data.values, predicted_mean.values)
            metrics.update(fin_metrics)
            
            # Plot results if requested
            if plot_results:
                self.plot_predictions(test_data, predicted_mean)
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return None
    
    def plot_predictions(self, actual, predictions):
        """
        Plot the predictions against the actual values.
        
        Args:
            actual (pandas.Series): Actual values.
            predictions (pandas.Series): Predicted values.
        """
        plt.figure(figsize=(12, 6))
        
        # Plot actual vs. predicted
        plt.plot(actual.index, actual, label='Actual', linewidth=1)
        plt.plot(predictions.index, predictions, label='Predicted', linewidth=1, color='red')
        
        # Add confidence intervals if available
        if hasattr(predictions, 'conf_int'):
            conf_int = predictions.conf_int()
            plt.fill_between(
                conf_int.index, 
                conf_int.iloc[:, 0], 
                conf_int.iloc[:, 1], 
                color='red', 
                alpha=0.1
            )
        
        plt.title(f'ARIMA Predictions for {self.ticker} - {self.prediction_horizon}d Ahead')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Create directory for plots if it doesn't exist
        plots_dir = os.path.join(MODELS_DIR, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Save the plot
        plot_file = os.path.join(plots_dir, f"{self.ticker}_arima_{self.prediction_horizon}d.png")
        plt.savefig(plot_file)
        plt.close()
        
        logger.info(f"Predictions plot saved to {plot_file}")
    
    def save(self):
        """
        Save the trained model to file.
        
        Returns:
            bool: True if model saved successfully, False otherwise.
        """
        if self.model_fit is not None:
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(self.model_file), exist_ok=True)
                
                # Save the model
                with open(self.model_file, 'wb') as f:
                    pickle.dump(self.model_fit, f)
                
                logger.info(f"Saved ARIMA model to {self.model_file}")
                return True
            except Exception as e:
                logger.error(f"Error saving ARIMA model: {e}")
                return False
        else:
            logger.error("No trained model to save")
            return False
    
    def load(self):
        """
        Load a trained model from file.
        
        Returns:
            bool: True if model loaded successfully, False otherwise.
        """
        if os.path.exists(self.model_file):
            try:
                with open(self.model_file, 'rb') as f:
                    self.model_fit = pickle.load(f)
                
                logger.info(f"Loaded ARIMA model from {self.model_file}")
                return True
            except Exception as e:
                logger.error(f"Error loading ARIMA model: {e}")
                return False
        else:
            logger.error(f"Model file not found: {self.model_file}")
            return False
    
    def predict_future(self, data, steps=30):
        """
        Make future predictions using the trained model.
        
        Args:
            data (pandas.DataFrame): Latest data to use for prediction.
            steps (int): Number of steps to predict into the future.
            
        Returns:
            pandas.DataFrame: DataFrame with future predictions.
        """
        logger.info(f"Making future predictions for {self.ticker} - {steps} steps ahead")
        
        if self.model_fit is None:
            logger.error("Model not trained or loaded")
            return None
        
        try:
            # Make forecast
            forecast = self.model_fit.forecast(steps=steps)
            
            # Create a DataFrame with the predictions
            last_date = data.index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq='B')
            
            predictions_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted_Value': forecast
            })
            
            predictions_df.set_index('Date', inplace=True)
            
            logger.info(f"Future predictions completed for {self.ticker}")
            
            return predictions_df
        
        except Exception as e:
            logger.error(f"Error making future predictions: {e}")
            return None
    
    def grid_search(self, train_data, p_values=range(0, 6), d_values=range(0, 3), q_values=range(0, 6)):
        """
        Perform grid search to find the best ARIMA parameters.
        
        Args:
            train_data (pandas.Series): Training data.
            p_values (range): Range of p values to try.
            d_values (range): Range of d values to try.
            q_values (range): Range of q values to try.
            
        Returns:
            tuple: Best parameters (p, d, q) and corresponding AIC value.
        """
        logger.info(f"Starting ARIMA grid search for {self.ticker}")
        
        best_aic = float('inf')
        best_params = None
        
        # Store results for all models
        results = []
        
        try:
            for p in p_values:
                for d in d_values:
                    for q in q_values:
                        # Skip if p=0 and q=0 (invalid model)
                        if p == 0 and q == 0:
                            continue
                        
                        try:
                            model = ARIMA(train_data, order=(p, d, q))
                            model_fit = model.fit()
                            
                            aic = model_fit.aic
                            bic = model_fit.bic
                            
                            # Log result
                            logger.info(f"ARIMA({p},{d},{q}) - AIC: {aic:.4f}, BIC: {bic:.4f}")
                            
                            # Store result
                            results.append({
                                'p': p,
                                'd': d,
                                'q': q,
                                'aic': aic,
                                'bic': bic
                            })
                            
                            # Update best model
                            if aic < best_aic:
                                best_aic = aic
                                best_params = (p, d, q)
                                
                        except Exception as e:
                            logger.warning(f"Error fitting ARIMA({p},{d},{q}): {e}")
                            continue
            
            # Sort results by AIC
            results_df = pd.DataFrame(results).sort_values('aic')
            
            # Save results
            results_dir = os.path.join(MODELS_DIR, 'grid_search')
            os.makedirs(results_dir, exist_ok=True)
            results_df.to_csv(
                os.path.join(results_dir, f"{self.ticker}_arima_grid_search.csv"),
                index=False
            )
            
            logger.info(f"Grid search completed - Best parameters: {best_params}, AIC: {best_aic:.4f}")
            
            # Set best parameters
            self.order = best_params
            
            return best_params, best_aic
        
        except Exception as e:
            logger.error(f"Error during grid search: {e}")
            return None, None


if __name__ == "__main__":
    # Simple test of the ARIMA model
    import pandas as pd
    
    # Load sample data or generate synthetic data for testing
    dates = pd.date_range(start='2020-01-01', periods=500, freq='B')
    data = pd.DataFrame({
        'Close': np.sin(np.linspace(0, 10, 500)) * 100 + 200 + np.random.normal(0, 5, 500)
    }, index=dates)
    
    # Initialize model
    model = ARIMAModel(
        ticker='TEST',
        order=(2, 1, 1),
        prediction_horizon=5
    )
    
    # Prepare data
    train_data, test_data = model.prepare_data(data, 'Close')
    
    if train_data is not None and test_data is not None:
        # Optional: Find best parameters
        # best_params, best_aic = model.grid_search(train_data, 
        #                                          p_values=range(0, 3), 
        #                                          d_values=range(0, 2), 
        #                                          q_values=range(0, 3))
        
        # Train model
        model_fit = model.train(train_data)
        
        if model_fit is not None:
            # Evaluate model
            metrics = model.evaluate(test_data, plot_results=True)
            
            # Make future predictions
            future_preds = model.predict_future(data, steps=30)
            
            if future_preds is not None:
                print("Future predictions:")
                print(future_preds.head()) 