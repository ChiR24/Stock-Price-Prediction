"""
Base model class for stock market prediction.
"""
import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from abc import ABC, abstractmethod
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

# Import project modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from utils.config import MODEL_DIR

# Set up logger
logger = setup_logger('model_base')

class ModelBase(ABC):
    """
    Abstract base class for all stock market prediction models.
    """
    def __init__(self, name='base_model', model_type='regression'):
        """
        Initialize the base model.

        Args:
            name: Name of the model. Defaults to 'base_model'.
            model_type: Type of model ('regression' or 'classification'). Defaults to 'regression'.
        """
        self.name = name
        self.model_type = model_type
        self.model = None
        self.feature_columns = None
        self.target_column = None
        self.scaler = None

    @abstractmethod
    def build_model(self, **kwargs):
        """
        Build the model architecture.

        Args:
            **kwargs: Additional arguments for model building.

        Returns:
            Model instance.
        """
        pass

    @abstractmethod
    def train(self, X_train, y_train, **kwargs):
        """
        Train the model.

        Args:
            X_train: Training features.
            y_train: Training targets.
            **kwargs: Additional arguments for training.

        Returns:
            Training history or trained model.
        """
        pass

    @abstractmethod
    def predict(self, X, **kwargs):
        """
        Make predictions with the model.

        Args:
            X: Features to predict on.
            **kwargs: Additional arguments for prediction.

        Returns:
            Predictions.
        """
        pass

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model performance.

        Args:
            X_test: Test features.
            y_test: Test targets.

        Returns:
            Dictionary of metrics.
        """
        # Make predictions
        y_pred = self.predict(X_test)

        # Calculate metrics based on model type
        metrics = {}

        if self.model_type == 'regression':
            # Regression metrics
            metrics['mse'] = mean_squared_error(y_test, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_test, y_pred)
            metrics['r2'] = r2_score(y_test, y_pred)

            logger.info(f"Regression metrics for {self.name}:")
            logger.info(f"  MSE: {metrics['mse']:.4f}")
            logger.info(f"  RMSE: {metrics['rmse']:.4f}")
            logger.info(f"  MAE: {metrics['mae']:.4f}")
            logger.info(f"  R²: {metrics['r2']:.4f}")

        elif self.model_type == 'classification':
            # Classification metrics
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['precision'] = precision_score(y_test, y_pred, average='weighted')
            metrics['recall'] = recall_score(y_test, y_pred, average='weighted')
            metrics['f1'] = f1_score(y_test, y_pred, average='weighted')

            logger.info(f"Classification metrics for {self.name}:")
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall: {metrics['recall']:.4f}")
            logger.info(f"  F1 Score: {metrics['f1']:.4f}")

        return metrics

    def calculate_financial_metrics(self, X_test, y_test, initial_investment=10000):
        """
        Calculate financial metrics for model evaluation.

        Args:
            X_test: Test features.
            y_test: Test targets (actual returns).
            initial_investment: Initial investment amount. Defaults to 10000.

        Returns:
            Dictionary of financial metrics.
        """
        # Make predictions
        y_pred = self.predict(X_test)

        # Check if y_test contains returns or prices
        if 'return' in str(self.target_column).lower():
            # If target is returns, use directly
            actual_returns = y_test
            predicted_returns = y_pred
        else:
            # If target is prices, calculate returns
            if isinstance(y_test, pd.Series):
                actual_prices = y_test.values
            else:
                actual_prices = y_test

            actual_returns = np.diff(actual_prices) / actual_prices[:-1]
            predicted_returns = np.diff(y_pred) / y_pred[:-1]

            # Adjust arrays to have the same length
            actual_returns = np.append(0, actual_returns)
            predicted_returns = np.append(0, predicted_returns)

        # Create a strategy where we go long if predicted return is positive
        strategy_returns = np.zeros_like(predicted_returns)
        for i in range(len(predicted_returns)):
            if predicted_returns[i] > 0:
                strategy_returns[i] = actual_returns[i]
            else:
                strategy_returns[i] = 0  # Stay in cash

        # Calculate cumulative returns
        cumulative_actual = np.cumprod(1 + actual_returns) * initial_investment
        cumulative_strategy = np.cumprod(1 + strategy_returns) * initial_investment

        # Calculate financial metrics
        metrics = {}

        # Total return
        metrics['total_return'] = (cumulative_strategy[-1] / initial_investment - 1) * 100

        # Calculate annualized return (assuming 252 trading days in a year)
        n_days = len(cumulative_strategy)
        metrics['annualized_return'] = (
            (cumulative_strategy[-1] / initial_investment) ** (252 / n_days) - 1
        ) * 100

        # Calculate Sharpe Ratio (assuming risk-free rate of 0)
        daily_returns = np.diff(cumulative_strategy) / cumulative_strategy[:-1]
        metrics['sharpe_ratio'] = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)

        # Calculate Maximum Drawdown
        peak = cumulative_strategy[0]
        max_drawdown = 0

        for value in cumulative_strategy:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)

        metrics['max_drawdown'] = max_drawdown * 100

        # Calculate Win Rate
        win_trades = sum(strategy_returns > 0)
        total_trades = sum(predicted_returns != 0)
        metrics['win_rate'] = (win_trades / total_trades) * 100 if total_trades > 0 else 0

        logger.info(f"Financial metrics for {self.name}:")
        logger.info(f"  Total Return: {metrics['total_return']:.2f}%")
        logger.info(f"  Annualized Return: {metrics['annualized_return']:.2f}%")
        logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        logger.info(f"  Maximum Drawdown: {metrics['max_drawdown']:.2f}%")
        logger.info(f"  Win Rate: {metrics['win_rate']:.2f}%")

        return metrics

    def save_model(self, filepath=None, include_timestamp=True):
        """
        Save the model to a file.

        Args:
            filepath: Path to save the model. Defaults to None (use default).
            include_timestamp: Whether to include timestamp in the filename. Defaults to True.

        Returns:
            Path of the saved model.
        """
        if not self.model:
            logger.error("No model to save")
            return None

        try:
            # Create model directory if it doesn't exist
            os.makedirs(MODEL_DIR, exist_ok=True)

            # Generate filename
            if filepath is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if include_timestamp else ""
                filepath = os.path.join(MODEL_DIR, f"{self.name}_{timestamp}.pkl")

            # Save model and metadata
            model_data = {
                'model': self.model,
                'name': self.name,
                'model_type': self.model_type,
                'feature_columns': self.feature_columns,
                'target_column': self.target_column,
                'scaler': self.scaler,
                'timestamp': datetime.now().isoformat()
            }

            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info(f"Model saved to {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return None

    def load_model(self, filepath=None):
        """
        Load a model from a file.

        Args:
            filepath: Path to load the model from. If None, tries to use a default path.

        Returns:
            True if successful, False otherwise.
        """
        if filepath is None:
            # Try to use a default path based on model name
            if hasattr(self, 'model_file'):
                filepath = self.model_file
            else:
                logger.error("No filepath provided and no default model_file attribute found")
                return False

        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.name = model_data['name']
            self.model_type = model_data['model_type']
            self.feature_columns = model_data['feature_columns']
            self.target_column = model_data['target_column']
            self.scaler = model_data['scaler']

            logger.info(f"Model loaded from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def feature_importance(self, X=None):
        """
        Get feature importance if available in the model.

        Args:
            X: Feature data for models that calculate importance dynamically. Defaults to None.

        Returns:
            Dictionary of feature importances or None if not available.
        """
        # Default implementation (override in subclasses if needed)
        try:
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                if self.feature_columns is not None:
                    result = dict(zip(self.feature_columns, importances))
                    # Sort by importance
                    result = {k: v for k, v in sorted(result.items(), key=lambda item: item[1], reverse=True)}
                    return result
                return importances
            elif hasattr(self.model, 'coef_'):
                importances = self.model.coef_
                if self.feature_columns is not None:
                    if importances.ndim > 1:
                        # For multi-class classification
                        importances = np.mean(np.abs(importances), axis=0)
                    result = dict(zip(self.feature_columns, importances))
                    # Sort by absolute importance
                    result = {k: v for k, v in sorted(result.items(), key=lambda item: abs(item[1]), reverse=True)}
                    return result
                return importances
        except Exception as e:
            logger.warning(f"Error getting feature importance: {e}")

        logger.warning("Feature importance not available for this model")
        return None