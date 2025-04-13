#!/usr/bin/env python
"""
Ensemble model for stock market prediction.
This module implements an ensemble model that combines predictions from multiple models.
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pickle

# Import project modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_models.model_base import ModelBase
from utils.logger import setup_logger
from utils.config import MODELS_DIR

# Set up logger
logger = setup_logger('ensemble_model')

class EnsembleModel(ModelBase):
    """
    Ensemble model for stock market prediction.
    This class implements an ensemble model that combines predictions from multiple base models.
    """

    def __init__(self, ticker, models=None, weights=None, method='weighted_average',
                 prediction_horizon=1, is_classification=False):
        """
        Initialize the ensemble model.

        Args:
            ticker (str): Stock ticker symbol.
            models (list): List of trained model objects to include in the ensemble.
            weights (list): List of weights for each model (must sum to 1). If None, equal weights are used.
            method (str): Ensemble method ('average', 'weighted_average', 'stacking').
            prediction_horizon (int): Number of days ahead to predict.
            is_classification (bool): Whether this is a classification or regression task.
        """
        super().__init__(ticker, 'ensemble')

        self.models = models if models is not None else []
        self.method = method
        self.prediction_horizon = prediction_horizon
        self.is_classification = is_classification

        # Define model file paths
        self.model_file = os.path.join(
            MODELS_DIR,
            f"{ticker}_ensemble_{prediction_horizon}d_{'class' if is_classification else 'reg'}.pkl"
        )

        # Set up weights
        if weights is None:
            if self.models:
                # Equal weights if not specified
                self.weights = [1.0 / len(self.models)] * len(self.models)
            else:
                self.weights = []
        else:
            # Validate weights
            if sum(weights) != 1.0:
                logger.warning("Weights do not sum to 1, normalizing...")
                self.weights = [w / sum(weights) for w in weights]
            else:
                self.weights = weights

        # Initialize meta-model for stacking
        self.meta_model = None

        logger.info(f"Ensemble model initialized for {ticker} - "
                    f"Method: {method}, "
                    f"Models: {len(self.models)}, "
                    f"Is Classification: {is_classification}")

    def add_model(self, model, weight=None):
        """
        Add a model to the ensemble.

        Args:
            model (object): Trained model object to add.
            weight (float): Weight for this model. If None, weights will be adjusted to be equal.

        Returns:
            bool: True if model was added successfully, False otherwise.
        """
        if model is None:
            logger.error("Cannot add None model to ensemble")
            return False

        # Add model to list
        self.models.append(model)

        # Adjust weights
        if weight is not None:
            # Add new weight and normalize
            new_weights = self.weights + [weight]
            self.weights = [w / sum(new_weights) for w in new_weights]
        else:
            # Set equal weights
            self.weights = [1.0 / len(self.models)] * len(self.models)

        logger.info(f"Added model to ensemble - Total models: {len(self.models)}")
        return True

    def remove_model(self, index):
        """
        Remove a model from the ensemble.

        Args:
            index (int): Index of the model to remove.

        Returns:
            bool: True if model was removed successfully, False otherwise.
        """
        if index < 0 or index >= len(self.models):
            logger.error(f"Invalid model index: {index}")
            return False

        # Remove model and its weight
        self.models.pop(index)
        self.weights.pop(index)

        # Renormalize weights if any models remain
        if self.models:
            self.weights = [w / sum(self.weights) for w in self.weights]

        logger.info(f"Removed model from ensemble - Total models: {len(self.models)}")
        return True

    def prepare_data(self, data, target_column='Close'):
        """
        Prepare data for ensemble model training and prediction.
        This is mainly used for stacking, as other ensemble methods use pre-trained models.

        Args:
            data (pandas.DataFrame): Input data.
            target_column (str): Target column name.

        Returns:
            tuple: (X_train, y_train, X_val, y_val, X_test, y_test) data split.
        """
        logger.info(f"Preparing data for ensemble model - Target: {target_column}")

        # Make a copy of the data
        df = data.copy()

        # Ensure target column exists
        if target_column not in df.columns:
            logger.error(f"Target column '{target_column}' not found in data")
            return None

        # Get individual predictions for each model
        all_train_preds = []
        all_val_preds = []
        all_test_preds = []

        # Split data for stacking (70%, 15%, 15%)
        train_size = int(len(df) * 0.7)
        val_size = int(len(df) * 0.15)

        train_data = df.iloc[:train_size]
        val_data = df.iloc[train_size:train_size+val_size]
        test_data = df.iloc[train_size+val_size:]

        # Extract target
        y_train = train_data[target_column]
        y_val = val_data[target_column]
        y_test = test_data[target_column]

        logger.info(f"Data split - Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")

        # For stacking, we need to get predictions from each model
        if self.method == 'stacking':
            for model in self.models:
                logger.info(f"Getting predictions from {model.__class__.__name__}")

                # Prepare data for current model
                if hasattr(model, 'prepare_data'):
                    model_data = model.prepare_data(df, target_column)

                    # Train the model if not trained
                    if not hasattr(model, 'model') or model.model is None:
                        model.train(train_data)

                # Get predictions
                train_preds = model.predict(train_data)
                val_preds = model.predict(val_data)
                test_preds = model.predict(test_data)

                # Store predictions
                all_train_preds.append(train_preds)
                all_val_preds.append(val_preds)
                all_test_preds.append(test_preds)

            # Stack predictions
            X_train = np.column_stack(all_train_preds)
            X_val = np.column_stack(all_val_preds)
            X_test = np.column_stack(all_test_preds)

            return X_train, y_train, X_val, y_val, X_test, y_test

        # For other methods, just return the data splits
        return train_data, y_train, val_data, y_val, test_data, y_test

    def build_model(self):
        """
        Build the ensemble model.
        For stacking, this builds a meta-model. For other methods, this is a no-op.

        Returns:
            object: Built model.
        """
        logger.info(f"Building ensemble model with method {self.method}")

        if self.method == 'stacking':
            # For stacking, build a meta-model
            if self.is_classification:
                from sklearn.linear_model import LogisticRegression
                self.meta_model = LogisticRegression(C=1.0, solver='liblinear')
                logger.info("Built logistic regression meta-model for stacking")
            else:
                from sklearn.linear_model import Ridge
                self.meta_model = Ridge(alpha=1.0)
                logger.info("Built ridge regression meta-model for stacking")

            return self.meta_model

        # For other methods, no need to build a specific model
        logger.info(f"Using {self.method} ensemble method")
        return None

    def train(self, train_data, y_train=None, val_data=None, y_val=None):
        """
        Train the ensemble model.
        For stacking, this trains the meta-model. For other methods, this is a no-op.

        Args:
            train_data: Training data (X for stacking, DataFrame for other methods).
            y_train: Target values for training.
            val_data: Validation data (X for stacking, DataFrame for other methods).
            y_val: Target values for validation.

        Returns:
            object: Trained model.
        """
        logger.info(f"Starting ensemble model training for {self.ticker}")

        if self.method == 'stacking':
            # For stacking, train the meta-model
            if self.meta_model is None:
                self.build_model()

            try:
                self.meta_model.fit(train_data, y_train)
                logger.info("Trained meta-model for stacking")

                # Save the model
                self.save()

                return self.meta_model

            except Exception as e:
                logger.error(f"Error training stacking meta-model: {e}")
                return None

        # For other methods, nothing to train
        logger.info(f"No training required for {self.method} ensemble")
        return None

    def predict(self, data):
        """
        Generate predictions from the ensemble model.

        Args:
            data: Data to predict on (X for stacking, DataFrame for other methods).

        Returns:
            numpy.array: Predicted values.
        """
        logger.info(f"Generating predictions with ensemble model using {self.method} method")

        if not self.models:
            logger.error("No models in ensemble")
            return None

        try:
            if self.method == 'stacking':
                # First, get predictions from each base model
                base_predictions = []

                for model in self.models:
                    preds = model.predict(data)
                    base_predictions.append(preds)

                # Stack predictions
                stacked_predictions = np.column_stack(base_predictions)

                # Use meta-model to make final predictions
                final_predictions = self.meta_model.predict(stacked_predictions)

                # For classification, ensure predictions are in the right format
                if self.is_classification:
                    # If meta-model outputs probabilities, convert to class labels
                    if hasattr(self.meta_model, 'predict_proba'):
                        proba = self.meta_model.predict_proba(stacked_predictions)
                        # Use threshold of 0.5
                        final_predictions = (proba[:, 1] >= 0.5).astype(int)

                return final_predictions

            elif self.method in ['average', 'weighted_average']:
                # Get predictions from each model
                all_predictions = []

                for i, model in enumerate(self.models):
                    # Get predictions from current model
                    preds = model.predict(data)

                    # Apply weight
                    if self.method == 'weighted_average':
                        preds = preds * self.weights[i]

                    all_predictions.append(preds)

                # Combine predictions
                if self.method == 'average':
                    combined_predictions = np.mean(all_predictions, axis=0)
                else:  # weighted_average
                    combined_predictions = np.sum(all_predictions, axis=0)

                # For classification, convert to binary predictions
                if self.is_classification:
                    combined_predictions = (combined_predictions >= 0.5).astype(int)

                return combined_predictions

            else:
                logger.error(f"Unsupported ensemble method: {self.method}")
                return None

        except Exception as e:
            logger.error(f"Error making ensemble predictions: {e}")
            return None

    def evaluate(self, test_data, y_test=None, plot_results=True):
        """
        Evaluate the ensemble model.

        Args:
            test_data: Test data (X for stacking, DataFrame for other methods).
            y_test: True target values (only needed for stacking).
            plot_results (bool): Whether to plot the results.

        Returns:
            dict: Evaluation metrics.
        """
        logger.info(f"Evaluating ensemble model for {self.ticker}")

        try:
            # Get predictions
            predictions = self.predict(test_data)

            if y_test is None and not isinstance(test_data, pd.DataFrame):
                logger.error("y_test must be provided for stacking evaluation")
                return None

            # For non-stacking methods, extract y_test from DataFrame
            if self.method != 'stacking' and y_test is None:
                # Find target column (assuming it's named target or is last column)
                if 'target' in test_data.columns:
                    y_test = test_data['target']
                else:
                    # Default to Close for stock data
                    y_test = test_data['Close'] if 'Close' in test_data.columns else test_data.iloc[:, -1]

            # Calculate metrics
            metrics = {}

            if self.is_classification:
                metrics['accuracy'] = accuracy_score(y_test, predictions)
                metrics['precision'] = precision_score(y_test, predictions, zero_division=0)
                metrics['recall'] = recall_score(y_test, predictions, zero_division=0)
                metrics['f1'] = f1_score(y_test, predictions, zero_division=0)

                logger.info(f"Classification metrics - "
                           f"Accuracy: {metrics['accuracy']:.4f}, "
                           f"Precision: {metrics['precision']:.4f}, "
                           f"Recall: {metrics['recall']:.4f}, "
                           f"F1: {metrics['f1']:.4f}")
            else:
                metrics['mse'] = mean_squared_error(y_test, predictions)
                metrics['rmse'] = np.sqrt(metrics['mse'])
                metrics['mae'] = mean_absolute_error(y_test, predictions)
                metrics['r2'] = r2_score(y_test, predictions)

                logger.info(f"Regression metrics - "
                           f"MSE: {metrics['mse']:.4f}, "
                           f"RMSE: {metrics['rmse']:.4f}, "
                           f"MAE: {metrics['mae']:.4f}, "
                           f"R²: {metrics['r2']:.4f}")

                # Calculate financial metrics
                fin_metrics = self.calculate_financial_metrics(y_test, predictions)
                metrics.update(fin_metrics)

            # Plot results if requested
            if plot_results:
                self.plot_predictions(y_test, predictions)

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating ensemble model: {e}")
            return None

    def plot_predictions(self, actual, predictions):
        """
        Plot the predictions against the actual values.

        Args:
            actual: Actual values.
            predictions: Predicted values.
        """
        plt.figure(figsize=(12, 6))

        # Create index for plotting
        index = range(len(actual)) if not hasattr(actual, 'index') else actual.index

        # Plot actual vs. predicted
        plt.plot(index, actual, label='Actual', linewidth=1)
        plt.plot(index, predictions, label='Predicted', linewidth=1, color='red')

        plt.title(f'Ensemble Predictions for {self.ticker} - {self.prediction_horizon}d Ahead')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Create directory for plots if it doesn't exist
        plots_dir = os.path.join(MODELS_DIR, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        # Save the plot
        plot_file = os.path.join(
            plots_dir,
            f"{self.ticker}_ensemble_{self.prediction_horizon}d_{'class' if self.is_classification else 'reg'}.png"
        )
        plt.savefig(plot_file)
        plt.close()

        logger.info(f"Predictions plot saved to {plot_file}")

    def save(self):
        """
        Save the ensemble model to file.

        Returns:
            bool: True if model saved successfully, False otherwise.
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_file), exist_ok=True)

            # Create state dictionary to save
            model_state = {
                'method': self.method,
                'weights': self.weights,
                'is_classification': self.is_classification,
                'prediction_horizon': self.prediction_horizon
            }

            # Add meta-model for stacking
            if self.method == 'stacking' and self.meta_model is not None:
                model_state['meta_model'] = self.meta_model

            # Save model state
            with open(self.model_file, 'wb') as f:
                pickle.dump(model_state, f)

            logger.info(f"Saved ensemble model to {self.model_file}")
            return True

        except Exception as e:
            logger.error(f"Error saving ensemble model: {e}")
            return False

    def load(self):
        """
        Load an ensemble model from file.
        Note: Base models need to be added separately.

        Returns:
            bool: True if model loaded successfully, False otherwise.
        """
        if os.path.exists(self.model_file):
            try:
                with open(self.model_file, 'rb') as f:
                    model_state = pickle.load(f)

                # Load model attributes
                self.method = model_state['method']
                self.weights = model_state['weights']
                self.is_classification = model_state['is_classification']
                self.prediction_horizon = model_state['prediction_horizon']

                # Load meta-model for stacking
                if self.method == 'stacking' and 'meta_model' in model_state:
                    self.meta_model = model_state['meta_model']

                logger.info(f"Loaded ensemble model from {self.model_file}")
                return True

            except Exception as e:
                logger.error(f"Error loading ensemble model: {e}")
                return False
        else:
            logger.error(f"Model file not found: {self.model_file}")
            return False

    def predict_future(self, data, steps=30):
        """
        Make future predictions using the ensemble model.

        Args:
            data (pandas.DataFrame): Latest data to use for prediction.
            steps (int): Number of steps to predict into the future.

        Returns:
            pandas.DataFrame: DataFrame with future predictions.
        """
        logger.info(f"Making future predictions for {self.ticker} - {steps} steps ahead")

        if not self.models:
            logger.error("No models in ensemble")
            return None

        try:
            # For each model, get future predictions
            futures = []

            for i, model in enumerate(self.models):
                # Get future predictions from current model
                model_future = model.predict_future(data, steps=steps)

                if model_future is not None:
                    # Extract predictions column (assuming it's the last column)
                    pred_col = model_future.columns[-1]
                    preds = model_future[pred_col].values

                    # Apply weight for weighted average
                    if self.method == 'weighted_average':
                        preds = preds * self.weights[i]

                    futures.append(preds)

            # Combine predictions
            if self.method == 'average':
                combined_future = np.mean(futures, axis=0)
            elif self.method == 'weighted_average':
                combined_future = np.sum(futures, axis=0)
            elif self.method == 'stacking':
                # For stacking, we need to transform the predictions first
                stacked_futures = np.column_stack(futures)
                combined_future = self.meta_model.predict(stacked_futures)
            else:
                logger.error(f"Unsupported ensemble method: {self.method}")
                return None

            # Create a DataFrame with the predictions
            last_date = data.index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq='B')

            predictions_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted_Value': combined_future
            })

            predictions_df.set_index('Date', inplace=True)

            logger.info(f"Future predictions completed for {self.ticker}")

            return predictions_df

        except Exception as e:
            logger.error(f"Error making future predictions: {e}")
            return None


if __name__ == "__main__":
    # Simple test of the ensemble model
    import numpy as np
    import pandas as pd
    from ml_models.arima_model import ARIMAModel

    # Load sample data or generate synthetic data for testing
    dates = pd.date_range(start='2020-01-01', periods=500, freq='B')
    data = pd.DataFrame({
        'Close': np.sin(np.linspace(0, 10, 500)) * 100 + 200 + np.random.normal(0, 5, 500)
    }, index=dates)

    # Create individual models
    arima1 = ARIMAModel(ticker='TEST', order=(1, 1, 1), prediction_horizon=5)
    arima2 = ARIMAModel(ticker='TEST', order=(2, 1, 2), prediction_horizon=5)

    # Prepare data and train models
    train_data1, test_data1 = arima1.prepare_data(data, 'Close')
    train_data2, test_data2 = arima2.prepare_data(data, 'Close')

    arima1.train(train_data1)
    arima2.train(train_data2)

    # Create ensemble model
    ensemble = EnsembleModel(
        ticker='TEST',
        models=[arima1, arima2],
        weights=[0.6, 0.4],
        method='weighted_average',
        prediction_horizon=5
    )

    # Evaluate ensemble
    metrics = ensemble.evaluate(data.iloc[-100:], plot_results=True)

    # Make future predictions
    future_preds = ensemble.predict_future(data, steps=20)

    if future_preds is not None:
        print("Future predictions:")
        print(future_preds.head())