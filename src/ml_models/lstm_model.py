#!/usr/bin/env python
"""
LSTM model for stock market prediction.
This module implements a Long Short-Term Memory (LSTM) neural network for stock price forecasting.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import pickle

# Import project modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_models.model_base import ModelBase
from utils.logger import setup_logger
from utils.config import MODELS_DIR

# Set up logger
logger = setup_logger('lstm_model')

class LSTMModel(ModelBase):
    """
    LSTM model for stock market prediction.
    This class implements a Long Short-Term Memory (LSTM) neural network for time series forecasting.
    """

    def __init__(self, ticker=None, sequence_length=60, units=50, layers=1, dropout=0.2,
                 prediction_horizon=1, batch_size=32, epochs=100, learning_rate=0.001,
                 features=None, name=None):
        """
        Initialize the LSTM model.

        Args:
            ticker (str): Stock ticker symbol.
            sequence_length (int): Number of previous time steps to use as input features.
            units (int): Number of LSTM units in each layer.
            layers (int): Number of LSTM layers.
            dropout (float): Dropout rate to prevent overfitting.
            prediction_horizon (int): Number of days ahead to predict.
            batch_size (int): Batch size for training.
            epochs (int): Maximum number of epochs for training.
            learning_rate (float): Learning rate for optimizer.
            features (list): List of feature column names to use. If None, uses only 'Close'.
            name (str): Name of the model. If None, uses ticker.
        """
        model_name = name if name is not None else ticker
        super().__init__(model_name, 'lstm')
        self.ticker = ticker

        self.sequence_length = sequence_length
        self.units = units
        self.layers = layers
        self.dropout = dropout
        self.prediction_horizon = prediction_horizon
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.features = features if features is not None else ['Close']

        # Initialize model
        self.model = None
        self.scaler = None
        self.feature_scalers = {}

        # Define model file paths
        model_name = f"{ticker}_lstm_{prediction_horizon}d" if ticker else f"lstm_{prediction_horizon}d"
        self.model_file = os.path.join(MODELS_DIR, f"{model_name}.pkl")
        self.scaler_file = os.path.join(MODELS_DIR, f"{model_name}_scaler.pkl")

        logger.info(f"LSTM model initialized for {ticker} with "
                   f"sequence_length={sequence_length}, "
                   f"units={units}, "
                   f"layers={layers}, "
                   f"prediction_horizon={prediction_horizon}")

    def prepare_data(self, data, target_column='Close'):
        """
        Prepare data for LSTM model training and testing.

        Args:
            data (pandas.DataFrame): Input data.
            target_column (str): Target column name.

        Returns:
            tuple: (X_train, y_train, X_val, y_val, X_test, y_test, scaler)
        """
        logger.info(f"Preparing data for LSTM model - Target: {target_column}")

        # Make a copy of the data
        df = data.copy()

        # Ensure target column exists
        if target_column not in df.columns:
            logger.error(f"Target column '{target_column}' not found in data")
            return None

        # Check if all feature columns exist
        for feature in self.features:
            if feature not in df.columns:
                logger.error(f"Feature column '{feature}' not found in data")
                return None

        # Initialize scalers
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        # Scale each feature separately
        scaled_features = {}
        for feature in self.features:
            if feature not in self.feature_scalers:
                self.feature_scalers[feature] = MinMaxScaler(feature_range=(0, 1))

            # Reshape data for scaling
            values = df[feature].values.reshape(-1, 1)
            scaled_values = self.feature_scalers[feature].fit_transform(values)
            scaled_features[feature] = scaled_values.flatten()

        # Create a new DataFrame with scaled features
        scaled_df = pd.DataFrame(scaled_features, index=df.index)

        # Prepare sequences
        X, y = self._create_sequences(scaled_df, target_column)

        # Split the data into training, validation, and testing sets (70%, 15%, 15%)
        train_size = int(len(X) * 0.7)
        val_size = int(len(X) * 0.15)

        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
        X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

        logger.info(f"Data split - Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

        # Save original data for later use
        self.original_data = df
        self.train_size = train_size
        self.val_size = val_size

        return X_train, y_train, X_val, y_val, X_test, y_test

    def _create_sequences(self, data, target_column):
        """
        Create sequences for LSTM input.

        Args:
            data (pandas.DataFrame): Scaled data.
            target_column (str): Target column name.

        Returns:
            tuple: (X, y) where X is the input sequences and y is the target values.
        """
        X, y = [], []

        # Total sequence length needs to account for prediction horizon
        total_seq_len = self.sequence_length + self.prediction_horizon

        # Create sequences
        for i in range(len(data) - total_seq_len + 1):
            # Extract sequence
            seq = data.iloc[i:i+self.sequence_length]

            # Get the target value (shifted by prediction_horizon)
            target_idx = i + self.sequence_length + self.prediction_horizon - 1
            target = data.iloc[target_idx][target_column]

            X.append(seq.values)
            y.append(target)

        return np.array(X), np.array(y)

    def build_model(self):
        """
        Build the LSTM model architecture.

        Returns:
            tensorflow.keras.models.Sequential: Built model.
        """
        logger.info(f"Building LSTM model with {self.layers} layers and {self.units} units")

        # Define input shape
        input_shape = (self.sequence_length, len(self.features))

        # Create model
        model = Sequential()

        # Add LSTM layers
        if self.layers == 1:
            model.add(LSTM(units=self.units, input_shape=input_shape))
            model.add(Dropout(self.dropout))
        else:
            # First layer
            model.add(LSTM(units=self.units, input_shape=input_shape, return_sequences=True))
            model.add(Dropout(self.dropout))

            # Middle layers
            for _ in range(self.layers - 2):
                model.add(LSTM(units=self.units, return_sequences=True))
                model.add(Dropout(self.dropout))

            # Last layer
            model.add(LSTM(units=self.units))
            model.add(Dropout(self.dropout))

        # Output layer
        model.add(Dense(units=1))

        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        # Print model summary
        model.summary(print_fn=logger.info)

        self.model = model
        return model

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the LSTM model.

        Args:
            X_train: Training data.
            y_train: Training targets.
            X_val: Validation data.
            y_val: Validation targets.

        Returns:
            tensorflow.keras.models.Sequential: Trained model.
        """
        logger.info(f"Training LSTM model for {self.ticker}")

        # Build model if not already built
        if self.model is None:
            self.build_model()

        # Create callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(filepath=self.model_file, save_best_only=True, monitor='val_loss')
        ]

        try:
            # Train model
            history = self.model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )

            # Plot training history
            self._plot_training_history(history)

            # Save scalers
            self._save_scalers()

            logger.info(f"LSTM model training completed - "
                       f"Final loss: {history.history['loss'][-1]:.6f}, "
                       f"Final val_loss: {history.history['val_loss'][-1]:.6f}")

            return self.model

        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            return None

    def _plot_training_history(self, history):
        """
        Plot training history.

        Args:
            history: Training history from model.fit().
        """
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'LSTM Training History - {self.ticker}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Create directory for plots if it doesn't exist
        plots_dir = os.path.join(MODELS_DIR, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        # Save the plot
        plt.savefig(os.path.join(plots_dir, f"{self.ticker}_lstm_training_history.png"))
        plt.close()

    def _save_scalers(self):
        """
        Save the feature scalers to file.
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.scaler_file), exist_ok=True)

            # Save scalers
            with open(self.scaler_file, 'wb') as f:
                pickle.dump({'feature_scalers': self.feature_scalers}, f)

            logger.info(f"Saved scalers to {self.scaler_file}")

        except Exception as e:
            logger.error(f"Error saving scalers: {e}")

    def _load_scalers(self):
        """
        Load the feature scalers from file.

        Returns:
            bool: True if scalers loaded successfully, False otherwise.
        """
        if os.path.exists(self.scaler_file):
            try:
                with open(self.scaler_file, 'rb') as f:
                    scalers_data = pickle.load(f)

                self.feature_scalers = scalers_data['feature_scalers']

                logger.info(f"Loaded scalers from {self.scaler_file}")
                return True

            except Exception as e:
                logger.error(f"Error loading scalers: {e}")
                return False
        else:
            logger.error(f"Scaler file not found: {self.scaler_file}")
            return False

    def predict(self, data):
        """
        Generate predictions from the LSTM model.

        Args:
            data (pandas.DataFrame): Data to predict on.

        Returns:
            numpy.array: Predicted values.
        """
        logger.info(f"Generating predictions with LSTM model for {self.ticker}")

        if self.model is None:
            logger.error("Model not trained yet")
            return None

        try:
            # Make a copy of the data
            df = data.copy()

            # Check if features exist
            for feature in self.features:
                if feature not in df.columns:
                    logger.error(f"Feature column '{feature}' not found in data")
                    return None

            # Scale features
            scaled_features = {}
            for feature in self.features:
                values = df[feature].values.reshape(-1, 1)

                # Check if we have a scaler for this feature
                if feature in self.feature_scalers:
                    scaled_values = self.feature_scalers[feature].transform(values)
                else:
                    # Create a new scaler if needed
                    self.feature_scalers[feature] = MinMaxScaler(feature_range=(0, 1))
                    scaled_values = self.feature_scalers[feature].fit_transform(values)

                scaled_features[feature] = scaled_values.flatten()

            # Create a new DataFrame with scaled features
            scaled_df = pd.DataFrame(scaled_features, index=df.index)

            # Create sequences for prediction
            sequences = []
            for i in range(len(scaled_df) - self.sequence_length + 1):
                seq = scaled_df.iloc[i:i+self.sequence_length]
                sequences.append(seq.values)

            # Convert to numpy array
            sequences = np.array(sequences)

            # Make predictions
            predictions = self.model.predict(sequences)

            # Reshape predictions
            predictions = predictions.reshape(-1, 1)

            # Inverse transform to get original scale
            target_feature = self.features[0]  # Default to first feature
            predictions = self.feature_scalers[target_feature].inverse_transform(predictions).flatten()

            # Shift predictions to align with original data
            all_predictions = np.full(len(df), np.nan)
            all_predictions[self.sequence_length+self.prediction_horizon-1:self.sequence_length+self.prediction_horizon-1+len(predictions)] = predictions

            return all_predictions

        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return None

    def evaluate(self, X_test, y_test, test_data=None, plot_results=True):
        """
        Evaluate the LSTM model.

        Args:
            X_test: Test data sequences.
            y_test: Test target values.
            test_data (pandas.DataFrame): Original test data (for plotting).
            plot_results (bool): Whether to plot the results.

        Returns:
            dict: Evaluation metrics.
        """
        logger.info(f"Evaluating LSTM model for {self.ticker}")

        if self.model is None:
            logger.error("Model not trained yet")
            return None

        try:
            # Make predictions
            predictions = self.model.predict(X_test).flatten()

            # Inverse transform predictions and actual values
            target_feature = self.features[0]  # Default to first feature

            predictions = self.feature_scalers[target_feature].inverse_transform(
                predictions.reshape(-1, 1)).flatten()

            y_test_inv = self.feature_scalers[target_feature].inverse_transform(
                y_test.reshape(-1, 1)).flatten()

            # Calculate metrics
            metrics = {}
            metrics['mse'] = mean_squared_error(y_test_inv, predictions)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_test_inv, predictions)
            metrics['r2'] = r2_score(y_test_inv, predictions)

            logger.info(f"Evaluation metrics - "
                       f"MSE: {metrics['mse']:.4f}, "
                       f"RMSE: {metrics['rmse']:.4f}, "
                       f"MAE: {metrics['mae']:.4f}, "
                       f"R²: {metrics['r2']:.4f}")

            # Calculate financial metrics
            if test_data is not None:
                # Extract the original values from test data
                original_values = test_data.iloc[self.train_size+self.val_size:]['Close'].values

                # We need to align the predictions with the original data
                aligned_preds = np.full_like(original_values, np.nan)
                aligned_preds[:len(predictions)] = predictions

                fin_metrics = self.calculate_financial_metrics(original_values, aligned_preds)
                metrics.update(fin_metrics)

            # Plot results if requested
            if plot_results and test_data is not None:
                self.plot_predictions(test_data, predictions, self.train_size + self.val_size)

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return None

    def plot_predictions(self, data, predictions, offset=0):
        """
        Plot the predictions against the actual values.

        Args:
            data (pandas.DataFrame): Original data.
            predictions (numpy.array): Predicted values.
            offset (int): Offset to align predictions with data.
        """
        plt.figure(figsize=(12, 6))

        # Create index for plotting
        index = data.index[offset:offset+len(predictions)]

        # Plot actual vs. predicted
        plt.plot(data.index, data['Close'], label='Actual', linewidth=1)
        plt.plot(index, predictions, label='Predicted', linewidth=1, color='red')

        plt.title(f'LSTM Predictions for {self.ticker} - {self.prediction_horizon}d Ahead')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Create directory for plots if it doesn't exist
        plots_dir = os.path.join(MODELS_DIR, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        # Save the plot
        plt.savefig(os.path.join(plots_dir, f"{self.ticker}_lstm_{self.prediction_horizon}d_predictions.png"))
        plt.close()

        logger.info(f"Predictions plot saved")

    def save_model(self, filepath=None):
        """
        Save the LSTM model to file.

        Args:
            filepath: Path to save the model. Defaults to None (use default).

        Returns:
            Path of the saved model or None if failed.
        """
        if self.model is None:
            logger.error("No model to save")
            return None

        try:
            # Set filepath if not provided
            if filepath is None:
                filepath = self.model_file

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Save model using the parent class method
            self.feature_columns = self.features
            self.target_column = 'Close'
            saved_path = super().save_model(filepath)

            # Save scalers
            self._save_scalers()

            logger.info(f"Saved model to {saved_path}")
            return saved_path

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return None

    def load_model(self, filepath=None):
        """
        Load a trained LSTM model from file.

        Args:
            filepath: Path to load the model from. If None, tries to use the default path.

        Returns:
            bool: True if model loaded successfully, False otherwise.
        """
        # Set filepath if not provided
        if filepath is None:
            filepath = self.model_file

        if os.path.exists(filepath):
            try:
                # Load model using parent class method
                success = super().load_model(filepath)

                if not success:
                    return False

                # Load scalers
                scaler_success = self._load_scalers()

                if not scaler_success:
                    logger.warning("Could not load scalers, model may not work correctly")

                logger.info(f"Loaded model from {filepath}")
                return True

            except Exception as e:
                logger.error(f"Error loading model: {e}")
                return False
        else:
            logger.error(f"Model file not found: {filepath}")
            return False

    def predict_future(self, data, steps=30):
        """
        Make future predictions using the LSTM model.

        Args:
            data (pandas.DataFrame): Latest data to use for prediction.
            steps (int): Number of steps to predict into the future.

        Returns:
            pandas.DataFrame: DataFrame with future predictions.
        """
        logger.info(f"Making future predictions for {self.ticker} - {steps} steps ahead")

        if self.model is None:
            logger.error("Model not trained yet")
            return None

        try:
            # Make a copy of the data
            df = data.copy()

            # Check if features exist
            for feature in self.features:
                if feature not in df.columns:
                    logger.error(f"Feature column '{feature}' not found in data")
                    return None

            # Scale features
            scaled_features = {}
            for feature in self.features:
                values = df[feature].values.reshape(-1, 1)
                scaled_values = self.feature_scalers[feature].transform(values)
                scaled_features[feature] = scaled_values.flatten()

            # Create a new DataFrame with scaled features
            scaled_df = pd.DataFrame(scaled_features, index=df.index)

            # Get the most recent sequence
            last_sequence = scaled_df.iloc[-self.sequence_length:].values

            # Initialize predictions list with the last sequence
            future_scaled = last_sequence.copy()
            future_preds = []

            # Make predictions for each future step
            for _ in range(steps):
                # Reshape for prediction
                pred_input = future_scaled[-self.sequence_length:].reshape(1, self.sequence_length, len(self.features))

                # Make prediction
                next_pred = self.model.predict(pred_input)[0][0]

                # Store prediction
                future_preds.append(next_pred)

                # Create a new row with the predicted value for all features
                new_row = np.full((1, len(self.features)), next_pred)

                # Append to the future data for next prediction
                future_scaled = np.vstack([future_scaled, new_row])

            # Inverse transform predictions
            target_feature = self.features[0]  # Default to first feature
            future_preds = self.feature_scalers[target_feature].inverse_transform(
                np.array(future_preds).reshape(-1, 1)).flatten()

            # Create a DataFrame with the predictions
            last_date = df.index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq='B')

            predictions_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted_Close': future_preds
            })

            predictions_df.set_index('Date', inplace=True)

            logger.info(f"Future predictions completed for {self.ticker}")

            return predictions_df

        except Exception as e:
            logger.error(f"Error making future predictions: {e}")
            return None


if __name__ == "__main__":
    # Simple test of the LSTM model
    import numpy as np
    import pandas as pd

    # Generate synthetic data for testing
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='B')

    # Generate sine wave with trend and noise
    t = np.linspace(0, 10, 1000)
    close = np.sin(t) * 50 + t * 20 + 100 + np.random.normal(0, 5, 1000)
    volume = np.abs(np.sin(t + 1)) * 10000 + 5000 + np.random.normal(0, 1000, 1000)

    data = pd.DataFrame({
        'Close': close,
        'Volume': volume
    }, index=dates)

    # Initialize the model
    lstm_model = LSTMModel(
        ticker='TEST',
        sequence_length=50,
        units=64,
        layers=2,
        dropout=0.2,
        prediction_horizon=5,
        batch_size=32,
        epochs=50,
        features=['Close', 'Volume']
    )

    # Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test = lstm_model.prepare_data(data, 'Close')

    # Train model
    lstm_model.train(X_train, y_train, X_val, y_val)

    # Evaluate model
    metrics = lstm_model.evaluate(X_test, y_test, data, plot_results=True)

    # Make future predictions
    future_preds = lstm_model.predict_future(data, steps=30)

    if future_preds is not None:
        print("Future predictions:")
        print(future_preds.head())