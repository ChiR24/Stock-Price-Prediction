"""
Module for benchmarking and comparing different machine learning models.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
    mean_squared_error, mean_absolute_error, r2_score
)

# Import project modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from utils.config import MODEL_DIR
from utils.spark_utils import create_spark_session, stop_spark_session, spark_to_dataframe
from ml_models.model_base import ModelBase
from ml_models.lstm_model import LSTMModel
from ml_models.spark_ml_models import (
    SparkRandomForestModel, SparkGradientBoostedTreesModel, SparkLinearModel
)

# Set up logger
logger = setup_logger('model_benchmark')

class ModelBenchmark:
    """
    Class for benchmarking and comparing different machine learning models.
    """
    def __init__(self, output_dir=None):
        """
        Initialize the ModelBenchmark.
        
        Args:
            output_dir: Directory to save benchmark results. Defaults to None (use MODEL_DIR).
        """
        self.output_dir = output_dir if output_dir else os.path.join(MODEL_DIR, 'benchmarks')
        os.makedirs(self.output_dir, exist_ok=True)
        self.models = []
        self.results = {}
        self.training_times = {}
        self.prediction_times = {}
        self.feature_importance = {}
        self.spark = None
    
    def add_model(self, model):
        """
        Add a model to the benchmark.
        
        Args:
            model: Model instance to add.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            if not isinstance(model, ModelBase):
                logger.error(f"Model must be an instance of ModelBase or its subclass")
                return False
            
            self.models.append(model)
            logger.info(f"Added model: {model.name} ({model.model_type})")
            
            return True
        
        except Exception as e:
            logger.error(f"Error adding model: {e}")
            return False
    
    def train_all_models(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train all models.
        
        Args:
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features. Defaults to None.
            y_val: Validation targets. Defaults to None.
            
        Returns:
            Dictionary of training results.
        """
        results = {}
        
        for model in self.models:
            try:
                logger.info(f"Training model: {model.name}")
                
                # Record start time
                start_time = datetime.now()
                
                # Initialize Spark session if needed
                if "Spark" in model.__class__.__name__ and not self.spark:
                    self.spark = create_spark_session()
                
                # Train the model
                if "Spark" in model.__class__.__name__:
                    # Prepare data for Spark models
                    train_data = model.prepare_data(X_train, y_train)
                    
                    if X_val is not None and y_val is not None:
                        val_data = model.prepare_data(X_val, y_val)
                    else:
                        val_data = None
                    
                    # Train the model
                    model.train(train_data[0], validation_df=train_data[1] if val_data is None else val_data[0])
                else:
                    # Train the model (non-Spark)
                    validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
                    model.train(X_train, y_train, validation_data=validation_data)
                
                # Record end time
                end_time = datetime.now()
                training_time = (end_time - start_time).total_seconds()
                
                self.training_times[model.name] = training_time
                
                logger.info(f"Trained model: {model.name} in {training_time:.2f} seconds")
                
                results[model.name] = {'status': 'success', 'training_time': training_time}
            
            except Exception as e:
                logger.error(f"Error training model {model.name}: {e}")
                results[model.name] = {'status': 'error', 'error': str(e)}
        
        self.results['training'] = results
        return results
    
    def evaluate_all_models(self, X_test, y_test):
        """
        Evaluate all models.
        
        Args:
            X_test: Test features.
            y_test: Test targets.
            
        Returns:
            Dictionary of evaluation results.
        """
        results = {}
        
        for model in self.models:
            try:
                logger.info(f"Evaluating model: {model.name}")
                
                # Record start time
                start_time = datetime.now()
                
                # Make predictions
                if "Spark" in model.__class__.__name__:
                    # Prepare data for Spark models
                    test_data = model.prepare_data(X_test, y_test)
                    
                    # Get predictions
                    predictions_df = model.predict(test_data[0])
                    
                    # Convert to numpy array
                    predictions = predictions_df.select('prediction').collect()
                    predictions = np.array([row['prediction'] for row in predictions])
                else:
                    # Make predictions (non-Spark)
                    predictions = model.predict(X_test)
                
                # Record end time
                end_time = datetime.now()
                prediction_time = (end_time - start_time).total_seconds()
                
                self.prediction_times[model.name] = prediction_time
                
                # Calculate metrics
                metrics = self._calculate_metrics(y_test, predictions, model.model_type)
                
                # Save results
                results[model.name] = {
                    'status': 'success',
                    'prediction_time': prediction_time,
                    'metrics': metrics
                }
                
                logger.info(f"Evaluated model: {model.name} in {prediction_time:.2f} seconds")
                
                # Get feature importance if available
                if hasattr(model.model, 'feature_importances_'):
                    self.feature_importance[model.name] = model.model.feature_importances_
                elif hasattr(model.model, 'coef_'):
                    self.feature_importance[model.name] = model.model.coef_
            
            except Exception as e:
                logger.error(f"Error evaluating model {model.name}: {e}")
                results[model.name] = {'status': 'error', 'error': str(e)}
        
        self.results['evaluation'] = results
        return results
    
    def _calculate_metrics(self, y_true, y_pred, model_type):
        """
        Calculate metrics for evaluation.
        
        Args:
            y_true: True target values.
            y_pred: Predicted values.
            model_type: Type of model ('regression' or 'classification').
            
        Returns:
            Dictionary of metrics.
        """
        metrics = {}
        
        try:
            if model_type == 'classification':
                # Classification metrics
                metrics['accuracy'] = accuracy_score(y_true, y_pred)
                metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                
                # ROC AUC if binary classification
                if len(np.unique(y_true)) == 2:
                    try:
                        metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
                    except Exception:
                        # If predictions are not probabilities, skip ROC AUC
                        metrics['roc_auc'] = None
            else:
                # Regression metrics
                metrics['mse'] = mean_squared_error(y_true, y_pred)
                metrics['rmse'] = np.sqrt(metrics['mse'])
                metrics['mae'] = mean_absolute_error(y_true, y_pred)
                metrics['r2'] = r2_score(y_true, y_pred)
        
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
        
        return metrics
    
    def generate_comparison_report(self, filename=None):
        """
        Generate a comparison report.
        
        Args:
            filename: Filename for the report. Defaults to None (auto-generate).
            
        Returns:
            Path to the saved report.
        """
        try:
            # Generate a filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"model_comparison_{timestamp}.html"
            
            filepath = os.path.join(self.output_dir, filename)
            
            # Create a report DataFrame
            if 'evaluation' in self.results:
                report_data = []
                
                for model_name, result in self.results['evaluation'].items():
                    if result['status'] == 'success':
                        row = {'Model': model_name}
                        
                        # Add training time
                        if model_name in self.training_times:
                            row['Training Time (s)'] = self.training_times[model_name]
                        
                        # Add prediction time
                        if model_name in self.prediction_times:
                            row['Prediction Time (s)'] = self.prediction_times[model_name]
                        
                        # Add metrics
                        for metric_name, metric_value in result['metrics'].items():
                            row[metric_name] = metric_value
                        
                        report_data.append(row)
                
                if report_data:
                    report_df = pd.DataFrame(report_data)
                    
                    # Generate the report
                    with open(filepath, 'w') as f:
                        f.write("<html>\n<head>\n")
                        f.write("<title>Model Comparison Report</title>\n")
                        f.write("<style>\n")
                        f.write("table { border-collapse: collapse; width: 100%; }\n")
                        f.write("th, td { text-align: left; padding: 8px; }\n")
                        f.write("tr:nth-child(even) { background-color: #f2f2f2; }\n")
                        f.write("th { background-color: #4CAF50; color: white; }\n")
                        f.write("</style>\n")
                        f.write("</head>\n<body>\n")
                        
                        f.write("<h1>Model Comparison Report</h1>\n")
                        f.write(f"<p>Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n")
                        
                        # Add the comparison table
                        f.write("<h2>Model Metrics</h2>\n")
                        f.write(report_df.to_html(index=False))
                        
                        # Add section for feature importance if available
                        if self.feature_importance:
                            f.write("<h2>Feature Importance</h2>\n")
                            
                            for model_name, importances in self.feature_importance.items():
                                f.write(f"<h3>{model_name}</h3>\n")
                                
                                # Save feature importance plot
                                plt.figure(figsize=(10, 6))
                                
                                # Get feature names if available
                                if hasattr(self.models[0], 'feature_columns') and self.models[0].feature_columns is not None:
                                    feature_names = self.models[0].feature_columns
                                else:
                                    feature_names = [f"Feature {i}" for i in range(len(importances))]
                                
                                # Plot feature importance
                                plt.barh(range(len(importances)), importances)
                                plt.yticks(range(len(importances)), feature_names)
                                plt.xlabel('Importance')
                                plt.title(f'Feature Importance - {model_name}')
                                
                                # Save the plot
                                plot_filename = f"feature_importance_{model_name}.png"
                                plot_filepath = os.path.join(self.output_dir, plot_filename)
                                plt.savefig(plot_filepath)
                                plt.close()
                                
                                f.write(f'<img src="{plot_filename}" alt="Feature Importance" width="800">\n')
                        
                        f.write("</body>\n</html>")
                    
                    logger.info(f"Generated comparison report: {filepath}")
                    
                    return filepath
                else:
                    logger.warning("No evaluation results available for the report")
                    return None
            else:
                logger.warning("No evaluation results available for the report")
                return None
        
        except Exception as e:
            logger.error(f"Error generating comparison report: {e}")
            return None
    
    def close(self):
        """
        Close all resources.
        """
        try:
            # Close Spark session if created
            if self.spark:
                stop_spark_session(self.spark)
                logger.info("Closed Spark session")
            
            # Close model resources if needed
            for model in self.models:
                if hasattr(model, 'close'):
                    model.close()
                    logger.info(f"Closed model: {model.name}")
        
        except Exception as e:
            logger.error(f"Error closing resources: {e}")

def main():
    """
    Main function to demonstrate model benchmarking.
    """
    try:
        # Create benchmark
        benchmark = ModelBenchmark()
        
        # Create models
        lstm_model = LSTMModel(name='lstm_model', model_type='classification')
        rf_model = SparkRandomForestModel(name='rf_model', model_type='classification')
        gbt_model = SparkGradientBoostedTreesModel(name='gbt_model', model_type='classification')
        linear_model = SparkLinearModel(name='linear_model', model_type='classification')
        
        # Add models to benchmark
        benchmark.add_model(lstm_model)
        benchmark.add_model(rf_model)
        benchmark.add_model(gbt_model)
        benchmark.add_model(linear_model)
        
        # Generate sample data
        X_train = np.random.rand(100, 10)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.rand(20, 10)
        y_test = np.random.randint(0, 2, 20)
        
        # Convert to DataFrame (for Spark models)
        X_train_df = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(10)])
        X_test_df = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(10)])
        
        # Train and evaluate models
        benchmark.train_all_models(X_train_df, y_train)
        benchmark.evaluate_all_models(X_test_df, y_test)
        
        # Generate report
        benchmark.generate_comparison_report()
        
        # Close resources
        benchmark.close()
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")


if __name__ == "__main__":
    main() 