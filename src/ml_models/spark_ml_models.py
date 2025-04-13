"""
Module for machine learning models using Spark MLlib.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, LogisticRegression
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor, LinearRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.sql.functions import col, expr, udf
from pyspark.sql.types import FloatType, IntegerType

# Import project modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from utils.config import MODEL_DIR
from utils.spark_utils import create_spark_session, stop_spark_session, spark_to_dataframe
from ml_models.model_base import ModelBase

# Set up logger
logger = setup_logger('spark_ml_models')

class SparkMLModel(ModelBase):
    """
    Base class for Spark ML models.
    """
    def __init__(self, name='spark_model', model_type='classification'):
        """
        Initialize the Spark ML model.
        
        Args:
            name: Name of the model. Defaults to 'spark_model'.
            model_type: Type of model ('regression' or 'classification'). Defaults to 'classification'.
        """
        super().__init__(name, model_type)
        self.spark = create_spark_session()
        self.model = None
        self.pipeline_model = None
        self.feature_columns = None
        self.target_column = None
    
    def prepare_data(self, X, y=None, train_test_split=0.8):
        """
        Prepare data for Spark ML model.
        
        Args:
            X: Features DataFrame or numpy array.
            y: Target array. Defaults to None.
            train_test_split: Fraction of data to use for training. Defaults to 0.8.
            
        Returns:
            Spark DataFrame or tuple of (train_df, test_df) if y is provided.
        """
        try:
            # Convert to Spark DataFrame
            if isinstance(X, pd.DataFrame):
                # Save feature columns
                self.feature_columns = X.columns.tolist()
                
                if y is not None:
                    # Add target to DataFrame
                    if isinstance(y, pd.Series):
                        self.target_column = y.name if y.name else 'target'
                        data = X.copy()
                        data[self.target_column] = y
                    else:
                        self.target_column = 'target'
                        data = X.copy()
                        data[self.target_column] = y
                    
                    # Convert to Spark DataFrame
                    spark_df = self.spark.createDataFrame(data)
                    
                    # Split into train and test
                    train_df, test_df = spark_df.randomSplit([train_test_split, 1.0 - train_test_split])
                    
                    logger.info(f"Prepared data: {spark_df.count()} rows, split into {train_df.count()} train and {test_df.count()} test")
                    
                    return train_df, test_df
                else:
                    # Convert to Spark DataFrame without splitting
                    spark_df = self.spark.createDataFrame(X)
                    
                    logger.info(f"Prepared data: {spark_df.count()} rows")
                    
                    return spark_df
            else:
                logger.error("X must be a pandas DataFrame")
                return None
        
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return None
    
    def close(self):
        """
        Close the Spark session.
        """
        stop_spark_session(self.spark)
    
    def save_model(self, filepath=None, include_timestamp=True):
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model. Defaults to None (use default path).
            include_timestamp: Whether to include timestamp in the filename. Defaults to True.
            
        Returns:
            Path to the saved model.
        """
        if not self.model:
            logger.error("Model not trained yet")
            return None
        
        try:
            # Create model directory if it doesn't exist
            os.makedirs(MODEL_DIR, exist_ok=True)
            
            # Generate filename
            if filepath is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if include_timestamp else ""
                filepath = os.path.join(MODEL_DIR, f"{self.name}_{timestamp}")
            
            # Save model
            self.model.write().overwrite().save(filepath)
            
            # Save metadata (feature and target columns)
            metadata = {
                'name': self.name,
                'model_type': self.model_type,
                'feature_columns': self.feature_columns,
                'target_column': self.target_column,
                'saved_at': datetime.now().isoformat()
            }
            
            metadata_path = filepath + "_metadata.json"
            with open(metadata_path, 'w') as f:
                import json
                json.dump(metadata, f)
            
            logger.info(f"Saved {self.model_type} model to {filepath}")
            
            return filepath
        
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return None
    
    def load_model(self, filepath):
        """
        Load a model from disk.
        
        Args:
            filepath: Path to the saved model.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            # Load metadata
            metadata_path = filepath + "_metadata.json"
            with open(metadata_path, 'r') as f:
                import json
                metadata = json.load(f)
            
            # Set metadata
            self.name = metadata['name']
            self.model_type = metadata['model_type']
            self.feature_columns = metadata['feature_columns']
            self.target_column = metadata['target_column']
            
            # Load model
            if self.model_type == 'classification':
                model_class = self._get_model_class(metadata['name'])
                self.model = model_class.load(filepath)
            else:
                model_class = self._get_model_class(metadata['name'], regression=True)
                self.model = model_class.load(filepath)
            
            logger.info(f"Loaded {self.model_type} model from {filepath}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def _get_model_class(self, name, regression=False):
        """
        Get the correct model class based on the model name.
        
        Args:
            name: Name of the model.
            regression: Whether the model is for regression. Defaults to False.
            
        Returns:
            Spark ML model class.
        """
        if regression:
            if 'rf' in name.lower() or 'randomforest' in name.lower():
                return RandomForestRegressor
            elif 'gbt' in name.lower():
                return GBTRegressor
            else:
                return LinearRegression
        else:
            if 'rf' in name.lower() or 'randomforest' in name.lower():
                return RandomForestClassifier
            elif 'gbt' in name.lower():
                return GBTClassifier
            else:
                return LogisticRegression


class SparkRandomForestModel(SparkMLModel):
    """
    Random Forest model using Spark MLlib.
    """
    def __init__(self, name='spark_rf', model_type='classification'):
        """
        Initialize the Random Forest model.
        
        Args:
            name: Name of the model. Defaults to 'spark_rf'.
            model_type: Type of model ('regression' or 'classification'). Defaults to 'classification'.
        """
        super().__init__(name, model_type)
    
    def build_model(self, params=None):
        """
        Build the Random Forest model.
        
        Args:
            params: Model parameters. Defaults to None (use default parameters).
            
        Returns:
            Random Forest model instance.
        """
        try:
            # Set default parameters if not provided
            if params is None:
                params = {
                    'numTrees': 100,
                    'maxDepth': 10,
                    'minInstancesPerNode': 1,
                    'seed': 42
                }
            
            # Create the model
            if self.model_type == 'classification':
                self.model = RandomForestClassifier(
                    featuresCol="features",
                    labelCol="label",
                    **params
                )
            else:
                self.model = RandomForestRegressor(
                    featuresCol="features",
                    labelCol="label",
                    **params
                )
            
            logger.info(f"Built Random Forest {self.model_type} model with parameters: {params}")
            
            return self.model
        
        except Exception as e:
            logger.error(f"Error building Random Forest model: {e}")
            return None
    
    def train(self, train_df, validation_df=None, params=None, **kwargs):
        """
        Train the Random Forest model.
        
        Args:
            train_df: Training data as Spark DataFrame.
            validation_df: Validation data as Spark DataFrame. Defaults to None.
            params: Model parameters. Defaults to None (use default parameters).
            **kwargs: Additional parameters for cross-validation.
            
        Returns:
            Trained model instance.
        """
        try:
            # Build the model
            self.build_model(params)
            
            # Train the model
            self.model = self.model.fit(train_df)
            
            logger.info(f"Trained Random Forest {self.model_type} model")
            
            # Evaluate the model
            if validation_df is not None:
                self.evaluate(validation_df)
            
            return self.model
        
        except Exception as e:
            logger.error(f"Error training Random Forest model: {e}")
            return None
    
    def predict(self, X):
        """
        Make predictions with the trained model.
        
        Args:
            X: Input data as Spark DataFrame.
            
        Returns:
            DataFrame with predictions.
        """
        if not self.model:
            logger.error("Model not trained yet")
            return None
        
        try:
            # Make predictions
            predictions = self.model.transform(X)
            
            logger.info(f"Made predictions on {X.count()} rows")
            
            return predictions
        
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return None
    
    def evaluate(self, test_df):
        """
        Evaluate the model.
        
        Args:
            test_df: Test data as Spark DataFrame.
            
        Returns:
            Dictionary with evaluation metrics.
        """
        if not self.model:
            logger.error("Model not trained yet")
            return None
        
        try:
            # Make predictions on test data
            predictions = self.model.transform(test_df)
            
            # Evaluate model
            metrics = {}
            
            if self.model_type == 'classification':
                evaluator = BinaryClassificationEvaluator(labelCol="label")
                
                # Calculate metrics
                auc = evaluator.evaluate(predictions)
                metrics['auc'] = auc
                
                # Set evaluator to calculate accuracy
                evaluator.setMetricName("areaUnderROC")
                metrics['areaUnderROC'] = evaluator.evaluate(predictions)
                
                evaluator.setMetricName("areaUnderPR")
                metrics['areaUnderPR'] = evaluator.evaluate(predictions)
                
                # Calculate accuracy manually
                from pyspark.sql.functions import when
                correct_preds = predictions.select(
                    when(col("prediction") == col("label"), 1).otherwise(0).alias("correct")
                )
                accuracy = correct_preds.agg({"correct": "mean"}).collect()[0][0]
                metrics['accuracy'] = accuracy
                
                logger.info(f"Classification metrics: AUC={auc:.4f}, Accuracy={accuracy:.4f}")
            else:
                # Regression metrics
                rmse_evaluator = RegressionEvaluator(labelCol="label", metricName="rmse")
                mae_evaluator = RegressionEvaluator(labelCol="label", metricName="mae")
                r2_evaluator = RegressionEvaluator(labelCol="label", metricName="r2")
                
                metrics['rmse'] = rmse_evaluator.evaluate(predictions)
                metrics['mae'] = mae_evaluator.evaluate(predictions)
                metrics['r2'] = r2_evaluator.evaluate(predictions)
                
                logger.info(f"Regression metrics: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, R2={metrics['r2']:.4f}")
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return None


class SparkGradientBoostedTreesModel(SparkMLModel):
    """
    Gradient Boosted Trees model using Spark MLlib.
    """
    def __init__(self, name='spark_gbt', model_type='classification'):
        """
        Initialize the Gradient Boosted Trees model.
        
        Args:
            name: Name of the model. Defaults to 'spark_gbt'.
            model_type: Type of model ('regression' or 'classification'). Defaults to 'classification'.
        """
        super().__init__(name, model_type)
    
    def build_model(self, params=None):
        """
        Build the Gradient Boosted Trees model.
        
        Args:
            params: Model parameters. Defaults to None (use default parameters).
            
        Returns:
            Gradient Boosted Trees model instance.
        """
        try:
            # Set default parameters if not provided
            if params is None:
                params = {
                    'maxIter': 20,
                    'maxDepth': 5,
                    'stepSize': 0.1,
                    'seed': 42
                }
            
            # Create the model
            if self.model_type == 'classification':
                self.model = GBTClassifier(
                    featuresCol="features",
                    labelCol="label",
                    **params
                )
            else:
                self.model = GBTRegressor(
                    featuresCol="features",
                    labelCol="label",
                    **params
                )
            
            logger.info(f"Built Gradient Boosted Trees {self.model_type} model with parameters: {params}")
            
            return self.model
        
        except Exception as e:
            logger.error(f"Error building Gradient Boosted Trees model: {e}")
            return None
    
    def train(self, train_df, validation_df=None, params=None, **kwargs):
        """
        Train the Gradient Boosted Trees model.
        
        Args:
            train_df: Training data as Spark DataFrame.
            validation_df: Validation data as Spark DataFrame. Defaults to None.
            params: Model parameters. Defaults to None (use default parameters).
            **kwargs: Additional parameters for cross-validation.
            
        Returns:
            Trained model instance.
        """
        try:
            # Build the model
            self.build_model(params)
            
            # Train the model
            self.model = self.model.fit(train_df)
            
            logger.info(f"Trained Gradient Boosted Trees {self.model_type} model")
            
            # Evaluate the model
            if validation_df is not None:
                self.evaluate(validation_df)
            
            return self.model
        
        except Exception as e:
            logger.error(f"Error training Gradient Boosted Trees model: {e}")
            return None
    
    def predict(self, X):
        """
        Make predictions with the trained model.
        
        Args:
            X: Input data as Spark DataFrame.
            
        Returns:
            DataFrame with predictions.
        """
        if not self.model:
            logger.error("Model not trained yet")
            return None
        
        try:
            # Make predictions
            predictions = self.model.transform(X)
            
            logger.info(f"Made predictions on {X.count()} rows")
            
            return predictions
        
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return None
    
    def evaluate(self, test_df):
        """
        Evaluate the model.
        
        Args:
            test_df: Test data as Spark DataFrame.
            
        Returns:
            Dictionary with evaluation metrics.
        """
        if not self.model:
            logger.error("Model not trained yet")
            return None
        
        try:
            # Make predictions on test data
            predictions = self.model.transform(test_df)
            
            # Evaluate model
            metrics = {}
            
            if self.model_type == 'classification':
                evaluator = BinaryClassificationEvaluator(labelCol="label")
                
                # Calculate metrics
                auc = evaluator.evaluate(predictions)
                metrics['auc'] = auc
                
                # Set evaluator to calculate accuracy
                evaluator.setMetricName("areaUnderROC")
                metrics['areaUnderROC'] = evaluator.evaluate(predictions)
                
                evaluator.setMetricName("areaUnderPR")
                metrics['areaUnderPR'] = evaluator.evaluate(predictions)
                
                # Calculate accuracy manually
                from pyspark.sql.functions import when
                correct_preds = predictions.select(
                    when(col("prediction") == col("label"), 1).otherwise(0).alias("correct")
                )
                accuracy = correct_preds.agg({"correct": "mean"}).collect()[0][0]
                metrics['accuracy'] = accuracy
                
                logger.info(f"Classification metrics: AUC={auc:.4f}, Accuracy={accuracy:.4f}")
            else:
                # Regression metrics
                rmse_evaluator = RegressionEvaluator(labelCol="label", metricName="rmse")
                mae_evaluator = RegressionEvaluator(labelCol="label", metricName="mae")
                r2_evaluator = RegressionEvaluator(labelCol="label", metricName="r2")
                
                metrics['rmse'] = rmse_evaluator.evaluate(predictions)
                metrics['mae'] = mae_evaluator.evaluate(predictions)
                metrics['r2'] = r2_evaluator.evaluate(predictions)
                
                logger.info(f"Regression metrics: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, R2={metrics['r2']:.4f}")
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return None


class SparkLinearModel(SparkMLModel):
    """
    Linear model (regression or logistic regression) using Spark MLlib.
    """
    def __init__(self, name='spark_linear', model_type='regression'):
        """
        Initialize the linear model.
        
        Args:
            name: Name of the model. Defaults to 'spark_linear'.
            model_type: Type of model ('regression' or 'classification'). Defaults to 'regression'.
        """
        super().__init__(name, model_type)
    
    def build_model(self, params=None):
        """
        Build the linear model.
        
        Args:
            params: Model parameters. Defaults to None (use default parameters).
            
        Returns:
            Linear model instance.
        """
        try:
            # Set default parameters if not provided
            if params is None:
                params = {
                    'maxIter': 100,
                    'regParam': 0.0,
                    'elasticNetParam': 0.0
                }
            
            # Create the model
            if self.model_type == 'classification':
                self.model = LogisticRegression(
                    featuresCol="features",
                    labelCol="label",
                    **params
                )
            else:
                self.model = LinearRegression(
                    featuresCol="features",
                    labelCol="label",
                    **params
                )
            
            logger.info(f"Built Linear {self.model_type} model with parameters: {params}")
            
            return self.model
        
        except Exception as e:
            logger.error(f"Error building Linear model: {e}")
            return None
    
    def train(self, train_df, validation_df=None, params=None, **kwargs):
        """
        Train the linear model.
        
        Args:
            train_df: Training data as Spark DataFrame.
            validation_df: Validation data as Spark DataFrame. Defaults to None.
            params: Model parameters. Defaults to None (use default parameters).
            **kwargs: Additional parameters for cross-validation.
            
        Returns:
            Trained model instance.
        """
        try:
            # Build the model
            self.build_model(params)
            
            # Train the model
            self.model = self.model.fit(train_df)
            
            logger.info(f"Trained Linear {self.model_type} model")
            
            # Evaluate the model
            if validation_df is not None:
                self.evaluate(validation_df)
            
            return self.model
        
        except Exception as e:
            logger.error(f"Error training Linear model: {e}")
            return None
    
    def predict(self, X):
        """
        Make predictions with the trained model.
        
        Args:
            X: Input data as Spark DataFrame.
            
        Returns:
            DataFrame with predictions.
        """
        if not self.model:
            logger.error("Model not trained yet")
            return None
        
        try:
            # Make predictions
            predictions = self.model.transform(X)
            
            logger.info(f"Made predictions on {X.count()} rows")
            
            return predictions
        
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return None
    
    def evaluate(self, test_df):
        """
        Evaluate the model.
        
        Args:
            test_df: Test data as Spark DataFrame.
            
        Returns:
            Dictionary with evaluation metrics.
        """
        if not self.model:
            logger.error("Model not trained yet")
            return None
        
        try:
            # Make predictions on test data
            predictions = self.model.transform(test_df)
            
            # Evaluate model
            metrics = {}
            
            if self.model_type == 'classification':
                evaluator = BinaryClassificationEvaluator(labelCol="label")
                
                # Calculate metrics
                auc = evaluator.evaluate(predictions)
                metrics['auc'] = auc
                
                # Set evaluator to calculate accuracy
                evaluator.setMetricName("areaUnderROC")
                metrics['areaUnderROC'] = evaluator.evaluate(predictions)
                
                evaluator.setMetricName("areaUnderPR")
                metrics['areaUnderPR'] = evaluator.evaluate(predictions)
                
                # Calculate accuracy manually
                from pyspark.sql.functions import when
                correct_preds = predictions.select(
                    when(col("prediction") == col("label"), 1).otherwise(0).alias("correct")
                )
                accuracy = correct_preds.agg({"correct": "mean"}).collect()[0][0]
                metrics['accuracy'] = accuracy
                
                logger.info(f"Classification metrics: AUC={auc:.4f}, Accuracy={accuracy:.4f}")
                
                # Get coefficients for logistic regression
                if hasattr(self.model, 'coefficients'):
                    metrics['coefficients'] = self.model.coefficients.toArray().tolist()
                
                if hasattr(self.model, 'intercept'):
                    metrics['intercept'] = self.model.intercept
                
            else:
                # Regression metrics
                rmse_evaluator = RegressionEvaluator(labelCol="label", metricName="rmse")
                mae_evaluator = RegressionEvaluator(labelCol="label", metricName="mae")
                r2_evaluator = RegressionEvaluator(labelCol="label", metricName="r2")
                
                metrics['rmse'] = rmse_evaluator.evaluate(predictions)
                metrics['mae'] = mae_evaluator.evaluate(predictions)
                metrics['r2'] = r2_evaluator.evaluate(predictions)
                
                logger.info(f"Regression metrics: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, R2={metrics['r2']:.4f}")
                
                # Get coefficients and intercept
                if hasattr(self.model, 'coefficients'):
                    metrics['coefficients'] = self.model.coefficients.toArray().tolist()
                
                if hasattr(self.model, 'intercept'):
                    metrics['intercept'] = self.model.intercept
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return None
            
            
def main():
    """
    Main function to demonstrate Spark ML models.
    """
    try:
        # Create a sample dataframe
        spark = create_spark_session()
        
        # Create sample data
        from pyspark.sql import Row
        data = [
            Row(features=[1.0, 0.0, 0.0], label=0.0),
            Row(features=[0.0, 1.0, 1.0], label=1.0),
            Row(features=[1.0, 1.0, 1.0], label=1.0),
            Row(features=[0.0, 0.0, 0.0], label=0.0)
        ]
        df = spark.createDataFrame(data)
        
        # Split data
        train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
        
        # Create and train a RandomForest model
        rf_model = SparkRandomForestModel(model_type='classification')
        rf_model.model = RandomForestClassifier(numTrees=10, maxDepth=5, seed=42)
        rf_model.model = rf_model.model.fit(train_df)
        
        # Make predictions
        predictions = rf_model.predict(test_df)
        predictions.show()
        
        # Evaluate model
        metrics = rf_model.evaluate(test_df)
        print(f"RandomForest metrics: {metrics}")
        
        # Create and train a GBT model
        gbt_model = SparkGradientBoostedTreesModel(model_type='classification')
        gbt_model.model = GBTClassifier(maxIter=10, maxDepth=5, seed=42)
        gbt_model.model = gbt_model.model.fit(train_df)
        
        # Make predictions
        predictions = gbt_model.predict(test_df)
        predictions.show()
        
        # Evaluate model
        metrics = gbt_model.evaluate(test_df)
        print(f"GBT metrics: {metrics}")
        
        # Create and train a Linear model
        linear_model = SparkLinearModel(model_type='classification')
        linear_model.model = LogisticRegression(maxIter=10, regParam=0.01)
        linear_model.model = linear_model.model.fit(train_df)
        
        # Make predictions
        predictions = linear_model.predict(test_df)
        predictions.show()
        
        # Evaluate model
        metrics = linear_model.evaluate(test_df)
        print(f"Linear model metrics: {metrics}")
        
        # Close Spark session
        stop_spark_session(spark)
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")


if __name__ == "__main__":
    main() 