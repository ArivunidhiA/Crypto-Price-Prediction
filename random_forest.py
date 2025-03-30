import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RandomForestModel:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        self.feature_importance = None
        self.metrics = {}
        
    def train(self, X_train, y_train):
        """Train the Random Forest model"""
        try:
            logger.info("Training Random Forest model...")
            self.model.fit(X_train, y_train)
            
            # Get feature importance
            self.feature_importance = dict(zip(
                ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'MACD'],
                self.model.feature_importances_
            ))
            
            logger.info("Model training completed successfully")
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def predict(self, X):
        """Make predictions using the trained model"""
        try:
            predictions = self.model.predict(X)
            return predictions
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        try:
            predictions = self.predict(X_test)
            
            # Calculate metrics
            self.metrics = {
                'mse': mean_squared_error(y_test, predictions),
                'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
                'mae': mean_absolute_error(y_test, predictions),
                'r2': r2_score(y_test, predictions)
            }
            
            logger.info(f"Model evaluation metrics: {self.metrics}")
            return self.metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
    
    def save_model(self, filepath):
        """Save the trained model"""
        try:
            joblib.dump(self.model, filepath)
            logger.info(f"Model saved successfully to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, filepath):
        """Load a trained model"""
        try:
            self.model = joblib.load(filepath)
            logger.info(f"Model loaded successfully from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def get_feature_importance(self):
        """Get feature importance scores"""
        return self.feature_importance
    
    def get_confidence_intervals(self, X, confidence=0.95):
        """Get prediction confidence intervals"""
        try:
            predictions = []
            for estimator in self.model.estimators_:
                predictions.append(estimator.predict(X))
            
            predictions = np.array(predictions)
            
            # Calculate confidence intervals
            lower = np.percentile(predictions, (1 - confidence) * 100, axis=0)
            upper = np.percentile(predictions, confidence * 100, axis=0)
            
            return {
                'lower_bound': lower,
                'upper_bound': upper,
                'mean_prediction': np.mean(predictions, axis=0)
            }
            
        except Exception as e:
            logger.error(f"Error calculating confidence intervals: {str(e)}")
            raise 