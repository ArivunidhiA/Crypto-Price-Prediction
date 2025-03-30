import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import joblib
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ARIMAModel:
    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.model = None
        self.metrics = {}
        
    def train(self, data):
        """Train the ARIMA model"""
        try:
            logger.info("Training ARIMA model...")
            
            # Fit the ARIMA model
            self.model = ARIMA(data, order=self.order)
            self.model = self.model.fit()
            
            logger.info("Model training completed successfully")
            return self.model
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def predict(self, steps=30):
        """Make predictions using the trained model"""
        try:
            if self.model is None:
                raise ValueError("Model not trained yet")
            
            # Get predictions
            forecast = self.model.forecast(steps=steps)
            return forecast
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def evaluate(self, test_data):
        """Evaluate model performance"""
        try:
            predictions = self.predict(steps=len(test_data))
            
            # Calculate metrics
            self.metrics = {
                'mse': mean_squared_error(test_data, predictions),
                'rmse': np.sqrt(mean_squared_error(test_data, predictions)),
                'mae': mean_absolute_error(test_data, predictions),
                'r2': r2_score(test_data, predictions)
            }
            
            logger.info(f"Model evaluation metrics: {self.metrics}")
            return self.metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
    
    def save_model(self, filepath):
        """Save the trained model"""
        try:
            if self.model is None:
                raise ValueError("No model to save")
            
            joblib.dump(self.model, filepath)
            logger.info(f"Model saved successfully to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, filepath):
        """Load a trained model"""
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Model file not found at {filepath}")
            
            self.model = joblib.load(filepath)
            logger.info(f"Model loaded successfully from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def get_confidence_intervals(self, steps=30, confidence=0.95):
        """Get prediction confidence intervals"""
        try:
            if self.model is None:
                raise ValueError("Model not trained yet")
            
            # Get forecast with confidence intervals
            forecast = self.model.get_forecast(steps=steps)
            forecast_mean = forecast.predicted_mean
            forecast_conf = forecast.conf_int(alpha=1-confidence)
            
            return {
                'lower_bound': forecast_conf.iloc[:, 0],
                'upper_bound': forecast_conf.iloc[:, 1],
                'mean_prediction': forecast_mean
            }
            
        except Exception as e:
            logger.error(f"Error calculating confidence intervals: {str(e)}")
            raise
    
    def get_model_summary(self):
        """Get the model summary statistics"""
        try:
            if self.model is None:
                raise ValueError("Model not trained yet")
            
            return self.model.summary()
            
        except Exception as e:
            logger.error(f"Error getting model summary: {str(e)}")
            raise
    
    def get_residuals(self):
        """Get model residuals"""
        try:
            if self.model is None:
                raise ValueError("Model not trained yet")
            
            return self.model.resid
            
        except Exception as e:
            logger.error(f"Error getting residuals: {str(e)}")
            raise 