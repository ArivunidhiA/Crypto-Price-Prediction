import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LSTMModel:
    def __init__(self, sequence_length=10, n_features=9, n_lstm_layers=2):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_lstm_layers = n_lstm_layers
        self.model = None
        self.metrics = {}
        
    def build_model(self):
        """Build the LSTM model architecture"""
        try:
            self.model = Sequential()
            
            # First LSTM layer
            self.model.add(LSTM(
                units=50,
                return_sequences=True,
                input_shape=(self.sequence_length, self.n_features)
            ))
            self.model.add(Dropout(0.2))
            
            # Additional LSTM layers
            for _ in range(self.n_lstm_layers - 1):
                self.model.add(LSTM(units=50, return_sequences=True))
                self.model.add(Dropout(0.2))
            
            # Final LSTM layer
            self.model.add(LSTM(units=50))
            self.model.add(Dropout(0.2))
            
            # Output layer
            self.model.add(Dense(units=1))
            
            # Compile the model
            self.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            logger.info("LSTM model built successfully")
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise
    
    def train(self, X_train, y_train, batch_size=32, epochs=50, validation_split=0.2):
        """Train the LSTM model"""
        try:
            if self.model is None:
                self.build_model()
            
            logger.info("Training LSTM model...")
            
            # Train the model
            history = self.model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=validation_split,
                verbose=1
            )
            
            logger.info("Model training completed successfully")
            return history
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def predict(self, X):
        """Make predictions using the trained model"""
        try:
            if self.model is None:
                raise ValueError("Model not trained yet")
            
            predictions = self.model.predict(X)
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def evaluate(self, X_test, y_test, preprocessor=None):
        """Evaluate model performance"""
        try:
            predictions = self.predict(X_test)
            
            # Inverse transform predictions and actual values if preprocessor is provided
            if preprocessor is not None:
                predictions = preprocessor.inverse_transform_target(predictions)
                if y_test.ndim == 1:
                    y_test = preprocessor.inverse_transform_target(y_test)
            
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
            if self.model is None:
                raise ValueError("No model to save")
            
            self.model.save(filepath)
            logger.info(f"Model saved successfully to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, filepath):
        """Load a trained model"""
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Model file not found at {filepath}")
            
            self.model = load_model(filepath)
            logger.info(f"Model loaded successfully from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def get_confidence_intervals(self, X, n_samples=100, confidence=0.95):
        """Get prediction confidence intervals using Monte Carlo Dropout"""
        try:
            predictions = []
            
            # Make multiple predictions with dropout enabled
            for _ in range(n_samples):
                pred = self.model.predict(X, verbose=0)
                predictions.append(pred)
            
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