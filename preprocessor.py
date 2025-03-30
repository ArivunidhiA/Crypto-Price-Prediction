import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import logging
import joblib
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_20', 'SMA_50', 'RSI', 'MACD'
        ]
        
    def prepare_data(self, df, target_column='Close', sequence_length=10, for_sequence_model=False, symbol=None):
        """
        Prepare data for model training
        """
        try:
            # Handle missing values
            df = self._handle_missing_values(df)
            
            # Create features
            features = df[self.feature_columns].values
            target = df[target_column].values.reshape(-1, 1)  # Reshape for scaler
            
            if for_sequence_model:
                # Create sequences for LSTM
                X, y = self._create_sequences(features, target, sequence_length)
                
                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )
                
                # Scale the features
                X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
                X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
                
                self.feature_scaler.fit(X_train_reshaped)
                self.target_scaler.fit(target)  # Fit target scaler on all target data
                
                # Save the scalers if symbol is provided
                if symbol:
                    self.save_scalers(symbol)
                
                X_train_scaled = self.feature_scaler.transform(X_train_reshaped).reshape(X_train.shape)
                X_test_scaled = self.feature_scaler.transform(X_test_reshaped).reshape(X_test.shape)
                
                # Scale targets
                y_train_scaled = self.target_scaler.transform(y_train.reshape(-1, 1)).ravel()
                y_test_scaled = self.target_scaler.transform(y_test.reshape(-1, 1)).ravel()
                
                return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled
            else:
                # For non-sequence models (Random Forest, etc.)
                # Use a sliding window to create features, but flatten the sequence
                X, y = self._create_sequences(features, target, sequence_length)
                X = X.reshape(X.shape[0], -1)  # Flatten the sequences
                
                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )
                
                # Scale the features
                self.feature_scaler.fit(X_train)
                
                # Save the scaler if symbol is provided
                if symbol:
                    self.save_scalers(symbol)
                
                X_train_scaled = self.feature_scaler.transform(X_train)
                X_test_scaled = self.feature_scaler.transform(X_test)
                
                # For non-sequence models, we don't scale the target
                y_train_flat = y_train.ravel()
                y_test_flat = y_test.ravel()
                
                return X_train_scaled, X_test_scaled, y_train_flat, y_test_flat
            
        except Exception as e:
            logger.error(f"Error in data preparation: {str(e)}")
            raise
    
    def _handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        # Forward fill missing values
        df = df.fillna(method='ffill')
        
        # Backward fill any remaining missing values
        df = df.fillna(method='bfill')
        
        return df
    
    def _create_sequences(self, features, target, sequence_length):
        """Create sequences for time series prediction"""
        X, y = [], []
        
        for i in range(len(features) - sequence_length):
            X.append(features[i:(i + sequence_length)])
            y.append(target[i + sequence_length])
            
        return np.array(X), np.array(y)
    
    def prepare_prediction_data(self, df, sequence_length=10, for_sequence_model=False, symbol=None):
        """Prepare data for making predictions"""
        try:
            # Load the scalers for the symbol
            if symbol:
                self.load_scalers(symbol)
            
            # Handle missing values
            df = self._handle_missing_values(df)
            
            # Get features
            features = df[self.feature_columns].values
            
            # Get the last sequence
            last_sequence = features[-sequence_length:]
            
            if for_sequence_model:
                # For LSTM: Reshape to (1, sequence_length, features)
                sequence_reshaped = last_sequence.reshape(-1, last_sequence.shape[-1])
                scaled_sequence = self.feature_scaler.transform(sequence_reshaped)
                return scaled_sequence.reshape(1, sequence_length, len(self.feature_columns))
            else:
                # For Random Forest: Flatten the sequence
                sequence_flat = last_sequence.reshape(1, -1)  # Flatten to (1, sequence_length * features)
                return self.feature_scaler.transform(sequence_flat)
            
        except Exception as e:
            logger.error(f"Error in preparing prediction data: {str(e)}")
            raise
    
    def inverse_transform_target(self, scaled_target):
        """Inverse transform scaled target back to original scale"""
        if scaled_target.ndim == 1:
            scaled_target = scaled_target.reshape(-1, 1)
        return self.target_scaler.inverse_transform(scaled_target).ravel()
    
    def save_scalers(self, symbol):
        """Save the fitted scalers for a cryptocurrency"""
        scaler_dir = 'models/scalers'
        os.makedirs(scaler_dir, exist_ok=True)
        
        feature_scaler_path = os.path.join(scaler_dir, f'feature_scaler_{symbol}.joblib')
        target_scaler_path = os.path.join(scaler_dir, f'target_scaler_{symbol}.joblib')
        
        joblib.dump(self.feature_scaler, feature_scaler_path)
        joblib.dump(self.target_scaler, target_scaler_path)
        logger.info(f"Saved scalers for {symbol}")
    
    def load_scalers(self, symbol):
        """Load the fitted scalers for a cryptocurrency"""
        feature_scaler_path = os.path.join('models/scalers', f'feature_scaler_{symbol}.joblib')
        target_scaler_path = os.path.join('models/scalers', f'target_scaler_{symbol}.joblib')
        
        self.feature_scaler = joblib.load(feature_scaler_path)
        self.target_scaler = joblib.load(target_scaler_path)
        logger.info(f"Loaded scalers for {symbol}") 