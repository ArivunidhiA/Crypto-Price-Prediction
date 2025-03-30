import os
from data.data_fetcher import CryptoDataFetcher
from data.preprocessor import DataPreprocessor
from models.random_forest import RandomForestModel
from models.lstm import LSTMModel
from models.arima import ARIMAModel
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def ensure_directory_exists(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def train_all_models():
    """Train all models for each cryptocurrency"""
    try:
        # Initialize components
        data_fetcher = CryptoDataFetcher()
        preprocessor = DataPreprocessor()
        
        # Create models directory
        model_dir = 'models/saved'
        ensure_directory_exists(model_dir)
        
        # Create scalers directory
        scaler_dir = 'models/scalers'
        ensure_directory_exists(scaler_dir)
        
        # Fetch historical data
        logger.info("Fetching historical data...")
        try:
            historical_data = data_fetcher.fetch_historical_data()
            if not historical_data:
                raise ValueError("No historical data retrieved")
            logger.info(f"Successfully fetched data for {len(historical_data)} cryptocurrencies")
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            raise
        
        # Train models for each cryptocurrency
        for symbol, df in historical_data.items():
            logger.info(f"\nProcessing {symbol}...")
            logger.info(f"Data shape for {symbol}: {df.shape}")
            
            try:
                # Verify data
                if df.empty:
                    logger.warning(f"Empty dataset for {symbol}, skipping...")
                    continue
                
                if len(df) < 100:  # Minimum required data points
                    logger.warning(f"Insufficient data for {symbol}, skipping...")
                    continue
                
                # Handle missing values
                df = df.fillna(method='ffill').fillna(method='bfill')
                
                # Prepare data
                logger.info(f"Preparing data for {symbol}...")
                
                # Prepare data for Random Forest
                X_train_rf, X_test_rf, y_train_rf, y_test_rf = preprocessor.prepare_data(
                    df, for_sequence_model=False, symbol=f"{symbol}_rf"
                )
                logger.info(f"Random Forest training data shape: {X_train_rf.shape}")
                logger.info(f"Random Forest testing data shape: {X_test_rf.shape}")
                
                # Prepare data for LSTM
                X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = preprocessor.prepare_data(
                    df, for_sequence_model=True, symbol=f"{symbol}_lstm"
                )
                logger.info(f"LSTM training data shape: {X_train_lstm.shape}")
                logger.info(f"LSTM testing data shape: {X_test_lstm.shape}")
                
                # Train Random Forest
                logger.info(f"Training Random Forest model for {symbol}...")
                rf_model = RandomForestModel()
                rf_model.train(X_train_rf, y_train_rf)
                rf_model.save_model(f"{model_dir}/rf_{symbol}.joblib")
                logger.info("Random Forest model saved successfully")
                
                # Train LSTM
                logger.info(f"Training LSTM model for {symbol}...")
                lstm_model = LSTMModel()
                lstm_model.train(X_train_lstm, y_train_lstm)
                lstm_model.save_model(f"{model_dir}/lstm_{symbol}.h5")
                logger.info("LSTM model saved successfully")
                
                # Train ARIMA
                logger.info(f"Training ARIMA model for {symbol}...")
                arima_model = ARIMAModel()
                arima_model.train(df['Close'])
                arima_model.save_model(f"{model_dir}/arima_{symbol}.joblib")
                logger.info("ARIMA model saved successfully")
                
                # Evaluate models
                logger.info(f"Evaluating models for {symbol}...")
                rf_metrics = rf_model.evaluate(X_test_rf, y_test_rf)
                lstm_metrics = lstm_model.evaluate(X_test_lstm, y_test_lstm, preprocessor)
                arima_metrics = arima_model.evaluate(y_test_rf)  # Use unscaled test data for ARIMA
                
                logger.info("\nModel evaluation results for {}:".format(symbol))
                logger.info("Random Forest - RMSE: {:.4f}, R²: {:.4f}".format(
                    rf_metrics['rmse'], rf_metrics['r2']
                ))
                logger.info("LSTM - RMSE: {:.4f}, R²: {:.4f}".format(
                    lstm_metrics['rmse'], lstm_metrics['r2']
                ))
                logger.info("ARIMA - RMSE: {:.4f}, R²: {:.4f}".format(
                    arima_metrics['rmse'], arima_metrics['r2']
                ))
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                continue
        
        logger.info("\nModel training completed successfully")
        
    except Exception as e:
        logger.error(f"Error in training process: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        train_all_models()
    except Exception as e:
        logger.error("Training failed:", exc_info=True)
        raise 