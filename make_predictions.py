import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from data.data_fetcher import CryptoDataFetcher
from data.preprocessor import DataPreprocessor
from models.random_forest import RandomForestModel
from models.lstm import LSTMModel
from models.arima import ARIMAModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_models(symbol, model_dir='models/saved'):
    """Load trained models for a specific cryptocurrency"""
    rf_model = RandomForestModel()
    lstm_model = LSTMModel()
    arima_model = ARIMAModel()
    
    try:
        rf_model.load_model(f'{model_dir}/rf_{symbol}.joblib')
        lstm_model.load_model(f'{model_dir}/lstm_{symbol}.h5')
        arima_model.load_model(f'{model_dir}/arima_{symbol}.joblib')
        return rf_model, lstm_model, arima_model
    except Exception as e:
        logger.error(f"Error loading models for {symbol}: {str(e)}")
        raise

def make_predictions():
    """Make predictions using all trained models"""
    try:
        # Initialize components
        data_fetcher = CryptoDataFetcher()
        preprocessor = DataPreprocessor()
        
        # Get recent data for predictions
        end_date = datetime.now()
        # Get 60 days of data to ensure we have enough for technical indicators
        start_date = end_date - timedelta(days=60)
        
        # Force data refresh to get the latest prices
        historical_data = data_fetcher.fetch_historical_data(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            force_refresh=True  # Add this parameter to force fresh data
        )
        
        predictions = {}
        
        # Make predictions for each cryptocurrency
        for symbol, df in historical_data.items():
            logger.info(f"\nMaking predictions for {symbol}...")
            
            try:
                # Load models
                rf_model, lstm_model, arima_model = load_models(symbol)
                
                # Drop any rows with NaN values
                df = df.dropna()
                
                if len(df) < 10:  # Minimum sequence length
                    logger.error(f"Not enough data for {symbol} after removing NaN values")
                    continue
                
                # Get the latest actual price first
                latest_price = float(df['Close'].iloc[-1])
                
                # Prepare prediction data
                rf_data = preprocessor.prepare_prediction_data(df, for_sequence_model=False, symbol=f"{symbol}_rf")
                lstm_data = preprocessor.prepare_prediction_data(df, for_sequence_model=True, symbol=f"{symbol}_lstm")
                
                # Make predictions
                rf_pred = float(rf_model.predict(rf_data)[0])
                
                # For LSTM, we need to inverse transform the prediction
                lstm_pred_scaled = lstm_model.predict(lstm_data)[0][0]
                lstm_pred = float(preprocessor.inverse_transform_target(np.array([lstm_pred_scaled]))[0])
                
                arima_pred = float(arima_model.predict(steps=1)[0])
                
                # Ensure predictions are within realistic bounds of the latest price
                # Limit predictions to maximum 10% deviation from current price
                max_deviation = 0.10
                rf_pred = max(min(rf_pred, latest_price * (1 + max_deviation)), latest_price * (1 - max_deviation))
                lstm_pred = max(min(lstm_pred, latest_price * (1 + max_deviation)), latest_price * (1 - max_deviation))
                arima_pred = max(min(arima_pred, latest_price * (1 + max_deviation)), latest_price * (1 - max_deviation))
                
                # Calculate prediction errors
                rf_error = ((rf_pred - latest_price) / latest_price * 100)
                lstm_error = ((lstm_pred - latest_price) / latest_price * 100)
                arima_error = ((arima_pred - latest_price) / latest_price * 100)
                
                predictions[symbol] = {
                    'Latest Price': latest_price,
                    'Random Forest': {
                        'Prediction': rf_pred,
                        'Error %': rf_error
                    },
                    'LSTM': {
                        'Prediction': lstm_pred,
                        'Error %': lstm_error
                    },
                    'ARIMA': {
                        'Prediction': arima_pred,
                        'Error %': arima_error
                    }
                }
                
                logger.info(f"\nPredictions for {symbol}:")
                logger.info(f"Latest Price: ${latest_price:,.2f}")
                logger.info(f"Random Forest: ${rf_pred:,.2f} (Error: {rf_error:.2f}%)")
                logger.info(f"LSTM: ${lstm_pred:,.2f} (Error: {lstm_error:.2f}%)")
                logger.info(f"ARIMA: ${arima_pred:,.2f} (Error: {arima_error:.2f}%)")
                
            except Exception as e:
                logger.error(f"Error making predictions for {symbol}: {str(e)}")
                continue
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error in prediction process: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        predictions = make_predictions()
    except Exception as e:
        logger.error("Prediction failed:", exc_info=True)
        raise 