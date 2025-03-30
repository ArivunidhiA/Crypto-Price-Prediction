from flask import Flask, render_template, jsonify, request
from data.data_fetcher import CryptoDataFetcher
from data.preprocessor import DataPreprocessor
from models.random_forest import RandomForestModel
from models.lstm import LSTMModel
from models.arima import ARIMAModel
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta
from make_predictions import make_predictions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize components
data_fetcher = CryptoDataFetcher()
preprocessor = DataPreprocessor()
rf_model = RandomForestModel()
lstm_model = LSTMModel()
arima_model = ARIMAModel()

# Model paths
MODEL_DIR = 'models/saved'
os.makedirs(MODEL_DIR, exist_ok=True)

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predictions')
def get_predictions():
    """Get predictions for all cryptocurrencies"""
    try:
        predictions = make_predictions()
        return jsonify({
            'status': 'success',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data': predictions
        })
    except Exception as e:
        logger.error(f"Error getting predictions: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/latest_prices')
def get_latest_prices():
    """Get latest cryptocurrency prices"""
    try:
        latest_prices = data_fetcher.get_latest_prices()
        return jsonify(latest_prices)
    except Exception as e:
        logger.error(f"Error fetching latest prices: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/historical_data')
def get_historical_data():
    """Get historical cryptocurrency data"""
    try:
        symbol = request.args.get('symbol', 'BTC')
        start_date = request.args.get('start_date', 
                                   (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
        end_date = request.args.get('end_date', datetime.now().strftime('%Y-%m-%d'))
        
        data = data_fetcher.fetch_historical_data(start_date, end_date)
        if symbol in data:
            df = data[symbol]
            return jsonify({
                'dates': df.index.strftime('%Y-%m-%d').tolist(),
                'prices': df['Close'].tolist(),
                'volumes': df['Volume'].tolist()
            })
        else:
            return jsonify({'error': f'No data found for {symbol}'}), 404
            
    except Exception as e:
        logger.error(f"Error fetching historical data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make price predictions using all models"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'BTC')
        days = data.get('days', 30)
        
        # Fetch historical data
        historical_data = data_fetcher.fetch_historical_data()
        if symbol not in historical_data:
            return jsonify({'error': f'No data found for {symbol}'}), 404
            
        df = historical_data[symbol]
        
        # Prepare data for prediction
        X = preprocessor.prepare_prediction_data(df)
        
        # Get predictions from all models
        predictions = {}
        
        # Random Forest predictions
        rf_predictions = rf_model.predict(X)
        rf_intervals = rf_model.get_confidence_intervals(X)
        predictions['random_forest'] = {
            'predictions': rf_predictions.tolist(),
            'confidence_intervals': {
                'lower': rf_intervals['lower_bound'].tolist(),
                'upper': rf_intervals['upper_bound'].tolist()
            }
        }
        
        # LSTM predictions
        lstm_predictions = lstm_model.predict(X)
        lstm_intervals = lstm_model.get_confidence_intervals(X)
        predictions['lstm'] = {
            'predictions': lstm_predictions.tolist(),
            'confidence_intervals': {
                'lower': lstm_intervals['lower_bound'].tolist(),
                'upper': lstm_intervals['upper_bound'].tolist()
            }
        }
        
        # ARIMA predictions
        arima_predictions = arima_model.predict(steps=days)
        arima_intervals = arima_model.get_confidence_intervals(steps=days)
        predictions['arima'] = {
            'predictions': arima_predictions.tolist(),
            'confidence_intervals': {
                'lower': arima_intervals['lower_bound'].tolist(),
                'upper': arima_intervals['upper_bound'].tolist()
            }
        }
        
        return jsonify(predictions)
        
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/train_models', methods=['POST'])
def train_models():
    """Train all models with latest data"""
    try:
        # Fetch historical data
        historical_data = data_fetcher.fetch_historical_data()
        
        for symbol, df in historical_data.items():
            # Prepare data
            X_train, X_test, y_train, y_test = preprocessor.prepare_data(df)
            
            # Train Random Forest
            rf_model.train(X_train, y_train)
            rf_model.save_model(f'{MODEL_DIR}/rf_{symbol}.joblib')
            
            # Train LSTM
            lstm_model.train(X_train, y_train)
            lstm_model.save_model(f'{MODEL_DIR}/lstm_{symbol}.h5')
            
            # Train ARIMA
            arima_model.train(df['Close'])
            arima_model.save_model(f'{MODEL_DIR}/arima_{symbol}.joblib')
            
        return jsonify({'message': 'Models trained successfully'})
        
    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_metrics')
def get_model_metrics():
    """Get performance metrics for all models"""
    try:
        symbol = request.args.get('symbol', 'BTC')
        historical_data = data_fetcher.fetch_historical_data()
        
        if symbol not in historical_data:
            return jsonify({'error': f'No data found for {symbol}'}), 404
            
        df = historical_data[symbol]
        X_train, X_test, y_train, y_test = preprocessor.prepare_data(df)
        
        metrics = {}
        
        # Random Forest metrics
        rf_model.load_model(f'{MODEL_DIR}/rf_{symbol}.joblib')
        metrics['random_forest'] = rf_model.evaluate(X_test, y_test)
        
        # LSTM metrics
        lstm_model.load_model(f'{MODEL_DIR}/lstm_{symbol}.h5')
        metrics['lstm'] = lstm_model.evaluate(X_test, y_test)
        
        # ARIMA metrics
        arima_model.load_model(f'{MODEL_DIR}/arima_{symbol}.joblib')
        metrics['arima'] = arima_model.evaluate(y_test)
        
        return jsonify(metrics)
        
    except Exception as e:
        logger.error(f"Error getting model metrics: {str(e)}")
        return jsonify({'error': str(e)}), 500

def load_models(symbol):
    """Load trained models for a specific cryptocurrency"""
    try:
        # Create model instances
        rf_model = RandomForestModel()
        lstm_model = LSTMModel()
        arima_model = ARIMAModel()
        
        # Try to load the models
        try:
            rf_model.load_model(f'{MODEL_DIR}/rf_{symbol}.joblib')
        except:
            logger.warning(f"Could not load Random Forest model for {symbol}")
            
        try:
            lstm_model.load_model(f'{MODEL_DIR}/lstm_{symbol}.h5')
        except:
            logger.warning(f"Could not load LSTM model for {symbol}")
            
        try:
            arima_model.load_model(f'{MODEL_DIR}/arima_{symbol}.joblib')
        except:
            logger.warning(f"Could not load ARIMA model for {symbol}")
        
        return rf_model, lstm_model, arima_model
    except Exception as e:
        logger.error(f"Error in load_models for {symbol}: {str(e)}")
        raise

@app.route('/predict', methods=['POST'])
def predict_future():
    """Generate future price predictions with realistic movements"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'BTC')
        
        if not symbol:
            return jsonify({
                'status': 'error',
                'message': 'Symbol is required'
            }), 400

        try:
            # Get historical data for the last 60 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)
            
            historical_data = data_fetcher.fetch_historical_data(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            if symbol not in historical_data:
                return jsonify({
                    'status': 'error',
                    'message': f'No data available for {symbol}'
                }), 404

            df = historical_data[symbol].copy()
            
            # Ensure we have enough data
            if len(df) < 2:
                return jsonify({
                    'status': 'error',
                    'message': 'Insufficient price data for predictions'
                }), 400

            # Get the latest price
            latest_price = df['Close'].iloc[-1]
            
            # Calculate daily returns
            df['Returns'] = df['Close'].pct_change()
            returns = df['Returns'].dropna().values
            
            # Calculate statistics
            mean_return = float(np.mean(returns))
            std_return = float(np.std(returns))
            
            # Number of days to predict
            prediction_days = 30
            
            def generate_prediction(start_price, days, mean_ret, vol):
                prices = [float(start_price)]
                
                # Set reasonable bounds for daily returns
                min_return = max(np.percentile(returns, 1), -0.15)
                max_return = min(np.percentile(returns, 99), 0.15)
                
                for _ in range(days):
                    # Generate random return
                    daily_return = float(np.random.normal(mean_ret, vol))
                    daily_return = max(min(daily_return, max_return), min_return)
                    
                    # Calculate next price
                    next_price = prices[-1] * (1 + daily_return)
                    prices.append(float(next_price))
                
                return prices[1:]

            # Generate predictions
            conservative = generate_prediction(
                latest_price, 
                prediction_days,
                mean_return * 0.8,  # More conservative mean
                std_return * 0.7    # Lower volatility
            )
            
            moderate = generate_prediction(
                latest_price,
                prediction_days,
                mean_return,        # Historical mean
                std_return         # Historical volatility
            )
            
            aggressive = generate_prediction(
                latest_price,
                prediction_days,
                mean_return * 1.2,  # More aggressive mean
                std_return * 1.3    # Higher volatility
            )

            # Generate dates
            historical_dates = df.index.strftime('%Y-%m-%d').tolist()
            future_dates = [(end_date + timedelta(days=i)).strftime('%Y-%m-%d') 
                           for i in range(1, prediction_days + 1)]
            
            # Convert historical prices to list of floats
            historical_prices = df['Close'].values.tolist()

            return jsonify({
                'status': 'success',
                'data': {
                    'dates': historical_dates + future_dates,
                    'predictions': {
                        'conservative': historical_prices + conservative,
                        'moderate': historical_prices + moderate,
                        'aggressive': historical_prices + aggressive
                    },
                    'latest_price': float(latest_price),
                    'symbol': symbol
                }
            })

        except Exception as e:
            logger.error(f"Error in prediction process: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'Prediction error: {str(e)}'
            }), 500

    except Exception as e:
        logger.error(f"Error in request processing: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 