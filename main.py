"""
Main execution script for Stock Market Neural Networks

This script demonstrates the complete pipeline for both regression and classification
tasks using neural networks with stock market data.
"""

import os
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append('src')

from data_processing import StockDataProcessor
from regression_model import StockPricePredictor
from classification_model import StockDirectionClassifier
from utils import (
    create_directories, plot_stock_data, print_data_summary,
    log_experiment, save_results_summary
)


def main():
    """
    Main function to run the complete pipeline.
    """
    print("="*80)
    print("STOCK MARKET NEURAL NETWORKS WITH KERAS")
    print("="*80)
    
    # Create necessary directories
    create_directories()
    
    # Configuration
    STOCK_SYMBOL = 'AAPL'  # Apple Inc.
    PERIOD = '2y'  # 2 years of data
    LOOKBACK_WINDOW = 10  # Days to look back for features
    
    print(f"\nConfiguration:")
    print(f"- Stock Symbol: {STOCK_SYMBOL}")
    print(f"- Data Period: {PERIOD}")
    print(f"- Lookback Window: {LOOKBACK_WINDOW} days")
    
    # Initialize data processor
    print("\n" + "="*50)
    print("STEP 1: DATA COLLECTION AND PROCESSING")
    print("="*50)
    
    processor = StockDataProcessor()
    
    # Fetch stock data
    print(f"\nFetching {STOCK_SYMBOL} stock data...")
    raw_data = processor.get_stock_data(STOCK_SYMBOL, period=PERIOD)
    
    if raw_data is None:
        print("Failed to fetch stock data. Exiting...")
        return
    
    print_data_summary(raw_data, "Raw Stock Data")
    
    # Add technical indicators
    print("\nCalculating technical indicators...")
    processed_data = processor.calculate_technical_indicators(raw_data)
    
    print_data_summary(processed_data, "Data with Technical Indicators")
    
    # Save processed data
    processed_data.to_csv('data/processed_stock_data.csv', index=False)
    print("Processed data saved to 'data/processed_stock_data.csv'")
    
    # Create interactive plot
    try:
        plot_stock_data(processed_data, STOCK_SYMBOL, 'data/stock_analysis.html')
    except Exception as e:
        print(f"Warning: Could not create interactive plot: {e}")
    
    # Prepare data for regression
    print("\n" + "="*50)
    print("STEP 2: REGRESSION MODEL (PRICE PREDICTION)")
    print("="*50)
    
    print("Preparing regression features...")
    X_reg, y_reg = processor.create_features_for_regression(
        processed_data, 
        lookback_window=LOOKBACK_WINDOW
    )
    
    print(f"Regression data shape: X={X_reg.shape}, y={y_reg.shape}")
    
    # Split and scale data for regression
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = processor.prepare_data_for_training(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    # Split training data for validation
    X_train_reg_split, X_val_reg, y_train_reg_split, y_val_reg = processor.prepare_data_for_training(
        X_train_reg, y_train_reg, test_size=0.2, random_state=42, scale_features=False
    )
    
    print(f"Training set: {X_train_reg_split.shape}")
    print(f"Validation set: {X_val_reg.shape}")
    print(f"Test set: {X_test_reg.shape}")
    
    # Initialize and train regression model
    print("\nInitializing regression model...")
    price_predictor = StockPricePredictor(
        hidden_layers=[128, 64, 32],
        dropout_rate=0.3,
        learning_rate=0.001
    )
    
    print("Training regression model...")
    price_predictor.train(
        X_train_reg_split, y_train_reg_split,
        X_val_reg, y_val_reg,
        epochs=100,
        batch_size=32,
        verbose=1
    )
    
    # Evaluate regression model
    print("\nEvaluating regression model...")
    regression_metrics = price_predictor.evaluate(X_test_reg, y_test_reg)
    
    # Plot training history and predictions
    try:
        price_predictor.plot_training_history()
        price_predictor.plot_predictions(X_test_reg, y_test_reg)
    except Exception as e:
        print(f"Warning: Could not create plots: {e}")
    
    # Save regression model
    price_predictor.save_model('models/stock_price_predictor')
    
    # Log regression experiment
    reg_params = {
        'model_type': 'regression',
        'hidden_layers': str(price_predictor.hidden_layers),
        'dropout_rate': price_predictor.dropout_rate,
        'learning_rate': price_predictor.learning_rate,
        'lookback_window': LOOKBACK_WINDOW
    }
    log_experiment('StockPricePredictor', regression_metrics, reg_params)
    
    # Prepare data for classification
    print("\n" + "="*50)
    print("STEP 3: CLASSIFICATION MODEL (DIRECTION PREDICTION)")
    print("="*50)
    
    print("Preparing classification features...")
    X_clf, y_clf = processor.create_features_for_classification(
        processed_data,
        lookback_window=LOOKBACK_WINDOW,
        threshold=0.02  # 2% threshold for direction classification
    )
    
    print(f"Classification data shape: X={X_clf.shape}, y={y_clf.shape}")
    
    # Check class distribution
    unique, counts = np.unique(y_clf, return_counts=True)
    class_dist = dict(zip(unique, counts))
    print(f"Class distribution: {class_dist}")
    
    # Split and scale data for classification
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = processor.prepare_data_for_training(
        X_clf, y_clf, test_size=0.2, random_state=42
    )
    
    # Split training data for validation
    X_train_clf_split, X_val_clf, y_train_clf_split, y_val_clf = processor.prepare_data_for_training(
        X_train_clf, y_train_clf, test_size=0.2, random_state=42, scale_features=False
    )
    
    print(f"Training set: {X_train_clf_split.shape}")
    print(f"Validation set: {X_val_clf.shape}")
    print(f"Test set: {X_test_clf.shape}")
    
    # Initialize and train classification model
    print("\nInitializing classification model...")
    direction_classifier = StockDirectionClassifier(
        hidden_layers=[128, 64, 32],
        dropout_rate=0.3,
        learning_rate=0.001
    )
    
    print("Training classification model...")
    direction_classifier.train(
        X_train_clf_split, y_train_clf_split,
        X_val_clf, y_val_clf,
        epochs=100,
        batch_size=32,
        verbose=1,
        use_class_weights=True
    )
    
    # Evaluate classification model
    print("\nEvaluating classification model...")
    classification_metrics = direction_classifier.evaluate(X_test_clf, y_test_clf)
    
    # Plot training history and confusion matrix
    try:
        direction_classifier.plot_training_history()
        direction_classifier.plot_confusion_matrix(X_test_clf, y_test_clf)
    except Exception as e:
        print(f"Warning: Could not create plots: {e}")
    
    # Save classification model
    direction_classifier.save_model('models/stock_direction_classifier')
    
    # Log classification experiment
    clf_params = {
        'model_type': 'classification',
        'hidden_layers': str(direction_classifier.hidden_layers),
        'dropout_rate': direction_classifier.dropout_rate,
        'learning_rate': direction_classifier.learning_rate,
        'lookback_window': LOOKBACK_WINDOW,
        'threshold': 0.02
    }
    log_experiment('StockDirectionClassifier', classification_metrics, clf_params)
    
    # Save comprehensive results summary
    print("\n" + "="*50)
    print("STEP 4: RESULTS SUMMARY")
    print("="*50)
    
    save_results_summary(regression_metrics, classification_metrics)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    print("\nModel Performance Summary:")
    print("-" * 40)
    print("Regression Model (Price Prediction):")
    for metric, value in regression_metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    print("\nClassification Model (Direction Prediction):")
    for metric, value in classification_metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    print("\nFiles Generated:")
    print("- data/processed_stock_data.csv")
    print("- data/stock_analysis.html")
    print("- models/stock_price_predictor.h5")
    print("- models/stock_direction_classifier.h5")
    print("- results/experiment_log.csv")
    print("- results/model_summary.txt")
    
    print("\nNext Steps:")
    print("1. Review the model performance metrics")
    print("2. Experiment with different hyperparameters")
    print("3. Try different stocks or time periods")
    print("4. Implement LSTM/GRU for better time series modeling")
    print("5. Add more technical indicators")
    
    print("\nDisclaimer:")
    print("These models are for educational purposes only.")
    print("Do not use for actual trading without proper validation!")


def demo_predictions():
    """
    Demonstrate making predictions with saved models.
    """
    print("\n" + "="*50)
    print("DEMO: MAKING PREDICTIONS WITH SAVED MODELS")
    print("="*50)
    
    # Check if models exist
    if not os.path.exists('models/stock_price_predictor.h5'):
        print("Models not found. Please run the main training first.")
        return
    
    # Initialize models
    price_predictor = StockPricePredictor()
    direction_classifier = StockDirectionClassifier()
    
    # Load trained models
    print("Loading trained models...")
    price_predictor.load_model('models/stock_price_predictor')
    direction_classifier.load_model('models/stock_direction_classifier')
    
    # Load test data (you would typically have new data here)
    processor = StockDataProcessor()
    data = processor.get_stock_data('AAPL', period='1mo')  # Last month
    
    if data is not None:
        processed_data = processor.calculate_technical_indicators(data)
        
        # Create features
        X_reg, y_reg = processor.create_features_for_regression(processed_data)
        X_clf, y_clf = processor.create_features_for_classification(processed_data)
        
        if len(X_reg) > 0:
            # Scale features (you would need to save and load the scaler in practice)
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_reg_scaled = scaler.fit_transform(X_reg)
            X_clf_scaled = scaler.fit_transform(X_clf)
            
            # Make predictions
            price_predictions = price_predictor.predict(X_reg_scaled[-5:])  # Last 5 predictions
            direction_predictions = direction_classifier.predict(X_clf_scaled[-5:])
            
            print(f"Recent price predictions: {price_predictions}")
            print(f"Recent direction predictions: {direction_predictions}")
            print("Direction: 0=Down, 1=Neutral, 2=Up")


if __name__ == "__main__":
    try:
        # Run main training pipeline
        main()
        
        # Optionally run prediction demo
        response = input("\nWould you like to run the prediction demo? (y/n): ")
        if response.lower() in ['y', 'yes']:
            demo_predictions()
            
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please check your internet connection and try again.")
    
    print("\nThank you for using the Stock Market Neural Networks project!") 