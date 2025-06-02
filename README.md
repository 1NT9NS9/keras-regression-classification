# Stock Market Neural Networks with Keras

A comprehensive machine learning project implementing neural networks for stock market analysis using Keras/TensorFlow.

## Project Overview

This project demonstrates two main neural network applications:

1. **Regression Model**: Predicts future stock prices based on historical data
2. **Classification Model**: Predicts stock movement direction (up/down/neutral)

## Features

- Data collection from Yahoo Finance using yfinance
- Technical indicators calculation
- Feature engineering for stock market data
- Neural network implementations for both regression and classification
- Model evaluation and visualization
- Comprehensive data preprocessing pipeline

## Project Structure

```
├── data/                   # Data storage directory
├── models/                 # Saved trained models
├── notebooks/              # Jupyter notebooks for exploration
├── src/                    # Source code
│   ├── data_processing.py  # Data collection and preprocessing
│   ├── regression_model.py # Stock price prediction model
│   ├── classification_model.py # Stock direction prediction model
│   └── utils.py           # Utility functions
├── requirements.txt        # Project dependencies
├── main.py                # Main execution script
└── README.md              # This file
```

## Installation

1. Clone this repository:
```bash
git clone <https://github.com/1NT9NS9/keras-regression-classification.git>
cd keras-regression-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start
```bash
python main.py
```

### Individual Components

1. **Data Processing**:
```python
from src.data_processing import StockDataProcessor
processor = StockDataProcessor()
data = processor.get_stock_data('AAPL', period='2y')
```

2. **Regression Model**:
```python
from src.regression_model import StockPricePredictor
predictor = StockPricePredictor()
predictor.train(data)
predictions = predictor.predict(test_data)
```

3. **Classification Model**:
```python
from src.classification_model import StockDirectionClassifier
classifier = StockDirectionClassifier()
classifier.train(data)
direction = classifier.predict(test_data)
```

## Model Performance

The models are evaluated using:
- **Regression**: MSE, MAE, R²
- **Classification**: Accuracy, Precision, Recall, F1-score

## Data Sources

- Yahoo Finance (via yfinance library)
- Technical indicators derived from price data

## Technical Indicators Used

- Simple Moving Averages (SMA)
- Exponential Moving Averages (EMA)
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- Bollinger Bands
- Volume indicators

## Neural Network Architectures

### Regression Model
- Dense layers with dropout for regularization
- ReLU activation functions
- Adam optimizer
- MSE loss function

### Classification Model
- Dense layers with dropout
- ReLU activation in hidden layers
- Softmax activation in output layer
- Adam optimizer
- Categorical crossentropy loss

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Disclaimer

This project is for educational purposes only. Do not use these models for actual trading decisions without proper validation and risk management.

## Future Enhancements

- LSTM/GRU implementations for time series analysis
- Ensemble methods
- Real-time data streaming
- Web interface for model interaction
- Additional technical indicators
- Portfolio optimization features 