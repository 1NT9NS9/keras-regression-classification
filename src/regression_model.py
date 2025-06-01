"""
Stock Price Prediction using Neural Networks

This module implements a neural network for regression tasks,
specifically for predicting future stock prices.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import os


class StockPricePredictor:
    """
    Neural Network model for stock price prediction (regression).
    """
    
    def __init__(self, hidden_layers=[128, 64, 32], dropout_rate=0.3, learning_rate=0.001):
        """
        Initialize the Stock Price Predictor.
        
        Args:
            hidden_layers (list): List of neurons in each hidden layer
            dropout_rate (float): Dropout rate for regularization
            learning_rate (float): Learning rate for optimizer
        """
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        self.is_trained = False
        
    def build_model(self, input_shape):
        """
        Build the neural network architecture.
        
        Args:
            input_shape (int): Number of input features
        """
        self.model = keras.Sequential([
            layers.Input(shape=(input_shape,)),
            layers.Dense(self.hidden_layers[0], activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
        ])
        
        # Add hidden layers
        for neurons in self.hidden_layers[1:]:
            self.model.add(layers.Dense(neurons, activation='relu'))
            self.model.add(layers.BatchNormalization())
            self.model.add(layers.Dropout(self.dropout_rate))
        
        # Output layer for regression (single continuous value)
        self.model.add(layers.Dense(1, activation='linear'))
        
        # Compile the model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        print("Model Architecture:")
        self.model.summary()
        
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, verbose=1):
        """
        Train the neural network model.
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training targets
            X_val (np.array): Validation features
            y_val (np.array): Validation targets
            epochs (int): Maximum number of training epochs
            batch_size (int): Batch size for training
            verbose (int): Verbosity level
        """
        if self.model is None:
            self.build_model(X_train.shape[1])
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        print(f"Training on {X_train.shape[0]} samples, validating on {X_val.shape[0]} samples")
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_trained = True
        print("Training completed!")
        
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X (np.array): Features for prediction
            
        Returns:
            np.array: Predicted values
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.model.predict(X)
        return predictions.flatten()
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model performance on test data.
        
        Args:
            X_test (np.array): Test features
            y_test (np.array): Test targets
            
        Returns:
            dict: Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Calculate percentage error
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
        
        print("=== Model Evaluation ===")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.6f}")
        
        return metrics
    
    def plot_training_history(self):
        """
        Plot training and validation loss curves.
        """
        if self.history is None:
            raise ValueError("No training history available")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.history.history['loss'], label='Training Loss', color='blue')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss', color='red')
        ax1.set_title('Model Loss During Training')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MAE plot
        ax2.plot(self.history.history['mae'], label='Training MAE', color='blue')
        ax2.plot(self.history.history['val_mae'], label='Validation MAE', color='red')
        ax2.set_title('Model MAE During Training')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Mean Absolute Error')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def plot_predictions(self, X_test, y_test, n_samples=100):
        """
        Plot actual vs predicted values.
        
        Args:
            X_test (np.array): Test features
            y_test (np.array): Actual test targets
            n_samples (int): Number of samples to plot
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before plotting predictions")
        
        y_pred = self.predict(X_test)
        
        # Select subset for plotting
        indices = np.random.choice(len(y_test), min(n_samples, len(y_test)), replace=False)
        y_test_subset = y_test[indices]
        y_pred_subset = y_pred[indices]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot
        ax1.scatter(y_test_subset, y_pred_subset, alpha=0.6, color='blue')
        ax1.plot([y_test_subset.min(), y_test_subset.max()], 
                [y_test_subset.min(), y_test_subset.max()], 
                'r--', lw=2, label='Perfect Prediction')
        ax1.set_xlabel('Actual Prices')
        ax1.set_ylabel('Predicted Prices')
        ax1.set_title('Actual vs Predicted Stock Prices')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Time series plot
        ax2.plot(range(len(y_test_subset)), y_test_subset, label='Actual', color='blue', alpha=0.7)
        ax2.plot(range(len(y_pred_subset)), y_pred_subset, label='Predicted', color='red', alpha=0.7)
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Stock Price')
        ax2.set_title('Stock Price Prediction Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Calculate and display correlation
        correlation = np.corrcoef(y_test, y_pred)[0, 1]
        print(f"Correlation between actual and predicted values: {correlation:.4f}")
    
    def save_model(self, filepath):
        """
        Save the trained model.
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the Keras model
        self.model.save(f"{filepath}.h5")
        
        # Save model configuration
        config = {
            'hidden_layers': self.hidden_layers,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'is_trained': self.is_trained
        }
        
        joblib.dump(config, f"{filepath}_config.pkl")
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model.
        
        Args:
            filepath (str): Path to the saved model
        """
        try:
            # Load the Keras model
            self.model = keras.models.load_model(f"{filepath}.h5")
            
            # Load model configuration
            config = joblib.load(f"{filepath}_config.pkl")
            self.hidden_layers = config['hidden_layers']
            self.dropout_rate = config['dropout_rate']
            self.learning_rate = config['learning_rate']
            self.is_trained = config['is_trained']
            
            print(f"Model loaded from {filepath}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def feature_importance_analysis(self, X_test, y_test, feature_names=None):
        """
        Analyze feature importance using permutation importance.
        
        Args:
            X_test (np.array): Test features
            y_test (np.array): Test targets
            feature_names (list): Names of features
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before feature importance analysis")
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(X_test.shape[1])]
        
        # Baseline performance
        baseline_score = r2_score(y_test, self.predict(X_test))
        
        # Calculate permutation importance
        importance_scores = []
        
        for i in range(X_test.shape[1]):
            X_permuted = X_test.copy()
            # Shuffle the feature
            np.random.shuffle(X_permuted[:, i])
            
            # Calculate new score
            permuted_predictions = self.predict(X_permuted)
            permuted_score = r2_score(y_test, permuted_predictions)
            
            # Importance is the decrease in performance
            importance = baseline_score - permuted_score
            importance_scores.append(importance)
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_scores
        }).sort_values('Importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(20)
        sns.barplot(data=top_features, x='Importance', y='Feature', palette='viridis')
        plt.title('Feature Importance (Top 20)')
        plt.xlabel('Importance Score (Decrease in R²)')
        plt.tight_layout()
        plt.show()
        
        return importance_df


# Example usage and testing
if __name__ == "__main__":
    # This would typically be called from main.py
    print("Stock Price Predictor module loaded successfully!")
    
    # Example of creating and displaying model architecture
    predictor = StockPricePredictor()
    # Note: build_model would be called automatically during training
    print("Model ready for training!") 