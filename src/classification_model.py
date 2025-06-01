"""
Stock Direction Classification using Neural Networks

This module implements a neural network for classification tasks,
specifically for predicting stock movement direction (up/down/neutral).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import joblib
import os


class StockDirectionClassifier:
    """
    Neural Network model for stock direction prediction (classification).
    """
    
    def __init__(self, hidden_layers=[128, 64, 32], dropout_rate=0.3, learning_rate=0.001):
        """
        Initialize the Stock Direction Classifier.
        
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
        self.num_classes = 3  # Down, Neutral, Up
        self.class_names = ['Down', 'Neutral', 'Up']
        
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
        
        # Output layer for classification (3 classes)
        self.model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        # Compile the model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model Architecture:")
        self.model.summary()
        
    def prepare_targets(self, y):
        """
        Convert targets to categorical format.
        
        Args:
            y (np.array): Target labels (0, 1, 2)
            
        Returns:
            np.array: One-hot encoded targets
        """
        return to_categorical(y, num_classes=self.num_classes)
    
    def calculate_class_weights(self, y):
        """
        Calculate class weights for imbalanced datasets.
        
        Args:
            y (np.array): Target labels
            
        Returns:
            dict: Class weights
        """
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y),
            y=y
        )
        return dict(enumerate(class_weights))
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, verbose=1, use_class_weights=True):
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
            use_class_weights (bool): Whether to use class weights for imbalanced data
        """
        if self.model is None:
            self.build_model(X_train.shape[1])
        
        # Prepare targets
        y_train_cat = self.prepare_targets(y_train)
        y_val_cat = self.prepare_targets(y_val)
        
        # Calculate class weights if needed
        class_weights = None
        if use_class_weights:
            class_weights = self.calculate_class_weights(y_train)
            print(f"Class weights: {class_weights}")
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
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
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weights,
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
            np.array: Predicted class labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.model.predict(X)
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities.
        
        Args:
            X (np.array): Features for prediction
            
        Returns:
            np.array: Prediction probabilities for each class
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
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
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }
        
        print("=== Model Evaluation ===")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.6f}")
        
        # Detailed classification report
        print("\n=== Detailed Classification Report ===")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        return metrics
    
    def plot_training_history(self):
        """
        Plot training and validation accuracy/loss curves.
        """
        if self.history is None:
            raise ValueError("No training history available")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.history.history['loss'], label='Training Loss', color='blue')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss', color='red')
        ax1.set_title('Model Loss During Training')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(self.history.history['accuracy'], label='Training Accuracy', color='blue')
        ax2.plot(self.history.history['val_accuracy'], label='Validation Accuracy', color='red')
        ax2.set_title('Model Accuracy During Training')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def plot_confusion_matrix(self, X_test, y_test):
        """
        Plot confusion matrix.
        
        Args:
            X_test (np.array): Test features
            y_test (np.array): Test targets
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before plotting confusion matrix")
        
        y_pred = self.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
        return cm
    
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
            'is_trained': self.is_trained,
            'num_classes': self.num_classes,
            'class_names': self.class_names
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
            self.num_classes = config['num_classes']
            self.class_names = config['class_names']
            
            print(f"Model loaded from {filepath}")
            
        except Exception as e:
            print(f"Error loading model: {e}")


# Example usage and testing
if __name__ == "__main__":
    # This would typically be called from main.py
    print("Stock Direction Classifier module loaded successfully!")
    
    # Example of creating and displaying model architecture
    classifier = StockDirectionClassifier()
    # Note: build_model would be called automatically during training
    print("Model ready for training!") 