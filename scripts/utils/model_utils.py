#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model utility functions for the Insurance Claim Duration Prediction project.
"""

import os
import pickle
from typing import Any, Dict

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def save_model(model: Any, filepath: str = 'models/best_model.pkl') -> None:
    """
    Save a trained model to disk
    
    Args:
        model: Trained model to save
        filepath: Path where to save the model
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save model
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved successfully to {filepath}")


def load_model(filepath: str = 'models/best_model.pkl') -> Any:
    """
    Load a trained model from disk
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Loaded model
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Model loaded successfully from {filepath}")
    return model


def calculate_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate common classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with calculated metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    return metrics


def calculate_business_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate business-specific metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with calculated business metrics
    """
    # Count various types of predictions
    long_cases = sum(y_true == 1)
    short_cases = sum(y_true == 0)
    
    # Long cases misclassified as short (high business impact)
    long_as_short = sum((y_true == 1) & (y_pred == 0))
    
    # Short cases misclassified as long (moderate business impact)
    short_as_long = sum((y_true == 0) & (y_pred == 1))
    
    # Calculate business metrics
    metrics = {
        'long_cases_correct_rate': 1 - (long_as_short / long_cases if long_cases > 0 else 0),
        'short_cases_correct_rate': 1 - (short_as_long / short_cases if short_cases > 0 else 0),
        'overall_correct_attribution_rate': 1 - ((long_as_short + short_as_long) / len(y_true))
    }
    
    return metrics 