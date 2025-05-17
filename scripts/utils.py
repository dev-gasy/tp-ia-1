#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for the Insurance Claim Duration Prediction project.
This file contains common utility functions used across the project.
"""

import os
import pickle
import re
from typing import Dict, Any, Set

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)


# Download NLTK resources
def download_nltk_resources(quiet: bool = False) -> None:
    """
    Download required NLTK resources if not already available
    
    Args:
        quiet: If True, suppress download messages
    """
    nltk.download('punkt', quiet=quiet)
    nltk.download('stopwords', quiet=quiet)


# Text processing functions
def clean_text(text: Any) -> str:
    """
    Clean and normalize text data
    
    Args:
        text: Text to clean and normalize
        
    Returns:
        Cleaned and normalized text
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words: Set[str] = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    return ' '.join(tokens)


# Feature engineering functions
def calculate_age(year_disability: int, year_birth: int) -> int:
    """
    Calculate age at time of disability
    
    Args:
        year_disability: Year when disability started
        year_birth: Year of birth
        
    Returns:
        Age in years
    """
    return year_disability - year_birth


def create_age_categories(df: pd.DataFrame, age_col: str = 'Age') -> pd.DataFrame:
    """
    Create age categories based on age
    
    Args:
        df: DataFrame containing age column
        age_col: Name of the age column
        
    Returns:
        DataFrame with age category column added
    """
    bins = [0, 25, 35, 45, 55, 65, 100]
    labels = ['25 or younger', '26-35', '36-45', '46-55', '56-65', '66+']

    df['Age_Category'] = pd.cut(df[age_col], bins=bins, labels=labels, right=False)

    return df


def create_salary_categories(df: pd.DataFrame, salary_col: str = 'Salaire_Annuel') -> pd.DataFrame:
    """
    Create salary categories and logarithm of salary
    
    Args:
        df: DataFrame containing salary column
        salary_col: Name of the salary column
        
    Returns:
        DataFrame with salary features added
    """
    # Log transformation of salary
    df['Salaire_Log'] = np.log1p(df[salary_col])

    # Salary categories
    salary_bins = [0, 20000, 40000, 60000, 100000, float('inf')]
    salary_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    df['Salary_Category'] = pd.cut(df[salary_col], bins=salary_bins, labels=salary_labels)

    return df


def create_seasonal_features(df: pd.DataFrame, month_col: str = 'Mois_Debut_Invalidite') -> pd.DataFrame:
    """
    Create seasonal features based on month
    
    Args:
        df: DataFrame containing month column
        month_col: Name of the month column
        
    Returns:
        DataFrame with seasonal features added
    """
    # Winter flag (December, January, February)
    df['Is_Winter'] = ((df[month_col] >= 12) | (df[month_col] <= 2)).astype(int)

    # Summer flag (June, July, August)
    df['Is_Summer'] = ((df[month_col] >= 6) & (df[month_col] <= 8)).astype(int)

    return df


# Visualization functions
def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = 'Confusion Matrix',
        filename: str = 'output/confusion_matrix.png'
) -> Dict[str, int]:
    """
    Calculate and visualize confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title
        filename: Output filename for saved plot
        
    Returns:
        Dictionary with confusion matrix metrics
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Extract values from confusion matrix
    tn, fp, fn, tp = cm.ravel()

    # Calculate confusion matrix metrics
    metrics = {
        'True Positives (TP)': tp,  # Cas longs prédits longs
        'False Negatives (FN)': fn,  # Cas longs prédits courts
        'True Negatives (TN)': tn,  # Cas courts prédits courts
        'False Positives (FP)': fp  # Cas courts prédits longs
    }

    # Visualize confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Court (<= 180 jours)', 'Long (> 180 jours)'],
                yticklabels=['Court (<= 180 jours)', 'Long (> 180 jours)'],
                annot_kws={"size": 12})
    plt.xlabel('Prédiction', fontsize=12)
    plt.ylabel('Réalité', fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    return metrics


def plot_roc_curve(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        title: str = 'ROC Curve',
        filename: str = 'output/roc_curve.png'
) -> float:
    """
    Generate and visualize ROC curve
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities for positive class
        title: Plot title
        filename: Output filename for saved plot
        
    Returns:
        ROC AUC score
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    return roc_auc


# Performance metrics functions
def calculate_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate standard classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with classification metrics
    """
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Store in dictionary
    metrics = {
        'Exactitude (Accuracy)': accuracy,
        'Précision (Precision)': precision,
        'Rappel (Recall)': recall,
        'Score F1': f1
    }

    return metrics


def calculate_business_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate business impact metrics relevant to the case attribution problem
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with business metrics
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Calculate business metrics
    total = tn + fp + fn + tp

    # Cases correctly assigned to appropriate employee type
    correct_attribution = (tn + tp) / total

    # Long cases incorrectly assigned to inexperienced employees (biggest business problem)
    long_to_inexperienced = fn / (fn + tp) if (fn + tp) > 0 else 0

    # Short cases incorrectly assigned to experienced employees (resource waste)
    short_to_experienced = fp / (tn + fp) if (tn + fp) > 0 else 0

    # Store business metrics
    business_metrics = {
        'Taux d\'attribution correct': correct_attribution,
        'Cas longs attribués aux employés inexpérimentés': long_to_inexperienced,
        'Cas courts attribués aux employés expérimentés': short_to_experienced
    }

    return business_metrics


# Data loading and saving functions
def load_claims_data(filepath: str = 'data/MODELING_DATA.csv') -> pd.DataFrame:
    """
    Load the insurance claims dataset
    
    Args:
        filepath: Path to the claims data file
        
    Returns:
        DataFrame with claims data
    """
    return pd.read_csv(filepath, sep=';')


def load_statcan_data(filepath: str = 'data/StatCanadaPopulationData.csv') -> pd.DataFrame:
    """
    Load the StatCanada population dataset
    
    Args:
        filepath: Path to the StatCanada data file
        
    Returns:
        DataFrame with StatCanada data
    """
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        # Try different encodings if utf-8 fails
        df = pd.read_csv(filepath, encoding='latin1')

    return df


def save_model(model: Any, filepath: str = 'models/best_model.pkl') -> None:
    """
    Save model to disk
    
    Args:
        model: Trained model to save
        filepath: Path where model will be saved
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model saved to {filepath}")


def load_model(filepath: str = 'models/best_model.pkl') -> Any:
    """
    Load model from disk
    
    Args:
        filepath: Path to saved model
        
    Returns:
        Loaded model
    """
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded successfully from {filepath}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None


def plot_confusion_matrix(y_true, y_pred, classes, model_name, output_dir='output'):
    """
    Plot and save confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: Class names
        model_name: Name of the model (for file naming)
        output_dir: Directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    output_path = os.path.join(output_dir, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    plt.savefig(output_path)
    plt.close()


def plot_roc_curve(y_true, y_proba, model_name, output_dir='output'):
    """
    Plot and save ROC curve
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities for the positive class
        model_name: Name of the model (for file naming)
        output_dir: Directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")

    output_path = os.path.join(output_dir, f'roc_curve_{model_name.lower().replace(" ", "_")}.png')
    plt.savefig(output_path)
    plt.close()


def calculate_metrics(y_true, y_pred, y_proba=None):
    """
    Calculate classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities for the positive class (optional)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }

    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        metrics['roc_auc'] = auc(fpr, tpr)

    return metrics
