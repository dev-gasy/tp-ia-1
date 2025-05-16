#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Training Script for Insurance Claim Duration Prediction
This script implements and evaluates multiple classification models
to predict claim duration class (short vs long).
"""

import os
import time
from typing import Dict, List, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from scripts import utils
from scripts.data_processing import main as process_data


def train_logistic_regression(
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        preprocessor: Any
) -> BaseEstimator:
    """
    Train a logistic regression model with hyperparameter tuning
    
    Args:
        X_train: Training features
        y_train: Training labels
        preprocessor: Data preprocessor
        
    Returns:
        Trained model
    """
    print("\nTraining Logistic Regression model...")

    # Create a pipeline with preprocessing and model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42))
    ])

    # Define parameter grid for grid search
    param_grid = {
        'classifier__C': [0.01, 0.1, 1.0, 10.0],
        'classifier__solver': ['liblinear', 'saga'],
        'classifier__penalty': ['l1', 'l2']
    }

    # Perform grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )

    # Train the model
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Print results
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    print(f"Training time: {training_time:.2f} seconds")

    return grid_search.best_estimator_


def train_decision_tree(
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        preprocessor: Any
) -> BaseEstimator:
    """
    Train a decision tree model with hyperparameter tuning
    
    Args:
        X_train: Training features
        y_train: Training labels
        preprocessor: Data preprocessor
        
    Returns:
        Trained model
    """
    print("\nTraining Decision Tree model...")

    # Create a pipeline with preprocessing and model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(random_state=42))
    ])

    # Define parameter grid for grid search
    param_grid = {
        'classifier__max_depth': [None, 5, 10, 15, 20],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__criterion': ['gini', 'entropy']
    }

    # Perform grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )

    # Train the model
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Print results
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    print(f"Training time: {training_time:.2f} seconds")

    return grid_search.best_estimator_


def train_random_forest(
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        preprocessor: Any
) -> BaseEstimator:
    """
    Train a random forest model with hyperparameter tuning
    
    Args:
        X_train: Training features
        y_train: Training labels
        preprocessor: Data preprocessor
        
    Returns:
        Trained model
    """
    print("\nTraining Random Forest model...")

    # Create a pipeline with preprocessing and model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Define parameter grid for grid search
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }

    # Perform grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )

    # Train the model
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Print results
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    print(f"Training time: {training_time:.2f} seconds")

    return grid_search.best_estimator_


def train_neural_network(
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        preprocessor: Any
) -> BaseEstimator:
    """
    Train a neural network model with hyperparameter tuning
    
    Args:
        X_train: Training features
        y_train: Training labels
        preprocessor: Data preprocessor
        
    Returns:
        Trained model
    """
    print("\nTraining Neural Network model...")

    # Create a pipeline with preprocessing and model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', MLPClassifier(random_state=42, max_iter=1000))
    ])

    # Define parameter grid for grid search
    param_grid = {
        'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'classifier__activation': ['relu', 'tanh'],
        'classifier__alpha': [0.0001, 0.001, 0.01],
        'classifier__learning_rate': ['constant', 'adaptive']
    }

    # Perform grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )

    # Train the model
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Print results
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    print(f"Training time: {training_time:.2f} seconds")

    return grid_search.best_estimator_


def evaluate_model(
        model: BaseEstimator,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        model_name: str
) -> Dict[str, Union[str, float]]:
    """
    Evaluate model performance on test set
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
        
    Returns:
        Dictionary with model performance metrics
    """
    print(f"\nEvaluating {model_name} model...")

    # Ensure output directory exists
    os.makedirs('output', exist_ok=True)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics using utils function
    metrics = utils.calculate_classification_metrics(y_test, y_pred)

    # Print metrics
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")

    # Create safe filename from model name
    safe_name = model_name.replace(" ", "_").lower()

    # Generate confusion matrix using utils function
    classes = ['Court (<= 180 jours)', 'Long (> 180 jours)']
    utils.plot_confusion_matrix(
        y_test,
        y_pred,
        classes=classes,
        model_name=model_name
    )

    # Print classification report
    from sklearn.metrics import classification_report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Court (<= 180 jours)', 'Long (> 180 jours)']))

    # ROC curve
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Create ROC curve using utils function
        utils.plot_roc_curve(
            y_test,
            y_pred_proba,
            model_name=model_name
        )
        # Calculate ROC AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
    except:
        print("Could not generate ROC curve (model may not support predict_proba)")
        roc_auc = None

    # Create result dictionary
    result = {
        'model_name': model_name,
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1']
    }

    if roc_auc is not None:
        result['roc_auc'] = roc_auc

    return result


def compare_models(model_results: List[Dict[str, Union[str, float]]]) -> str:
    """
    Compare the performance of all models
    
    Args:
        model_results: List of dictionaries with model results
        
    Returns:
        Name of the best model
    """
    print("\nComparing model performance:")

    # Create comparison dataframe
    results_df = pd.DataFrame(model_results)
    print(results_df)

    # Ensure output directory exists
    os.makedirs('output', exist_ok=True)

    # Plot comparison
    plt.figure(figsize=(12, 8))
    metrics = ['accuracy', 'precision', 'recall', 'f1']

    # Set width of bars
    barWidth = 0.2

    # Set position of bars on X axis
    r = np.arange(len(results_df))

    # Make the plot
    for i, metric in enumerate(metrics):
        plt.bar(r + i * barWidth, results_df[metric], width=barWidth,
                edgecolor='grey', label=metric)

    # Add labels and title
    plt.xlabel('Models', fontweight='bold')
    plt.ylabel('Scores', fontweight='bold')
    plt.xticks(r + 0.3, results_df['model_name'])
    plt.title('Performance Comparison of Different Models')
    plt.legend()
    plt.savefig('output/model_comparison.png')

    # Find the best model based on F1 score
    best_model_idx = results_df['f1'].idxmax()
    best_model_name = results_df.loc[best_model_idx, 'model_name']
    print(f"\nBest model based on F1 score: {best_model_name}")

    return best_model_name


def main() -> BaseEstimator:
    """
    Main function to execute the model training and evaluation pipeline
    
    Returns:
        Trained best model
    """
    # Process data (calling the main function from data_processing.py)
    X_train, X_test, y_train, y_test, preprocessor, feature_names = process_data()

    # Train different models
    models: Dict[str, BaseEstimator] = {}

    # Logistic Regression
    models['logistic_regression'] = train_logistic_regression(X_train, y_train, preprocessor)

    # Decision Tree
    models['decision_tree'] = train_decision_tree(X_train, y_train, preprocessor)

    # Random Forest
    models['random_forest'] = train_random_forest(X_train, y_train, preprocessor)

    # Neural Network
    models['neural_network'] = train_neural_network(X_train, y_train, preprocessor)

    # Evaluate all models
    model_results = []
    for model_name, model in models.items():
        # Convert model_name to display format
        display_name = model_name.replace('_', ' ').title()
        model_result = evaluate_model(model, X_test, y_test, display_name)
        model_results.append(model_result)

    # Compare models
    best_model_name = compare_models(model_results)

    # Save the best model
    best_model_key = best_model_name.lower().replace(' ', '_')
    best_model = models[best_model_key]

    # Use utils function to save model
    utils.save_model(best_model, 'models/best_model.pkl')

    print(f"\nBest model ({best_model_name}) saved as 'models/best_model.pkl'")

    return best_model


if __name__ == "__main__":
    main()
