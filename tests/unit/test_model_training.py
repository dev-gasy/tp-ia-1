#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for the model training module.
"""

import pytest
from sklearn.base import BaseEstimator

from scripts.model_training import (
    train_logistic_regression, train_decision_tree, evaluate_model, compare_models
)


class TestModelTraining:
    """Tests for model training functionality."""

    @pytest.mark.slow
    def test_train_logistic_regression(self, train_test_data, preprocessor):
        """Test training of logistic regression model."""
        X_train, _, y_train, _ = train_test_data

        # Train the model
        model = train_logistic_regression(X_train, y_train, preprocessor)

        # Check that a model is returned
        assert model is not None
        assert isinstance(model, BaseEstimator)

        # Check that the model can make predictions
        X_sample = X_train.iloc[:5]
        preds = model.predict(X_sample)
        assert len(preds) == len(X_sample)
        assert all(pred in [0, 1] for pred in preds)

    @pytest.mark.slow
    def test_train_decision_tree(self, train_test_data, preprocessor):
        """Test training of decision tree model."""
        X_train, _, y_train, _ = train_test_data

        # Train the model
        model = train_decision_tree(X_train, y_train, preprocessor)

        # Check that a model is returned
        assert model is not None
        assert isinstance(model, BaseEstimator)

        # Check that the model can make predictions
        X_sample = X_train.iloc[:5]
        preds = model.predict(X_sample)
        assert len(preds) == len(X_sample)
        assert all(pred in [0, 1] for pred in preds)

    @pytest.mark.slow
    def test_train_random_forest(self, train_test_data, preprocessor):
        """Test training of random forest model."""
        X_train, _, y_train, _ = train_test_data

        # Train the model with reduced hyperparameters for testing
        # We override the params to make tests faster
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline

        # Create a simplified pipeline for testing
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=10, max_depth=5, random_state=42))
        ])

        # Fit the model
        pipeline.fit(X_train, y_train)

        # Check that the model can make predictions
        X_sample = X_train.iloc[:5]
        preds = pipeline.predict(X_sample)
        assert len(preds) == len(X_sample)
        assert all(pred in [0, 1] for pred in preds)

    def test_evaluate_model(self, train_test_data, preprocessor):
        """Test model evaluation."""
        X_train, X_test, y_train, y_test = train_test_data

        # Create a simple model for testing
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline

        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=10, max_depth=3, random_state=42))
        ])

        # Train the model
        model.fit(X_train, y_train)

        # Evaluate the model
        results = evaluate_model(model, X_test, y_test, "RandomForest")

        # Check that results are returned
        assert isinstance(results, dict)
        assert "model_name" in results
        assert "accuracy" in results
        assert "precision" in results
        assert "recall" in results
        assert "f1_score" in results
        assert "training_time" in results

        # Check that metric values are in the expected range
        for metric in ["accuracy", "precision", "recall", "f1_score"]:
            assert 0 <= results[metric] <= 1

        # Check that confusion matrix metrics were calculated
        for metric in ["true_positives", "false_positives", "true_negatives", "false_negatives"]:
            assert metric in results

    def test_compare_models(self):
        """Test model comparison."""
        # Create sample model results
        model_results = [
            {
                "model_name": "LogisticRegression",
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.88,
                "f1_score": 0.85,
                "training_time": 1.5,
                "inference_time": 0.01
            },
            {
                "model_name": "RandomForest",
                "accuracy": 0.90,
                "precision": 0.92,
                "recall": 0.89,
                "f1_score": 0.90,
                "training_time": 3.0,
                "inference_time": 0.02
            }
        ]

        # Compare the models
        best_model = compare_models(model_results)

        # Check that a best model name is returned
        assert isinstance(best_model, str)
        assert best_model in ["LogisticRegression", "RandomForest"]

        # The model with the highest F1 score should be selected
        expected_best = max(model_results, key=lambda x: x["f1_score"])["model_name"]
        assert best_model == expected_best
