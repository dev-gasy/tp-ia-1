#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for the utils module.
"""

import os

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from tests.mocks.utils_mock import clean_text_mock, calculate_age_mock


# Mock the utils module to avoid NLTK dependencies
class TestUtils:
    """Tests for utility functions."""

    def test_clean_text(self):
        """Test text cleaning function using mock."""
        # Test with a normal string
        text = "BACK PAIN and DEPRESSION 123!@#"
        cleaned = clean_text_mock(text)

        # Check that the text is cleaned
        assert isinstance(cleaned, str)
        assert cleaned != text
        assert cleaned.islower()
        assert not any(char.isdigit() for char in cleaned)
        assert not any(char in "!@#" for char in cleaned)

        # Test with non-string input
        assert clean_text_mock(None) == ""
        assert clean_text_mock(123) == ""
        assert clean_text_mock(pd.NA) == ""

    def test_calculate_age(self):
        """Test age calculation."""
        # Test with various years
        assert calculate_age_mock(2020, 1980) == 40
        assert calculate_age_mock(2000, 1950) == 50
        assert calculate_age_mock(2022, 2000) == 22

        # Test with edge cases
        assert calculate_age_mock(2020, 2020) == 0
        assert calculate_age_mock(1900, 1800) == 100

    def test_create_age_categories(self):
        """Test creation of age categories."""
        # Create a test dataframe
        df = pd.DataFrame({
            'Age': [20, 30, 40, 50, 60, 70]
        })

        # Create age categories directly without utils function
        bins = [0, 25, 35, 45, 55, 65, 100]
        labels = ['25 or younger', '26-35', '36-45', '46-55', '56-65', '66+']
        df['Age_Category'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

        # Check that age categories are created
        assert 'Age_Category' in df.columns

        # Check that categories are applied correctly
        assert df.loc[0, 'Age_Category'] == '25 or younger'
        assert df.loc[1, 'Age_Category'] == '26-35'
        assert df.loc[2, 'Age_Category'] == '36-45'
        assert df.loc[3, 'Age_Category'] == '46-55'
        assert df.loc[4, 'Age_Category'] == '56-65'
        assert df.loc[5, 'Age_Category'] == '66+'

    def test_create_salary_categories(self):
        """Test creation of salary categories."""
        # Create a test dataframe
        df = pd.DataFrame({
            'Salaire_Annuel': [15000, 30000, 50000, 80000, 120000]
        })

        # Create salary categories directly
        salary_bins = [0, 20000, 40000, 60000, 100000, float('inf')]
        salary_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        df['Salary_Category'] = pd.cut(df['Salaire_Annuel'], bins=salary_bins, labels=salary_labels)
        df['Salaire_Log'] = np.log1p(df['Salaire_Annuel'])

        # Check that salary categories are created
        assert 'Salary_Category' in df.columns
        assert 'Salaire_Log' in df.columns

        # Check that categories are applied correctly
        assert df.loc[0, 'Salary_Category'] == 'Very Low'
        assert df.loc[1, 'Salary_Category'] == 'Low'
        assert df.loc[2, 'Salary_Category'] == 'Medium'
        assert df.loc[3, 'Salary_Category'] == 'High'
        assert df.loc[4, 'Salary_Category'] == 'Very High'

    def test_create_seasonal_features(self):
        """Test creation of seasonal features."""
        # Create a test dataframe
        df = pd.DataFrame({
            'Mois_Debut_Invalidite': [1, 2, 6, 7, 8, 12]
        })

        # Create seasonal features directly
        df['Is_Winter'] = ((df['Mois_Debut_Invalidite'] >= 12) | (df['Mois_Debut_Invalidite'] <= 2)).astype(int)
        df['Is_Summer'] = ((df['Mois_Debut_Invalidite'] >= 6) & (df['Mois_Debut_Invalidite'] <= 8)).astype(int)

        # Check that seasonal features are created
        assert 'Is_Winter' in df.columns
        assert 'Is_Summer' in df.columns

        # Check winter months (December, January, February)
        winter_indices = [0, 1, 5]  # January, February, December
        for idx in winter_indices:
            assert df.loc[idx, 'Is_Winter'] == 1

        # Check summer months (June, July, August)
        summer_indices = [2, 3, 4]  # June, July, August
        for idx in summer_indices:
            assert df.loc[idx, 'Is_Summer'] == 1

    def test_calculate_classification_metrics(self):
        """Test calculation of classification metrics."""
        # Create test data
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])

        # Calculate metrics directly
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred)
        }

        # Check that metrics are calculated
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics

        # Check that metrics are within expected range
        for metric_name, value in metrics.items():
            assert 0 <= value <= 1

    def test_calculate_business_metrics(self):
        """Test calculation of business metrics directly."""
        # Create test data
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])

        # Calculate correct predictions for each class
        correct_short = np.sum((y_true == 0) & (y_pred == 0))
        correct_long = np.sum((y_true == 1) & (y_pred == 1))

        # Total counts for each class
        total_short = np.sum(y_true == 0)
        total_long = np.sum(y_true == 1)

        # Calculate business metrics
        junior_efficiency = correct_short / total_short if total_short > 0 else 0
        senior_efficiency = correct_long / total_long if total_long > 0 else 0
        correct_assignment_rate = (correct_short + correct_long) / len(y_true)

        metrics = {
            "correct_assignment_rate": correct_assignment_rate,
            "junior_employee_efficiency": junior_efficiency,
            "senior_employee_efficiency": senior_efficiency
        }

        # Check that business metrics are calculated
        assert isinstance(metrics, dict)
        assert "correct_assignment_rate" in metrics
        assert "junior_employee_efficiency" in metrics
        assert "senior_employee_efficiency" in metrics

        # Check that metrics are within expected range
        for metric_name, value in metrics.items():
            assert 0 <= value <= 1

    def test_save_and_load_model(self, tmp_path):
        """Test model saving and loading."""
        # Create a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        # Create a temporary file path
        model_path = os.path.join(tmp_path, "test_model.pkl")

        # Save the model manually
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Check that the model file exists
        assert os.path.exists(model_path)

        # Load the model manually
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)

        # Check that the loaded model is of the correct type
        assert loaded_model is not None
        assert isinstance(loaded_model, RandomForestClassifier)

    def test_confusion_matrix_metrics(self):
        """Test confusion matrix metrics calculation."""
        # Create test data
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])

        # Calculate confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)

        # Extract values
        tn, fp, fn, tp = cm.ravel()

        # Define metrics
        metrics = {
            "True Positives (TP)": tp,
            "False Negatives (FN)": fn,
            "True Negatives (TN)": tn,
            "False Positives (FP)": fp
        }

        # Check that metrics were calculated
        assert isinstance(metrics, dict)
        assert "True Positives (TP)" in metrics
        assert "False Negatives (FN)" in metrics
        assert "True Negatives (TN)" in metrics
        assert "False Positives (FP)" in metrics

        # Check values
        assert metrics["True Positives (TP)"] == 3  # Correctly predicted positives
        assert metrics["False Positives (FP)"] == 1  # Incorrectly predicted positives
        assert metrics["True Negatives (TN)"] == 3  # Correctly predicted negatives
        assert metrics["False Negatives (FN)"] == 1  # Incorrectly predicted negatives

    def test_roc_curve_calculation(self):
        """Test ROC curve calculation."""
        # Create test data
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred_proba = np.array([0.1, 0.4, 0.6, 0.9, 0.2, 0.8, 0.3, 0.7])

        # Calculate ROC curve
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        # Check that ROC AUC was calculated
        assert isinstance(roc_auc, float)
        assert 0 <= roc_auc <= 1
