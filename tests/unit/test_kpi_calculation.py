#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for the KPI calculation module.
"""

import os

import numpy as np
import pandas as pd
import pytest

from scripts.kpi_calculation import (
    calculate_confusion_matrix,
    calculate_key_metrics,
    calculate_class_distribution,
    calculate_business_impact,
    generate_roc_curve,
    generate_summary_report,
    generate_formatted_confusion_matrix,
    generate_performance_metrics_table,
    load_test_data
)


class TestKPICalculation:
    """Tests for KPI calculation functionality."""

    def test_calculate_confusion_matrix(self, tmp_path):
        """Test confusion matrix calculation."""
        # Create test data
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])

        # Create output directory
        output_dir = str(tmp_path)
        os.environ['OUTPUT_DIR'] = output_dir

        # Calculate confusion matrix
        metrics = calculate_confusion_matrix(y_true, y_pred)

        # Check that metrics are returned
        assert isinstance(metrics, dict)
        assert 'True Positives (TP)' in metrics
        assert 'False Positives (FP)' in metrics
        assert 'True Negatives (TN)' in metrics
        assert 'False Negatives (FN)' in metrics

        # Check metric values
        assert metrics['True Positives (TP)'] == 3  # Correctly predicted 1s
        assert metrics['False Positives (FP)'] == 1  # Incorrectly predicted 1s
        assert metrics['True Negatives (TN)'] == 3  # Correctly predicted 0s
        assert metrics['False Negatives (FN)'] == 1  # Incorrectly predicted 0s

        # Check that file was created
        assert os.path.exists(os.path.join(output_dir, 'kpi_confusion_matrix.png'))

    def test_calculate_key_metrics(self, tmp_path):
        """Test key metrics calculation."""
        # Create test data
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])

        # Create output directory
        output_dir = str(tmp_path)
        os.environ['OUTPUT_DIR'] = output_dir

        # Calculate key metrics
        metrics = calculate_key_metrics(y_true, y_pred)

        # Check that metrics are returned
        assert isinstance(metrics, dict)
        assert 'Exactitude (Accuracy)' in metrics
        assert 'Précision (Precision)' in metrics
        assert 'Rappel (Recall)' in metrics
        assert 'Score F1' in metrics

        # Check metric values (based on the test data)
        assert round(metrics['Exactitude (Accuracy)'], 2) == 0.75
        assert round(metrics['Précision (Precision)'], 2) == 0.75
        assert round(metrics['Rappel (Recall)'], 2) == 0.75
        assert round(metrics['Score F1'], 2) == 0.75

        # Check that file was created
        assert os.path.exists(os.path.join(output_dir, 'kpi_metrics.png'))

    def test_calculate_class_distribution(self, tmp_path):
        """Test class distribution calculation."""
        # Create test data
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 1])

        # Create output directory
        output_dir = str(tmp_path)
        os.environ['OUTPUT_DIR'] = output_dir

        # Calculate class distribution
        dist_df, error_rates = calculate_class_distribution(y_true, y_pred)

        # Check that a dataframe is returned
        assert isinstance(dist_df, pd.DataFrame)
        assert isinstance(error_rates, dict)

        # Check dataframe structure
        assert dist_df.shape == (2, 2)
        assert 'Réel' in dist_df.columns
        assert 'Prédit' in dist_df.columns

        # Check that error rates are calculated
        assert 'Taux d\'erreur global' in error_rates

        # Check that file was created
        assert os.path.exists(os.path.join(output_dir, 'class_distribution.png'))

    def test_calculate_business_impact(self, tmp_path):
        """Test business impact calculation."""
        # Create test data
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])

        # Create output directory
        output_dir = str(tmp_path)
        os.environ['OUTPUT_DIR'] = output_dir

        # Calculate business impact
        impact_metrics = calculate_business_impact(y_true, y_pred)

        # Check that metrics are returned
        assert isinstance(impact_metrics, dict)
        assert 'Taux d\'attribution correcte' in impact_metrics
        assert 'Efficacité des employés juniors' in impact_metrics
        assert 'Efficacité des employés seniors' in impact_metrics

        # Check that file was created
        assert os.path.exists(os.path.join(output_dir, 'business_impact.png'))

    def test_generate_roc_curve(self, tmp_path):
        """Test ROC curve generation."""
        # Create test data
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred_proba = np.array([0.1, 0.4, 0.6, 0.9, 0.2, 0.8, 0.3, 0.7])

        # Create output directory
        output_dir = str(tmp_path)
        os.environ['OUTPUT_DIR'] = output_dir

        # Generate ROC curve
        auc_score = generate_roc_curve(y_true, y_pred_proba)

        # Check that AUC score is returned
        assert isinstance(auc_score, float)
        assert 0 <= auc_score <= 1

        # Check that file was created
        assert os.path.exists(os.path.join(output_dir, 'roc_curve.png'))

    def test_generate_summary_report(self):
        """Test summary report generation."""
        # Create test metrics data
        metrics_dict = [
            {
                'model_name': 'LogisticRegression',
                'accuracy': 0.85,
                'precision': 0.82,
                'recall': 0.88,
                'f1_score': 0.85,
                'true_positives': 40,
                'false_positives': 10,
                'true_negatives': 30,
                'false_negatives': 5
            },
            {
                'model_name': 'RandomForest',
                'accuracy': 0.90,
                'precision': 0.92,
                'recall': 0.89,
                'f1_score': 0.90,
                'true_positives': 45,
                'false_positives': 5,
                'true_negatives': 35,
                'false_negatives': 5
            }
        ]

        # Generate summary report
        report_df = generate_summary_report(metrics_dict)

        # Check that a dataframe is returned
        assert isinstance(report_df, pd.DataFrame)

        # Check dataframe structure
        assert report_df.shape[0] == 2  # Two models
        assert 'Modèle' in report_df.columns
        assert 'Exactitude' in report_df.columns
        assert 'Précision' in report_df.columns
        assert 'Rappel' in report_df.columns
        assert 'Score F1' in report_df.columns

        # Check that model names are in the dataframe
        assert 'LogisticRegression' in report_df['Modèle'].values
        assert 'RandomForest' in report_df['Modèle'].values

    def test_generate_formatted_confusion_matrix(self):
        """Test formatted confusion matrix generation."""
        # Create test data
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])

        # Generate formatted confusion matrix
        cm_df = generate_formatted_confusion_matrix(y_true, y_pred)

        # Check that a dataframe is returned
        assert isinstance(cm_df, pd.DataFrame)

        # Check dataframe structure
        assert cm_df.shape == (2, 2)

        # Check column and index names
        assert "Prédiction: Cas Courts (≤ 180 jours)" in cm_df.columns
        assert "Prédiction: Cas Longs (> 180 jours)" in cm_df.columns
        assert "Réalité: Cas Courts (≤ 180 jours)" in cm_df.index
        assert "Réalité: Cas Longs (> 180 jours)" in cm_df.index

        # Check confusion matrix values
        assert cm_df.loc["Réalité: Cas Courts (≤ 180 jours)", "Prédiction: Cas Courts (≤ 180 jours)"] == 3
        assert cm_df.loc["Réalité: Cas Courts (≤ 180 jours)", "Prédiction: Cas Longs (> 180 jours)"] == 1
        assert cm_df.loc["Réalité: Cas Longs (> 180 jours)", "Prédiction: Cas Courts (≤ 180 jours)"] == 1
        assert cm_df.loc["Réalité: Cas Longs (> 180 jours)", "Prédiction: Cas Longs (> 180 jours)"] == 3

    def test_generate_performance_metrics_table(self):
        """Test performance metrics table generation."""
        # Create test data
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])

        # Generate performance metrics table
        metrics_df = generate_performance_metrics_table(y_true, y_pred)

        # Check that a dataframe is returned
        assert isinstance(metrics_df, pd.DataFrame)

        # Check dataframe structure - should have metrics as rows and values/descriptions as columns
        assert metrics_df.shape[0] >= 4  # At least 4 core metrics
        assert 'Métrique' in metrics_df.columns
        assert 'Valeur' in metrics_df.columns
        assert 'Description' in metrics_df.columns

        # Check that core metrics are included
        metric_names = metrics_df['Métrique'].tolist()
        assert 'Exactitude (Accuracy)' in metric_names
        assert 'Précision (Precision)' in metric_names
        assert 'Rappel (Recall)' in metric_names
        assert 'Score F1' in metric_names

    def test_load_test_data(self, processed_sample_data, test_data_dir):
        """Test loading of test data."""
        # Save processed data to test directory
        test_data_path = os.path.join(test_data_dir, 'test_processed_data.csv')
        processed_sample_data.to_csv(test_data_path, index=False)

        # Load test data
        X, y = load_test_data(test_data_path)

        # Check that data is loaded
        assert X is not None
        assert y is not None
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series) or isinstance(y, np.ndarray)

        # Check that required categorical columns exist
        required_categorical_features = [
            'Sexe', 'Age_Category', 'Salary_Category'
        ]
        for col in required_categorical_features:
            assert col in X.columns or any(col in feat for feat in X.columns)
