#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integration tests for the machine learning pipeline.
Tests how different components work together.
"""

import os
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from scripts import utils
from scripts.data_processing import (
    clean_data, feature_engineering, preprocess_pipeline, split_data
)
from scripts.kpi_calculation import (
    calculate_confusion_matrix, calculate_key_metrics, generate_summary_report
)
from scripts.model_training import (
    evaluate_model
)


class TestPipelineIntegration:
    """Integration tests for the ML pipeline."""

    @pytest.mark.slow
    def test_data_processing_to_model_training(self, sample_claims_data, sample_statcan_data):
        """Test integration of data processing and model training."""
        # Process the data
        cleaned_claims, cleaned_statcan = clean_data(sample_claims_data, sample_statcan_data)
        assert isinstance(cleaned_claims, pd.DataFrame)

        # Apply feature engineering
        enhanced_df = feature_engineering(cleaned_claims)
        assert isinstance(enhanced_df, pd.DataFrame)
        assert 'Age' in enhanced_df.columns
        assert 'Age_Category' in enhanced_df.columns
        assert 'Salary_Category' in enhanced_df.columns

        # Create preprocessing pipeline
        preprocessor, num_features, cat_features = preprocess_pipeline(enhanced_df)
        assert preprocessor is not None

        # Split the data
        X_train, X_test, y_train, y_test, feature_names = split_data(enhanced_df, preprocessor)
        assert len(X_train) > 0
        assert len(X_test) > 0

        # Train a simple logistic regression model for testing
        from sklearn.linear_model import LogisticRegression
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ])

        # Train the model
        pipeline.fit(X_train, y_train)

        # Make predictions
        y_pred = pipeline.predict(X_test)

        # Check that we get predictions
        assert len(y_pred) == len(y_test)
        assert all(pred in [0, 1] for pred in y_pred)

    def test_model_evaluation_to_kpi_calculation(self, train_test_data, preprocessor, tmp_path):
        """Test integration of model evaluation and KPI calculation."""
        X_train, X_test, y_train, y_test = train_test_data

        # Create a simple RandomForest model for testing
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=10, max_depth=3, random_state=42))
        ])

        # Train the model
        pipeline.fit(X_train, y_train)

        # Make predictions
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

        # Evaluate the model
        results = evaluate_model(pipeline, X_test, y_test, "RandomForest")
        assert isinstance(results, dict)

        # Setup output directory for KPI calculations
        output_dir = str(tmp_path)
        os.environ['OUTPUT_DIR'] = output_dir

        # Calculate confusion matrix
        cm_metrics = calculate_confusion_matrix(y_test, y_pred)
        assert isinstance(cm_metrics, dict)

        # Calculate key performance metrics
        kpi_metrics = calculate_key_metrics(y_test, y_pred)
        assert isinstance(kpi_metrics, dict)

        # Generate summary report
        metrics_list = [results]
        summary_df = generate_summary_report(metrics_list)
        assert isinstance(summary_df, pd.DataFrame)
        assert summary_df.shape[0] == 1

        # Check that KPI files were created
        assert os.path.exists(os.path.join(output_dir, 'kpi_confusion_matrix.png'))
        assert os.path.exists(os.path.join(output_dir, 'kpi_metrics.png'))

    def test_model_save_and_load(self, train_test_data, preprocessor, tmp_path):
        """Test model serialization and deserialization."""
        X_train, X_test, y_train, y_test = train_test_data

        # Create a simple model
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=10, max_depth=3, random_state=42))
        ])

        # Train the model
        pipeline.fit(X_train, y_train)

        # Make predictions before saving
        y_pred_before = pipeline.predict(X_test)

        # Save the model
        model_path = os.path.join(tmp_path, "test_integration_model.pkl")
        utils.save_model(pipeline, model_path)

        # Check that the model file exists
        assert os.path.exists(model_path)

        # Load the model
        loaded_pipeline = utils.load_model(model_path)

        # Check that the model is loaded correctly
        assert loaded_pipeline is not None
        assert isinstance(loaded_pipeline, Pipeline)

        # Make predictions after loading
        y_pred_after = loaded_pipeline.predict(X_test)

        # Check that predictions are the same
        assert np.array_equal(y_pred_before, y_pred_after)

    def test_end_to_end_simple_workflow(self, sample_claims_data, tmp_path):
        """Test a simplified end-to-end workflow."""
        # Create test directories
        model_dir = os.path.join(tmp_path, "models")
        os.makedirs(model_dir, exist_ok=True)

        output_dir = os.path.join(tmp_path, "output")
        os.makedirs(output_dir, exist_ok=True)

        os.environ['OUTPUT_DIR'] = output_dir

        # Step 1: Feature Engineering
        df = sample_claims_data.copy()
        df['Age'] = df['An_Debut_Invalidite'] - df['Annee_Naissance']
        df['Is_Winter'] = ((df['Mois_Debut_Invalidite'] >= 12) | (df['Mois_Debut_Invalidite'] <= 2)).astype(int)
        df['Is_Summer'] = ((df['Mois_Debut_Invalidite'] >= 6) & (df['Mois_Debut_Invalidite'] <= 8)).astype(int)

        # Create categorical features
        bins = [0, 25, 35, 45, 55, 65, 100]
        labels = ['25 or younger', '26-35', '36-45', '46-55', '56-65', '66+']
        df['Age_Category'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

        salary_bins = [0, 20000, 40000, 60000, 100000, float('inf')]
        salary_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        df['Salary_Category'] = pd.cut(df['Salaire_Annuel'], bins=salary_bins, labels=salary_labels)

        # Ensure target variable exists
        df['Classe_Employe'] = (df['Duree_Invalidite'] > 180).astype(int)

        # Step 2: Create preprocessing pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder

        numerical_features = ['Age', 'Salaire_Annuel', 'Duree_Delai_Attente']
        categorical_features = ['Sexe', 'Code_Emploi', 'Age_Category', 'Salary_Category', 'Is_Winter']

        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'
        )

        # Step 3: Split data
        from sklearn.model_selection import train_test_split

        X = df.drop(['Duree_Invalidite', 'Classe_Employe'], axis=1, errors='ignore')
        y = df['Classe_Employe']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Step 4: Train model
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=10, max_depth=3, random_state=42))
        ])

        pipeline.fit(X_train, y_train)

        # Step 5: Make predictions
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

        # Step 6: Calculate metrics
        metrics = utils.calculate_classification_metrics(y_test, y_pred)
        assert isinstance(metrics, dict)
        assert 0 <= metrics['accuracy'] <= 1

        # Step 7: Save model
        model_path = os.path.join(model_dir, "test_end_to_end_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(pipeline, f)

        # Step 8: Load model and verify it works
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)

        # Make predictions with loaded model
        loaded_pred = loaded_model.predict(X_test)
        assert np.array_equal(y_pred, loaded_pred)
