#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for the data processing module.
"""

import numpy as np
import pandas as pd
import pytest

from scripts.data_processing import (
    clean_data, create_synthetic_claims_data,
    feature_engineering, preprocess_pipeline, split_data, merge_datasets
)


class TestDataProcessing:
    """Tests for data processing functionality."""

    def test_create_synthetic_claims_data(self):
        """Test creation of synthetic claims data."""
        df = create_synthetic_claims_data()

        # Check dataframe structure
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert df.shape[0] > 0
        assert df.shape[1] > 0

        # Check if required columns exist
        required_columns = [
            'An_Debut_Invalidite', 'Mois_Debut_Invalidite', 'Duree_Delai_Attente',
            'FSA', 'Sexe', 'Annee_Naissance', 'Code_Emploi',
            'Description_Invalidite', 'Salaire_Annuel', 'Duree_Invalidite'
        ]
        for col in required_columns:
            assert col in df.columns

    def test_clean_data(self, sample_claims_data, sample_statcan_data):
        """Test data cleaning process."""
        cleaned_claims, cleaned_statcan = clean_data(sample_claims_data, sample_statcan_data)

        # Check that cleaned dataframes are returned
        assert isinstance(cleaned_claims, pd.DataFrame)
        assert isinstance(cleaned_statcan, pd.DataFrame)

        # Check that no NaN values exist in essential columns
        essential_cols = [
            'An_Debut_Invalidite', 'Mois_Debut_Invalidite', 'Duree_Delai_Attente',
            'Sexe', 'Annee_Naissance', 'Code_Emploi', 'Salaire_Annuel', 'Duree_Invalidite'
        ]
        for col in essential_cols:
            if col in cleaned_claims.columns:
                assert cleaned_claims[col].isna().sum() == 0

    def test_feature_engineering(self, processed_sample_data):
        """Test feature engineering process."""
        # Test with a sample of processed data
        processed_df = processed_sample_data.copy()

        # Apply feature engineering
        enhanced_df = feature_engineering(processed_df)

        # Check that new features were created
        new_features = [
            'Age', 'Age_Category', 'Salary_Category', 'Is_Winter', 'Is_Summer'
        ]
        for feature in new_features:
            assert feature in enhanced_df.columns

        # Validate Age calculation
        assert 'Age' in enhanced_df.columns
        assert (enhanced_df['Age'] ==
                enhanced_df['An_Debut_Invalidite'] - enhanced_df['Annee_Naissance']).all()

        # Validate seasonal features
        winter_months = [12, 1, 2]
        summer_months = [6, 7, 8]

        winter_mask = enhanced_df['Mois_Debut_Invalidite'].isin(winter_months)
        summer_mask = enhanced_df['Mois_Debut_Invalidite'].isin(summer_months)

        assert (enhanced_df.loc[winter_mask, 'Is_Winter'] == 1).all()
        assert (enhanced_df.loc[summer_mask, 'Is_Summer'] == 1).all()

    def test_preprocess_pipeline(self, processed_sample_data):
        """Test preprocessing pipeline creation."""
        # Get a sample of processed data
        df = processed_sample_data.copy()

        # Create preprocessing pipeline
        preprocessor, num_features, cat_features = preprocess_pipeline(df)

        # Check that we get the expected output types
        assert preprocessor is not None
        assert isinstance(num_features, list)
        assert isinstance(cat_features, list)

        # Check that feature lists are not empty
        assert len(num_features) > 0
        assert len(cat_features) > 0

    def test_split_data(self, processed_sample_data, preprocessor):
        """Test train/test data splitting."""
        # Get a sample of processed data
        df = processed_sample_data.copy()

        # Split the data
        X_train, X_test, y_train, y_test, feature_names = split_data(df, preprocessor)

        # Check that we get the expected output types
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, np.ndarray) or isinstance(y_train, pd.Series)
        assert isinstance(y_test, np.ndarray) or isinstance(y_test, pd.Series)
        assert isinstance(feature_names, list)

        # Check that splits are not empty
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) > 0
        assert len(y_test) > 0

        # Check that the categorical columns exist (this was a bug fix)
        required_categorical_features = [
            'Age_Category', 'Sexe', 'Salary_Category'
        ]
        for col in required_categorical_features:
            assert col in X_train.columns or any(col in feat for feat in X_train.columns)

    def test_merge_datasets(self, sample_claims_data, sample_statcan_data):
        """Test dataset merging."""
        # Merge the datasets
        merged_df = merge_datasets(sample_claims_data, sample_statcan_data)

        # Check that a dataframe is returned
        assert isinstance(merged_df, pd.DataFrame)

        # The merged dataframe should have at least the same number of rows as claims_data
        assert merged_df.shape[0] >= sample_claims_data.shape[0]

        # Merged dataframe should have columns from both dataframes
        claims_cols = sample_claims_data.columns.tolist()
        for col in claims_cols:
            assert col in merged_df.columns
