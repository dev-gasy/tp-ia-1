#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pytest configuration file with common fixtures.
"""

import os

# Import the NLTK mock setup function
from tests.mocks.nltk_mock import setup_nltk_mock

# Set up the comprehensive NLTK mock before any imports
setup_nltk_mock()

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Original mock fixture kept for compatibility but it's not strictly needed anymore
@pytest.fixture(scope="session", autouse=True)
def mock_nltk():
    """Mock NLTK to avoid requiring actual NLTK resources."""
    # Our setup_nltk_mock() has already installed all the mocks
    yield


@pytest.fixture(scope="session")
def test_data_dir():
    """Fixture for test data directory."""
    test_dir = os.path.join(os.path.dirname(__file__), "test_data")
    os.makedirs(test_dir, exist_ok=True)
    return test_dir


@pytest.fixture(scope="session")
def sample_claims_data():
    """Fixture for sample claims data."""
    np.random.seed(42)
    n_samples = 100

    # Generate synthetic data
    data = {
        'An_Debut_Invalidite': np.random.randint(2000, 2022, n_samples),
        'Mois_Debut_Invalidite': np.random.randint(1, 13, n_samples),
        'Duree_Delai_Attente': np.random.choice([14, 90, 119, 180, 182], n_samples),
        'FSA': np.random.choice(['M5V', 'T2P', 'V6C'], n_samples),
        'Sexe': np.random.choice(['M', 'F'], n_samples),
        'Annee_Naissance': np.random.randint(1950, 2000, n_samples),
        'Code_Emploi': np.random.randint(1, 6, n_samples),
        'Description_Invalidite': np.random.choice([
            'MAJOR DEPRESSION', 'BACK PAIN', 'KNEE INJURY'
        ], n_samples),
        'Salaire_Annuel': np.random.normal(45000, 15000, n_samples).astype(int),
        'Duree_Invalidite': np.random.choice([90, 120, 200, 250], n_samples)
    }

    df = pd.DataFrame(data)
    df['Classe_Employe'] = (df['Duree_Invalidite'] > 180).astype(int)

    return df


@pytest.fixture(scope="session")
def sample_statcan_data():
    """Fixture for sample StatCanada data."""
    # Create minimal StatCanada data for testing
    data = {
        'Province': ['Ontario', 'Quebec', 'British Columbia'],
        'FSA_Prefix': ['M', 'T', 'V'],
        'Population': [100000, 80000, 70000],
        'Median_Age': [40, 42, 38],
        'Median_Income': [55000, 48000, 60000]
    }

    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def preprocessor():
    """Fixture for a sample preprocessor."""
    # Define feature types
    numerical_features = ['Age', 'Salaire_Annuel', 'Duree_Delai_Attente', 'Mois_Debut_Invalidite']
    categorical_features = ['Sexe', 'Code_Emploi', 'Is_Winter', 'Is_Summer']

    # Create transformers
    numerical_transformer = Pipeline(steps=[
        ('imputer', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    return preprocessor


@pytest.fixture(scope="session")
def processed_sample_data(sample_claims_data):
    """Fixture for processed sample data."""
    df = sample_claims_data.copy()

    # Add additional columns needed for preprocessing
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

    return df


@pytest.fixture(scope="session")
def train_test_data(processed_sample_data):
    """Fixture for train/test split."""
    from sklearn.model_selection import train_test_split

    df = processed_sample_data.copy()
    X = df.drop(['Duree_Invalidite', 'Classe_Employe', 'Description_Invalidite'], axis=1, errors='ignore')
    y = df['Classe_Employe']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test
