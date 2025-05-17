#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to generate test data for the Insurance Claim Duration Prediction project.
"""

import os

import numpy as np
import pandas as pd


def create_test_claims_data(n_samples=100, output_file=None):
    """
    Create test claims data for testing.
    
    Args:
        n_samples: Number of samples to generate
        output_file: Path to save the generated data
        
    Returns:
        DataFrame with synthetic claims data
    """
    np.random.seed(42)

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

    # Add derived features
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

    # Save data if output file is specified
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")

    return df


def create_test_statcan_data(output_file=None):
    """
    Create test StatCanada data for testing.
    
    Args:
        output_file: Path to save the generated data
        
    Returns:
        DataFrame with synthetic StatCanada data
    """
    # Create minimal StatCanada data for testing
    data = {
        'Province': ['Ontario', 'Quebec', 'British Columbia', 'Alberta', 'Manitoba'],
        'FSA_Prefix': ['M', 'H', 'V', 'T', 'R'],
        'Population': [100000, 80000, 70000, 60000, 50000],
        'Median_Age': [40, 42, 38, 36, 39],
        'Median_Income': [55000, 48000, 60000, 58000, 45000]
    }

    df = pd.DataFrame(data)

    # Save data if output file is specified
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")

    return df


if __name__ == '__main__':
    # Create test data directory
    test_data_dir = os.path.dirname(os.path.abspath(__file__))

    # Generate claims data
    claims_data = create_test_claims_data(
        n_samples=100,
        output_file=os.path.join(test_data_dir, 'test_claims_data.csv')
    )

    # Generate StatCanada data
    statcan_data = create_test_statcan_data(
        output_file=os.path.join(test_data_dir, 'test_statcan_data.csv')
    )

    print("Test data generation complete.")
