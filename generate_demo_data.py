#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demo Data Generator Script
This script creates demo data and model for testing the KPI calculation functionality.
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Create directories if they don't exist
for directory in ['data', 'output', 'models']:
    os.makedirs(directory, exist_ok=True)


# Generate synthetic data
def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)

    # Generate features
    features = {
        'An_Debut_Invalidite': np.random.randint(2000, 2023, n_samples),
        'Mois_Debut_Invalidite': np.random.randint(1, 13, n_samples),
        'Duree_Delai_Attente': np.random.randint(0, 200, n_samples),
        'FSA': np.random.choice(['H3Z', 'T2P', 'M5V', 'V6C'], n_samples),
        'Sexe': np.random.choice(['M', 'F'], n_samples),
        'Annee_Naissance': np.random.randint(1940, 2000, n_samples),
        'Code_Emploi': np.random.randint(1, 6, n_samples),
        'Salaire_Annuel': np.random.uniform(20000, 100000, n_samples),
        'Age': np.zeros(n_samples),
        'Age_Squared': np.zeros(n_samples),
        'Age_Category': np.random.choice(['25 or younger', '26-35', '36-45', '46-55', '56-65', '66+'], n_samples),
        'Salaire_Log': np.zeros(n_samples),
        'Salary_Category': np.random.choice(['Very Low', 'Low', 'Medium', 'High', 'Very High'], n_samples),
        'Is_Winter': np.random.randint(0, 2, n_samples),
        'Is_Summer': np.random.randint(0, 2, n_samples),
        'Description_Word_Count': np.random.randint(1, 15, n_samples)
    }

    # Create DataFrame
    df = pd.DataFrame(features)

    # Calculate age and salaire_log
    df['Age'] = df['An_Debut_Invalidite'] - df['Annee_Naissance']
    df['Age_Squared'] = df['Age'] ** 2
    df['Salaire_Log'] = np.log1p(df['Salaire_Annuel'])

    # Generate target
    probabilities = 1 / (1 + np.exp(-(df['Age'] / 10 - 4 + np.random.normal(0, 1, n_samples))))
    df['Classe_Employe'] = (probabilities > 0.5).astype(int)

    # Generate Duration based on Class
    short_durations = np.random.randint(30, 180, n_samples)
    long_durations = np.random.randint(181, 1000, n_samples)
    df['Duree_Invalidite'] = np.where(df['Classe_Employe'] == 0, short_durations, long_durations)

    # Add text description field
    descriptions = [
        "LOWER BACK PAIN", "DEPRESSION", "ANXIETY", "BROKEN LEG", "FLU",
        "HEART ATTACK", "STROKE", "CANCER", "DIABETES", "PREGNANCY"
    ]
    df['Description_Invalidite'] = np.random.choice(descriptions, n_samples)

    return df


# Train a simple model
def train_demo_model(df):
    # Features and target
    X = df.drop(['Duree_Invalidite', 'Classe_Employe', 'Description_Invalidite'], axis=1)
    y = df['Classe_Employe']

    # Convert categorical features to dummy variables
    X = pd.get_dummies(X, drop_first=True)

    # Create pipeline with preprocessing and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Train model
    pipeline.fit(X, y)

    return pipeline, X.columns.tolist()


if __name__ == "__main__":
    print("Generating synthetic data...")
    df = generate_synthetic_data()

    # Save processed data
    df.to_csv('data/processed_data.csv', index=False)
    print(f"Saved synthetic data to data/processed_data.csv with {len(df)} samples")

    # Train model
    print("\nTraining demo model...")
    model, feature_names = train_demo_model(df)

    # Debug: Print feature names used by the model
    print("\nFeature names used by the model:")
    print(feature_names)

    # Save model
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Saved demo model to models/best_model.pkl")

    print("\nDemo data generation completed!")
