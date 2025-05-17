#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to generate visualizations with updated font sizes.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14

# Make sure the output directory exists
os.makedirs('output', exist_ok=True)


def generate_synthetic_data(n_samples=100):
    """Generate synthetic data for visualization tests."""
    np.random.seed(42)

    # Generate synthetic data
    data = pd.DataFrame({
        'Age': np.random.normal(40, 10, n_samples).astype(int),
        'Age_Squared': np.random.normal(1600, 800, n_samples).astype(int),
        'Age_Category': np.random.choice(['Jeune', 'Moyen', 'Senior'], n_samples),
        'Salaire_Annuel': np.random.normal(45000, 15000, n_samples).astype(int),
        'Salaire_Log': np.random.normal(10.5, 0.5, n_samples),
        'Mois_Debut_Invalidite': np.random.randint(1, 13, n_samples),
        'An_Debut_Invalidite': np.random.randint(2015, 2023, n_samples),
        'Annee_Naissance': np.random.randint(1960, 2000, n_samples),
        'Duree_Delai_Attente': np.random.gamma(5, 2, n_samples).astype(int),
        'Duree_Invalidite': np.random.gamma(10, 30, n_samples).astype(int),
        'FSA': np.random.choice(['M5V', 'H2X', 'V6B', 'K1P'], n_samples),
        'Sexe': np.random.choice(['M', 'F'], n_samples),
        'Code_Emploi': np.random.choice(['A1', 'B2', 'C3', 'D4'], n_samples),
        'Salary_Category': np.random.choice(['Bas', 'Moyen', 'Haut'], n_samples),
        'Is_Winter': np.random.choice([0, 1], n_samples),
        'Is_Summer': np.random.choice([0, 1], n_samples),
        'Description_Word_Count': np.random.randint(5, 50, n_samples),
        'Description_Invalidite': np.random.choice(['Dépression', 'Anxiété', 'Blessure au dos', 'Fracture'], n_samples)
    })

    # Add a target variable (Classification)
    data['Classe_Employe'] = np.where(data['Duree_Invalidite'] > 180, 1, 0)

    return data


def generate_correlation_matrix(data, title, filename, annot_kws=None):
    """Generate and save a correlation matrix visualization."""
    # Calculate correlation matrix
    corr = data.corr()

    # Create figure
    plt.figure(figsize=(14, 12))

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set default annotation parameters if not provided
    if annot_kws is None:
        annot_kws = {"size": 10}

    # Create heatmap
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        vmin=-1,
        vmax=1,
        center=0,
        annot_kws=annot_kws
    )

    # Add title and adjust layout
    plt.title(title, fontsize=16)
    plt.tight_layout()

    # Save the figure
    plt.savefig(f"output/{filename}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Correlation matrix saved to output/{filename}.png")


def generate_confusion_matrix(title, filename):
    """Generate and save a confusion matrix visualization."""
    # Create synthetic prediction data
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create figure
    plt.figure(figsize=(8, 6))

    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=['Court (≤180 jours)', 'Long (>180 jours)'],
        yticklabels=['Court (≤180 jours)', 'Long (>180 jours)'],
        annot_kws={"size": 12}
    )

    # Add labels and title
    plt.xlabel('Prédiction', fontsize=12)
    plt.ylabel('Réalité', fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()

    # Save the figure
    plt.savefig(f"output/{filename}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to output/{filename}.png")


def generate_roc_curve(title, filename):
    """Generate and save a ROC curve visualization."""
    # Create synthetic ROC data
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_score = np.random.random(100)

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Create figure
    plt.figure(figsize=(8, 6))

    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # Set axis limits
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    # Add labels and title
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the figure
    plt.savefig(f"output/{filename}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to output/{filename}.png")


def main():
    """Generate all test visualizations."""
    print("Generating test visualizations with updated font sizes...")

    # Generate synthetic data
    data = generate_synthetic_data(n_samples=100)

    # Generate correlation matrix
    generate_correlation_matrix(
        data=data.select_dtypes(include=['int64', 'float64']),
        title="Matrice de Corrélation - Variables Quantitatives",
        filename="correlation_matrix_quantitative_test"
    )

    # Generate correlation matrix for all variables
    # First, encode categorical variables
    data_encoded = data.copy()
    for col in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data_encoded[col] = le.fit_transform(data[col])

    generate_correlation_matrix(
        data=data_encoded,
        title="Matrice de Corrélation - Toutes Variables",
        filename="correlation_matrix_test"
    )

    # Generate confusion matrix
    generate_confusion_matrix(
        title="Matrice de Confusion",
        filename="confusion_matrix_test"
    )

    # Generate ROC curve
    generate_roc_curve(
        title="Courbe ROC",
        filename="roc_curve_test"
    )

    print("All test visualizations generated successfully!")


if __name__ == "__main__":
    main()
