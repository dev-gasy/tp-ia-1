#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script d'analyse des données pour le projet de prédiction de durée d'invalidité
Ce script génère des visualisations et des analyses des données pour le rapport
"""

# Set matplotlib to use French locale
import locale
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

try:
    locale.setlocale(locale.LC_ALL, 'fr_FR.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_ALL, 'fr_FR')
    except:
        pass  # If French locale is not available, use default

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# Make sure the output directory exists
os.makedirs('output', exist_ok=True)


def load_data():
    """
    Charge les données pour l'analyse
    """
    try:
        data = pd.read_csv('data/processed_data.csv')
        print(f"Données chargées: {len(data)} lignes, {len(data.columns)} colonnes")
        return data
    except Exception as e:
        print(f"Erreur lors du chargement des données: {str(e)}")
        return None


def analyze_data_types(data):
    """
    Analyse et affiche les types de données
    """
    # Get data types
    data_types = data.dtypes

    # Count by type
    type_counts = data_types.value_counts()

    # Print summary
    print("\nDistribution des types de données:")
    print(type_counts)

    # Identify categorical and numerical columns
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

    print(f"\nColonnes catégorielles ({len(categorical_cols)}):")
    print(categorical_cols)

    print(f"\nColonnes numériques ({len(numerical_cols)}):")
    print(numerical_cols)

    return categorical_cols, numerical_cols


def plot_correlation_quantitative(data, numerical_cols):
    """
    Génère une matrice de corrélation pour les variables quantitatives
    """
    # Select only numerical columns
    num_data = data[numerical_cols]

    # Calculate correlation
    corr = num_data.corr()

    # Plot correlation matrix
    plt.figure(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, annot=True, fmt=".2f", annot_kws={"size": 8})

    plt.title('Matrice de Corrélation - Variables Quantitatives', fontsize=16)
    plt.tight_layout()
    plt.savefig('output/correlation_quantitative.png', dpi=300)
    print("Matrice de corrélation pour variables quantitatives enregistrée")
    return corr


def plot_correlation_qualitative(data, categorical_cols):
    """
    Génère une matrice de corrélation pour les variables qualitatives
    en utilisant le coefficient de Cramer's V
    """
    # Create a copy of the data with only categorical columns
    cat_data = data[categorical_cols].copy()

    # Encode categorical variables
    encoders = {}
    for col in cat_data.columns:
        encoders[col] = LabelEncoder()
        cat_data[col] = encoders[col].fit_transform(cat_data[col].astype(str))

    # Function to calculate Cramer's V
    def cramers_v(x, y):
        from scipy.stats import chi2_contingency
        confusion_matrix = pd.crosstab(x, y)
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

    # Calculate Cramer's V for each pair of categorical variables
    n_cols = len(categorical_cols)
    cramer_matrix = np.zeros((n_cols, n_cols))

    for i in range(n_cols):
        for j in range(i, n_cols):
            if i == j:
                cramer_matrix[i, j] = 1
            else:
                cramer_v = cramers_v(cat_data[categorical_cols[i]], cat_data[categorical_cols[j]])
                cramer_matrix[i, j] = cramer_v
                cramer_matrix[j, i] = cramer_v

    # Create a dataframe from the matrix
    cramer_df = pd.DataFrame(cramer_matrix, index=categorical_cols, columns=categorical_cols)

    # Plot the matrix
    plt.figure(figsize=(14, 10))
    mask = np.triu(np.ones_like(cramer_df, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(cramer_df, mask=mask, cmap=cmap, vmax=1, vmin=0, center=0.5,
                square=True, linewidths=.5, annot=True, fmt=".2f", annot_kws={"size": 8})

    plt.title('Matrice de Corrélation (Cramer\'s V) - Variables Qualitatives', fontsize=16)
    plt.tight_layout()
    plt.savefig('output/correlation_qualitative.png', dpi=300)
    print("Matrice de corrélation pour variables qualitatives enregistrée")
    return cramer_df


def plot_correlation_all(data, target='Classe_Employe'):
    """
    Génère une matrice de corrélation pour toutes les variables,
    en convertissant les catégorielles en dummies
    """
    # Get dummies for categorical variables (one-hot encoding)
    data_encoded = pd.get_dummies(data)

    # Calculate correlation
    corr = data_encoded.corr()

    # Sort by correlation with target if specified
    if target in corr.columns:
        corr_target = corr[target].abs().sort_values(ascending=False)
        print(f"\nTop 10 variables corrélées avec {target}:")
        print(corr_target.head(10))

    # Plot correlation matrix
    plt.figure(figsize=(20, 16))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # If there are too many variables, don't show annotations
    annot = len(corr.columns) <= 30

    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, annot=annot, fmt=".2f", annot_kws={"size": 7})

    plt.title('Matrice de Corrélation - Toutes Variables', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('output/correlation_all.png', dpi=300)
    print("Matrice de corrélation pour toutes les variables enregistrée")
    return corr


def main():
    """
    Fonction principale pour l'analyse des données
    """
    print("=" * 80)
    print("ANALYSE DES DONNÉES")
    print("=" * 80)

    # Load data
    data = load_data()
    if data is None:
        return

    # Analyze data types
    categorical_cols, numerical_cols = analyze_data_types(data)

    # Generate correlation matrices
    plot_correlation_quantitative(data, numerical_cols)
    plot_correlation_qualitative(data, categorical_cols)
    plot_correlation_all(data, target='Classe_Employe')

    print("\nAnalyse des données terminée!")


if __name__ == "__main__":
    main()
