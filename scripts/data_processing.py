#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de Traitement des Données pour la Prédiction de Durée des Réclamations d'Assurance
Ce script gère l'acquisition, le nettoyage, et la transformation des données pour
le projet de prédiction de durée des réclamations d'assurance.
"""

import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from scripts import utils

# Téléchargement des ressources NLTK (à exécuter une seule fois)
nltk.download('punkt')
nltk.download('stopwords')


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Charger les jeux de données internes et externes
    
    Returns:
        Tuple contenant le DataFrame des réclamations et le DataFrame de StatCanada
    """
    print("Loading datasets...")

    # Chargement des données internes (réclamations d'assurance)
    try:
        # Paramètres mis à jour pour les versions récentes de pandas
        claims_data = pd.read_csv('data/MODELING_DATA.csv', delimiter=';', on_bad_lines='skip')
        print(f"Loaded claims data with {claims_data.shape[0]} rows and {claims_data.shape[1]} columns")
    except Exception as e:
        print(f"Error loading claims data: {e}")
        # Essayer des approches alternatives si la première méthode échoue
        try:
            # Lire le fichier ligne par ligne pour gérer les lignes problématiques
            with open('data/MODELING_DATA.csv', 'r', encoding='utf-8') as file:
                lines = file.readlines()

            # Analyser l'en-tête
            header = lines[0].strip().split(';')

            # Analyser les lignes de données, en ignorant celles qui posent problème
            data_rows = []
            for i, line in enumerate(lines[1:], 1):
                try:
                    values = line.strip().split(';')
                    # S'assurer que le nombre de champs est correct
                    if len(values) == len(header):
                        data_rows.append(values)
                    else:
                        print(f"Skipping line {i + 1}: expected {len(header)} fields but got {len(values)}")
                except Exception as e:
                    print(f"Error parsing line {i + 1}: {e}")

            # Créer le DataFrame
            claims_data = pd.DataFrame(data_rows, columns=header)
            print(f"Loaded claims data manually: {claims_data.shape[0]} rows, {claims_data.shape[1]} columns")
        except Exception as e2:
            print(f"Second attempt failed: {e2}")
            # Dernier recours: créer des données synthétiques pour les tests
            print("Creating synthetic test data")
            claims_data = create_synthetic_claims_data()

    # Convertir les colonnes numériques de chaînes de caractères aux types appropriés
    numerical_cols = ['An_Debut_Invalidite', 'Mois_Debut_Invalidite', 'Duree_Delai_Attente',
                      'Annee_Naissance', 'Salaire_Annuel', 'Duree_Invalidite']

    for col in numerical_cols:
        if col in claims_data.columns:
            claims_data[col] = pd.to_numeric(claims_data[col], errors='coerce')

    # Chargement des données externes (données de population de StatCanada)
    try:
        statcan_data = pd.read_csv('data/StatCanadaPopulationData.csv')
        print(f"Loaded StatCanada data with {statcan_data.shape[0]} rows and {statcan_data.shape[1]} columns")
    except Exception as e:
        print(f"Error loading StatCanada data: {e}")
        # Créer un DataFrame vide comme solution de repli
        statcan_data = pd.DataFrame()
        print("Created empty StatCanada DataFrame as fallback")

    return claims_data, statcan_data


def create_synthetic_claims_data() -> pd.DataFrame:
    """
    Créer des données synthétiques pour les tests lorsque les données réelles ne peuvent pas être chargées
    
    Returns:
        DataFrame avec des données synthétiques
    """
    print("Generating synthetic claims data for testing")
    np.random.seed(42)
    n_samples = 1000

    # Générer des données synthétiques
    data = {
        'An_Debut_Invalidite': np.random.randint(1995, 2007, n_samples),
        'Mois_Debut_Invalidite': np.random.randint(1, 13, n_samples),
        'Duree_Delai_Attente': np.random.choice([14, 90, 119, 180, 182], n_samples),
        'FSA': np.random.choice(['M5V', 'T2P', 'V6C'], n_samples),
        'Sexe': np.random.choice(['M', 'F'], n_samples),
        'Annee_Naissance': np.random.randint(1940, 1980, n_samples),
        'Code_Emploi': np.random.randint(1, 6, n_samples),
        'Description_Invalidite': np.random.choice([
            'MAJOR DEPRESSION', 'BACK PAIN', 'KNEE INJURY', 'CANCER',
            'HEART DISEASE', 'BROKEN ARM', 'ANXIETY DISORDER'
        ], n_samples),
        'Salaire_Annuel': np.random.normal(45000, 15000, n_samples).astype(int),
        'Duree_Invalidite': np.random.gamma(5, 40, n_samples).astype(int)
    }

    # Créer le DataFrame
    df = pd.DataFrame(data)

    # Sauvegarder les données synthétiques pour une utilisation future
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/synthetic_claims_data.csv', index=False)

    return df


def explore_data(claims_data: pd.DataFrame, statcan_data: pd.DataFrame) -> pd.DataFrame:
    """
    Effectuer une analyse exploratoire initiale des données
    
    Args:
        claims_data: DataFrame avec les données de réclamations
        statcan_data: DataFrame avec les données de StatCanada
        
    Returns:
        DataFrame des réclamations traité
    """
    print("\nExploring claims data:")
    print(claims_data.info())
    print("\nSample of claims data:")
    print(claims_data.head())

    print("\nExploring StatCanada data:")
    print(statcan_data.info())
    print("\nSample of StatCanada data:")
    print(statcan_data.head())

    # Vérifier les valeurs manquantes
    print("\nMissing values in claims data:")
    print(claims_data.isnull().sum())

    # Statistiques descriptives pour les variables numériques
    print("\nDescriptive statistics for numerical variables:")
    numerical_cols = ['Duree_Delai_Attente', 'Annee_Naissance', 'Salaire_Annuel', 'Duree_Invalidite']
    print(claims_data[numerical_cols].describe())

    # Créer la variable cible
    claims_data['Classe_Employe'] = (claims_data['Duree_Invalidite'] > 180).astype(int)
    print("\nClass distribution (Classe_Employe):")
    print(claims_data['Classe_Employe'].value_counts(normalize=True))

    # S'assurer que le répertoire de sortie existe
    os.makedirs('output', exist_ok=True)

    # Visualiser les corrélations
    plt.figure(figsize=(10, 8))
    correlation_matrix = claims_data[numerical_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Numerical Variables')
    plt.savefig('output/correlation_matrix.png')

    # Visualiser la distribution de la cible
    plt.figure(figsize=(8, 6))
    sns.histplot(claims_data['Duree_Invalidite'], bins=30, kde=True)
    plt.axvline(x=180, color='red', linestyle='--', label='Limite 180 jours')
    plt.title('Distribution de la durée d\'invalidité')
    plt.xlabel('Durée (jours)')
    plt.ylabel('Fréquence')
    plt.legend()
    plt.savefig('output/target_distribution.png')

    return claims_data


def clean_data(
        claims_data: pd.DataFrame,
        statcan_data: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Nettoyer et préparer les données
    
    Args:
        claims_data: DataFrame avec les données de réclamations
        statcan_data: DataFrame avec les données de StatCanada
        
    Returns:
        Tuple contenant le DataFrame des réclamations et le DataFrame de StatCanada nettoyés
    """
    print("\nCleaning data...")

    # Gestion des valeurs manquantes
    claims_data = claims_data.dropna(subset=['Duree_Invalidite'])  # Supprimer les enregistrements avec cible manquante

    # Gestion des autres valeurs manquantes avec imputation (dans le pipeline de prétraitement plus tard)

    # Calculer l'âge au moment de l'invalidité
    claims_data['Age'] = claims_data.apply(
        lambda row: utils.calculate_age(row['An_Debut_Invalidite'], row['Annee_Naissance']),
        axis=1
    )

    # Créer des catégories d'âge
    claims_data = utils.create_age_categories(claims_data)

    # Traiter les données textuelles (Description_Invalidite)
    claims_data['Description_Clean'] = claims_data['Description_Invalidite'].apply(utils.clean_text)

    # Créer la variable cible
    claims_data['Classe_Employe'] = (claims_data['Duree_Invalidite'] > 180).astype(int)

    return claims_data, statcan_data


def merge_datasets(claims_data: pd.DataFrame, statcan_data: pd.DataFrame) -> pd.DataFrame:
    """
    Fusionner les données de réclamations avec les données de StatCanada en utilisant FSA (code postal) comme clé
    
    Args:
        claims_data: DataFrame avec les données de réclamations
        statcan_data: DataFrame avec les données de StatCanada
        
    Returns:
        DataFrame fusionné
    """
    print("\nMerging datasets...")

    # Vérifier si les données StatCanada sont vides
    if statcan_data.empty:
        print("StatCanada data is empty. Skipping merge and returning claims data.")
        return claims_data

    # Renommer et sélectionner les colonnes pertinentes de StatCanada pour la fusion
    if 'Géo_Code' in statcan_data.columns and 'FSA' not in statcan_data.columns:
        statcan_data = statcan_data.rename(columns={'Géo_Code': 'FSA'})

    # Sélectionner les colonnes pertinentes des données StatCanada
    # Cette sélection doit être ajustée en fonction des colonnes réelles du jeu de données
    try:
        population_cols = [col for col in statcan_data.columns if 'population' in col.lower()]
        density_cols = [col for col in statcan_data.columns if 'densité' in col.lower() or 'density' in col.lower()]
        relevant_cols = ['FSA'] + population_cols + density_cols

        # Vérifier que la colonne FSA existe
        if 'FSA' not in statcan_data.columns:
            print("FSA column not found in StatCanada data. Skipping merge and returning claims data.")
            return claims_data

        statcan_subset = statcan_data[relevant_cols].drop_duplicates(subset=['FSA'])
    except Exception as e:
        print(f"Error in processing StatCanada data: {e}. Using a simplified approach.")
        if 'FSA' not in statcan_data.columns:
            print("FSA column not found in StatCanada data. Skipping merge and returning claims data.")
            return claims_data

        statcan_subset = statcan_data.copy()
        if statcan_subset.shape[1] > 1:
            statcan_subset = statcan_subset.iloc[:, :5]  # Prendre seulement les premières colonnes

    # Fusionner les jeux de données
    try:
        merged_data = pd.merge(claims_data, statcan_subset, on='FSA', how='left')
        print(f"Merged dataset has {merged_data.shape[0]} rows and {merged_data.shape[1]} columns")
    except Exception as e:
        print(f"Error during merge: {e}. Returning claims data without merge.")
        merged_data = claims_data

    return merged_data


def feature_engineering(merged_data: pd.DataFrame) -> pd.DataFrame:
    """
    Créer de nouvelles caractéristiques pour améliorer la performance du modèle
    
    Args:
        merged_data: DataFrame avec les données de réclamations et StatCanada fusionnées
        
    Returns:
        DataFrame avec les nouvelles caractéristiques
    """
    print("\nPerforming feature engineering...")

    # Make a copy to avoid modifying the original
    merged_data = merged_data.copy()

    # Transformations du salaire en utilisant les fonctions utils
    merged_data = utils.create_salary_categories(merged_data)

    # Créer des caractéristiques polynomiales pour l'âge
    merged_data['Age_Squared'] = merged_data['Age'] ** 2

    # Créer des caractéristiques saisonnières en utilisant les fonctions utils
    merged_data = utils.create_seasonal_features(merged_data)

    # Check if Description_Clean exists, if not create it
    if 'Description_Clean' not in merged_data.columns and 'Description_Invalidite' in merged_data.columns:
        merged_data['Description_Clean'] = merged_data['Description_Invalidite'].apply(utils.clean_text)

    # Caractéristiques textuelles simples (peut être étendu avec du NLP plus avancé)
    if 'Description_Clean' in merged_data.columns:
        merged_data['Description_Word_Count'] = merged_data['Description_Clean'].apply(lambda x: len(str(x).split()))
    else:
        print("Warning: 'Description_Clean' column not found. Skipping text feature creation.")
        merged_data['Description_Word_Count'] = 0  # Default value

    return merged_data


def preprocess_pipeline(
        merged_data: pd.DataFrame
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Créer des pipelines de prétraitement pour les caractéristiques numériques et catégorielles
    
    Args:
        merged_data: DataFrame avec les caractéristiques
        
    Returns:
        Tuple contenant le préprocesseur, la liste des caractéristiques numériques, et la liste des caractéristiques catégorielles
    """
    print("\nCreating preprocessing pipeline...")

    # Définir les types de caractéristiques
    numeric_features = [
        'Duree_Delai_Attente', 'Age', 'Salaire_Annuel', 'Salaire_Log',
        'Age_Squared', 'Description_Word_Count'
    ]

    categorical_features = [
        'Sexe', 'Code_Emploi', 'Age_Category', 'Salary_Category',
        'Is_Winter', 'Is_Summer'
    ]

    # Définir le prétraitement pour les caractéristiques numériques
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Définir le prétraitement pour les caractéristiques catégorielles
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combiner les étapes de prétraitement
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # S'assurer que toutes les colonnes nécessaires existent
    for col in numeric_features + categorical_features:
        if col not in merged_data.columns:
            print(f"Warning: Column {col} not found in data. Adding empty column.")
            merged_data[col] = np.nan

    return preprocessor, numeric_features, categorical_features


def split_data(
        merged_data: pd.DataFrame,
        preprocessor: ColumnTransformer
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """
    Diviser les données en ensembles d'entraînement et de test et prétraiter
    
    Args:
        merged_data: DataFrame avec les caractéristiques
        preprocessor: Transformateur de prétraitement
        
    Returns:
        Tuple contenant les caractéristiques d'entraînement, les caractéristiques de test, les étiquettes d'entraînement,
        les étiquettes de test, et les noms des caractéristiques
    """
    print("\nSplitting data into training and testing sets...")

    # S'assurer que les caractéristiques catégorielles requises existent
    required_categorical_features = [
        'Sexe', 'Code_Emploi', 'Age_Category', 'Salary_Category',
        'Is_Winter', 'Is_Summer'
    ]

    # Ajouter les colonnes manquantes avec des valeurs par défaut si elles n'existent pas
    for col in required_categorical_features:
        if col not in merged_data.columns:
            print(f"Warning: Required column {col} not found in data. Adding with default values.")
            if col == 'Age_Category':
                merged_data = utils.create_age_categories(merged_data)
            elif col == 'Salary_Category':
                merged_data = utils.create_salary_categories(merged_data)
            elif col in ['Is_Winter', 'Is_Summer']:
                merged_data = utils.create_seasonal_features(merged_data)
            else:
                merged_data[col] = 'Unknown'  # Valeur par défaut

    # Préparer la matrice de caractéristiques et le vecteur cible
    X = merged_data.drop(['Duree_Invalidite', 'Classe_Employe', 'Description_Invalidite'], axis=1, errors='ignore')
    y = merged_data['Classe_Employe']

    # Diviser les données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")

    # Sauvegarder les noms des colonnes pour l'interprétation ultérieure
    feature_names = X.columns.tolist()

    return X_train, X_test, y_train, y_test, feature_names


def main() -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, ColumnTransformer, List[str]]:
    """
    Fonction principale pour exécuter le pipeline de traitement des données
    
    Returns:
        Tuple contenant les caractéristiques d'entraînement, les caractéristiques de test, les étiquettes d'entraînement,
        les étiquettes de test, le préprocesseur, et les noms des caractéristiques
    """
    # Charger les données
    claims_data, statcan_data = load_data()

    # Explorer les données
    claims_data = explore_data(claims_data, statcan_data)

    # Nettoyer les données
    claims_data, statcan_data = clean_data(claims_data, statcan_data)

    # Fusionner les jeux de données
    merged_data = merge_datasets(claims_data, statcan_data)

    # Ingénierie des caractéristiques
    merged_data = feature_engineering(merged_data)

    # Créer le pipeline de prétraitement
    preprocessor, numeric_features, categorical_features = preprocess_pipeline(merged_data)

    # Diviser et prétraiter les données
    X_train, X_test, y_train, y_test, feature_names = split_data(merged_data, preprocessor)

    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs('data', exist_ok=True)

    print("\nData processing completed successfully!")

    return X_train, X_test, y_train, y_test, preprocessor, feature_names


if __name__ == "__main__":
    main()
