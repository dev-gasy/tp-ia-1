#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script d'Entraînement de Modèle pour la Prédiction de Durée des Réclamations d'Assurance
Ce script implémente et évalue plusieurs modèles de classification
pour prédire la classe de durée des réclamations (courte vs longue).
"""

import os
import time
from typing import Dict, List, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from scripts import utils
from scripts.data_processing import main as process_data


def train_logistic_regression(
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        preprocessor: Any
) -> BaseEstimator:
    """
    Entraîner un modèle de régression logistique avec optimisation des hyperparamètres
    
    Args:
        X_train: Caractéristiques d'entraînement
        y_train: Étiquettes d'entraînement
        preprocessor: Préprocesseur de données
        
    Returns:
        Modèle entraîné
    """
    print("\nTraining Logistic Regression model...")

    # Créer un pipeline avec prétraitement et modèle
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42))
    ])

    # Définir la grille de paramètres pour la recherche en grille
    param_grid = {
        'classifier__C': [0.01, 0.1, 1.0, 10.0],
        'classifier__solver': ['liblinear', 'saga'],
        'classifier__penalty': ['l1', 'l2']
    }

    # Effectuer la recherche en grille
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )

    # Entraîner le modèle
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Afficher les résultats
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    print(f"Training time: {training_time:.2f} seconds")

    return grid_search.best_estimator_


def train_decision_tree(
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        preprocessor: Any
) -> BaseEstimator:
    """
    Entraîner un modèle d'arbre de décision avec optimisation des hyperparamètres
    
    Args:
        X_train: Caractéristiques d'entraînement
        y_train: Étiquettes d'entraînement
        preprocessor: Préprocesseur de données
        
    Returns:
        Modèle entraîné
    """
    print("\nTraining Decision Tree model...")

    # Créer un pipeline avec prétraitement et modèle
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(random_state=42))
    ])

    # Définir la grille de paramètres pour la recherche en grille
    param_grid = {
        'classifier__max_depth': [None, 5, 10, 15, 20],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__criterion': ['gini', 'entropy']
    }

    # Effectuer la recherche en grille
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )

    # Entraîner le modèle
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Afficher les résultats
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    print(f"Training time: {training_time:.2f} seconds")

    return grid_search.best_estimator_


def train_random_forest(
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        preprocessor: Any
) -> BaseEstimator:
    """
    Entraîner un modèle de forêt aléatoire avec optimisation des hyperparamètres
    
    Args:
        X_train: Caractéristiques d'entraînement
        y_train: Étiquettes d'entraînement
        preprocessor: Préprocesseur de données
        
    Returns:
        Modèle entraîné
    """
    print("\nTraining Random Forest model...")

    # Créer un pipeline avec prétraitement et modèle
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Définir la grille de paramètres pour la recherche en grille
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }

    # Effectuer la recherche en grille
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )

    # Entraîner le modèle
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Afficher les résultats
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    print(f"Training time: {training_time:.2f} seconds")

    return grid_search.best_estimator_


def train_neural_network(
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        preprocessor: Any
) -> BaseEstimator:
    """
    Entraîner un modèle de réseau de neurones avec optimisation des hyperparamètres
    
    Args:
        X_train: Caractéristiques d'entraînement
        y_train: Étiquettes d'entraînement
        preprocessor: Préprocesseur de données
        
    Returns:
        Modèle entraîné
    """
    print("\nTraining Neural Network model...")

    # Créer un pipeline avec prétraitement et modèle
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', MLPClassifier(random_state=42, max_iter=1000))
    ])

    # Définir la grille de paramètres pour la recherche en grille
    param_grid = {
        'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'classifier__activation': ['relu', 'tanh'],
        'classifier__alpha': [0.0001, 0.001, 0.01],
        'classifier__learning_rate': ['constant', 'adaptive']
    }

    # Effectuer la recherche en grille
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )

    # Entraîner le modèle
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Afficher les résultats
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    print(f"Training time: {training_time:.2f} seconds")

    return grid_search.best_estimator_


def evaluate_model(
        model: BaseEstimator,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        model_name: str
) -> Dict[str, Union[str, float]]:
    """
    Évaluer la performance du modèle sur l'ensemble de test
    
    Args:
        model: Modèle entraîné
        X_test: Caractéristiques de test
        y_test: Étiquettes de test
        model_name: Nom du modèle
        
    Returns:
        Dictionnaire avec les métriques de performance du modèle
    """
    print(f"\nEvaluating {model_name} model...")

    # S'assurer que le répertoire de sortie existe
    os.makedirs('output', exist_ok=True)

    # Faire des prédictions
    y_pred = model.predict(X_test)

    # Calculer les métriques en utilisant la fonction utils
    metrics = utils.calculate_classification_metrics(y_test, y_pred)

    # Afficher les métriques
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")

    # Créer un nom de fichier sécurisé à partir du nom du modèle
    safe_name = model_name.replace(" ", "_").lower()

    # Générer la matrice de confusion en utilisant la fonction utils
    classes = ['Court (<= 180 jours)', 'Long (> 180 jours)']
    utils.plot_confusion_matrix(
        y_test,
        y_pred,
        classes=classes,
        model_name=model_name
    )

    # Afficher le rapport de classification
    from sklearn.metrics import classification_report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Court (<= 180 jours)', 'Long (> 180 jours)']))

    # Courbe ROC
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Créer la courbe ROC en utilisant la fonction utils
        utils.plot_roc_curve(
            y_test,
            y_pred_proba,
            model_name=model_name
        )
        # Calculer l'AUC ROC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
    except:
        print("Could not generate ROC curve (model may not support predict_proba)")
        roc_auc = None

    # Créer le dictionnaire de résultats
    result = {
        'model_name': model_name,
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1']
    }

    if roc_auc is not None:
        result['roc_auc'] = roc_auc

    return result


def compare_models(model_results: List[Dict[str, Union[str, float]]]) -> str:
    """
    Comparer la performance de tous les modèles
    
    Args:
        model_results: Liste de dictionnaires avec les résultats des modèles
        
    Returns:
        Nom du meilleur modèle
    """
    print("\nComparing model performance:")

    # Créer un DataFrame de comparaison
    results_df = pd.DataFrame(model_results)
    print(results_df)

    # S'assurer que le répertoire de sortie existe
    os.makedirs('output', exist_ok=True)

    # Tracer la comparaison
    plt.figure(figsize=(12, 8))
    metrics = ['accuracy', 'precision', 'recall', 'f1']

    # Définir la largeur des barres
    barWidth = 0.2

    # Définir la position des barres sur l'axe X
    r = np.arange(len(results_df))

    # Créer le graphique
    for i, metric in enumerate(metrics):
        plt.bar(r + i * barWidth, results_df[metric], width=barWidth,
                edgecolor='grey', label=metric)

    # Ajouter les étiquettes et le titre
    plt.xlabel('Models', fontweight='bold')
    plt.ylabel('Scores', fontweight='bold')
    plt.xticks(r + 0.3, results_df['model_name'])
    plt.title('Performance Comparison of Different Models')
    plt.legend()
    plt.savefig('output/model_comparison.png')

    # Trouver le meilleur modèle basé sur le score F1
    best_model_idx = results_df['f1'].idxmax()
    best_model_name = results_df.loc[best_model_idx, 'model_name']
    print(f"\nBest model based on F1 score: {best_model_name}")

    return best_model_name


def main() -> BaseEstimator:
    """
    Fonction principale pour exécuter le pipeline d'entraînement et d'évaluation du modèle
    
    Returns:
        Meilleur modèle entraîné
    """
    # Traiter les données (en appelant la fonction principale de data_processing.py)
    X_train, X_test, y_train, y_test, preprocessor, feature_names = process_data()

    # Entraîner différents modèles
    models: Dict[str, BaseEstimator] = {}

    # Régression Logistique
    models['logistic_regression'] = train_logistic_regression(X_train, y_train, preprocessor)

    # Arbre de Décision
    models['decision_tree'] = train_decision_tree(X_train, y_train, preprocessor)

    # Forêt Aléatoire
    models['random_forest'] = train_random_forest(X_train, y_train, preprocessor)

    # Réseau de Neurones
    models['neural_network'] = train_neural_network(X_train, y_train, preprocessor)

    # Évaluer tous les modèles
    model_results = []
    for model_name, model in models.items():
        # Convertir le nom du modèle au format d'affichage
        display_name = model_name.replace('_', ' ').title()
        model_result = evaluate_model(model, X_test, y_test, display_name)
        model_results.append(model_result)

    # Comparer les modèles
    best_model_name = compare_models(model_results)

    # Sauvegarder le meilleur modèle
    best_model_key = best_model_name.lower().replace(' ', '_')
    best_model = models[best_model_key]

    # Utiliser la fonction utils pour sauvegarder le modèle
    utils.save_model(best_model, 'models/best_model.pkl')

    print(f"\nBest model ({best_model_name}) saved as 'models/best_model.pkl'")

    return best_model


if __name__ == "__main__":
    main()
