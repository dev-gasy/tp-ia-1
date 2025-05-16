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
from scripts.ethics_governance import EthicsGovernance


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
    Fonction principale pour entraîner et évaluer les modèles.
    
    Returns:
        Le meilleur modèle sélectionné
    """
    print("Starting model training and evaluation...")
    
    # Traiter les données
    X_train, X_test, y_train, y_test, preprocessor, feature_names = process_data()
    
    # Entraîner différents modèles
    models = []
    
    # 1. Régression Logistique
    logistic_model = train_logistic_regression(X_train, y_train, preprocessor)
    models.append(("Régression Logistique", logistic_model))
    
    # 2. Arbre de Décision
    decision_tree_model = train_decision_tree(X_train, y_train, preprocessor)
    models.append(("Arbre de Décision", decision_tree_model))
    
    # 3. Random Forest
    random_forest_model = train_random_forest(X_train, y_train, preprocessor)
    models.append(("Random Forest", random_forest_model))
    
    # 4. Réseau de Neurones
    neural_network_model = train_neural_network(X_train, y_train, preprocessor)
    models.append(("Réseau de Neurones", neural_network_model))
    
    # Évaluer et comparer les modèles
    model_results = []
    
    for name, model in models:
        result = evaluate_model(model, X_test, y_test, name)
        model_results.append(result)
    
    # Comparer les modèles et sélectionner le meilleur
    best_model_name = compare_models(model_results)
    
    # Trouver le meilleur modèle parmi ceux entraînés
    best_model = None
    for name, model in models:
        if name == best_model_name:
            best_model = model
            break
    
    if best_model is None:
        raise ValueError(f"Le modèle {best_model_name} n'a pas été trouvé parmi les modèles entraînés.")
    
    # Implémenter les considérations éthiques et de gouvernance
    print("\nExécution des analyses éthiques et de gouvernance...")
    
    # Initialiser le module d'éthique et de gouvernance
    ethics = EthicsGovernance(
        model=best_model,
        preprocessor=preprocessor,
        feature_names=feature_names,
        protected_attributes=['Sexe', 'Age_Category', 'FSA']
    )
    
    # Vérifier si les colonnes protégées existent dans les données de test
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    # Générer des prédictions pour les analyses d'éthique
    best_model_pipeline = best_model if isinstance(best_model, Pipeline) else Pipeline([('preprocessor', preprocessor), ('classifier', best_model)])
    y_pred = best_model_pipeline.predict(X_test)
    
    try:
        # Analyser les biais potentiels (si les attributs protégés sont présents)
        protected_cols = [col for col in ethics.protected_attributes if col in X_test_df.columns]
        if protected_cols:
            print(f"Analyse de biais sur les attributs protégés disponibles: {protected_cols}")
            bias_stats = ethics.detect_bias(X_test_df, y_test, y_pred)
            print("Analyse de biais terminée. Visualisations enregistrées dans output/ethics/")
        else:
            print("Aucun attribut protégé disponible dans les données pour l'analyse de biais")
        
        # Générer des explications pour les prédictions 
        try:
            shap_values = ethics.explain_predictions(X_test_df)
            print("Explications SHAP générées. Visualisations enregistrées dans output/ethics/")
        except Exception as e:
            print(f"Erreur lors de la génération des explications SHAP: {str(e)}")
        
        # Générer un rapport d'éthique complet
        report_path = ethics.generate_ethics_report()
        print(f"Rapport d'éthique généré: {report_path}")
        
    except Exception as e:
        print(f"Erreur lors de l'analyse éthique: {str(e)}")
    
    print("\nModel training and evaluation completed!")
    
    return best_model


if __name__ == "__main__":
    main()
