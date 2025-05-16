#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de comparaison avancée des modèles
Ce script génère des visualisations supplémentaires pour comparer les modèles
"""

# Set matplotlib to use French locale
import locale
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import cross_val_score

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

# Ensure output directory exists
os.makedirs('output', exist_ok=True)


def load_data():
    """
    Charge les données pour l'évaluation des modèles
    """
    try:
        data = pd.read_csv('data/processed_data.csv')
        print(f"Données chargées: {len(data)} lignes, {len(data.columns)} colonnes")

        # Split features and target
        X = data.drop(['Duree_Invalidite', 'Classe_Employe', 'Description_Invalidite'], axis=1, errors='ignore')
        y = data['Classe_Employe']

        # Process the data (one-hot encoding)
        X_processed = pd.get_dummies(X)

        return X_processed, y
    except Exception as e:
        print(f"Erreur lors du chargement des données: {str(e)}")
        return None, None


def train_and_evaluate_models(X, y, models_config, random_state=42):
    """
    Entraîne et évalue plusieurs modèles
    
    Args:
        X: Features
        y: Target
        models_config: Liste de configurations de modèles
        random_state: Graine aléatoire
        
    Returns:
        Dictionnaire avec les modèles entraînés et les résultats
    """
    results = []
    trained_models = {}

    for name, model_class, params in models_config:
        print(f"\nEntraînement du modèle: {name}")

        # Create a copy of params and add random_state if the model supports it
        model_params = params.copy()
        if 'random_state' not in params and hasattr(model_class, 'random_state'):
            model_params['random_state'] = random_state

        # Create model
        model = model_class(**model_params)

        # Train and evaluate model
        start_time = time.time()
        model.fit(X, y)
        training_time = time.time() - start_time

        # Cross-validation scores
        cv_start_time = time.time()
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
        cv_time = time.time() - cv_start_time

        # Predictions
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]

        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        # Generate confusion matrix
        cm = confusion_matrix(y, y_pred)

        # Store results
        result = {
            'name': name,
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cv_scores': cv_scores,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'training_time': training_time,
            'cv_time': cv_time,
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'confusion_matrix': cm
        }

        results.append(result)
        trained_models[name] = model

        print(f"  Exactitude: {accuracy:.4f}")
        print(f"  Précision: {precision:.4f}")
        print(f"  Rappel: {recall:.4f}")
        print(f"  Score F1: {f1:.4f}")
        print(f"  Moyenne CV (F1): {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
        print(f"  Temps d'entraînement: {training_time:.2f} secondes")
        print(f"  Temps de validation croisée: {cv_time:.2f} secondes")

    return results, trained_models


def plot_roc_comparison(results):
    """
    Génère un graphique de comparaison des courbes ROC pour tous les modèles
    """
    plt.figure(figsize=(10, 8))

    for result in results:
        name = result['name']
        fpr = result['fpr']
        tpr = result['tpr']
        roc_auc = result['roc_auc']

        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')

    # Plot diagonal line
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs')
    plt.ylabel('Taux de Vrais Positifs')
    plt.title('Comparaison des Courbes ROC')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('output/roc_comparison.png', dpi=300)
    print("Graphique de comparaison des courbes ROC enregistré")


def plot_metrics_comparison(results):
    """
    Génère un graphique de comparaison des métriques pour tous les modèles
    """
    # Create DataFrame from results
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metrics_data = []

    for result in results:
        metrics_data.append({
            'model': result['name'],
            'accuracy': result['accuracy'],
            'precision': result['precision'],
            'recall': result['recall'],
            'f1': result['f1']
        })

    metrics_df = pd.DataFrame(metrics_data)

    # Plot metrics comparison
    plt.figure(figsize=(14, 8))

    # Set width of bars
    barWidth = 0.2

    # Set positions of bars on X axis
    r = np.arange(len(metrics_df))

    # Plot bars
    for i, metric in enumerate(metrics):
        plt.bar(r + i * barWidth, metrics_df[metric], width=barWidth, label=metric)

    # Add xticks on the middle of the group bars
    plt.xlabel('Modèle', fontweight='bold')
    plt.ylabel('Score', fontweight='bold')
    plt.title('Comparaison des Métriques par Modèle')
    plt.xticks([r + barWidth * (len(metrics) - 1) / 2 for r in range(len(metrics_df))], metrics_df['model'])
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('output/metrics_comparison.png', dpi=300)
    print("Graphique de comparaison des métriques enregistré")


def plot_training_time_comparison(results):
    """
    Génère un graphique de comparaison des temps d'entraînement
    """
    # Prepare data
    models = [result['name'] for result in results]
    training_times = [result['training_time'] for result in results]
    cv_times = [result['cv_time'] for result in results]

    # Plot
    plt.figure(figsize=(12, 6))

    # Set width of bars
    barWidth = 0.4

    # Set positions of bars on X axis
    r1 = np.arange(len(models))
    r2 = [x + barWidth for x in r1]

    # Create bars
    plt.bar(r1, training_times, width=barWidth, label='Temps d\'entraînement')
    plt.bar(r2, cv_times, width=barWidth, label='Temps de validation croisée')

    # Add labels
    plt.xlabel('Modèle', fontweight='bold')
    plt.ylabel('Temps (secondes)', fontweight='bold')
    plt.title('Comparaison des Temps d\'Entraînement et de Validation')
    plt.xticks([r + barWidth / 2 for r in range(len(models))], models)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('output/training_time_comparison.png', dpi=300)
    print("Graphique de comparaison des temps d'entraînement enregistré")


def plot_cv_scores_comparison(results):
    """
    Génère un graphique de comparaison des scores de validation croisée
    """
    # Prepare data
    models = [result['name'] for result in results]
    cv_means = [result['cv_mean'] for result in results]
    cv_stds = [result['cv_std'] for result in results]

    # Plot
    plt.figure(figsize=(12, 6))
    plt.errorbar(models, cv_means, yerr=cv_stds, fmt='o')
    plt.axhline(y=np.max(cv_means), color='r', linestyle='--', alpha=0.5)

    # Add labels
    plt.xlabel('Modèle', fontweight='bold')
    plt.ylabel('Score F1 Moyen (Validation Croisée)', fontweight='bold')
    plt.title('Comparaison des Scores de Validation Croisée')
    plt.ylim(min(cv_means) - 0.1, max(cv_means) + 0.1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('output/cv_scores_comparison.png', dpi=300)
    print("Graphique de comparaison des scores de validation croisée enregistré")


def create_radar_chart(results):
    """
    Génère un graphique radar pour comparer les modèles sur plusieurs métriques
    """
    # Prepare data
    metrics = ['accuracy', 'precision', 'recall', 'f1']

    # Extract metrics for each model
    model_data = {}
    for result in results:
        model_data[result['name']] = [
            result['accuracy'],
            result['precision'],
            result['recall'],
            result['f1']
        ]

    # Create radar chart
    num_models = len(results)
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for i, (model_name, values) in enumerate(model_data.items()):
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, label=model_name)
        ax.fill(angles, values, alpha=0.1)

    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)

    # Add grid and legend
    ax.grid(True)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Comparaison des Modèles sur Plusieurs Métriques', size=15, y=1.1)
    plt.tight_layout()
    plt.savefig('output/radar_chart_comparison.png', dpi=300)
    print("Graphique radar de comparaison enregistré")


def create_detailed_comparison_table(results):
    """
    Crée un tableau détaillé de comparaison des modèles
    """
    # Create data for DataFrame
    comparison_data = []

    # Fill data
    for result in results:
        comparison_data.append({
            'Modèle': result['name'],
            'Exactitude': result['accuracy'],
            'Précision': result['precision'],
            'Rappel': result['recall'],
            'Score F1': result['f1'],
            'AUC': result['roc_auc'],
            'CV Score Moyen': result['cv_mean'],
            'CV Écart-type': result['cv_std'],
            'Temps d\'entraînement (s)': result['training_time'],
            'Temps de validation (s)': result['cv_time']
        })

    # Create DataFrame
    comparison_df = pd.DataFrame(comparison_data)

    # Create an HTML representation for the report
    html_table = comparison_df.to_html(index=False, float_format=lambda x: f"{x:.4f}")
    with open('output/detailed_comparison.html', 'w') as f:
        f.write(html_table)

    print("Tableau détaillé de comparaison enregistré")

    return comparison_df


def main():
    """
    Fonction principale
    """
    print("=" * 80)
    print("COMPARAISON AVANCÉE DES MODÈLES")
    print("=" * 80)

    # Load data
    X, y = load_data()
    if X is None or y is None:
        return

    # Define models to compare
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier

    models_config = [
        ('Régression Logistique', LogisticRegression, {'C': 1.0, 'solver': 'liblinear'}),
        ('Arbre de Décision', DecisionTreeClassifier, {'max_depth': 10}),
        ('Random Forest', RandomForestClassifier, {'n_estimators': 100, 'max_depth': 10}),
        ('Réseau de Neurones', MLPClassifier, {'hidden_layer_sizes': (50,), 'max_iter': 1000})
    ]

    # Train and evaluate models
    results, trained_models = train_and_evaluate_models(X, y, models_config)

    # Generate visualizations
    plot_roc_comparison(results)
    plot_metrics_comparison(results)
    plot_training_time_comparison(results)
    plot_cv_scores_comparison(results)
    create_radar_chart(results)

    # Create detailed comparison table
    detailed_comparison = create_detailed_comparison_table(results)
    print("\nTableau de comparaison détaillé:")
    print(detailed_comparison)

    print("\nComparaison avancée des modèles terminée!")


if __name__ == "__main__":
    main()
