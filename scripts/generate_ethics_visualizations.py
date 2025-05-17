#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de génération des visualisations pour les aspects éthiques et de gouvernance
Ce script crée des exemples de visualisations pour illustrer les aspects éthiques 
dans le rapport.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Configurer le style des visualisations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Assurer que le répertoire de sortie existe
os.makedirs('output/ethics', exist_ok=True)


def generate_anonymization_example():
    """Génère une visualisation du processus d'anonymisation des données."""

    # Créer des données d'exemple
    data = {
        'Données originales': [
            'FSA: M5V 2B5',
            'Année de naissance: 1975',
            'Description: MAJOR DEPRESSION',
            'Sexe: F'
        ],
        'Données anonymisées': [
            'FSA: M**',
            'Année de naissance: 1970',
            'Description: ANONYMIZED_MEDICAL_CONDITION',
            'Sexe: F'
        ]
    }

    # Créer une figure
    fig, ax = plt.figure(figsize=(10, 6)), plt.subplot(111)

    # Cacher les axes
    ax.set_axis_off()

    # Créer une table
    table = ax.table(
        cellText=[[data['Données originales'][i], data['Données anonymisées'][i]] for i in
                  range(len(data['Données originales']))],
        colLabels=['Données originales', 'Données anonymisées'],
        loc='center',
        cellLoc='center',
        colWidths=[0.4, 0.4]
    )

    # Styliser la table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)

    # Colorer les cellules pour mettre en évidence les différences
    for i in range(len(data['Données originales'])):
        table[(i + 1, 0)].set_facecolor('#f0f0f0')
        table[(i + 1, 1)].set_facecolor('#e0f0ff')

    # Ajouter un titre
    plt.suptitle('Anonymisation des données sensibles', fontsize=16)
    plt.tight_layout()

    # Sauvegarder l'image
    plt.savefig('output/ethics/anonymization_example.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Visualisation d'anonymisation générée")


def generate_bias_analysis():
    """Génère une visualisation d'analyse de biais par groupe démographique."""

    # Créer des données synthétiques pour l'analyse de biais
    np.random.seed(42)

    categories = ['Femme', 'Homme']
    metrics = ['Précision', 'Rappel', 'Exactitude']

    # Valeurs légèrement différentes pour montrer de petits écarts
    values = {
        'Femme': {
            'Précision': 0.94 + np.random.normal(0, 0.02),
            'Rappel': 0.91 + np.random.normal(0, 0.02),
            'Exactitude': 0.93 + np.random.normal(0, 0.02)
        },
        'Homme': {
            'Précision': 0.91 + np.random.normal(0, 0.02),
            'Rappel': 0.94 + np.random.normal(0, 0.02),
            'Exactitude': 0.92 + np.random.normal(0, 0.02)
        }
    }

    # Créer un DataFrame pour le graphique
    data = []
    for cat in categories:
        for metric in metrics:
            data.append({
                'Groupe': cat,
                'Métrique': metric,
                'Valeur': values[cat][metric]
            })

    df = pd.DataFrame(data)

    # Créer le graphique
    plt.figure(figsize=(10, 6))
    chart = sns.barplot(x='Groupe', y='Valeur', hue='Métrique', data=df)

    # Ajouter des annotations
    for i, bar in enumerate(chart.patches):
        chart.annotate(f'{bar.get_height():.2f}',
                       (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       ha='center', va='bottom',
                       fontsize=9)

    # Ajouter un titre et ajuster la mise en page
    plt.title('Analyse comparative des performances par genre', fontsize=14)
    plt.xlabel('Genre')
    plt.ylabel('Valeur de la métrique')
    plt.ylim(0.8, 1.0)  # Zoom sur la région pertinente
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Métrique')
    plt.tight_layout()

    # Sauvegarder l'image
    plt.savefig('output/ethics/bias_analysis_sexe.png', dpi=300)
    plt.close()
    print("Visualisation d'analyse de biais générée")


def generate_shap_example():
    """Génère une visualisation SHAP fictive pour montrer l'importance des caractéristiques."""

    # Définir les caractéristiques et leurs valeurs SHAP
    features = [
        'Âge', 'Salaire', 'Délai d\'attente', 'Durée moyenne FSA',
        'Code Emploi 2', 'Code Emploi 4', 'Sexe_M', 'Saison_Hiver',
        'Densité Population', 'Revenu moyen région'
    ]

    # Valeurs SHAP moyennes (positives et négatives)
    shap_values = np.array([
        0.35, 0.28, -0.22, 0.18, 0.15, -0.14, 0.08, -0.07, 0.06, -0.05
    ])

    # Créer le graphique
    plt.figure(figsize=(12, 7))

    # Définir un colormap personnalisé (rouge pour négatif, bleu pour positif)
    colors = ['#ff4d4d', '#4d94ff']
    cmap = LinearSegmentedColormap.from_list('custom_diverging', colors, N=256)

    # Tracer les barres
    bars = plt.barh(features, shap_values, color=[cmap(0.8) if x > 0 else cmap(0.2) for x in shap_values])

    # Ajouter des annotations
    for bar in bars:
        width = bar.get_width()
        if width > 0:
            label_pos = width + 0.01
        else:
            label_pos = width - 0.06
        plt.text(label_pos, bar.get_y() + bar.get_height() / 2, f'{width:.2f}',
                 va='center', fontsize=10)

    # Ajouter une ligne verticale à zéro
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

    # Ajuster le graphique
    plt.title('Importance des caractéristiques (valeurs SHAP)', fontsize=16)
    plt.xlabel('Impact sur la prédiction (valeur SHAP)', fontsize=12)
    plt.ylabel('Caractéristique', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Sauvegarder l'image
    plt.savefig('output/ethics/shap_summary.png', dpi=300)
    plt.close()
    print("Visualisation SHAP générée")


def main():
    """Fonction principale pour générer toutes les visualisations."""
    print("Génération des visualisations pour le rapport d'éthique...")

    generate_anonymization_example()
    generate_bias_analysis()
    generate_shap_example()

    print("Génération des visualisations d'éthique terminée!")


if __name__ == "__main__":
    main()
