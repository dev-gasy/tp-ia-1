#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Génère un graphique radar comparant les performances des différents modèles.
Module optimisé suivant le principe DRY (Don't Repeat Yourself).
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class RadarChartGenerator:
    """Classe pour générer des graphiques radar comparant des modèles."""

    def __init__(self, output_dir: str = 'output'):
        """
        Initialise le générateur de graphiques radar.

        Args:
            output_dir: Répertoire de sortie pour les images générées
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Configuration du style de visualisation
        self._setup_visualization_style()
        
        # Données de performance des modèles (basées sur le tableau du rapport)
        self.models = [
            'Régression Logistique',
            'Arbre de Décision',
            'Random Forest',
            'Réseau de Neurones'
        ]
        
        # Métriques (exactitude, précision, rappel, score F1)
        self.metrics = ['Exactitude', 'Précision', 'Rappel', 'Score F1']
        
        # Valeurs des métriques pour chaque modèle
        self.values = np.array([
            # Exactitude, Précision, Rappel, F1
            [0.871, 0.871, 0.873, 0.872],  # Régression Logistique
            [0.983, 0.980, 0.986, 0.983],  # Arbre de Décision
            [0.997, 1.000, 0.994, 0.997],  # Random Forest
            [0.833, 0.773, 0.946, 0.851],  # Réseau de Neurones
        ])

    def _setup_visualization_style(self) -> None:
        """Configure le style de visualisation pour les graphiques."""
        sns.set_style('whitegrid')
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False

    def create_radar_chart(self, 
                          filename: str = 'radar_chart_comparison.png', 
                          figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Crée un graphique radar comparant les performances des modèles.
        
        Args:
            filename: Nom du fichier de sortie
            figsize: Dimensions de la figure (largeur, hauteur)
        """
        try:
            # Nombre de variables
            N = len(self.metrics)
            
            # Angle pour chaque axe
            angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
            angles += angles[:1]  # Fermer le polygone
            
            # Configuration du graphique
            fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
            
            # Couleurs pour chaque modèle
            colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
            
            # Tracer chaque modèle
            for i, model in enumerate(self.models):
                values_model = self.values[i].tolist()
                values_model += values_model[:1]  # Fermer le polygone
                
                ax.plot(angles, values_model, 'o-', linewidth=2, label=model, color=colors[i])
                ax.fill(angles, values_model, alpha=0.1, color=colors[i])
            
            # Étiquettes pour les axes
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(self.metrics, fontsize=12)
            
            # Limites de l'axe y
            ax.set_ylim(0.7, 1.02)
            ax.set_yticks([0.7, 0.8, 0.9, 1.0])
            
            # Titre et légende
            plt.title('Comparaison des performances des modèles', size=16, pad=20)
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
            
            # Ajuster la mise en page
            plt.tight_layout()
            
            # Sauvegarder le graphique
            output_path = os.path.join(self.output_dir, filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Graphique radar sauvegardé dans {output_path}")
        except Exception as e:
            print(f"Erreur lors de la création du graphique radar: {str(e)}")


def create_radar_chart() -> None:
    """Point d'entrée principal pour la génération du graphique radar."""
    generator = RadarChartGenerator()
    generator.create_radar_chart()


if __name__ == "__main__":
    create_radar_chart()
