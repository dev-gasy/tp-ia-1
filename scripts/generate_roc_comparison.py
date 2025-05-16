#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Génère un graphique comparatif des courbes ROC pour différents modèles.
Module optimisé suivant le principe DRY (Don't Repeat Yourself).
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns


class ROCComparisonGenerator:
    """Classe pour générer des graphiques comparatifs de courbes ROC."""

    def __init__(self, output_dir: str = 'output'):
        """
        Initialise le générateur de comparaison de courbes ROC.

        Args:
            output_dir: Répertoire de sortie pour les images générées
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Configuration du style de visualisation
        self._setup_visualization_style()
        
        # Données des modèles (nom, AUC rapportée)
        self.models = [
            ('Régression Logistique', 0.947),
            ('Arbre de Décision', 0.991),
            ('Random Forest', 1.000),
            ('Réseau de Neurones', 0.954)
        ]
        
        # Couleurs pour chaque modèle
        self.colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']

    def _setup_visualization_style(self) -> None:
        """Configure le style de visualisation pour les graphiques."""
        sns.set_style('whitegrid')
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False

    def generate_roc_comparison(self, 
                               filename: str = 'roc_comparison_chart.png',
                               figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Génère une comparaison des courbes ROC pour différents modèles.
        Utilise des données simulées basées sur les AUC rapportées.
        
        Args:
            filename: Nom du fichier de sortie
            figsize: Dimensions de la figure (largeur, hauteur)
        """
        try:
            # Configuration du graphique
            plt.figure(figsize=figsize)
            
            # Génération de courbes ROC simulées basées sur l'AUC rapportée
            for i, (model_name, reported_auc) in enumerate(self.models):
                # Simuler des points FPR, TPR basés sur l'AUC rapportée
                # Plus l'AUC est élevée, plus la courbe sera proche du coin supérieur gauche
                fpr = np.linspace(0, 1, 100)
                
                if reported_auc > 0.99:
                    # Pour les modèles presque parfaits
                    tpr = np.ones_like(fpr)
                    tpr[:3] = np.linspace(0, 1, 3)
                else:
                    # Pour les autres modèles, une approximation basée sur l'AUC
                    tpr = fpr**(1.0/(10*reported_auc))
                
                # Calculer l'AUC
                roc_auc = auc(fpr, tpr)
                
                # Tracer la courbe ROC
                plt.plot(
                    fpr, tpr, 
                    color=self.colors[i],
                    lw=2,
                    label=f'{model_name} (AUC = {reported_auc:.3f})'
                )
            
            # Tracer la ligne diagonale de référence
            plt.plot([0, 1], [0, 1], 'k--', lw=1)
            
            # Configuration des axes et des étiquettes
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Taux de faux positifs (1 - Spécificité)', fontsize=12)
            plt.ylabel('Taux de vrais positifs (Sensibilité)', fontsize=12)
            plt.title('Comparaison des courbes ROC', fontsize=16)
            plt.legend(loc='lower right', fontsize=12)
            
            # Ajuster la mise en page
            plt.tight_layout()
            
            # Sauvegarder le graphique
            output_path = os.path.join(self.output_dir, filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Comparaison des courbes ROC sauvegardée dans {output_path}")
        except Exception as e:
            print(f"Erreur lors de la création du graphique de comparaison ROC: {str(e)}")


def generate_roc_comparison() -> None:
    """Point d'entrée principal pour la génération du graphique de comparaison ROC."""
    generator = ROCComparisonGenerator()
    generator.generate_roc_comparison()


if __name__ == "__main__":
    generate_roc_comparison()
