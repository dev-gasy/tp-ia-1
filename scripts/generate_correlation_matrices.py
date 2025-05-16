#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Génère différentes matrices de corrélation pour les variables du dataset.
Module optimisé suivant le principe DRY (Don't Repeat Yourself).
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


class CorrelationMatrixGenerator:
    """Classe pour générer différents types de matrices de corrélation."""

    def __init__(self, output_dir: str = 'output'):
        """
        Initialise le générateur de matrices de corrélation.

        Args:
            output_dir: Répertoire de sortie pour les images générées
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Configuration du style de visualisation
        self._setup_visualization_style()

    def _setup_visualization_style(self) -> None:
        """Configure le style de visualisation pour toutes les matrices."""
        sns.set_style('whitegrid')
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False

    def load_data(self, data_path: str = 'data/processed_data.csv') -> pd.DataFrame:
        """
        Charge les données depuis un fichier CSV.

        Args:
            data_path: Chemin vers le fichier de données

        Returns:
            DataFrame contenant les données chargées
        """
        try:
            df = pd.read_csv(data_path)
            print(f"Données chargées avec succès: {df.shape[0]} lignes, {df.shape[1]} colonnes")
            return df
        except Exception as e:
            print(f"Erreur lors du chargement des données: {str(e)}")
            return self._create_demo_data()

    def _create_demo_data(self, n_samples: int = 100) -> pd.DataFrame:
        """
        Crée un jeu de données de démonstration.

        Args:
            n_samples: Nombre d'échantillons à générer

        Returns:
            DataFrame contenant les données générées
        """
        print("Création d'un jeu de données de démonstration...")

        # Fixer la graine pour la reproductibilité
        np.random.seed(42)
        
        # Variables quantitatives
        df = pd.DataFrame({
            'Age': np.random.normal(40, 10, n_samples).astype(int),
            'Salaire': np.random.normal(45000, 15000, n_samples).astype(int),
            'Duree_Invalidite': np.random.gamma(5, 30, n_samples).astype(int),
            'Delai_Traitement': np.random.gamma(2, 20, n_samples).astype(int),
            
            # Variables qualitatives
            'Sexe': np.random.choice(['M', 'F'], n_samples),
            'Code_Emploi': np.random.choice(['A1', 'B2', 'C3', 'D4'], n_samples),
            'Age_Category': np.random.choice(['Jeune', 'Moyen', 'Senior'], n_samples),
            'Salary_Category': np.random.choice(['Bas', 'Moyen', 'Élevé'], n_samples),
            'Is_Winter': np.random.choice([0, 1], n_samples),
            'Is_Summer': np.random.choice([0, 1], n_samples)
        })
        
        # Ajouter une variable cible
        df['Classe_Employe'] = np.where(df['Duree_Invalidite'] > 180, 1, 0)
        
        return df

    def _encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode les variables catégorielles pour les calculs de corrélation.

        Args:
            df: DataFrame contenant les données à encoder

        Returns:
            DataFrame avec les variables catégorielles encodées
        """
        df_encoded = df.copy()
        
        # Identifier les colonnes catégorielles
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) == 0:
            return df_encoded
            
        # Encoder chaque colonne catégorielle
        encoder = LabelEncoder()
        for col in categorical_cols:
            df_encoded[col] = encoder.fit_transform(df[col])
            
        return df_encoded

    def _get_variable_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Identifie les variables quantitatives et qualitatives dans le DataFrame.

        Args:
            df: DataFrame à analyser

        Returns:
            Tuple contenant les listes des variables quantitatives et qualitatives
        """
        quant_vars = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        qual_vars = df.select_dtypes(include=['object']).columns.tolist()
        return quant_vars, qual_vars

    def _generate_correlation_matrix(
        self,
        df: pd.DataFrame,
        title: str,
        filename: str,
        figsize: Tuple[int, int] = (12, 10),
        annot_kws: Optional[Dict] = None
    ) -> None:
        """
        Génère et sauvegarde une matrice de corrélation.

        Args:
            df: DataFrame contenant les données pour la matrice de corrélation
            title: Titre à afficher sur la matrice
            filename: Nom du fichier de sortie (sans extension)
            figsize: Dimensions de la figure (largeur, hauteur)
            annot_kws: Paramètres pour les annotations
        """
        plt.figure(figsize=figsize)
        
        # Calculer la matrice de corrélation
        corr_matrix = df.corr()
        
        # Masque pour afficher seulement la moitié inférieure
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Paramètres par défaut pour les annotations
        if annot_kws is None:
            annot_kws = {}
            
        # Créer le heatmap
        sns.heatmap(
            corr_matrix, 
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
        
        plt.title(title, fontsize=16)
        plt.tight_layout()
        
        # Sauvegarder la figure
        output_path = os.path.join(self.output_dir, f"{filename}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Matrice de corrélation sauvegardée: {output_path}")

    def generate_all_correlation_matrices(self, df: pd.DataFrame) -> None:
        """
        Génère toutes les matrices de corrélation pour le DataFrame.

        Args:
            df: DataFrame contenant les données
        """
        # Identifier les types de variables
        quant_vars, qual_vars = self._get_variable_types(df)
        
        # Encoder les variables catégorielles
        df_encoded = self._encode_categorical_variables(df)
        
        # 1. Matrice pour variables quantitatives
        if quant_vars:
            self._generate_correlation_matrix(
                df=df[quant_vars],
                title="Matrice de Corrélation - Variables Quantitatives",
                filename="correlation_matrix_quantitative"
            )
        
        # 2. Matrice pour variables qualitatives
        if qual_vars:
            self._generate_correlation_matrix(
                df=df_encoded[qual_vars],
                title="Matrice de Corrélation - Variables Qualitatives",
                filename="correlation_matrix_qualitative"
            )
        
        # 3. Matrice pour toutes les variables
        self._generate_correlation_matrix(
            df=df_encoded,
            title="Matrice de Corrélation - Toutes Variables",
            filename="correlation_matrix",
            figsize=(14, 12),
            annot_kws={"size": 8}
        )

    def run(self):
        """
        Génère toutes les matrices de corrélation
        """
        print("Generating correlation matrices...")
        
        # Load data
        df = self.load_data()
        if df is None:
            # Create synthetic data if real data loading fails
            df = self._create_demo_data()
        
        # Generate correlation matrices
        self.generate_all_correlation_matrices(df)
        print("All correlation matrices generated.")


def main() -> None:
    """Point d'entrée principal du script."""
    try:
        # Créer l'instance du générateur
        generator = CorrelationMatrixGenerator()
        
        # Générer les matrices
        generator.run()
        
        print("Génération des matrices de corrélation terminée avec succès.")
    except Exception as e:
        print(f"Erreur lors de la génération des matrices: {str(e)}")


if __name__ == "__main__":
    main()
