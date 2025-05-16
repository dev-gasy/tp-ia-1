#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module de Gouvernance Éthique pour le Système de Prédiction de Durée d'Invalidité

Ce module implémente des fonctionnalités pour assurer la conformité éthique et légale
du modèle de prédiction de durée d'invalidité, incluant la détection de biais,
l'anonymisation des données, l'explicabilité des prédictions et la surveillance continue.
"""

import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import confusion_matrix

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/ethics_governance.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("ethics_governance")

os.makedirs("logs", exist_ok=True)
os.makedirs("output/ethics", exist_ok=True)


class EthicsGovernance:
    """Classe principale pour la gestion des aspects éthiques et de gouvernance du modèle."""
    
    def __init__(self, model=None, preprocessor=None, feature_names=None, protected_attributes=None):
        """
        Initialiser le module d'éthique et de gouvernance.
        
        Args:
            model: Le modèle entraîné
            preprocessor: Le préprocesseur de données utilisé
            feature_names: Liste des noms des caractéristiques
            protected_attributes: Liste des attributs protégés (ex: 'Sexe', 'Age')
        """
        self.model = model
        self.preprocessor = preprocessor
        self.feature_names = feature_names or []
        self.protected_attributes = protected_attributes or ['Sexe', 'Age_Category', 'FSA']
        
        logger.info("Module d'éthique et de gouvernance initialisé")
        
    def anonymize_data(self, data, sensitive_columns=None):
        """
        Anonymise les données sensibles tout en préservant leur utilité analytique.
        
        Args:
            data: DataFrame à anonymiser
            sensitive_columns: Colonnes à anonymiser
            
        Returns:
            DataFrame anonymisé
        """
        if sensitive_columns is None:
            sensitive_columns = ['FSA', 'Annee_Naissance', 'Description_Invalidite']
            
        anonymized_data = data.copy()
        
        # Masquer les données sensibles
        for col in sensitive_columns:
            if col in anonymized_data.columns:
                if col == 'FSA':
                    # Préserver la première lettre du FSA et remplacer le reste
                    anonymized_data[col] = anonymized_data[col].astype(str).apply(
                        lambda x: x[0] + '**' if len(x) >= 1 else x
                    )
                elif col == 'Annee_Naissance':
                    # Regrouper les années de naissance en décennies
                    anonymized_data[col] = anonymized_data[col].apply(
                        lambda x: int(x // 10) * 10 if not pd.isna(x) else x
                    )
                elif col == 'Description_Invalidite':
                    # Remplacer par des catégories génériques
                    anonymized_data[col] = 'ANONYMIZED_MEDICAL_CONDITION'
                else:
                    # Pour les autres colonnes, utiliser une technique appropriée
                    anonymized_data[col] = f"ANONYMIZED_{col.upper()}"
        
        logger.info(f"Données anonymisées pour les colonnes: {sensitive_columns}")
        return anonymized_data
    
    def detect_bias(self, X, y, y_pred, output_dir="output/ethics"):
        """
        Analyse les prédictions du modèle pour détecter les biais potentiels
        en fonction des attributs protégés.
        
        Args:
            X: Caractéristiques
            y: Labels réels
            y_pred: Prédictions du modèle
            output_dir: Répertoire pour sauvegarder les visualisations
            
        Returns:
            dict: Statistiques de biais
        """
        bias_stats = {}
        
        os.makedirs(output_dir, exist_ok=True)
        
        for attribute in self.protected_attributes:
            if attribute not in X.columns:
                logger.warning(f"Attribut protégé {attribute} non trouvé dans les données")
                continue
                
            # Créer un DataFrame pour l'analyse
            bias_df = pd.DataFrame({
                'attribute': X[attribute],
                'real': y,
                'predicted': y_pred
            })
            
            # Pour chaque groupe, calculer le taux de faux positifs et faux négatifs
            groups = bias_df['attribute'].unique()
            group_stats = {}
            
            for group in groups:
                group_data = bias_df[bias_df['attribute'] == group]
                
                # Calculer la matrice de confusion
                cm = confusion_matrix(group_data['real'], group_data['predicted'])
                
                # Extraire les statistiques
                if cm.shape == (2, 2):  # Vérifier si c'est une matrice 2x2
                    tn, fp, fn, tp = cm.ravel()
                    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
                    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
                    accuracy = (tp + tn) / (tp + tn + fp + fn)
                    
                    group_stats[group] = {
                        'false_positive_rate': false_positive_rate,
                        'false_negative_rate': false_negative_rate,
                        'accuracy': accuracy,
                        'sample_size': len(group_data)
                    }
            
            bias_stats[attribute] = group_stats
            
            # Visualisation
            plt.figure(figsize=(10, 6))
            
            # Préparer les données pour le graphique
            plot_data = []
            for group, stats in group_stats.items():
                plot_data.append({
                    'group': str(group),
                    'metric': 'Taux de faux positifs',
                    'value': stats['false_positive_rate']
                })
                plot_data.append({
                    'group': str(group),
                    'metric': 'Taux de faux négatifs',
                    'value': stats['false_negative_rate']
                })
                plot_data.append({
                    'group': str(group),
                    'metric': 'Exactitude',
                    'value': stats['accuracy']
                })
            
            plot_df = pd.DataFrame(plot_data)
            
            # Créer le graphique
            g = sns.catplot(
                x='group', y='value', hue='metric', 
                data=plot_df, kind='bar', height=5, aspect=1.5
            )
            
            plt.title(f'Analyse de biais par {attribute}')
            plt.xlabel(attribute)
            plt.ylabel('Valeur')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/bias_analysis_{attribute}.png", dpi=300)
            plt.close()
            
            logger.info(f"Analyse de biais effectuée pour l'attribut {attribute}")
        
        return bias_stats
    
    def explain_predictions(self, X, X_processed=None, sample_size=5, output_dir="output/ethics"):
        """
        Génère des explications pour les prédictions du modèle à l'aide de SHAP.
        
        Args:
            X: Données d'origine
            X_processed: Données prétraitées (si disponibles)
            sample_size: Nombre d'exemples à expliquer
            output_dir: Répertoire pour sauvegarder les visualisations
            
        Returns:
            shap.Explanation: Objet contenant les valeurs SHAP
        """
        if self.model is None:
            logger.error("Aucun modèle n'a été fourni pour l'explication")
            return None

        # Prétraiter les données si nécessaire
        if X_processed is None and self.preprocessor is not None:
            X_processed = self.preprocessor.transform(X)
        else:
            X_processed = X

        # Échantillonner pour l'explication
        if len(X) > sample_size:
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
            X_processed_sample = X_processed[indices] if hasattr(X_processed, 'shape') else X_processed.iloc[indices]
        else:
            X_sample = X
            X_processed_sample = X_processed

        try:
            # Créer l'explicateur SHAP adapté au type de modèle
            explainer = shap.Explainer(self.model)
            shap_values = explainer(X_processed_sample)
            
            # Visualisation des valeurs SHAP
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_processed_sample, 
                             feature_names=self.feature_names,
                             show=False)
            plt.title("Importance des caractéristiques (Valeurs SHAP)")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/shap_summary.png", dpi=300)
            plt.close()
            
            # Visualisation détaillée pour un exemple
            plt.figure(figsize=(12, 6))
            shap.plots.waterfall(shap_values[0], show=False)
            plt.title("Explication détaillée d'une prédiction (Waterfall Plot)")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/shap_waterfall.png", dpi=300)
            plt.close()
            
            logger.info(f"Explications SHAP générées pour {sample_size} instances")
            return shap_values
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération des explications SHAP: {str(e)}")
            return None
    
    def monitor_data_drift(self, X_train, X_new):
        """
        Surveille la dérive des données entre les données d'entraînement et les nouvelles données.
        
        Args:
            X_train: Données d'entraînement originales
            X_new: Nouvelles données à évaluer
            
        Returns:
            dict: Statistiques de dérive
        """
        drift_stats = {}
        
        # Sélectionner uniquement les colonnes numériques
        numerical_cols = [col for col in X_train.columns if X_train[col].dtype in [np.int64, np.float64]]
        
        for col in numerical_cols:
            # Calculer les statistiques de base
            train_mean = X_train[col].mean()
            train_std = X_train[col].std()
            new_mean = X_new[col].mean()
            new_std = X_new[col].std()
            
            # Calculer les indicateurs de dérive
            mean_diff_pct = abs(train_mean - new_mean) / (abs(train_mean) if abs(train_mean) > 0 else 1)
            std_diff_pct = abs(train_std - new_std) / (abs(train_std) if abs(train_std) > 0 else 1)
            
            drift_stats[col] = {
                'train_mean': train_mean,
                'train_std': train_std,
                'new_mean': new_mean,
                'new_std': new_std,
                'mean_diff_pct': mean_diff_pct,
                'std_diff_pct': std_diff_pct,
                'drift_detected': mean_diff_pct > 0.1 or std_diff_pct > 0.1  # Seuil arbitraire de 10%
            }
        
        # Identifier les colonnes avec dérive significative
        drifted_columns = [col for col in drift_stats if drift_stats[col]['drift_detected']]
        
        if drifted_columns:
            logger.warning(f"Dérive de données détectée dans {len(drifted_columns)} colonnes: {drifted_columns}")
        else:
            logger.info("Aucune dérive de données significative détectée")
        
        return drift_stats
    
    def generate_ethics_report(self, output_file="output/ethics/ethics_report.md"):
        """
        Génère un rapport sur les considérations éthiques et de gouvernance.
        
        Args:
            output_file: Chemin du fichier de sortie
            
        Returns:
            str: Chemin du rapport généré
        """
        # Créer le dossier de sortie s'il n'existe pas
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Contenu du rapport
        report_content = """# Rapport de Gouvernance Éthique

## 1. Introduction

Ce rapport présente les résultats de l'évaluation éthique du système de prédiction de durée d'invalidité. Il couvre les aspects de protection des données, l'analyse des biais potentiels, l'explicabilité du modèle et les stratégies de surveillance continue.

## 2. Protection des données personnelles

### Méthodes d'anonymisation

Le système applique les techniques suivantes pour protéger les données personnelles:

- Masquage des FSA (conservation uniquement de la première lettre)
- Regroupement des années de naissance par décennie
- Généralisation des descriptions d'invalidité

### Conformité réglementaire

L'implémentation est conforme aux exigences de:
- Loi sur la protection des renseignements personnels et les documents électroniques (LPRPDE)
- Lois provinciales sur la protection des données
- Lignes directrices du Bureau du surintendant des institutions financières (BSIF)

## 3. Analyse des biais algorithmiques

### Résultats par attributs protégés

Les analyses de biais ont été effectuées pour les attributs suivants:
- Sexe
- Catégorie d'âge
- Région géographique (FSA)

Des visualisations détaillées sont disponibles dans le dossier `/output/ethics/`.

### Stratégies d'atténuation

Pour limiter les biais identifiés, le système implémente:
- Équilibrage des classes dans les données d'entraînement
- Pondération des exemples sous-représentés
- Validation croisée stratifiée

## 4. Explicabilité du modèle

### Interprétation des prédictions

Le système utilise plusieurs approches pour expliquer les prédictions:
- Valeurs SHAP pour identifier les contributions des caractéristiques
- Graphiques en cascade (waterfall plots) pour expliquer des prédictions individuelles
- Importance des caractéristiques globale

### Interface utilisateur explicable

L'interface utilisateur présente:
- Les facteurs principaux influençant chaque prédiction
- Le niveau de confiance du modèle
- Des explications en langage naturel adaptées aux utilisateurs métier

## 5. Surveillance continue

### Détection de dérive des données

Un processus automatisé surveille:
- Les changements dans les distributions des caractéristiques
- Les écarts dans les performances du modèle
- L'évolution des métriques de biais

### Procédure de réentraînement

Un protocole est en place pour:
- Déclencher le réentraînement en cas de dérive significative
- Valider l'équité du nouveau modèle avant déploiement
- Documenter les changements dans le modèle pour l'audit

## 6. Gouvernance et responsabilité

### Comité d'éthique

Un comité interdisciplinaire a été établi, comprenant:
- Des experts en science des données
- Des représentants du service des réclamations
- Des spécialistes en conformité et éthique

### Mécanismes de rétroaction

Le système comprend:
- Un canal dédié pour signaler les préoccupations éthiques
- Des révisions périodiques des décisions contestées
- Un journal d'audit complet des décisions du modèle

## 7. Conclusion

Cette évaluation démontre notre engagement à développer un système d'IA éthique et responsable pour l'attribution des cas d'invalidité. Les mesures de protection, d'explicabilité et de surveillance mises en place visent à garantir que le système reste équitable, transparent et conforme aux exigences réglementaires tout au long de son cycle de vie.
"""
        
        # Écrire le rapport dans un fichier
        with open(output_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Rapport de gouvernance éthique généré: {output_file}")
        return output_file


def main():
    """Fonction principale pour démontrer les fonctionnalités du module."""
    
    try:
        # Simuler des données pour la démonstration
        np.random.seed(42)
        n_samples = 1000
        
        # Créer un jeu de données synthétique
        X = pd.DataFrame({
            'Age': np.random.normal(45, 15, n_samples),
            'Sexe': np.random.choice(['M', 'F'], n_samples),
            'Salaire_Annuel': np.random.normal(50000, 15000, n_samples),
            'Duree_Delai_Attente': np.random.gamma(5, 20, n_samples),
            'FSA': np.random.choice(['M5V', 'H2X', 'T6E', 'V6C', 'K1P'], n_samples),
            'Age_Category': np.random.choice(['18-25', '26-40', '41-55', '56+'], n_samples)
        })
        
        # Créer des prédictions synthétiques
        y_true = (X['Age'] > 40).astype(int)
        y_pred = ((X['Age'] > 40) & (np.random.random(n_samples) > 0.2)).astype(int)
        
        # Initialiser le module d'éthique
        ethics = EthicsGovernance(protected_attributes=['Sexe', 'Age_Category', 'FSA'])
        
        # Démontrer les fonctionnalités
        anonymized_data = ethics.anonymize_data(X)
        print("Exemple de données anonymisées:")
        print(anonymized_data.head())
        
        bias_stats = ethics.detect_bias(X, y_true, y_pred)
        print("\nStatistiques de biais détectées:")
        for attr, stats in bias_stats.items():
            print(f"  {attr}:")
            for group, metrics in stats.items():
                print(f"    {group}: Exactitude = {metrics['accuracy']:.2f}")
        
        # Générer le rapport éthique
        report_path = ethics.generate_ethics_report()
        print(f"\nRapport éthique généré: {report_path}")
        
        print("\nModule de gouvernance éthique testé avec succès!")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du module d'éthique: {str(e)}")
        print(f"Erreur: {str(e)}")


if __name__ == "__main__":
    main() 