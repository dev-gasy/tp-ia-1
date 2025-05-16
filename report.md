# Rapport d'Analyse: Système de Prédiction de Durée d'Invalidité

## Introduction

Ce rapport présente les résultats du projet d'intelligence artificielle développé pour le service des réclamations d'une compagnie d'assurances vie. L'objectif principal était de concevoir un outil d'aide à la décision permettant d'attribuer efficacement les dossiers d'invalidité aux employés selon leur niveau d'expérience, en se basant sur une prédiction de la durée probable de l'invalidité.

Le système développé classifie les cas d'invalidité en deux catégories:

- **Cas courts** (≤ 180 jours): attribués aux employés ayant peu d'expérience
- **Cas longs** (> 180 jours): attribués aux employés expérimentés

# 1. Compréhension du domaine d'affaires

## 1.1 Compréhension du contexte d'affaires

Le contexte de ce projet s'inscrit dans l'amélioration de l'efficacité opérationnelle du service des réclamations d'une compagnie d'assurances vie. Actuellement, l'attribution des dossiers d'invalidité aux employés se fait sans tenir compte de la complexité potentielle des cas, ce qui entraîne une allocation sous-optimale des ressources humaines.

La problématique principale réside dans la prédiction de la durée probable d'une invalidité dès l'ouverture du dossier, afin d'orienter son attribution vers l'employé dont le niveau d'expérience correspond à la complexité anticipée. En effet, les cas de longue durée (supérieure à 180 jours) nécessitent une expertise plus approfondie et devraient être confiés aux employés expérimentés, tandis que les cas de courte durée peuvent être gérés efficacement par des employés moins expérimentés.

Une meilleure distribution des cas, basée sur cette prédiction de durée, permettrait d'améliorer :

- Le temps de traitement des dossiers (réduction des délais)
- La satisfaction des assurés grâce à une prise en charge plus adaptée
- L'efficacité globale du service par une utilisation optimale des compétences disponibles

## 1.2 Définition des objectifs

L'objectif principal était de développer un modèle capable de prédire la durée d'une invalidité afin d'orienter son attribution vers l'employé le plus adapté. La variable cible `Classe_Employe` est définie comme:

- Valeur 1: durée d'invalidité > 180 jours (cas longs)
- Valeur 0: durée d'invalidité ≤ 180 jours (cas courts)

## 1.3 Mesure de performance

Les KPIs utilisés pour évaluer le modèle ont été:

| Métrique           | Description                                                 |
| ------------------ | ----------------------------------------------------------- |
| Exactitude globale | Proportion totale de prédictions correctes                  |
| Précision          | Proportion de cas réellement longs parmi ceux prédits longs |
| Rappel             | Proportion de cas longs correctement identifiés             |
| Score F1           | Moyenne harmonique entre précision et rappel                |

<div style="text-align: center;">
<img src="output/performance_metrics_table.png" alt="Tableau des métriques de performance" width="600px">
<p><em>Figure 1: Tableau des métriques de performance</em></p>
</div>

## 1.4 Inventaire des données disponibles

Deux sources de données ont été utilisées:

1. **Données internes** (`MODELING_DATA.csv`): 5000 enregistrements de cas d'invalidité (1996-2006)
2. **Données externes** (`StatCanadaPopulationData.csv`): Données démographiques de Statistique Canada

## 1.5 Besoins de gouvernances (considérations légales ou éthiques)

Des mesures ont été mises en place pour adresser les aspects éthiques et de gouvernance:

- **Protection des données**: Techniques d'anonymisation appliquées aux données sensibles
- **Détection des biais**: Analyses pour identifier les discriminations potentielles
- **Explicabilité**: Utilisation de SHAP pour rendre les prédictions interprétables
- **Conformité réglementaire**: Respect des cadres légaux applicables
- **Gouvernance des modèles**: Processus de surveillance continue

<div style="text-align: center;">
<img src="output/ethics/anonymization_example.png" alt="Exemple d'anonymisation des données" width="600px">
<p><em>Figure 2: Exemple d'anonymisation des données sensibles</em></p>
</div>

# 2. Acquisition et Compréhension des données

## 2.1 Acquisition et chargement des données

### Sources de données

Les données ont été chargées depuis deux fichiers principaux :

1. **Données internes** (`MODELING_DATA.csv`)

   - Format CSV avec séparateur point-virgule
   - 5000 enregistrements de cas d'invalidité sur la période 1996-2006

2. **Données externes** (`StatCanadaPopulationData.csv`)
   - Format CSV avec séparateur virgule
   - Données démographiques de Statistique Canada liées par FSA

### Considérations techniques

Le traitement des données a impliqué :

- Uniformisation des séparateurs CSV
- Gestion des encodages de caractères
- Traitement par lots dans un environnement Python

## 2.2 Exploration initiale et statistiques descriptives

### Distribution des classes

La distribution des classes montre un déséquilibre significatif:

- Cas longs (> 180 jours): 84.02%
- Cas courts (≤ 180 jours): 15.98%

<div style="text-align: center;">
<img src="output/target_distribution.png" alt="Distribution des classes" width="600px">
<p><em>Figure 3: Distribution des classes dans le jeu de données original</em></p>
</div>

### Variables numériques clés

- Durée moyenne d'invalidité: 538 jours
- Délai d'attente moyen: 101 jours
- Salaire annuel moyen: 41 656 $

### Analyse des corrélations

<div style="text-align: center;">
<img src="output/correlation_matrix.png" alt="Matrice de corrélation globale" width="600px">
<p><em>Figure 4: Matrice de corrélation entre toutes les variables</em></p>
</div>

<div class="grid-container" style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px;">
  <div>
    <div style="text-align: center;">
    <img src="output/correlation_matrix_quantitative.png" alt="Matrice de corrélation - Variables Quantitatives" width="100%">
    <p><em>Figure 5a: Matrice de corrélation entre variables quantitatives</em></p>
    </div>
  </div>
  <div>
    <div style="text-align: center;">
    <img src="output/correlation_matrix_qualitative.png" alt="Matrice de corrélation - Variables Qualitatives" width="100%">
    <p><em>Figure 5b: Matrice de corrélation entre variables qualitatives</em></p>
    </div>
  </div>
</div>

Les facteurs les plus corrélés avec la classe d'employé sont:

- Durée d'invalidité (0.82)
- Âge (0.76)
- Âge au carré (0.74)
- Année de naissance (0.72)

## 2.3 Nettoyage des données

Les opérations de nettoyage suivantes ont été réalisées:

- Suppression des valeurs manquantes
- Correction des valeurs aberrantes
- Uniformisation des formats
- Gestion des problèmes d'encodage

## 2.4 Transformation et ingénierie des caractéristiques

Les transformations suivantes ont été appliquées:

- Création de variables d'âge à partir de l'année de naissance
- Encodage des variables catégorielles
- Transformation logarithmique du salaire
- Extraction de caractéristiques des descriptions textuelles d'invalidité
- Création d'indicateurs saisonniers

# 3. Modélisation

## 3.1 Ingénierie des caractéristiques

Les caractéristiques suivantes ont été dérivées pour améliorer la performance des modèles:

- Variables polynomiales d'âge (âge², âge³)
- Variables d'interaction (âge × salaire)
- Encodage one-hot des catégories
- Vectorisation des descriptions d'invalidité
- Agrégation des données démographiques par FSA

## 3.2 Entraînement des modèles

Quatre modèles de classification ont été entraînés:

1. **Régression Logistique**
2. **Arbre de Décision**
3. **Random Forest**
4. **Réseau de Neurones**

Pour l'entraînement, les données ont été divisées comme suit:

- 70% pour l'entraînement
- 15% pour la validation
- 15% pour le test

La validation croisée stratifiée à 5 plis a été utilisée pour assurer la robustesse des résultats.

## 3.3 Discussion des types d'approches possibles pour la modélisation

Les approches suivantes ont été évaluées:

### Approches linéaires

- **Régression Logistique**: Fournit une grande interprétabilité et des résultats de base solides

### Approches non-linéaires

- **Arbres de Décision**: Capturent efficacement les relations non-linéaires
- **Random Forest**: Offre une robustesse contre le surapprentissage
- **Réseaux de Neurones**: Modélise des relations extrêmement complexes

Nous avons adopté une approche progressive, commençant par des modèles simples avant de passer à des modèles plus complexes, tout en maintenant l'explicabilité requise pour le secteur de l'assurance.

## 3.4 Proposition d'approches et de mesures pour évaluer la qualité

Notre stratégie d'évaluation a intégré:

- **Validation croisée stratifiée**: Pour gérer le déséquilibre des classes
- **Validation temporelle**: Test sur les années plus récentes (2005-2006)
- **Sur-échantillonnage**: Techniques SMOTE pour équilibrer les classes

Les métriques spécifiques au contexte d'affaires incluaient:

- Matrice de coût asymétrique (FN plus coûteux que FP)
- Taux d'attribution optimale
- Analyses par sous-groupes démographiques

## 3.5 Évaluation des modèles

### Métriques de performance

<div style="text-align: center;">
<img src="output/kpi_metrics.png" alt="Métriques de performance détaillées" width="600px">
<p><em>Figure 6: Métriques de performance détaillées par modèle</em></p>
</div>

### Comparaison des performances

<div class="grid-container" style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px;">
  <div>
    <div style="text-align: center;">
    <img src="output/model_comparison.png" alt="Comparaison des modèles" width="100%">
    <p><em>Figure 7a: Comparaison des performances des modèles</em></p>
    </div>
  </div>
  <div>
    <div style="text-align: center;">
    <img src="output/radar_chart_comparison.png" alt="Comparaison radar des modèles" width="100%">
    <p><em>Figure 7b: Visualisation radar des performances multimétriques</em></p>
    </div>
  </div>
</div>

Le tableau ci-dessous présente les performances détaillées:

| Modèle                | Exactitude | Précision | Rappel | Score F1 | AUC   | CV Score | Temps (s) |
| --------------------- | ---------- | --------- | ------ | -------- | ----- | -------- | --------- |
| Régression Logistique | 0.871      | 0.871     | 0.873  | 0.872    | 0.947 | 0.863    | 0.004     |
| Arbre de Décision     | 0.983      | 0.980     | 0.986  | 0.983    | 0.991 | 0.795    | 0.004     |
| Random Forest         | 0.997      | 1.000     | 0.994  | 0.997    | 1.000 | 0.854    | 0.068     |
| Réseau de Neurones    | 0.833      | 0.773     | 0.946  | 0.851    | 0.954 | 0.839    | 0.037     |

### Courbes ROC

<div style="text-align: center;">
<img src="output/roc_comparison_chart.png" alt="Comparaison des courbes ROC" width="600px">
<p><em>Figure 8: Courbes ROC comparatives des différents modèles</em></p>
</div>

### Matrices de confusion

<div class="grid-container" style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px;">
  <div>
    <div style="text-align: center;">
    <img src="output/confusion_matrix_random_forest.png" alt="Matrice de confusion - Random Forest" width="100%">
    <p><em>Figure 9a: Matrice de confusion - Random Forest</em></p>
    </div>
  </div>
  <div>
    <div style="text-align: center;">
    <img src="output/confusion_matrix_neural_network.png" alt="Matrice de confusion - Réseau de Neurones" width="100%">
    <p><em>Figure 9b: Matrice de confusion - Réseau de Neurones</em></p>
    </div>
  </div>
</div>

<div class="grid-container" style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px;">
  <div>
    <div style="text-align: center;">
    <img src="output/confusion_matrix_decision_tree.png" alt="Matrice de confusion - Arbre de Décision" width="100%">
    <p><em>Figure 9c: Matrice de confusion - Arbre de Décision</em></p>
    </div>
  </div>
  <div>
    <div style="text-align: center;">
    <img src="output/confusion_matrix_logistic_regression.png" alt="Matrice de confusion - Régression Logistique" width="100%">
    <p><em>Figure 9d: Matrice de confusion - Régression Logistique</em></p>
    </div>
  </div>
</div>

### Impact sur les opérations

<div class="grid-container" style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px;">
  <div>
    <div style="text-align: center;">
    <img src="output/kpi_business_impact.png" alt="Impact opérationnel du modèle" width="100%">
    <p><em>Figure 10a: Impact opérationnel du modèle sur le processus d'attribution</em></p>
    </div>
  </div>
  <div>
    <div style="text-align: center;">
    <img src="output/kpi_class_distribution.png" alt="Distribution des classes prédites" width="100%">
    <p><em>Figure 10b: Distribution des classes prédites vs. réelles</em></p>
    </div>
  </div>
</div>

### Importance des caractéristiques

<div style="text-align: center;">
<img src="output/ethics/feature_importance_fallback.png" alt="Importance des caractéristiques" width="600px">
<p><em>Figure 11: Analyse de l'importance relative des caractéristiques pour la prédiction. Cette visualisation montre les variables qui ont le plus d'influence sur les prédictions du modèle Random Forest.</em></p>
</div>

### Analyse éthique des modèles

#### Explicabilité avec SHAP

<div style="text-align: center;">
<img src="output/ethics/shap_summary.png" alt="Analyse SHAP" width="600px">
<p><em>Figure 12: Analyse SHAP des contributions des caractéristiques, montrant l'impact de chaque variable sur la prédiction finale selon sa valeur.</em></p>
</div>

#### Analyse des biais par caractéristiques démographiques

L'analyse des biais démographiques évalue comment le modèle performe à travers différents groupes de population. Les graphiques ci-dessous présentent trois métriques clés par groupe démographique:

- **Taux de faux positifs**: Proportion de cas courts incorrectement classés comme longs
- **Taux de faux négatifs**: Proportion de cas longs incorrectement classés comme courts
- **Exactitude**: Proportion globale de prédictions correctes

<div style="text-align: center; margin-bottom: 30px;">
<img src="output/ethics/bias_analysis_sexe.png" alt="Analyse des biais par sexe" width="80%">
<p><em>Figure 13a: Analyse des biais selon le sexe des assurés. Les barres montrent le taux de faux positifs, faux négatifs et l'exactitude pour chaque groupe.</em></p>
</div>

<div class="grid-container" style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px;">
  <div>
    <div style="text-align: center;">
    <img src="output/ethics/bias_analysis_Age_Category.png" alt="Analyse des biais par catégorie d'âge" width="100%">
    <p><em>Figure 13b: Analyse des biais selon la catégorie d'âge. Cette analyse permet d'identifier si certaines tranches d'âge sont désavantagées par le modèle.</em></p>
    </div>
  </div>
  <div>
    <div style="text-align: center;">
    <img src="output/ethics/bias_analysis_FSA.png" alt="Analyse des biais par région (FSA)" width="100%">
    <p><em>Figure 13c: Analyse des biais selon la région géographique (FSA). Cette analyse permet de vérifier l'équité géographique du modèle.</em></p>
    </div>
  </div>
</div>

# 4. Déploiement

## 4.1 Aspects clés du déploiement

### Mise en production

Le modèle a été déployé sous forme d'API REST permettant l'intégration avec les systèmes existants du service des réclamations. Les principales caractéristiques incluent:

- Endpoint de prédiction pour les nouveaux cas
- Interface pour l'explication des décisions
- Logging des prédictions pour l'audit et la conformité

### Développement API

- Documentation interactive via Swagger UI
- Authentification sécurisée
- Validation des données entrantes

### Stratégie de déploiement

- Environnement conteneurisé via Docker
- Déploiement progressif avec tests A/B

## 4.2 Intégration spécifique au projet

L'intégration du modèle dans le processus métier comprend:

- Connexion au système de gestion des réclamations
- Interface utilisateur pour les gestionnaires
- Tableau de bord de suivi des performances
- Pipeline automatisé de réentraînement mensuel

Le modèle est maintenant en production et contribue activement à l'optimisation de l'attribution des cas d'invalidité.
