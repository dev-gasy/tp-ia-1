# Système de Prédiction de Durée d'Invalidité

Ce projet implémente un système de prédiction de la durée des cas d'invalidité pour optimiser l'attribution des dossiers aux employés selon leur niveau d'expérience.

## Objectif

L'objectif principal est d'attribuer efficacement les cas d'invalidité selon leur durée prévue :

- **Cas courts** (≤ 180 jours) : attribués aux employés peu expérimentés
- **Cas longs** (> 180 jours) : attribués aux employés expérimentés

## Architecture du Système

Le système est composé de plusieurs composants organisés dans une architecture moderne:

- **Backend API**: FastAPI pour servir les prédictions du modèle
- **Frontend**: Application React avec Vite pour une interface utilisateur interactive
- **Infrastructure**: Docker et Docker Compose pour la conteneurisation et l'orchestration
- **Gouvernance éthique**: Modules intégrés pour garantir l'équité, la transparence et la conformité

## Structure du Projet

```
├── backend/                  # API FastAPI
│   ├── app.py                # Application principale
│   ├── Dockerfile            # Configuration Docker
│   └── requirements.txt      # Dépendances Python
├── frontend/                 # Application React
│   ├── src/                  # Code source
│   │   ├── api/              # Services API
│   │   ├── components/       # Composants réutilisables
│   │   └── pages/            # Pages de l'application
│   ├── Dockerfile            # Configuration Docker
│   └── nginx.conf            # Configuration du serveur web
├── data/                     # Données
│   ├── MODELING_DATA.csv     # Données internes
│   ├── StatCanadaPopulationData.csv  # Données externes
│   └── processed_data.csv    # Données prétraitées
├── models/                   # Modèles entraînés
│   └── best_model.pkl        # Meilleur modèle
├── scripts/                  # Scripts Python originaux
│   ├── data_processing.py    # Traitement des données
│   ├── model_training.py     # Entraînement des modèles
│   ├── kpi_calculation.py    # Calcul des KPIs
│   ├── model_deployment.py   # Version Flask originale
│   ├── utils.py              # Fonctions utilitaires
│   ├── ethics_governance.py  # Module de gouvernance éthique
│   └── generate_ethics_visualizations.py # Visualisations éthiques
├── output/                   # Résultats et visualisations
│   ├── ethics/               # Visualisations et analyses éthiques
│       ├── feature_importance_fallback.png  # Importance des caractéristiques
│       ├── shap_summary.png  # Analyse SHAP
│       ├── bias_analysis_*.png  # Analyses de biais
│       └── ethics_report.md  # Rapport d'éthique
├── logs/                     # Journaux d'audit et de gouvernance
├── docker-compose.yml        # Configuration Docker Compose
├── setup.sh                  # Script d'installation
├── generate_demo_data.py     # Génération de données démo
├── main.py                   # Script principal d'exécution du pipeline
├── description.md            # Description méthodologique complète
├── report.md                 # Rapport technique
├── description_document.docx # Documentation méthodologique (Word)
├── technical_report_final.docx # Rapport technique complet (Word)
└── README.md                 # Documentation
```

## Installation

### Prérequis

- Docker
- Docker Compose
- Python 3.9 ou supérieur
- Pandoc (pour la génération de documents Word)

### Installation avec Docker

1. Cloner le dépôt:

```bash
git clone <URL_DU_DEPOT>
cd tp-ia-1
```

2. Exécuter le script d'installation:

```bash
chmod +x setup.sh
./setup.sh
```

Le script va:

- Vérifier les prérequis
- Créer les répertoires nécessaires
- Générer des données de démonstration
- Construire et démarrer les conteneurs Docker

3. Accéder à l'application:
   - Frontend: http://localhost
   - Backend API: http://localhost:8000
   - Documentation API: http://localhost:8000/docs

### Installation manuelle (pour développement)

#### Backend

1. Créer un environnement virtuel:

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Installer les dépendances:

```bash
pip install -r requirements.txt
```

3. Exécuter l'API:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend

1. Installer les dépendances:

```bash
cd frontend
npm install
```

2. Démarrer le serveur de développement:

```bash
npm run dev
```

### Exécution du pipeline complet

Pour exécuter le pipeline de données, d'entraînement et d'évaluation:

```bash
python main.py
```

Options disponibles:

- `--skip-processing` : Sauter l'étape de traitement des données
- `--skip-training` : Sauter l'étape d'entraînement des modèles
- `--skip-kpi` : Sauter l'étape de calcul des KPIs
- `--skip-visualization` : Sauter l'étape de génération des visualisations

## Utilisation

### Prédiction via l'interface web

1. Accéder à l'application web à l'adresse http://localhost
2. Naviguer vers la page "Prédiction"
3. Remplir le formulaire avec les informations du cas d'invalidité
4. Soumettre le formulaire pour obtenir une prédiction
5. Consulter le résultat avec la durée prévue et la recommandation d'attribution

### Appel direct à l'API

Exemple de requête avec curl:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "data": {
             "Age": 35,
             "Sexe": "M",
             "Code_Emploi": 2,
             "Salaire_Annuel": 45000,
             "Duree_Delai_Attente": 30,
             "Description_Invalidite": "FRACTURE JAMBE",
             "Mois_Debut_Invalidite": 6,
             "Is_Winter": 0
           }
         }'
```

## Performances du Modèle

Le système atteint les performances suivantes avec le modèle Random Forest:

- **Exactitude (Accuracy)** : 99.7%
- **Précision (Precision)** : 100.0%
- **Rappel (Recall)** : 99.4%
- **Score F1** : 99.7%
- **AUC** : 100.0%

Ces performances dépassent largement les objectifs initiaux de 60% et les objectifs améliorés de 80% d'attributions correctes.

## Documentation Générée

### Documents Word

Deux documents Word complets ont été générés pour une utilisation professionnelle:

- **Description Méthodologique** (`description_document.docx`): Document détaillant le contexte d'affaires, les objectifs et la méthodologie
- **Rapport Technique** (`technical_report_final.docx`): Rapport complet avec toutes les visualisations, analyses de performances et résultats techniques

Pour regénérer ces documents à partir des fichiers Markdown:

```bash
pip install pandoc
pandoc description.md -o description_document.docx --toc --toc-depth=3
pandoc technical_report_pandoc.md -o technical_report_final.docx --toc --toc-depth=3
```

### Rapports

- **Description Méthodologique**: [description.md](description.md)
- **Rapport Technique**: [report.md](report.md)
- **Rapport d'Éthique**: [output/ethics/ethics_report.md](output/ethics/ethics_report.md)
- **Documentation API**: http://localhost:8000/docs

## Visualisations Disponibles

Le projet génère automatiquement de nombreuses visualisations:

### Analyses de Données

- Distribution des classes cibles
- Matrices de corrélation
- Visualisations des caractéristiques importantes

### Performances des Modèles

- Courbes ROC de comparaison
- Matrices de confusion
- Graphiques radar des performances
- Comparaisons des modèles

### Analyses Éthiques

- Importance des caractéristiques (feature importance)
- Visualisations SHAP pour l'explicabilité
- Analyses de biais par groupe démographique
- Exemples d'anonymisation des données

## Auteurs

Développé par Tolotra RAHARISON (2584264) pour le cours d'IA appliquée au CEGEP Sainte-Foy.

## Licence

Ce projet est distribué sous licence MIT.

## Aspects Éthiques et de Gouvernance

Le système intègre plusieurs mécanismes pour garantir une utilisation éthique et responsable:

### Protection des Données

- **Anonymisation**: Masquage automatique des informations identifiantes (FSA, années de naissance)
- **Conformité LPRPDE**: Respect des lois canadiennes sur la protection des données personnelles

### Équité et Transparence

- **Détection de biais**: Analyse automatique des disparités de performance entre groupes démographiques
- **Explicabilité**: Génération d'explications SHAP pour rendre les prédictions compréhensibles

### Gouvernance

- **Journal d'audit**: Enregistrement de toutes les prédictions et leurs justifications
- **Surveillance continue**: Détection automatique de la dérive des données et des performances
