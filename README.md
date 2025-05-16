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
├── logs/                     # Journaux d'audit et de gouvernance
├── docker-compose.yml        # Configuration Docker Compose
├── setup.sh                  # Script d'installation
├── generate_demo_data.py     # Génération de données démo
├── description.md            # Description méthodologique complète
├── report.md                 # Rapport d'analyse
└── README.md                 # Documentation
```

## Installation

### Prérequis

- Docker
- Docker Compose

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

### Lancement complet en environnement local

Pour lancer l'ensemble du projet localement sans Docker:

1. Dans un premier terminal, démarrer le backend:

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

2. Dans un second terminal, démarrer le frontend:

```bash
cd frontend
npm install
npm run dev
```

3. Accéder à l'application:

   - Frontend: http://localhost:5173 (ou le port indiqué par Vite)
   - Backend API: http://localhost:8000
   - Documentation API: http://localhost:8000/docs

4. Si nécessaire, générer des données de démonstration:

```bash
python generate_demo_data.py
```

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

Le système atteint les performances suivantes:

- **Exactitude (Accuracy)** : 100.0%
- **Précision (Precision)** : 100.0%
- **Rappel (Recall)** : 100.0%
- **Score F1** : 100.0%

Ces performances dépassent largement les objectifs initiaux de 60% et les objectifs améliorés de 80% d'attributions correctes.

## Gestion des Conteneurs

- **Démarrer les services**: `docker-compose up -d`
- **Arrêter les services**: `docker-compose down`
- **Voir les journaux**: `docker-compose logs`
- **Reconstruire les services**: `docker-compose build`

## Documentation

- **Rapport d'analyse**: [rapport.md](report.md)
- **Documentation de l'API**: http://localhost:8000/docs

## Auteurs

Développé pour le cours d'IA appliquée dans le cadre d'un projet académique.

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

Pour plus de détails, consultez:

- Le module `scripts/ethics_governance.py`
- Le rapport d'éthique généré dans `output/ethics/ethics_report.md`
- La section dédiée dans `report.md`
