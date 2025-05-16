# 1. Compréhension du domaine d'affaires

Cette première phase du cycle de vie d'un projet en intelligence artificielle est fondamentale. Un projet mal défini ou mal compris risque de ne pas répondre au besoin réel. Il est donc impératif de bien saisir le contexte, les objectifs, les attentes en matière de performance et les données disponibles.

## 1.1 Compréhension du contexte d'affaires

Le contexte de ce mandat est centré sur l'amélioration de l'efficacité opérationnelle au sein du service des réclamations d'une compagnie d'assurances vie. L'objectif est de faciliter le processus d'attribution des cas d'invalidités futurs aux employés de ce service.

Actuellement, cette attribution se fait potentiellement de manière suboptimale. Une meilleure distribution des cas, basée sur la complexité estimée de l'invalidité et l'expérience des employés, pourrait améliorer :

- Le temps de traitement
- La satisfaction des assurés
- L'efficacité globale du service

La solution développée sera un outil d'aide à la décision, utilisé par le service des réclamations, potentiellement par les gestionnaires ou directement par les employés lors de l'ouverture d'un nouveau dossier de réclamation. L'attribution se ferait ainsi au moment où les informations initiales sur l'invalidité sont disponibles.

## 1.2 Définition des objectifs

L'objectif principal est de bâtir un outil d'aide à la décision basé sur l'intelligence artificielle pour attribuer les cas d'invalidités futurs. Plus spécifiquement, l'outil doit pouvoir prédire la durée future d'une invalidité afin de la classer dans l'une des deux catégories prédéfinies :

1. **Invalidités courtes** : Durée prédite inférieure à six mois
   - Attribuées aux employés ayant peu d'expérience
2. **Invalidités longues** : Durée prédite supérieure à six mois
   - Attribuées aux employés ayant beaucoup d'expérience

La variable clé à modéliser est la durée de l'invalidité (`Duree_Invalidite`). Pour répondre au besoin de classification binaire (< 6 mois vs > 6 mois), cette variable quantitative sera transformée en une nouvelle variable binaire `Classe_Employe` :

- Valeur 1 : durée d'invalidité > 180 jours
- Valeur 0 : durée d'invalidité ≤ 180 jours

Cette variable binaire deviendra notre variable cible. La problématique est donc clairement un problème de classification de type "Two-Class Classification".

## 1.3 Mesure de performance pour connaître le niveau de succès du projet (KPI)

Pour évaluer le succès de cet outil, il est crucial de définir des indicateurs de performance clés (KPI) respèctant les critères SMART.

Le KPI principal doit refléter la capacité du modèle à prédire correctement la catégorie de durée d'invalidité pour permettre une attribution adéquate des dossiers.

Une mesure de performance pertinente sera de comparer la durée d'invalidité prédite (la classe binaire) à la durée réelle une fois que le dossier est clôturé.

### Matrice de confusion

Pour un problème de classification, la matrice de confusion est un outil essentiel. Elle permet d'analyser :

| Type de prédiction | Description               |
| ------------------ | ------------------------- |
| Vrais-Positifs     | Cas longs prédits longs   |
| Faux-Négatifs      | Cas longs prédits courts  |
| Vrais-Négatifs     | Cas courts prédits courts |
| Faux-Positifs      | Cas courts prédits longs  |

### Métriques de performance

En analysant cette matrice, on peut calculer :

- L'exactitude globale
- La précision (proportion de prédictions positives correctes)
- Le rappel (proportion de cas réels positifs correctement identifiés)
- Le score F1

Ces métriques permettent de quantifier le pourcentage de dossiers faussement ou correctement attribués aux employés expérimentés ou peu expérimentés.

### Objectifs de performance

Des objectifs de performance pourront être fixés dans le temps :

- Phase initiale : 60% de taux d'attribution correct
- Phase d'amélioration : 80% de taux d'attribution correct

## 1.4 Inventaire des données disponibles

Deux sources de données principales sont disponibles pour ce projet :

### 1.4.1 Données internes

Fichier : `MODELING_DATA.csv`

- Source : Département des réclamations
- Volume : 5000 enregistrements (1996-2006)
- Variables :
  - Année et mois de début de l'invalidité
  - Délai d'attente avant le début des prestations
  - FSA (trois premiers caractères du code postal)
  - Sexe
  - Année de naissance de l'assuré
  - Code d'emploi
  - Description textuelle de l'invalidité
  - Salaire annuel de l'assuré
  - Duree_Invalidite (variable cible)

### 1.4.2 Données externes

Fichier : `StatCanadaPopulationData.csv`

- Source : Statistique Canada
- Type : Données ouvertes
- Variables potentielles :
  - Densité de population
  - Nombre de personnes mariées
  - Nombre moyen d'enfants par famille
- Clé de liaison : FSA (RTA)

# 2. Acquisition et Compréhension des données

Cette deuxième phase du cycle de vie d'un projet en intelligence artificielle vise à acquérir les données identifiées lors de l'étape précédente et à les explorer en profondeur pour en comprendre la structure, le contenu, la qualité et les relations potentielles entre les variables.

C'est une étape fondamentale, car la qualité des données a un impact direct sur la performance des modèles qui seront développés.
Cette phase se décline en plusieurs sous-étapes interdépendantes :

## 2.1 Acquisition et chargement des données

### Sources de données

1. **Données internes**

   - Fichier : `MODELING_DATA.csv`
   - Volume : 5000 enregistrements
   - Format : CSV (séparateur point-virgule)

2. **Données externes**
   - Fichier : `StatCanadaPopulationData.csv`
   - Source : Statistique Canada
   - Format : CSV (séparateur virgule)

### Considérations techniques

- Traitement par lots (batch)
- Uniformisation des séparateurs
- Environnement : local ou cloud
- Outils : R ou Python (recommandés)

## 2.2 Exploration initiale et statistiques descriptives

### Variables quantitatives

Analyse des mesures de position centrale et de dispersion pour :

- `Duree_Delai_Attente`
- `Annee_Naissance`
- `Salaire_Annuel`
- `Duree_Invalidite`

### Variables qualitatives

Analyse des distributions de fréquence pour :

- `An_Debut_Invalidite`
- `Mois_Debut_Invalidite`
- `FSA`
- `Sexe`
- `Code`
- `Emploi`
- `Description_Invalidite`

### Visualisations

- Graphiques de relation avec `Duree_Invalidite`
- Diagrammes en boîte
- Analyses par catégories (année, sexe, emploi, âge)

### Corrélations observées

- Corrélation quasi nulle : âge vs durée d'invalidité
- Corrélation modérée : délai d'attente vs durée
- Corrélation modérée : salaire vs durée

## 2.3 Nettoyage des données

Points de nettoyage à considérer :

1. Uniformisation des séparateurs CSV
2. Gestion des problèmes d'encodage
3. Traitement des valeurs aberrantes
4. Gestion des valeurs manquantes

### Analyse des valeurs manquantes

## 2.4 Transformation et ingénierie des caractéristiques

### Transformations prévues

1. **Variables démographiques**

   - Calcul de l'âge
   - Indicateurs par catégorie d'âge
   - Variables polynomiales de l'âge

2. **Variables financières**

   - Catégorisation du salaire
   - Transformations mathématiques (logarithme, fonction carrée)

3. **Traitement textuel**

   - Nettoyage des caractères
   - Tokenisation
   - Suppression des mots vides
   - Normalisation

4. **Enrichissement externe**

   - Jointure avec les données de Statistique Canada
   - Agrégation par région

5. **Variable cible**
   - Binarisation de `Duree_Invalidite` en `Classe_Employe`

# 3. Modélisation

Cette phase cruciale consiste à choisir et construire les modèles qui tenteront de résoudre la problématique définie lors de l'étape 1, en utilisant les données préparées lors de l'étape 2.

C'est ici que l'on passe de la compréhension des données à la création de l'intelligence artificielle elle-même.
Pour notre projet, qui vise à prédire la durée de l'invalidité pour classer les cas (Classe_Employe : moins ou plus de 180 jours), il s'agit d'un problème de classification binaire. On se concentrera sur les approches supervisées dans le cadre de ce cours pour ce projet.

Cette phase se décompose généralement en trois sous-étapes principales :

## 3.1 Ingénierie des caractéristiques (Feature Engineering)

Bien que la préparation et la transformation initiale des données soient effectuées à l'étape 2, l'ingénierie des caractéristiques en tant que première sous-étape de la modélisation met l'accent sur la création de nouvelles variables à partir des données existantes dans le but spécifique d'améliorer la performance des modèles.

Cela implique de transformer les données brutes en attributs (caractéristiques) pertinents que les algorithmes d'apprentissage automatique peuvent mieux comprendre et utiliser. Pour notre projet, cela a été abordé lors de la description de l'étape 2 et inclut notamment :

- Le calcul de l'âge (`Age`) à partir de l'année de début de l'invalidité et de l'année de naissance
- La création d'indicateurs liés à l'âge (par exemple, un indicateur pour les assurés de 25 ans ou moins)
- La catégorisation ou la transformation mathématique du `Salaire_Annuel`
- Le traitement du texte du champ `Description_Invalidite`, qui est identifié comme pivotal20. Ce traitement est complexe et nécessite des étapes comme le retrait de caractères, la tokenisation, la suppression des mots vides (stop words), la normalisation ou le stemming, et la création de représentations textuelles (comme des indicateurs de mots clés ou le bag-of-words). L'objectif est de rendre ces informations textuelles utilisables par un modèle
- L'intégration et la dérivation de variables à partir des données externes de Statistique Canada en utilisant le FSA, potentiellement en regroupant les FSA
- La binarisation de la variable cible `Duree_Invalidite` pour créer la variable `Classe_Employe` (1 si > 180 jours, 0 sinon), nécessaire pour la classification binaire

L'objectif est d'obtenir un jeu de données final avec les caractéristiques les plus pertinentes pour l'entraînement des modèles.

## 3.2 Entraînement des modèles (Model Training)

Une fois que les caractéristiques sont prêtes, on procède à l'entraînement des algorithmes.

### Séparation des données

Pour évaluer la capacité du modèle à généraliser sur de nouvelles données, il est crucial de diviser l'ensemble de données en sous-ensembles distincts :

- Ensemble d'entraînement : ajustement des paramètres du modèle
- Ensemble de validation : sélection des hyperparamètres
- Ensemble de test : évaluation finale de la performance

Des techniques comme la validation croisée (K-fold) sont couramment utilisées pour cela.

### Choix des algorithmes

Il existe plusieurs algorithmes d'apprentissage supervisé adaptés aux problèmes de classification. Pour ce projet, nous allons évaluer :

1. Régression Logistique (point de départ simple et interprétable)
2. Arbres de Décision
3. Forêts Aléatoires (Random Forests)
4. Réseaux de Neurones

L'approche recommandée est de commencer simple et potentiellement d'explorer des algorithmes plus complexes par la suite.

### Entraînement

On nourrit l'algorithme choisi avec les données de l'ensemble d'entraînement. L'algorithme "apprend" des données pour trouver les relations entre les caractéristiques et la variable cible (`Classe_Employe`).

## 3.3 Évaluation des modèles (Model Evaluation)

Après l'entraînement, il est essentiel d'évaluer la performance du ou des modèles développés.

### Mesure de la performance

On utilise l'ensemble de test (qui n'a servi ni à l'entraînement ni à la sélection des hyperparamètres) pour mesurer la précision du modèle. Les métriques de performance (KPI) définies lors de l'étape 1 sont utilisées ici. Pour un problème de classification binaire, cela inclut :

- Calcul du taux d'échec ou de réussite
- Utilisation d'une matrice de confusion

### Comparaison des modèles

Si plusieurs algorithmes ont été entraînés, leurs performances sur l'ensemble de test sont comparées en utilisant les métriques définies.

### Sélection du modèle final

L'algorithme jugé le plus performant est retenu pour la phase suivante selon :

- Les critères définis
- La capacité à généraliser
- L'interprétabilité

La phase de Modélisation est itérative ; il peut être nécessaire de revenir aux étapes précédentes (comme l'ingénierie des caractéristiques ou le nettoyage des données) si les performances du modèle ne sont pas satisfaisantes.

# 4. Déploiement

Cette phase représente le point culminant du projet, où la solution développée lors des étapes précédentes est mise en œuvre pour être utilisée dans un environnement opérationnel. L'objectif est de rendre le modèle accessible aux utilisateurs finaux ou à d'autres systèmes, afin qu'il puisse commencer à générer la valeur d'affaires initialement définie.

## 4.1 Aspects clés du déploiement

### Mise en production

- Intégration du modèle dans l'infrastructure existante
- Mise en place du pipeline de données
- Rendre opérationnels l'algorithme et le processus de préparation des données

### Développement API

- Interface de programmation pour l'interaction avec le modèle
- Intégration avec les applications existantes
- Accès aux prédictions pour les utilisateurs finaux

### Stratégie de déploiement

- Choix de l'environnement (cloud vs local)
- Considérations techniques (AWS, Google Cloud, Azure)
- Plan de transition organisationnelle

### Stratégie de maintenance

- Suivi des performances du modèle
- Détection des dégradations
- Processus de réentraînement périodique

## 4.2 Intégration spécifique au projet

Pour notre projet de prédiction de la durée d'invalidité :

- Intégration dans l'outil de gestion des réclamations
- Attribution automatique des cas d'invalidité
- Suivi continu de la performance prédictive
- Mises à jour régulières du modèle

Le déploiement marque la transition du projet d'IA du stade expérimental vers une utilisation concrète et la génération de valeur.
