# 1. Compréhension du domaine d'affaires

Cette première phase du cycle de vie d'un projet en intelligence artificielle est fondamentale. Un projet mal défini ou mal compris risque de ne pas répondre au besoin réel. Il est donc impératif de bien saisir le contexte, les objectifs, les attentes en matière de performance et les données disponibles.

## 1.1 Compréhension du contexte d'affaires

Le contexte de ce mandat est centré sur l'amélioration de l'efficacité opérationnelle au sein du service des réclamations d'une compagnie d'assurances vie. Actuellement, l'attribution des dossiers d'invalidité aux employés se fait sans tenir compte de la complexité potentielle des cas, ce qui entraîne une allocation sous-optimale des ressources humaines.

La problématique principale réside dans la prédiction de la durée probable d'une invalidité dès l'ouverture du dossier, afin d'orienter son attribution vers l'employé dont le niveau d'expérience correspond à la complexité anticipée. En effet, les cas d'invalidité de longue durée (supérieure à 180 jours) nécessitent une expertise plus approfondie et devraient être confiés aux employés expérimentés, tandis que les cas de courte durée peuvent être gérés efficacement par des employés moins expérimentés.

Une meilleure distribution des cas, basée sur cette prédiction de durée, permettrait d'améliorer :

- Le temps de traitement des dossiers (réduction des délais)
- La satisfaction des assurés grâce à une prise en charge plus adaptée
- L'efficacité globale du service par une utilisation optimale des compétences disponibles

La solution développée sera un outil d'aide à la décision, utilisé par les gestionnaires ou directement par les employés lors de l'ouverture d'un nouveau dossier. Ce système de classification interviendrait dès la réception des informations initiales sur l'invalidité, optimisant ainsi l'ensemble du processus de traitement des réclamations.

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

## 1.5 Besoins de gouvernances (considérations légales ou éthiques)

L'utilisation de l'intelligence artificielle pour la prise de décisions dans le domaine de l'assurance soulève d'importantes considérations légales et éthiques qui doivent être prises en compte tout au long du projet. Ces considérations sont fondamentales pour garantir la conformité réglementaire et maintenir la confiance des assurés.

### 1.5.1 Protection des données personnelles

Notre projet traite des données sensibles relatives à la santé des assurés, ce qui implique :

- Conformité avec la **Loi sur la protection des renseignements personnels et les documents électroniques (LPRPDE)** au niveau fédéral
- Respect des lois provinciales sur la protection des données (comme la Loi sur la protection des renseignements personnels dans le secteur privé au Québec)
- Mise en place de mesures de sécurité adéquates pour protéger les données médicales confidentielles
- Anonymisation des données d'entraînement pour éviter l'identification des individus

### 1.5.2 Biais et discrimination algorithmique

Les modèles d'IA peuvent involontairement perpétuer ou amplifier des biais existants, notamment :

- Biais de genre ou d'âge dans la prédiction de la durée d'invalidité
- Discrimination géographique basée sur le FSA (code postal)
- Biais liés au niveau de revenu (variable Salaire_Annuel)

Pour atténuer ces risques, nous prévoyons :

- Analyse des données d'entraînement pour détecter les biais potentiels
- Tests réguliers du modèle pour identifier les disparités de performance entre différents groupes démographiques
- Techniques d'équité algorithmique pour équilibrer les prédictions

### 1.5.3 Transparence et explicabilité

Dans le secteur de l'assurance, les décisions doivent être justifiables et explicables :

- Privilégier des modèles interprétables lorsque possible (comme la régression logistique ou les arbres de décision)
- Développer des méthodes d'explication locales pour les modèles complexes (comme SHAP ou LIME)
- Documenter clairement la logique de prise de décision pour justifier l'attribution des cas
- Permettre la contestation humaine des décisions algorithmiques

### 1.5.4 Conformité réglementaire spécifique à l'assurance

Le secteur de l'assurance est fortement réglementé :

- Respect des lignes directrices du Bureau du surintendant des institutions financières (BSIF)
- Conformité avec les lois provinciales sur les assurances
- Documentation adéquate des modèles pour les audits réglementaires
- Mise en place d'un processus de révision humaine pour les cas limites ou contestés

### 1.5.5 Gouvernance des modèles

Pour assurer une utilisation éthique et responsable de notre solution, nous établirons :

- Un comité d'éthique interdisciplinaire pour superviser le développement et l'utilisation du modèle
- Des protocoles de validation régulière et de surveillance continue
- Un processus documenté de développement et de déploiement conforme aux normes internes
- Des mécanismes de rétroaction permettant aux employés et aux assurés de signaler des problèmes potentiels

L'intégration de ces considérations éthiques et légales dès la phase de conception garantira non seulement la conformité du projet, mais contribuera également à développer une solution d'IA responsable qui renforce la confiance des parties prenantes.

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

## 3.3 Discussion des types d'approches possibles pour la modélisation et recommandation

La sélection de l'approche de modélisation la plus appropriée est une étape cruciale qui influencera directement la qualité des prédictions et l'utilité de notre solution. Pour notre problème de classification binaire de la durée d'invalidité, plusieurs approches sont envisageables, chacune avec ses avantages et limitations.

### Approches linéaires vs non-linéaires

#### Approches linéaires

**Régression Logistique**

_Pourquoi la choisir ?_

- Excellente interprétabilité : les coefficients indiquent clairement l'influence de chaque variable sur la probabilité
- Simplicité d'implémentation et rapidité d'exécution
- Base solide pour établir un modèle de référence (baseline)
- Particulièrement adaptée aux contextes où l'explicabilité est exigée par les régulateurs

_Limitations :_

- Hypothèse de linéarité des relations entre variables qui peut être inappropriée pour certaines données d'assurance
- Difficulté à capturer les interactions complexes entre les variables (ex: relation entre âge, type d'invalidité et durée)

#### Approches non-linéaires

**Arbres de Décision**

_Pourquoi les choisir ?_

- Capacité à modéliser des relations non-linéaires naturellement présentes dans les données d'invalidité
- Bonne interprétabilité grâce à la structure visuelle des règles de décision
- Gestion automatique des variables catégorielles sans encodage préalable
- Robustesse aux valeurs aberrantes fréquentes dans les données d'assurance

_Limitations :_

- Tendance au surapprentissage, particulièrement problématique avec nos 5000 enregistrements
- Sensibilité aux petites variations dans les données

**Forêts Aléatoires (Random Forests)**

_Pourquoi les choisir ?_

- Robustesse contre le surapprentissage grâce à l'agrégation de multiples arbres
- Excellente performance prédictive dans de nombreux problèmes de classification
- Mesure intégrée de l'importance des caractéristiques, utile pour comprendre les facteurs influençant la durée d'invalidité
- Gestion efficace des caractéristiques hautement dimensionnelles (comme les descripteurs textuels extraits des descriptions d'invalidité)

_Limitations :_

- Moins interprétable qu'un arbre de décision unique ou qu'une régression logistique
- Temps d'entraînement plus long
- Nécessite des méthodes additionnelles (comme SHAP) pour atteindre le niveau d'explicabilité requis dans le secteur de l'assurance

**Réseaux de Neurones**

_Pourquoi les choisir ?_

- Capacité supérieure à modéliser des relations extrêmement complexes et non-linéaires
- Particulièrement efficaces pour l'intégration de données textuelles non structurées (descriptions d'invalidité)
- Potentiel de performance prédictive élevée avec un réglage approprié

_Limitations :_

- Opacité du processus décisionnel ("boîte noire"), problématique dans le secteur réglementé de l'assurance
- Nécessité de grands volumes de données pour performance optimale (limitation avec 5000 enregistrements)
- Risque élevé de surapprentissage sans régularisation appropriée
- Complexité du réglage des hyperparamètres

### Approches d'ensemble vs modèles uniques

**Méthodes d'ensemble (boosting, stacking)**

_Pourquoi les choisir ?_

- Amélioration significative des performances par rapport aux modèles individuels
- Réduction de la variance et du biais
- Capacité à combiner les forces de différents types de modèles

_Limitations :_

- Complexité accrue du modèle final
- Augmentation du temps de calcul
- Réduction de l'interprétabilité

### Recommandation d'approche

Compte tenu de notre contexte spécifique (secteur de l'assurance, besoin d'explicabilité, volume de données modéré, importance des considérations éthiques), nous recommandons une **approche progressive** :

1. **Phase initiale : Modèle de base interprétable**

   - Commencer par une régression logistique comme modèle de référence
   - Évaluer soigneusement les performances et identifier les limitations

2. **Phase intermédiaire : Arbres de décision et forêts aléatoires**

   - Implémenter un arbre de décision pour améliorer la capture des relations non-linéaires
   - Déployer des forêts aléatoires pour améliorer la précision tout en conservant un niveau raisonnable d'interprétabilité
   - Utiliser des techniques comme SHAP pour maintenir l'explicabilité requise

3. **Phase avancée (optionnelle) : Techniques d'ensemble**

   - Si les performances ne sont pas satisfaisantes, explorer le gradient boosting (XGBoost, LightGBM)
   - Considérer les approches de stacking combinant plusieurs modèles

4. **Phase expérimentale : Réseaux de neurones**
   - Uniquement si l'explicabilité peut être garantie par des techniques comme LIME ou SHAP
   - Particulièrement utile si les descriptions textuelles des invalidités s'avèrent déterminantes

Cette approche progressive nous permettra de trouver le meilleur équilibre entre performance prédictive et explicabilité, tout en respectant les contraintes éthiques et réglementaires du secteur de l'assurance.

## 3.4 Proposition d'approches et de mesures pour évaluer la qualité de la modélisation

L'évaluation rigoureuse de la qualité des modèles est essentielle pour garantir que notre solution d'IA atteigne les objectifs d'affaires définis. Dans le contexte de notre projet d'assurance invalidité, cette évaluation doit être particulièrement méthodique compte tenu des enjeux éthiques, financiers et humains impliqués.

### Approches d'évaluation recommandées

#### 1. Validation croisée stratifiée (Stratified K-Fold Cross-Validation)

_Pourquoi cette approche ?_

- Préserve la distribution des classes dans chaque fold, critique avec notre déséquilibre de classes (84% cas longs, 16% cas courts)
- Réduit la variance de l'estimation de performance
- Maximise l'utilisation de nos données limitées (5000 enregistrements)
- Permet d'évaluer la stabilité du modèle face à différents échantillons

_Implémentation recommandée :_

- 5 à 10 folds pour un bon compromis entre biais et variance
- Stratification selon la variable cible `Classe_Employe`
- Calcul des métriques de performance sur chaque fold puis moyenne

#### 2. Validation temporelle (Time-Based Validation)

_Pourquoi cette approche ?_

- Respecte la temporalité des données d'assurance (1996-2006)
- Simule les conditions réelles d'utilisation où le modèle prédit des cas futurs
- Détecte les dérives temporelles potentielles dans les caractéristiques des invalidités

_Implémentation recommandée :_

- Division chronologique: entraînement sur 1996-2004, validation sur 2005, test sur 2006
- Évaluation de la stabilité des performances à travers différentes périodes

#### 3. Validation avec sur-échantillonnage équilibré

_Pourquoi cette approche ?_

- Contourne le problème de déséquilibre des classes (16% seulement de cas courts)
- Évite que le modèle ne se spécialise uniquement sur la classe majoritaire
- Évalue la capacité du modèle à identifier correctement les cas moins fréquents

_Implémentation recommandée :_

- Techniques comme SMOTE ou ADASYN pour l'ensemble d'entraînement
- Conservation de la distribution originale dans les ensembles de validation et de test
- Comparaison des performances avec et sans sur-échantillonnage

### Mesures d'évaluation de la qualité

#### 1. Métriques de classification générale

**Matrice de confusion complète**

- Vrais positifs (TP): Cas longs correctement identifiés
- Faux positifs (FP): Cas courts incorrectement classés comme longs
- Vrais négatifs (TN): Cas courts correctement identifiés
- Faux négatifs (FN): Cas longs incorrectement classés comme courts

**Exactitude (Accuracy)**

- Pourcentage global de prédictions correctes: (TP + TN) / (TP + TN + FP + FN)
- Pertinence: Mesure de base, mais insuffisante avec classes déséquilibrées
- Objectif minimum: Dépasser les 80% pour la phase d'amélioration

**Précision et Rappel par classe**

- Précision (cas longs): TP / (TP + FP) - Proportion de cas réellement longs parmi ceux prédits longs
- Rappel (cas longs): TP / (TP + FN) - Proportion de cas longs correctement identifiés
- Précision et rappel similaires pour les cas courts

**Score F1**

- Moyenne harmonique entre précision et rappel: 2 _ (Précision _ Rappel) / (Précision + Rappel)
- Particulièrement pertinent pour notre problème avec classes déséquilibrées

#### 2. Métriques spécifiques au contexte d'affaires

**Coût d'erreur asymétrique**

- Impact financier différent selon le type d'erreur: un cas long attribué à un employé junior (FN) est potentiellement plus coûteux qu'un cas court attribué à un senior (FP)
- Création d'une matrice de coût spécifique aux conséquences métier:
  - FN: Coût élevé (retard de traitement, insatisfaction client, réattribution nécessaire)
  - FP: Coût modéré (inefficacité des ressources)

**Taux d'attribution optimale**

- Pourcentage de cas attribués au bon niveau d'expérience d'employé
- Directement aligné avec l'objectif d'affaires

**Courbe ROC et aire sous la courbe (AUC)**

- Évaluation de la capacité discriminative du modèle à différents seuils
- AUC > 0.8 considéré comme bon, > 0.9 comme excellent

**Courbe Précision-Rappel**

- Plus informative que ROC avec classes déséquilibrées
- Aire sous la courbe précision-rappel (PR-AUC)

#### 3. Évaluation de la robustesse du modèle

**Analyse de sensibilité**

- Test de la stabilité des prédictions face à de petites variations des caractéristiques
- Particulièrement important pour les variables sensibles comme l'âge, le sexe ou le FSA

**Tests sur des sous-groupes démographiques**

- Évaluation des performances sur différents segments (hommes/femmes, tranches d'âge, régions)
- Détection des biais potentiels dans les prédictions

**Évaluation de l'explicabilité**

- Utilisation de SHAP ou LIME pour quantifier la cohérence des explications
- Validation par des experts métier de la logique des décisions

### Intégration dans le processus de développement

Les approches et mesures décrites ci-dessus seront intégrées dans un processus d'évaluation complet:

1. **Phase préliminaire**: Évaluation des modèles de base avec validation croisée stratifiée
2. **Phase d'optimisation**: Affinement des hyperparamètres avec validation temporelle
3. **Phase de sélection**: Comparaison des modèles finaux avec toutes les métriques
4. **Phase de test final**: Évaluation sur l'ensemble de test indépendant et calcul du coût d'erreur métier
5. **Phase de monitoring**: Surveillance continue des performances après déploiement

Cette méthodologie d'évaluation rigoureuse nous permettra de sélectionner et d'optimiser le modèle le plus adapté au problème d'attribution des cas d'invalidité, tout en assurant sa conformité avec les exigences métier, éthiques et réglementaires.

## 3.5 Évaluation des modèles (Model Evaluation)

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
