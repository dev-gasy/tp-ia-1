#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
API FastAPI pour la prédiction de durée d'invalidité
Ce script crée une API REST pour servir le modèle entraîné et faire des prédictions
sur de nouvelles réclamations d'assurance.
"""

import pickle
import re
from typing import Optional

import nltk
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from pydantic import BaseModel, Field

# Télécharger les ressources NLTK (à exécuter une seule fois)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


# Définition des modèles Pydantic pour la validation des données
class ClaimData(BaseModel):
    """Modèle pour les données d'entrée de réclamation"""
    Age: Optional[int] = None
    Sexe: Optional[str] = None
    Code_Emploi: Optional[int] = None
    Salaire_Annuel: Optional[float] = None
    Duree_Delai_Attente: Optional[int] = None
    Description_Invalidite: Optional[str] = None
    Mois_Debut_Invalidite: Optional[int] = None
    Is_Winter: Optional[int] = None
    An_Debut_Invalidite: Optional[int] = None
    Annee_Naissance: Optional[int] = None
    FSA: Optional[str] = None


class PredictionRequest(BaseModel):
    """Modèle pour la requête de prédiction"""
    data: ClaimData


class PredictionResponse(BaseModel):
    """Modèle pour la réponse de prédiction"""
    prediction: int
    prediction_label: str
    employee_assignment: str
    probability: Optional[float] = None


# Initialisation de l'application FastAPI
app = FastAPI(
    title="API de Prédiction de Durée d'Invalidité",
    description="Cette API permet de prédire si un cas d'invalidité sera de courte (≤ 180 jours) ou longue (> 180 jours) durée",
    version="1.0.0"
)

# Configuration CORS pour permettre les requêtes depuis le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales
model = None


def load_model(model_path='../models/best_model.pkl'):
    """
    Charger le modèle entraîné depuis le disque
    """
    global model
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Modèle chargé avec succès depuis {model_path}")
        return True
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {str(e)}")
        return False


def clean_text(text):
    """
    Nettoyer et normaliser les données textuelles
    """
    if not isinstance(text, str):
        return ""

    # Convertir en minuscules
    text = text.lower()

    # Supprimer les caractères spéciaux et les chiffres
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenisation
    tokens = word_tokenize(text)

    # Supprimer les mots vides
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    return ' '.join(tokens)


def calculate_age(year_disability, year_birth):
    """
    Calculer l'âge au moment de l'invalidité
    """
    if year_disability is None or year_birth is None:
        return None
    return year_disability - year_birth


def preprocess_claim_data(data_dict):
    """
    Prétraiter les nouvelles données de réclamation pour la prédiction
    """
    # Créer un DataFrame à partir des données d'entrée
    df = pd.DataFrame([data_dict])

    # Nettoyer les données textuelles
    if 'Description_Invalidite' in df.columns:
        df['Description_Clean'] = df['Description_Invalidite'].apply(clean_text)
        df['Description_Word_Count'] = df['Description_Clean'].apply(lambda x: len(str(x).split()))

    # Calculer l'âge
    if 'Age' not in df.columns and 'An_Debut_Invalidite' in df.columns and 'Annee_Naissance' in df.columns:
        df['Age'] = calculate_age(df['An_Debut_Invalidite'], df['Annee_Naissance'])

    if 'Age' in df.columns:
        df['Age_Squared'] = df['Age'] ** 2

        # Créer des catégories d'âge
        bins = [0, 25, 35, 45, 55, 65, 100]
        labels = ['25 or younger', '26-35', '36-45', '46-55', '56-65', '66+']
        df['Age_Category'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

    # Transformations du salaire
    if 'Salaire_Annuel' in df.columns:
        df['Salaire_Log'] = np.log1p(df['Salaire_Annuel'])

        # Créer des catégories de salaire
        salary_bins = [0, 20000, 40000, 60000, 100000, float('inf')]
        salary_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        df['Salary_Category'] = pd.cut(df['Salaire_Annuel'], bins=salary_bins, labels=salary_labels)

    # Créer des caractéristiques saisonnières
    if 'Is_Winter' not in df.columns and 'Mois_Debut_Invalidite' in df.columns:
        df['Is_Winter'] = ((df['Mois_Debut_Invalidite'] >= 12) |
                           (df['Mois_Debut_Invalidite'] <= 2)).astype(int)

        df['Is_Summer'] = ((df['Mois_Debut_Invalidite'] >= 6) &
                           (df['Mois_Debut_Invalidite'] <= 8)).astype(int)

    # Supprimer la colonne de texte originale qui n'est pas nécessaire pour la prédiction
    if 'Description_Invalidite' in df.columns:
        df = df.drop(['Description_Invalidite'], axis=1, errors='ignore')

    # Encodage one-hot des variables catégorielles
    if 'Sexe' in df.columns:
        df = pd.get_dummies(df, columns=['Sexe'], drop_first=True)

    if 'Age_Category' in df.columns:
        df = pd.get_dummies(df, columns=['Age_Category'], drop_first=True)

    if 'Salary_Category' in df.columns:
        df = pd.get_dummies(df, columns=['Salary_Category'], drop_first=True)

    # Suppression des colonnes qui ne sont pas utilisées pour la prédiction
    cols_to_drop = ['An_Debut_Invalidite', 'Annee_Naissance', 'FSA', 'Description_Clean']
    df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1, errors='ignore')

    return df


@app.on_event("startup")
async def startup_event():
    """
    Événement exécuté au démarrage de l'application
    """
    load_model()


@app.get("/")
async def root():
    """
    Endpoint racine pour vérifier que l'API est en cours d'exécution
    """
    return {"message": "API de Prédiction de Durée d'Invalidité"}


@app.get("/health")
async def health():
    """
    Endpoint de vérification de l'état de santé
    """
    if model is not None:
        return {"status": "healthy", "model_loaded": True}
    else:
        raise HTTPException(status_code=503, detail="Model not loaded")


@app.get("/model_info")
async def model_info():
    """
    Endpoint pour obtenir des informations sur le modèle
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Extraire des informations sur le modèle
    try:
        model_type = type(model).__name__
        if hasattr(model, 'steps'):
            classifier_type = type(model.steps[-1][1]).__name__
        else:
            classifier_type = model_type

        return {
            "model_type": model_type,
            "classifier_type": classifier_type,
            "description": "Modèle de classification pour prédire la durée d'invalidité (courte vs longue)"
        }
    except Exception as e:
        return {"model_type": "Unknown", "error": str(e)}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Endpoint pour faire des prédictions
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convertir les données Pydantic en dictionnaire
        data_dict = request.data.dict()

        # Prétraiter les données
        processed_data = preprocess_claim_data(data_dict)

        # Faire la prédiction
        prediction = model.predict(processed_data)

        # Obtenir la probabilité si disponible
        try:
            probability = float(model.predict_proba(processed_data)[0][1])
        except:
            probability = None

        # Préparer la réponse
        result = {
            "prediction": int(prediction[0]),
            "prediction_label": "Long (> 180 jours)" if prediction[0] == 1 else "Court (≤ 180 jours)",
            "employee_assignment": "Employé expérimenté" if prediction[0] == 1 else "Employé peu expérimenté",
            "probability": probability
        }

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sample_input")
async def sample_input():
    """
    Fournir un exemple de format d'entrée pour l'API
    """
    sample = {
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
    }

    return sample


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
