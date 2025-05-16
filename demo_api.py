#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de démonstration de l'API de prédiction de durée d'invalidité
Ce script montre comment utiliser l'API pour prédire la durée d'une invalidité.
"""

import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure output directory exists
os.makedirs('output', exist_ok=True)

# URL de l'API (lorsqu'elle est lancée localement)
API_URL = "http://localhost:5001"

def test_api_health():
    """Vérifier si l'API est en cours d'exécution"""
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            print("✅ API en fonctionnement")
            return True
        else:
            print(f"❌ API non disponible (code: {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ Erreur de connexion à l'API: {str(e)}")
        return False

def get_model_info():
    """Obtenir des informations sur le modèle déployé"""
    try:
        response = requests.get(f"{API_URL}/model_info")
        if response.status_code == 200:
            model_info = response.json()
            print("\n=== INFORMATIONS SUR LE MODÈLE ===")
            for key, value in model_info.items():
                print(f"{key}: {value}")
            return model_info
        else:
            print(f"❌ Impossible d'obtenir les informations du modèle (code: {response.status_code})")
            return None
    except Exception as e:
        print(f"❌ Erreur: {str(e)}")
        return None

def predict_single_case(case_data):
    """
    Prédire la classe d'un cas d'invalidité
    
    Args:
        case_data: Dictionnaire avec les caractéristiques du cas
        
    Returns:
        Résultat de la prédiction
    """
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"data": case_data}
        )
        
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            print(f"❌ Erreur lors de la prédiction (code: {response.status_code})")
            print(response.text)
            return None
    except Exception as e:
        print(f"❌ Erreur: {str(e)}")
        return None

def display_prediction_result(result, case_data):
    """Afficher le résultat de la prédiction de manière formatée"""
    if not result:
        return
    
    print("\n=== RÉSULTAT DE LA PRÉDICTION ===")
    
    # Afficher les caractéristiques du cas
    print("\nCaractéristiques du cas:")
    for key, value in case_data.items():
        print(f"- {key}: {value}")
    
    # Afficher la prédiction
    prediction = result.get("prediction")
    probability = result.get("probability", 0) * 100
    
    print("\nPrédiction:")
    if prediction == 1:
        print(f"🔴 Cas LONG (> 180 jours) - Confiance: {probability:.2f}%")
        print("📋 Action recommandée: Attribuer à un employé EXPÉRIMENTÉ")
    else:
        print(f"🟢 Cas COURT (≤ 180 jours) - Confiance: {100-probability:.2f}%")
        print("📋 Action recommandée: Attribuer à un employé PEU EXPÉRIMENTÉ")
    
    # Créer une visualisation de la prédiction
    plt.figure(figsize=(10, 3))
    categories = ['Court (≤ 180 jours)', 'Long (> 180 jours)']
    probabilities = [100-probability, probability]
    colors = ['lightgreen', 'lightcoral']
    
    # Afficher les probabilités
    ax = sns.barplot(x=categories, y=probabilities, palette=colors)
    
    # Ajouter une ligne horizontale à 50%
    plt.axhline(y=50, color='black', linestyle='--', alpha=0.7)
    
    # Personnaliser le graphique
    plt.ylim(0, 100)
    plt.ylabel('Probabilité (%)')
    plt.title('Probabilité de la durée d\'invalidité')
    
    # Ajouter les valeurs sur les barres
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{p.get_height():.1f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'bottom',
                    xytext = (0, 5), textcoords = 'offset points')
    
    plt.tight_layout()
    plt.savefig('output/prediction_result.png')
    print("\n📊 Graphique de prédiction enregistré dans 'output/prediction_result.png'")

def predict_multiple_cases(cases):
    """
    Prédire plusieurs cas et générer un rapport de synthèse
    
    Args:
        cases: Liste de dictionnaires avec les caractéristiques des cas
        
    Returns:
        DataFrame avec les résultats
    """
    results = []
    
    print(f"\nPrédiction de {len(cases)} cas...")
    
    for i, case in enumerate(cases):
        print(f"Cas {i+1}/{len(cases)}...", end="\r")
        result = predict_single_case(case)
        if result:
            # Ajouter le résultat et les caractéristiques du cas
            result_dict = {
                "cas_id": i+1,
                "prediction": "Long (> 180 jours)" if result["prediction"] == 1 else "Court (≤ 180 jours)",
                "probabilite": result["probability"],
                "attribution": "Employé EXPÉRIMENTÉ" if result["prediction"] == 1 else "Employé PEU EXPÉRIMENTÉ"
            }
            # Ajouter les caractéristiques du cas
            result_dict.update(case)
            results.append(result_dict)
    
    print(f"✅ Prédiction de {len(results)}/{len(cases)} cas terminée")
    
    # Créer un DataFrame avec les résultats
    if results:
        results_df = pd.DataFrame(results)
        
        # Enregistrer les résultats au format CSV
        results_df.to_csv('output/multiple_predictions.csv', index=False)
        print("📄 Résultats enregistrés dans 'output/multiple_predictions.csv'")
        
        # Créer une visualisation des attributions
        plt.figure(figsize=(10, 6))
        attribution_counts = results_df['attribution'].value_counts()
        colors = ['lightcoral' if 'EXPÉRIMENTÉ' in x else 'lightgreen' for x in attribution_counts.index]
        attribution_counts.plot(kind='bar', color=colors)
        plt.title("Répartition des attributions")
        plt.ylabel("Nombre de cas")
        plt.xlabel("Type d'attribution")
        plt.tight_layout()
        plt.savefig('output/attribution_distribution.png')
        print("📊 Distribution des attributions enregistrée dans 'output/attribution_distribution.png'")
        
        return results_df
    
    return None

def main():
    """Fonction principale de démonstration"""
    print("\n====================================")
    print("DÉMONSTRATEUR D'API DE PRÉDICTION")
    print("====================================\n")
    
    # Vérifier si l'API est en cours d'exécution
    if not test_api_health():
        print("\n⚠️ L'API n'est pas accessible. Veuillez lancer le script model_deployment.py d'abord.")
        print("    Commande: python scripts/model_deployment.py")
        return
    
    # Obtenir des informations sur le modèle
    model_info = get_model_info()
    
    # Exemple de cas d'invalidité (courte durée)
    cas_court = {
        "Age": 35,
        "Sexe": "M",
        "Code_Emploi": 2,
        "Salaire_Annuel": 45000,
        "Duree_Delai_Attente": 30,
        "Description_Invalidite": "FRACTURE JAMBE",
        "Mois_Debut_Invalidite": 6,
        "Is_Winter": 0
    }
    
    # Exemple de cas d'invalidité (longue durée)
    cas_long = {
        "Age": 58,
        "Sexe": "F",
        "Code_Emploi": 4,
        "Salaire_Annuel": 62000,
        "Duree_Delai_Attente": 119,
        "Description_Invalidite": "CANCER",
        "Mois_Debut_Invalidite": 1,
        "Is_Winter": 1
    }
    
    # Prédire un cas court
    print("\n--- PRÉDICTION D'UN CAS COURT ---")
    result_court = predict_single_case(cas_court)
    display_prediction_result(result_court, cas_court)
    
    # Prédire un cas long
    print("\n--- PRÉDICTION D'UN CAS LONG ---")
    result_long = predict_single_case(cas_long)
    display_prediction_result(result_long, cas_long)
    
    # Prédire plusieurs cas
    print("\n--- PRÉDICTION DE PLUSIEURS CAS ---")
    
    # Générer quelques cas aléatoires pour la démonstration
    import numpy as np
    
    multiple_cases = []
    np.random.seed(42)
    
    for i in range(20):
        # Générer aléatoirement des caractéristiques
        age = np.random.randint(25, 65)
        sexe = np.random.choice(["M", "F"])
        code_emploi = np.random.randint(1, 6)
        salaire = np.random.randint(30000, 100000)
        delai = np.random.choice([30, 60, 90, 119])
        mois = np.random.randint(1, 13)
        is_winter = 1 if mois in [12, 1, 2, 3] else 0
        
        # Liste de descriptions possibles
        descriptions = [
            "DEPRESSION", "ANXIETE", "FRACTURE", "CANCER", 
            "MAL DE DOS", "CHIRURGIE", "ACCIDENT", "BURN OUT"
        ]
        description = np.random.choice(descriptions)
        
        # Créer le cas
        case = {
            "Age": age,
            "Sexe": sexe,
            "Code_Emploi": code_emploi,
            "Salaire_Annuel": salaire,
            "Duree_Delai_Attente": delai,
            "Description_Invalidite": description,
            "Mois_Debut_Invalidite": mois,
            "Is_Winter": is_winter
        }
        
        multiple_cases.append(case)
    
    # Prédire les cas multiples
    results_df = predict_multiple_cases(multiple_cases)
    
    if results_df is not None:
        print("\n=== RÉSUMÉ DES PRÉDICTIONS MULTIPLES ===")
        print(f"Nombre total de cas: {len(results_df)}")
        prediction_counts = results_df['prediction'].value_counts()
        print("\nRépartition des prédictions:")
        for label, count in prediction_counts.items():
            print(f"- {label}: {count} cas ({count/len(results_df)*100:.1f}%)")

    print("\n====================================")
    print("FIN DE LA DÉMONSTRATION")
    print("====================================")

if __name__ == "__main__":
    main() 