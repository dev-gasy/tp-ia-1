#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Deployment Script for Insurance Claim Duration Prediction
This script creates a REST API to serve the trained model for making predictions
on new insurance claims.
"""

import os
import pickle
import re

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Global variables
model = None
preprocessor = None


def load_model(model_path='models/best_model.pkl'):
    """
    Load the trained model from disk
    """
    global model
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded successfully from {model_path}")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False


def clean_text(text):
    """
    Clean and normalize text data - simplified version without NLTK
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Split into words
    tokens = text.split()

    # Basic English stopwords
    stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
                  'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
                  'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
                  'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
                  'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                  'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
                  'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
                  'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
                  'with', 'about', 'against', 'between', 'into', 'through', 'during',
                  'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
                  'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once'}

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    return ' '.join(tokens)


def calculate_age(year_disability, year_birth):
    """
    Calculate age at time of disability
    """
    return year_disability - year_birth


def preprocess_claim_data(data):
    """
    Preprocess new claim data for prediction
    """
    # Create a DataFrame from the input data
    df = pd.DataFrame([data])

    # Clean text data
    if 'Description_Invalidite' in df.columns:
        df['Description_Clean'] = df['Description_Invalidite'].apply(clean_text)
        df['Description_Word_Count'] = df['Description_Clean'].apply(lambda x: len(str(x).split()))

    # Calculate age
    if 'An_Debut_Invalidite' in df.columns and 'Annee_Naissance' in df.columns:
        df['Age'] = calculate_age(df['An_Debut_Invalidite'], df['Annee_Naissance'])
        df['Age_Squared'] = df['Age'] ** 2

    # Create age categories
    if 'Age' in df.columns:
        bins = [0, 25, 35, 45, 55, 65, 100]
        labels = ['25 or younger', '26-35', '36-45', '46-55', '56-65', '66+']
        df['Age_Category'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

    # Salary transformations
    if 'Salaire_Annuel' in df.columns:
        df['Salaire_Log'] = np.log1p(df['Salaire_Annuel'])

        # Create salary categories
        salary_bins = [0, 20000, 40000, 60000, 100000, float('inf')]
        salary_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        df['Salary_Category'] = pd.cut(df['Salaire_Annuel'], bins=salary_bins, labels=salary_labels)

    # Create seasonal features
    if 'Mois_Debut_Invalidite' in df.columns:
        df['Is_Winter'] = ((df['Mois_Debut_Invalidite'] >= 12) |
                           (df['Mois_Debut_Invalidite'] <= 2)).astype(int)

        df['Is_Summer'] = ((df['Mois_Debut_Invalidite'] >= 6) &
                           (df['Mois_Debut_Invalidite'] <= 8)).astype(int)

    # Drop the original text column which is not needed for prediction
    if 'Description_Invalidite' in df.columns:
        df = df.drop(['Description_Invalidite'], axis=1, errors='ignore')

    return df


@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for making predictions
    """
    try:
        # Get data from request
        data = request.get_json(force=True)

        # Preprocess data
        processed_data = preprocess_claim_data(data.get('data', data))

        # Make prediction
        prediction = model.predict(processed_data)

        # Get probability if available
        try:
            probability = model.predict_proba(processed_data)[0][1].tolist()
        except:
            probability = None

        # Prepare response
        result = {
            'prediction': int(prediction[0]),
            'prediction_label': 'Long (> 180 jours)' if prediction[0] == 1 else 'Court (<= 180 jours)',
            'employee_assignment': 'Employé expérimenté' if prediction[0] == 1 else 'Employé peu expérimenté',
            'probability': probability
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint
    """
    if model is not None:
        return jsonify({'status': 'healthy', 'model_loaded': True}), 200
    else:
        return jsonify({'status': 'unhealthy', 'model_loaded': False}), 503


@app.route('/model_info', methods=['GET'])
def model_info():
    """
    Endpoint to get information about the model
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503

    # Get model information
    try:
        model_type = type(model).__name__

        # Try to get model parameters if available
        try:
            params = model.get_params()
        except:
            params = {"info": "Model parameters not available"}

        return jsonify({
            'model_type': model_type,
            'model_parameters': params,
            'features_used': 15,  # Simplified
            'last_updated': '2023-04-15',  # Example date
            'version': '1.0.0'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/sample_input', methods=['GET'])
def sample_input():
    """
    Provide a sample input format for the API
    """
    sample = {
        'An_Debut_Invalidite': 2022,
        'Mois_Debut_Invalidite': 6,
        'Duree_Delai_Attente': 30,
        'FSA': 'H3Z',
        'Sexe': 'F',
        'Annee_Naissance': 1980,
        'Code_Emploi': 2,
        'Description_Invalidite': 'LOWER BACK PAIN',
        'Salaire_Annuel': 45000
    }

    return jsonify(sample)


def main():
    """
    Main function to start the Flask application
    """
    # Load the model
    success = load_model()

    if not success:
        print("Failed to load model. Application will not start.")
        return

    # Start Flask application
    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port, debug=False)

    print(f"API is running on port {port}")


if __name__ == "__main__":
    # Use port 5002 to avoid conflicts
    os.environ['PORT'] = '5002'
    main()
