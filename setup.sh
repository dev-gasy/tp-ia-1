#!/bin/bash

# Script de configuration pour le système de prédiction de durée d'invalidité
# Ce script va préparer l'environnement, générer les données de démonstration et construire les conteneurs

set -e

# Couleurs pour les messages
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Configuration du système de prédiction de durée d'invalidité ===${NC}"

# Vérifier si Docker est installé
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker n'est pas installé. Veuillez l'installer avant de continuer.${NC}"
    exit 1
fi

# Vérifier si Docker Compose est installé
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Docker Compose n'est pas installé. Veuillez l'installer avant de continuer.${NC}"
    exit 1
fi

# Créer les répertoires nécessaires s'ils n'existent pas
echo -e "${YELLOW}Création des répertoires...${NC}"
mkdir -p data models output

# Générer les données de démonstration
echo -e "${YELLOW}Génération des données de démonstration...${NC}"
if [ -f "models/best_model.pkl" ]; then
    echo -e "${YELLOW}Le modèle existe déjà. Voulez-vous le régénérer? (y/n)${NC}"
    read -r response
    if [ "$response" = "y" ]; then
        echo -e "${YELLOW}Génération du modèle...${NC}"
        python generate_demo_data.py
    fi
else
    echo -e "${YELLOW}Génération du modèle...${NC}"
    python generate_demo_data.py
fi

# Construire et démarrer les conteneurs avec Docker Compose
echo -e "${YELLOW}Construction et démarrage des conteneurs...${NC}"
docker-compose build
docker-compose up -d

# Vérifier si les conteneurs sont en cours d'exécution
echo -e "${YELLOW}Vérification des conteneurs...${NC}"
sleep 5

if docker-compose ps | grep -q "invalidite_backend" && docker-compose ps | grep -q "invalidite_frontend"; then
    echo -e "${GREEN}Les conteneurs sont en cours d'exécution!${NC}"
    echo -e "${GREEN}Frontend disponible à l'adresse: http://localhost${NC}"
    echo -e "${GREEN}Backend API disponible à l'adresse: http://localhost:8000${NC}"
    echo -e "${GREEN}Documentation API: http://localhost:8000/docs${NC}"
else
    echo -e "${RED}Certains conteneurs ne sont pas en cours d'exécution. Vérifiez les journaux avec 'docker-compose logs'.${NC}"
    exit 1
fi

echo -e "${GREEN}=== Installation terminée! ===${NC}"
echo -e "${YELLOW}Commandes utiles:${NC}"
echo -e "- 'docker-compose logs' pour voir les journaux"
echo -e "- 'docker-compose down' pour arrêter les conteneurs"
echo -e "- 'docker-compose up -d' pour redémarrer les conteneurs" 