import axios from "axios";

// Configuration de l'URL de base de l'API
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// Création d'une instance axios avec la configuration de base
const api = axios.create({
  baseURL: API_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

// Interface pour les données de réclamation
export interface ClaimData {
  Age?: number;
  Sexe?: string;
  Code_Emploi?: number;
  Salaire_Annuel?: number;
  Duree_Delai_Attente?: number;
  Description_Invalidite?: string;
  Mois_Debut_Invalidite?: number;
  Is_Winter?: number;
  An_Debut_Invalidite?: number;
  Annee_Naissance?: number;
  FSA?: string;
}

// Interface pour la réponse de prédiction
export interface PredictionResponse {
  prediction: number;
  prediction_label: string;
  employee_assignment: string;
  probability: number | null;
}

// Interface pour les informations du modèle
export interface ModelInfo {
  model_type: string;
  classifier_type: string;
  description: string;
}

// Service API
const apiService = {
  // Vérifier si l'API est en fonctionnement
  checkHealth: async (): Promise<{ status: string; model_loaded: boolean }> => {
    const response = await api.get("/health");
    return response.data;
  },

  // Obtenir des informations sur le modèle
  getModelInfo: async (): Promise<ModelInfo> => {
    const response = await api.get("/model_info");
    return response.data;
  },

  // Faire une prédiction
  predict: async (data: ClaimData): Promise<PredictionResponse> => {
    const response = await api.post("/predict", { data });
    return response.data;
  },

  // Obtenir un exemple de données d'entrée
  getSampleInput: async (): Promise<{ data: ClaimData }> => {
    const response = await api.get("/sample_input");
    return response.data;
  },
};

export default apiService;
