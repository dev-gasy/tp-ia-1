#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script principal pour générer toutes les visualisations nécessaires au rapport.
Ce script coordonne la génération de toutes les visualisations en un seul endroit.
"""

import os
import importlib.util
import sys
from typing import Dict, List, Optional


def ensure_module_exists(module_name: str, file_path: str) -> bool:
    """
    Vérifie si un module existe et l'importe si nécessaire.
    
    Args:
        module_name: Nom du module à vérifier
        file_path: Chemin vers le fichier du module
        
    Returns:
        True si le module existe ou a été importé avec succès, False sinon
    """
    try:
        # Vérifier si le module est déjà importé
        if module_name in sys.modules:
            return True
            
        # Vérifier si le fichier existe
        if not os.path.exists(file_path):
            print(f"Le fichier {file_path} n'existe pas.")
            return False
            
        # Importer le module dynamiquement
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            print(f"Impossible de charger le module {module_name} depuis {file_path}.")
            return False
            
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        return True
    except Exception as e:
        print(f"Erreur lors de l'importation du module {module_name}: {str(e)}")
        return False


def generate_all_visualizations(output_dir: str = 'output') -> Dict[str, bool]:
    """
    Génère toutes les visualisations nécessaires pour le rapport.
    
    Args:
        output_dir: Répertoire de sortie pour les visualisations
        
    Returns:
        Dictionnaire indiquant quelles visualisations ont été générées avec succès
    """
    # Assurer que le répertoire de sortie existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Résultat des opérations
    results = {}
    
    # Liste des modules à exécuter
    modules = [
        {
            'name': 'generate_correlation_matrices',
            'path': 'scripts/generate_correlation_matrices.py',
            'function': 'main'
        },
        {
            'name': 'generate_radar_chart',
            'path': 'scripts/generate_radar_chart.py',
            'function': 'create_radar_chart'
        },
        {
            'name': 'generate_roc_comparison',
            'path': 'scripts/generate_roc_comparison.py',
            'function': 'generate_roc_comparison'
        }
    ]
    
    # Exécuter chaque module
    for module_info in modules:
        module_name = module_info['name']
        file_path = module_info['path']
        function_name = module_info['function']
        
        print(f"\n=== Génération de {module_name} ===")
        
        # Vérifier et importer le module
        if ensure_module_exists(module_name, file_path):
            try:
                # Exécuter la fonction principale du module
                function = getattr(sys.modules[module_name], function_name)
                function()
                results[module_name] = True
                print(f"✓ {module_name} généré avec succès")
            except Exception as e:
                results[module_name] = False
                print(f"✗ Erreur lors de la génération de {module_name}: {str(e)}")
        else:
            results[module_name] = False
            print(f"✗ Module {module_name} non disponible")
    
    return results


def print_summary(results: Dict[str, bool]) -> None:
    """
    Affiche un résumé des résultats de la génération.
    
    Args:
        results: Dictionnaire des résultats de génération
    """
    print("\n=== Résumé de la génération ===")
    successful = [name for name, success in results.items() if success]
    failed = [name for name, success in results.items() if not success]
    
    if successful:
        print(f"Générations réussies ({len(successful)}):")
        for name in successful:
            print(f"  ✓ {name}")
    
    if failed:
        print(f"Générations échouées ({len(failed)}):")
        for name in failed:
            print(f"  ✗ {name}")
    
    success_rate = len(successful) / len(results) * 100 if results else 0
    print(f"\nTaux de réussite: {success_rate:.1f}% ({len(successful)}/{len(results)})")


if __name__ == "__main__":
    try:
        # Générer toutes les visualisations
        results = generate_all_visualizations()
        
        # Afficher le résumé
        print_summary(results)
        
        # Code de sortie basé sur le succès
        success = all(results.values())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Erreur lors de la génération des visualisations: {str(e)}")
        sys.exit(1) 