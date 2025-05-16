#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Insurance Claim Duration Prediction Project - Main Script

This script runs the complete project workflow:
1. Data Processing
2. Model Training
3. KPI Calculation
4. Visualization Generation
5. Ethics and Governance Analysis
6. Model Deployment (optional)
"""

import os
import subprocess
import time
import argparse


def run_data_processing():
    """Run the data processing script."""
    print("\n" + "="*80)
    print("STEP 1: DATA PROCESSING")
    print("="*80)
    
    try:
        from scripts import data_processing
        X_train, X_test, y_train, y_test, preprocessor, feature_names = data_processing.main()
        print("Data processing completed successfully!")
        return True
    except Exception as e:
        print(f"Error during data processing: {str(e)}")
        return False


def run_model_training():
    """Run the model training script."""
    print("\n" + "="*80)
    print("STEP 2: MODEL TRAINING")
    print("="*80)
    
    try:
        from scripts import model_training
        best_model = model_training.main()
        print("Model training completed successfully!")
        return True
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        return False


def run_kpi_calculation():
    """Run the KPI calculation script."""
    print("\n" + "="*80)
    print("STEP 3: KPI CALCULATION")
    print("="*80)
    
    try:
        from scripts import kpi_calculation
        summary_df = kpi_calculation.main()
        print("KPI calculation completed successfully!")
        return True
    except Exception as e:
        print(f"Error during KPI calculation: {str(e)}")
        return False


def run_visualization_generation():
    """Run the visualization generation script."""
    print("\n" + "="*80)
    print("STEP 4: VISUALIZATION GENERATION")
    print("="*80)
    
    try:
        from scripts import visualizations
        visualizations.generate_all_visualizations()
        print("Visualization generation completed successfully!")
        return True
    except Exception as e:
        print(f"Error during visualization generation: {str(e)}")
        return False


def run_ethics_governance():
    """Run the ethics governance analysis and visualizations."""
    print("\n" + "="*80)
    print("STEP 5: ETHICS AND GOVERNANCE ANALYSIS")
    print("="*80)
    
    try:
        # Generate ethics visualizations
        from scripts import generate_ethics_visualizations
        generate_ethics_visualizations.main()
        
        # Try to load model and preprocessor for SHAP explanations
        import pickle
        import os
        import pandas as pd
        
        # Load best model if available
        model = None
        preprocessor = None
        feature_names = None
        
        try:
            if os.path.exists('models/best_model.pkl'):
                with open('models/best_model.pkl', 'rb') as f:
                    model = pickle.load(f)
                print("Model loaded for ethics analysis")
                
                # Extract preprocessor from pipeline if possible
                if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
                    preprocessor = model.named_steps['preprocessor']
                    print("Preprocessor extracted from model pipeline")
            
            # Try to load synthetic data for feature names
            if os.path.exists('data/processed_data.csv'):
                data = pd.read_csv('data/processed_data.csv')
                feature_names = data.drop(['Duree_Invalidite', 'Classe_Employe', 'Description_Invalidite'], 
                                         axis=1, errors='ignore').columns.tolist()
                print(f"Feature names loaded: {len(feature_names)} features")
        except Exception as e:
            print(f"Note: Could not load model artifacts for SHAP: {str(e)}")
            print("Ethics analysis will continue without SHAP explanations")
        
        # Run ethics governance analysis
        from scripts import ethics_governance
        ethics_governance.main(model=model, preprocessor=preprocessor, feature_names=feature_names)
        
        print("Ethics and governance analysis completed successfully!")
        return True
    except Exception as e:
        print(f"Error during ethics and governance analysis: {str(e)}")
        return False


def run_model_deployment(run_in_background=True):
    """Run the model deployment script."""
    print("\n" + "="*80)
    print("STEP 6: MODEL DEPLOYMENT")
    print("="*80)
    
    try:
        if run_in_background:
            # Run Flask application in a subprocess
            print("Starting model deployment API in the background...")
            deployment_process = subprocess.Popen(
                ["python", "scripts/model_deployment.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Give the server a moment to start
            time.sleep(2)
            
            # Check if the process is still running
            if deployment_process.poll() is None:
                print("API server is running in the background. Access it at http://localhost:5000")
                print("To stop the server, you'll need to terminate the process manually.")
                
                # Test the API health endpoint
                import requests
                try:
                    response = requests.get("http://localhost:5000/health")
                    if response.status_code == 200:
                        print("API health check: Healthy")
                    else:
                        print(f"API health check: Unhealthy (Status Code: {response.status_code})")
                except:
                    print("Could not connect to API. Make sure it's running correctly.")
                
                return True
            else:
                stdout, stderr = deployment_process.communicate()
                print(f"API server failed to start: {stderr.decode('utf-8')}")
                return False
        else:
            # Run the deployment directly in this process
            from scripts import model_deployment
            model_deployment.main()
            return True
    except Exception as e:
        print(f"Error during model deployment: {str(e)}")
        return False


def main():
    """Run the complete project workflow."""
    parser = argparse.ArgumentParser(description='Run the complete Insurance Claim Duration Prediction project workflow.')
    parser.add_argument('--skip-processing', action='store_true', help='Skip the data processing step')
    parser.add_argument('--skip-training', action='store_true', help='Skip the model training step')
    parser.add_argument('--skip-kpi', action='store_true', help='Skip the KPI calculation step')
    parser.add_argument('--skip-visualization', action='store_true', help='Skip the visualization generation step')
    parser.add_argument('--skip-ethics', action='store_true', help='Skip the ethics and governance analysis step')
    parser.add_argument('--deploy', action='store_true', help='Run the model deployment step')
    parser.add_argument('--foreground', action='store_true', help='Run deployment in foreground instead of background')
    
    args = parser.parse_args()
    
    print("\n" + "*"*80)
    print("INSURANCE CLAIM DURATION PREDICTION PROJECT")
    print("*"*80)
    
    # Create necessary directories
    for directory in ['data', 'output', 'models', 'output/ethics', 'logs']:
        os.makedirs(directory, exist_ok=True)
    
    # Track successful steps
    steps_success = {
        "Data Processing": None,
        "Model Training": None,
        "KPI Calculation": None,
        "Visualization Generation": None,
        "Ethics and Governance": None,
        "Model Deployment": None
    }
    
    # Run data processing
    if not args.skip_processing:
        steps_success["Data Processing"] = run_data_processing()
    else:
        print("\nSkipping data processing step...")
    
    # Run model training
    if not args.skip_training:
        if args.skip_processing or steps_success["Data Processing"]:
            steps_success["Model Training"] = run_model_training()
        else:
            print("\nSkipping model training because data processing failed...")
    else:
        print("\nSkipping model training step...")
    
    # Run KPI calculation
    if not args.skip_kpi:
        if args.skip_training or steps_success["Model Training"] or args.skip_processing:
            steps_success["KPI Calculation"] = run_kpi_calculation()
        else:
            print("\nSkipping KPI calculation because previous steps failed...")
    else:
        print("\nSkipping KPI calculation step...")
    
    # Run visualization generation
    if not args.skip_visualization:
        if args.skip_kpi or steps_success["KPI Calculation"] or args.skip_training:
            steps_success["Visualization Generation"] = run_visualization_generation()
        else:
            print("\nSkipping visualization generation because previous steps failed...")
    else:
        print("\nSkipping visualization generation step...")
    
    # Run ethics and governance analysis
    if not args.skip_ethics:
        if args.skip_visualization or steps_success["Visualization Generation"] or args.skip_kpi:
            steps_success["Ethics and Governance"] = run_ethics_governance()
        else:
            print("\nSkipping ethics and governance analysis because previous steps failed...")
    else:
        print("\nSkipping ethics and governance analysis step...")
    
    # Run model deployment
    if args.deploy:
        steps_success["Model Deployment"] = run_model_deployment(not args.foreground)
    else:
        print("\nSkipping model deployment step. Use --deploy to run the deployment.")
    
    # Print summary
    print("\n" + "="*80)
    print("PROJECT WORKFLOW SUMMARY")
    print("="*80)
    
    for step, success in steps_success.items():
        if success is None:
            status = "SKIPPED"
        elif success:
            status = "SUCCESS"
        else:
            status = "FAILED"
        print(f"{step}: {status}")
    
    print("\nProject workflow completed!")


if __name__ == "__main__":
    main() 