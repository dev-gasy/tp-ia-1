#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility package for the Insurance Claim Duration Prediction project.
This package contains common utility functions and modules used across the project.
"""

# Import all utility functions to make them available at the package level
from .data_utils import (
    download_nltk_resources,
    clean_text,
    calculate_age,
    create_age_categories,
    create_salary_categories,
    create_seasonal_features,
    load_claims_data,
    load_statcan_data
)

from .model_utils import (
    save_model,
    load_model,
    calculate_classification_metrics,
    calculate_business_metrics
)

from .visualization_utils import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_correlation_matrix,
    plot_radar_chart,
    plot_metrics_comparison
)

__all__ = [
    # Data utilities
    'download_nltk_resources',
    'clean_text',
    'calculate_age',
    'create_age_categories',
    'create_salary_categories',
    'create_seasonal_features',
    'load_claims_data',
    'load_statcan_data',
    
    # Model utilities
    'save_model',
    'load_model',
    'calculate_classification_metrics',
    'calculate_business_metrics',
    
    # Visualization utilities
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_correlation_matrix',
    'plot_radar_chart',
    'plot_metrics_comparison'
] 