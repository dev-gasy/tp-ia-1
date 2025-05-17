#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization utility functions for the Insurance Claim Duration Prediction project.
"""

import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix


def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        classes: List[str] = ['Court (<= 180 jours)', 'Long (> 180 jours)'],
        model_name: str = 'Model',
        output_dir: str = 'output'
) -> Dict[str, int]:
    """
    Plot and save confusion matrix visualization.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: Class labels
        model_name: Name of the model
        output_dir: Directory to save the output
        
    Returns:
        Dictionary with confusion matrix metrics
    """
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create a safe filename
    safe_name = model_name.lower().replace(' ', '_')

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 12})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14)
    plt.tight_layout()

    # Save the plot
    output_path = os.path.join(output_dir, f'confusion_matrix_{safe_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Confusion matrix saved to {output_path}")

    # Extract metrics from confusion matrix
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics = {
            'true_negative': tn,
            'false_positive': fp,
            'false_negative': fn,
            'true_positive': tp
        }
        return metrics
    else:
        return {"error": "Confusion matrix is not 2x2"}


def plot_roc_curve(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        model_name: str = 'Model',
        output_dir: str = 'output'
) -> float:
    """
    Plot and save ROC curve visualization.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        model_name: Name of the model
        output_dir: Directory to save the output
        
    Returns:
        Area Under the Curve (AUC) value
    """
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create a safe filename
    safe_name = model_name.lower().replace(' ', '_')

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    output_path = os.path.join(output_dir, f'roc_curve_{safe_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"ROC curve saved to {output_path}")

    return roc_auc


def plot_correlation_matrix(
        data: pd.DataFrame,
        title: str = 'Correlation Matrix',
        mask_upper: bool = True,
        figsize: Tuple[int, int] = (12, 10),
        output_path: str = 'output/correlation_matrix.png'
) -> None:
    """
    Plot and save correlation matrix visualization.
    
    Args:
        data: DataFrame with data to plot correlation
        title: Title of the plot
        mask_upper: Whether to mask the upper triangle
        figsize: Figure size (width, height)
        output_path: Path to save the output file
    """
    # Make sure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Compute correlation matrix
    corr = data.corr()

    # Create mask for the upper triangle
    mask = np.zeros_like(corr, dtype=bool) if not mask_upper else np.triu(np.ones_like(corr, dtype=bool))

    # Create figure
    plt.figure(figsize=figsize)

    # Plot heatmap
    sns.heatmap(
        corr, mask=mask, annot=True, fmt='.2f',
        cmap='coolwarm', center=0, linewidths=0.5,
        cbar_kws={'shrink': 0.8},
        annot_kws={"size": 10}  # Set font size for annotations
    )

    # Add title and adjust layout
    plt.title(title, fontsize=16)
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Correlation matrix saved to {output_path}")


def plot_radar_chart(
        metrics_data: Dict[str, List[float]],
        categories: List[str],
        title: str = 'Model Comparison',
        figsize: Tuple[int, int] = (10, 8),
        output_path: str = 'output/radar_chart_comparison.png'
) -> None:
    """
    Generate and save a radar chart comparing models across metrics.
    
    Args:
        metrics_data: Dictionary with model names as keys and lists of metric values as values
        categories: Names of the metrics/categories to display on the radar
        title: Title of the plot
        figsize: Figure size (width, height)
        output_path: Path to save the output file
    """
    # Make sure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Number of variables
    N = len(categories)

    # Set up the angles for each category
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

    # Make it a full circle by repeating the first angle
    angles += angles[:1]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    # Colors for each model
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_data)))

    # Plot each model
    for i, (model_name, metric_values) in enumerate(metrics_data.items()):
        # Make sure the values form a complete circle
        values = metric_values.copy()
        values += values[:1]

        # Plot the model metrics
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)

    # Add legend, title and adjust layout
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title(title, size=16, pad=20)

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Radar chart saved to {output_path}")


def plot_metrics_comparison(
        metrics_df: pd.DataFrame,
        x_col: str,
        metrics_cols: List[str],
        title: str = 'Model Metrics Comparison',
        figsize: Tuple[int, int] = (12, 8),
        output_path: str = 'output/metrics_comparison.png'
) -> None:
    """
    Generate and save a bar chart comparing model metrics.
    
    Args:
        metrics_df: DataFrame with model metrics
        x_col: Column name to use for x-axis (typically model name)
        metrics_cols: List of column names for the metrics to display
        title: Title of the plot
        figsize: Figure size (width, height)
        output_path: Path to save the output file
    """
    # Make sure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create figure
    plt.figure(figsize=figsize)

    # Plot grouped bar chart
    metrics_df.plot(
        x=x_col, y=metrics_cols, kind='bar',
        ax=plt.gca(), rot=0, width=0.8
    )

    # Add title and labels
    plt.title(title, fontsize=16)
    plt.xlabel('')
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0, 1.1)

    # Add grid
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add legend and adjust layout
    plt.legend(title='Metrics', fontsize=10)
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Metrics comparison chart saved to {output_path}")


def plot_roc_comparison(
        roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
        title: str = 'ROC Curve Comparison',
        figsize: Tuple[int, int] = (10, 8),
        output_path: str = 'output/roc_comparison_chart.png'
) -> None:
    """
    Generate and save a comparison of ROC curves for multiple models.
    
    Args:
        roc_data: Dictionary with model names as keys and tuples of (fpr, tpr, auc) as values
        title: Title of the plot
        figsize: Figure size (width, height)
        output_path: Path to save the output file
    """
    # Make sure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create figure
    plt.figure(figsize=figsize)

    # Colors for each model
    colors = plt.cm.tab10(np.linspace(0, 1, len(roc_data)))

    # Plot each ROC curve
    for i, (model_name, (fpr, tpr, roc_auc)) in enumerate(roc_data.items()):
        plt.plot(
            fpr, tpr, lw=2,
            label=f'{model_name} (AUC = {roc_auc:.3f})',
            color=colors[i]
        )

    # Add diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', lw=1)

    # Set axis limits
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    # Add labels and title
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=16)

    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"ROC comparison chart saved to {output_path}")


# Set common visualization style
def set_visualization_style() -> None:
    """Set global visualization style for consistent plots."""
    sns.set_style('whitegrid')
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 14  # Increase the default font size
