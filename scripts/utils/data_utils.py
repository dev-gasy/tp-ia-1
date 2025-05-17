#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data utility functions for the Insurance Claim Duration Prediction project.
"""

import re
from typing import Any, Set

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


# Download NLTK resources
def download_nltk_resources(quiet: bool = False) -> None:
    """
    Download required NLTK resources if not already available
    
    Args:
        quiet: If True, suppress download messages
    """
    nltk.download('punkt', quiet=quiet)
    nltk.download('stopwords', quiet=quiet)


# Text processing functions
def clean_text(text: Any) -> str:
    """
    Clean and normalize text data
    
    Args:
        text: Text to clean and normalize
        
    Returns:
        Cleaned and normalized text
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words: Set[str] = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    return ' '.join(tokens)


# Feature engineering functions
def calculate_age(year_disability: int, year_birth: int) -> int:
    """
    Calculate age at time of disability
    
    Args:
        year_disability: Year when disability started
        year_birth: Year of birth
        
    Returns:
        Age in years
    """
    return year_disability - year_birth


def create_age_categories(df: pd.DataFrame, age_col: str = 'Age') -> pd.DataFrame:
    """
    Create age categories based on age
    
    Args:
        df: DataFrame containing age column
        age_col: Name of the age column
        
    Returns:
        DataFrame with age category column added
    """
    bins = [0, 25, 35, 45, 55, 65, 100]
    labels = ['25 or younger', '26-35', '36-45', '46-55', '56-65', '66+']

    df['Age_Category'] = pd.cut(df[age_col], bins=bins, labels=labels, right=False)

    return df


def create_salary_categories(df: pd.DataFrame, salary_col: str = 'Salaire_Annuel') -> pd.DataFrame:
    """
    Create salary categories and logarithm of salary
    
    Args:
        df: DataFrame containing salary column
        salary_col: Name of the salary column
        
    Returns:
        DataFrame with salary features added
    """
    # Log transformation of salary
    df['Salaire_Log'] = np.log1p(df[salary_col])

    # Salary categories
    salary_bins = [0, 20000, 40000, 60000, 100000, float('inf')]
    salary_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    df['Salary_Category'] = pd.cut(df[salary_col], bins=salary_bins, labels=salary_labels)

    return df


def create_seasonal_features(df: pd.DataFrame, month_col: str = 'Mois_Debut_Invalidite') -> pd.DataFrame:
    """
    Create seasonal features based on month
    
    Args:
        df: DataFrame containing month column
        month_col: Name of the month column
        
    Returns:
        DataFrame with seasonal features added
    """
    # Winter flag (December, January, February)
    df['Is_Winter'] = ((df[month_col] >= 12) | (df[month_col] <= 2)).astype(int)

    # Summer flag (June, July, August)
    df['Is_Summer'] = ((df[month_col] >= 6) & (df[month_col] <= 8)).astype(int)

    return df


def load_claims_data(filepath: str = 'data/MODELING_DATA.csv') -> pd.DataFrame:
    """
    Load and prepare the insurance claims dataset
    
    Args:
        filepath: Path to the claims data file
        
    Returns:
        Loaded claims DataFrame
    """
    df = pd.read_csv(filepath)
    print(f"Loaded claims data with {df.shape[0]} rows and {df.shape[1]} columns")
    return df


def load_statcan_data(filepath: str = 'data/StatCanadaPopulationData.csv') -> pd.DataFrame:
    """
    Load and prepare the Statistics Canada population dataset
    
    Args:
        filepath: Path to the StatCan data file
        
    Returns:
        Loaded StatCan DataFrame
    """
    df = pd.read_csv(filepath)
    print(f"Loaded StatCan data with {df.shape[0]} rows and {df.shape[1]} columns")
    return df
