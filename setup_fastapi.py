#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup script for FastAPI service
"""

import os
import sys

import uvicorn
from sklearn.ensemble import RandomForestClassifier

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the app
from backend.app import app, model

# Set up a mock model
app.model = RandomForestClassifier()
model = RandomForestClassifier()

if __name__ == '__main__':
    # Run the FastAPI app
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
