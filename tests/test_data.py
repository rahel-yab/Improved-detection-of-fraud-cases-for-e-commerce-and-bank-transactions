import pytest
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

def test_fraud_data_loading():
    """Test loading of processed fraud data"""
    fraud_df = pd.read_csv('data/processed/fraud_data_processed.csv')
    assert fraud_df.shape[0] > 0, "Fraud data should not be empty"
    assert 'class' in fraud_df.columns, "Target column should be present"

def test_credit_data_loading():
    """Test loading of processed credit card data"""
    credit_df = pd.read_csv('data/processed/creditcard_processed.csv')
    assert credit_df.shape[0] > 0, "Credit data should not be empty"
    assert 'Class' in credit_df.columns, "Target column should be present"

def test_smote_data_shapes():
    """Test that SMOTE balanced the classes"""
    X_fraud = joblib.load('data/processed/X_fraud_smote.pkl')
    y_fraud = joblib.load('data/processed/y_fraud_smote.pkl')
    
    assert X_fraud.shape[0] == y_fraud.shape[0], "X and y should have same number of samples"
    assert len(np.unique(y_fraud)) == 2, "Should have 2 classes after SMOTE"
    
    # Check class balance
    unique, counts = np.unique(y_fraud, return_counts=True)
    assert counts[0] == counts[1], "Classes should be balanced after SMOTE"

def test_credit_smote_data_shapes():
    """Test credit card SMOTE data"""
    X_credit = joblib.load('data/processed/X_credit_smote.pkl')
    y_credit = joblib.load('data/processed/y_credit_smote.pkl')
    
    assert X_credit.shape[0] == y_credit.shape[0], "X and y should have same number of samples"
    assert len(np.unique(y_credit)) == 2, "Should have 2 classes"
    
    unique, counts = np.unique(y_credit, return_counts=True)
    assert counts[0] == counts[1], "Classes should be balanced"

def test_preprocessor_loading():
    """Test loading of preprocessing models"""
    preprocessor = joblib.load('models/fraud_preprocessor.pkl')
    scaler = joblib.load('models/credit_scaler.pkl')
    
    assert preprocessor is not None, "Preprocessor should load"
    assert scaler is not None, "Scaler should load"
    assert isinstance(scaler, StandardScaler), "Scaler should be StandardScaler"