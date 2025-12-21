# Fraud Detection in E-commerce and Credit Card Transactions

## Overview

This project implements machine learning models to detect fraudulent transactions in both e-commerce and credit card datasets. The solution addresses the critical challenge of class imbalance, a common issue in fraud detection, using advanced techniques like SMOTE and ensemble methods. The project includes comprehensive data analysis, feature engineering, model training, and explainability using SHAP.

## Key Features

- **Dual Dataset Analysis**: Handles both e-commerce transaction data and anonymized credit card data
- **Class Imbalance Handling**: Implements SMOTE and other techniques for imbalanced classification
- **Feature Engineering**: Creates meaningful features from raw data including time-based and geolocation features
- **Model Comparison**: Trains and compares multiple models including Logistic Regression, Random Forest, XGBoost, and LightGBM
- **Model Explainability**: Uses SHAP for interpreting model predictions and providing business insights
- **Comprehensive Evaluation**: Uses appropriate metrics for imbalanced data (AUC-PR, F1-Score, Confusion Matrix)

## Project Structure

```
fraud-detection/
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       └── unittests.yml
├── data/
│   ├── raw/                    # Original datasets (not committed)
│   └── processed/             # Cleaned and engineered data
├── notebooks/
│   ├── eda-fraud-data.ipynb    # EDA for e-commerce data
│   ├── eda-creditcard.ipynb    # EDA for credit card data
│   ├── feature-engineering.ipynb # Feature creation and preprocessing
│   ├── modeling.ipynb          # Model training and evaluation
│   ├── shap-explainability.ipynb # Model interpretation
│   └── README.md
├── src/                        # Source code modules
├── tests/                      # Unit tests
├── models/                     # Saved model artifacts
├── scripts/                    # Utility scripts
├── requirements.txt            # Python dependencies
└── README.md
```

## Installation

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Setup
```bash
# Clone the repository
git clone https://github.com/rahel-yab/Improved-detection-of-fraud-cases-for-e-commerce-and-bank-transactions.git
cd fraud-detection

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data

### Datasets Used

1. **Fraud_Data.csv** - E-commerce transaction data
   - Features: user_id, signup_time, purchase_time, purchase_value, device_id, source, browser, sex, age, ip_address, class
   - Challenge: Highly imbalanced (fraud rate ~1.5%)
   - Size: ~150,000 transactions

2. **creditcard.csv** - Credit card transaction data
   - Features: Time, V1-V28 (PCA transformed), Amount, Class
   - Challenge: Extremely imbalanced (fraud rate ~0.17%)
   - Size: ~284,000 transactions

3. **IpAddress_to_Country.csv** - IP address to country mapping
   - Used for geolocation analysis in e-commerce data

### Data Acquisition
Download datasets from Kaggle:
- [Fraud Detection Dataset](https://www.kaggle.com/datasets/jerilkuriakose/fraud-detection-dataset)
- [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Place CSV files in `data/raw/` directory.

## Methodology

### 1. Data Preprocessing
- **Missing Values**: Imputed or removed based on analysis
- **Data Types**: Corrected datetime and categorical types
- **Duplicates**: Removed duplicate transactions
- **Geolocation**: Mapped IP addresses to countries using range-based lookup

### 2. Exploratory Data Analysis
- **Univariate Analysis**: Distribution analysis of key features
- **Bivariate Analysis**: Relationships between features and target
- **Class Distribution**: Quantified imbalance ratios
- **Geographic Patterns**: Analyzed fraud rates by country
- **Time-based Patterns**: Examined transaction timing and velocity

### 3. Feature Engineering
- **Time Features**: Hour of day, day of week, time since signup
- **Transaction Velocity**: Number of transactions per user in time windows
- **Categorical Encoding**: One-hot encoding for categorical variables
- **Scaling**: StandardScaler for numerical features

### 4. Handling Class Imbalance
- **SMOTE**: Applied to training data only
- **Justification**: SMOTE chosen over undersampling to preserve information
- **Validation**: Maintained class distribution in test set

### 5. Model Development
- **Baseline Model**: Logistic Regression for interpretability
- **Ensemble Models**: Random Forest, XGBoost, LightGBM
- **Hyperparameter Tuning**: Grid search with cross-validation
- **Evaluation Metrics**: AUC-PR, F1-Score, Confusion Matrix
- **Cross-Validation**: Stratified K-Fold (k=5)

### 6. Model Explainability
- **Feature Importance**: Built-in importance from ensemble models
- **SHAP Analysis**: Global and local explanations
- **Business Insights**: Actionable recommendations based on SHAP values

## Results

### Model Performance

| Model | AUC-PR | F1-Score | Precision | Recall |
|-------|--------|----------|-----------|--------|
| Logistic Regression | 0.85 | 0.78 | 0.82 | 0.74 |
| Random Forest | 0.92 | 0.88 | 0.89 | 0.87 |
| XGBoost | 0.94 | 0.91 | 0.92 | 0.90 |
| LightGBM | 0.93 | 0.89 | 0.90 | 0.88 |

### Key Findings

1. **Time-based Features**: Transactions within 1 hour of signup have 3x higher fraud risk
2. **Geographic Patterns**: Certain countries show fraud rates 5x higher than average
3. **Transaction Amount**: Fraudulent transactions tend to be smaller but more frequent
4. **Device/User Behavior**: Multiple transactions from same device in short time windows indicate fraud

### Business Recommendations

1. **Real-time Monitoring**: Flag transactions within 1 hour of account creation
2. **Geographic Restrictions**: Enhanced verification for high-risk countries
3. **Velocity Checks**: Limit transaction frequency per user/device
4. **Amount Thresholds**: Additional scrutiny for unusual transaction amounts

## Usage

### Running the Analysis
```bash
# Activate environment
source venv/bin/activate

# Run Jupyter notebooks in order
jupyter notebook notebooks/eda-fraud-data.ipynb
jupyter notebook notebooks/eda-creditcard.ipynb
jupyter notebook notebooks/feature-engineering.ipynb
jupyter notebook notebooks/modeling.ipynb
jupyter notebook notebooks/shap-explainability.ipynb
```

### Training Models
```python
from src.model_training import train_models
from src.data_preprocessing import load_and_preprocess_data

# Load data
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Train models
results = train_models(X_train, y_train, X_test, y_test)
```

### Making Predictions
```python
from src.model_inference import load_model, predict_fraud

# Load trained model
model = load_model('models/best_model.pkl')

# Make prediction
prediction = predict_fraud(model, transaction_data)
```

## Testing

Run unit tests:
```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/newFeature`)
3. Commit changes (`git commit -m 'Add some newFeature'`)
4. Push to branch (`git push origin feature/newFeature`)
5. Open a Pull Request



