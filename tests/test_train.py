import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

def test_train_model_file_exists():
    """Vérifie que le fichier churn_model.pkl est créé dans data/ après exécution de train.py"""
    assert os.path.exists('data/churn_model.pkl'), (
        "Le fichier churn_model.pkl n'existe pas après l'exécution de train.py."
    )

def test_train_model_loading():
    """Vérifie que le fichier sauvegardé contient un modèle Random Forest"""
    model = joblib.load('data/churn_model.pkl')
    assert isinstance(model, LogisticRegression), (
        "Le fichier churn_model.pkl ne contient pas un modèle LogisticRegression."
    )

def test_model_prediction_int():
    """Vérifie que le modèle entraîné peut prédire sur un sous-ensemble des données"""
    model = joblib.load('data/churn_model.pkl')
    data = pd.read_csv('data/customer_churn.csv')
    X = data[["Age", "Years", "Num_Sites", "Account_Manager"]]

    prediction = model.predict(X[:1])
    assert prediction in [0, 1], "La prédiction n'est pas 0 ou 1."