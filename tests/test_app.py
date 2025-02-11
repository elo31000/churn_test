import pytest

import sys
import os

# Ajouter le chemin de la racine du projet au PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
from app import app

@pytest.fixture
def client():
    """Fixture pour initialiser le client Flask en mode test"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_route(client):
    """Vérifie que la route principale (/) retourne un statut 200"""
    response = client.get('/')
    assert response.status_code == 200, "La route principale (/) ne retourne pas un statut 200."

def test_predict_route_valid(client):
    """Vérifie que la route /predict retourne une prédiction pour des données valides"""
    data = {
        'Age': 30,
        'Account_Manager': 1,
        'Years': 5,
        'Num_Sites': 20
    }
    # Envoyer les données en tant que formulaire encodé
    response = client.post('/predict', data=data)
    assert response.status_code == 200, "La route /predict ne retourne pas un statut 200."

    # Vérifiez que la réponse contient une clé 'churn_prediction'
    json_data = response.get_json()
    assert 'churn_prediction' in json_data, "La réponse ne contient pas de clé 'churn_prediction'."

    # Vérifiez que la valeur 'churn_prediction' est un entier
    json_data = response.get_json()
    assert isinstance(json_data['churn_prediction'], int), "La prédiction n'est pas un enier."
