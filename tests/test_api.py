from fastapi.testclient import TestClient
from source.main import app

client = TestClient(app)

# Données d'un client type (exemple du README)
CLIENT_NORMAL = {
    "credit_score": 619,
    "country": "France",
    "gender": "Female",
    "age": 42,
    "tenure": 2,
    "balance": 0.0,
    "products_number": 1,
    "credit_card": 1,
    "active_member": 1,
    "estimated_salary": 101348.88
}

# Données d'un client à fort risque de churn
CLIENT_CHURN = {
    "credit_score": 400,
    "country": "Germany",
    "gender": "Female",
    "age": 55,
    "tenure": 1,
    "balance": 120000.0,
    "products_number": 4,
    "credit_card": 0,
    "active_member": 0,
    "estimated_salary": 50000.0
}


# --- Test 1 : Route racine ---
def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


# --- Test 2 : Prédiction retourne le bon format ---
def test_prediction_format():
    response = client.post("/predire", json=CLIENT_NORMAL)
    assert response.status_code == 200
    data = response.json()
    assert "churn_prediction" in data
    assert "churn_probability" in data
    assert "status" in data


# --- Test 3 : La probabilité est bien entre 0 et 1 ---
def test_probabilite_valide():
    response = client.post("/predire", json=CLIENT_NORMAL)
    proba = response.json()["churn_probability"]
    assert 0.0 <= proba <= 1.0


# --- Test 4 : La prédiction est bien 0 ou 1 ---
def test_prediction_binaire():
    response = client.post("/predire", json=CLIENT_NORMAL)
    prediction = response.json()["churn_prediction"]
    assert prediction in [0, 1]


# --- Test 5 : Client à fort risque a une proba élevée ---
def test_client_churn_proba_elevee():
    response = client.post("/predire", json=CLIENT_CHURN)
    proba = response.json()["churn_probability"]
    assert proba > 0.5


# --- Test 6 : Données manquantes retournent une erreur 422 ---
def test_donnees_manquantes():
    response = client.post("/predire", json={"credit_score": 619})
    assert response.status_code == 422


# --- Test 7 : Le status correspond bien à la probabilité ---
def test_status_coherent():
    response = client.post("/predire", json=CLIENT_CHURN)
    data = response.json()
    if data["churn_probability"] > 0.7:
        assert data["status"] == "Alerte critique : Risque de départ très élevé."
    elif data["churn_probability"] > 0.4:
        assert data["status"] == "Attention : Risque modéré, client à surveiller."
    else:
        assert data["status"] == "Fidélité stable : Faible probabilité de départ."