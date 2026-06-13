import sys
import os
import __main__
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from source import feature_engineering

# Patch pour permettre à joblib de reconstruire la classe personnalisée
__main__.FeatureEngineering = feature_engineering.FeatureEngineering

# Initialisons l'application FastAPI
app = FastAPI(
    title="API de Prédiction du Churn Bancaire",
    description="Cette API renvoie la probabilité qu'un client quitte la ABC Multinational Bank.",
    version="1.0.0"
)

# Chargeons le modèle pré-entraîné au démarrage de l'API
try:
    # chargement du modèle
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = joblib.load(os.path.join(BASE_DIR, 'modele', 'bank_churn_model.joblib'))
except Exception as e:
    raise RuntimeError(f"Impossible de charger le fichier du modèle : {e}")

# Définissons le schéma des données d'entrée attendues
class CustomerData(BaseModel):
    credit_score: int = Field(..., description="Score de crédit du client")
    country: str = Field(..., description="Pays de résidence (France, Spain, Germany)")
    gender: str = Field(..., description="Sexe du client (Male, Female)")
    age: int = Field(..., description="Âge du client")
    tenure: int = Field(..., description="Nombre d'années dans la banque")
    balance: float = Field(..., description="Solde actuel du compte")
    products_number: int = Field(..., description="Nombre de produits bancaires détenus")
    credit_card: int = Field(..., description="Possède une carte de crédit (1=Oui, 0=Non)")
    active_member: int = Field(..., description="Membre actif (1=Oui, 0=Non)")
    estimated_salary: float = Field(..., description="Salaire annuel estimé")

    # Configuration d'un exemple de l'API
    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }

# Définissons le schéma de la réponse (Sortie avec les scores)
class PredictionResponse(BaseModel):
    churn_prediction: int = Field(..., description="1 si le client va partir, 0 s'il reste")
    churn_probability: float = Field(..., description="Le score / probabilité de départ (entre 0 et 1)")
    status: str = Field(..., description="Message d'interprétation métier")

# Route racine pour vérifier le bon fonctionnement
@app.get("/")
def home():
    return {"message": "L'API de Churn est en ligne. Accédez à /docs pour tester."}

# La route principale : /predire
@app.post("/predire", response_model=PredictionResponse)
def prediction_churn(customer: CustomerData):
    try:
        # Convertir les données d'entrée Pydantic en DataFrame Pandas
        input_data = pd.DataFrame([customer.model_dump()])

        # Obtenir la prédiction binaire (0 ou 1)
        prediction = int(model.predict(input_data)[0])

        # Obtenir les scores/probabilités (proba d'appartenir à la classe 1 : Churn)
        probabilities = model.predict_proba(input_data)[0]
        churn_proba = float(probabilities[1])

        # Petit message d'aide à la décision
        if churn_proba > 0.7:
            status = "Alerte critique : Risque de départ très élevé."
        elif churn_proba > 0.4:
            status = "Attention : Risque modéré, client à surveiller."
        else:
            status = "Fidélité stable : Faible probabilité de départ."

        return PredictionResponse(
            churn_prediction=prediction,
            churn_probability=round(churn_proba, 4),
            status=status
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")