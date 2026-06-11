# <div align="center">Bank Customer Churn Prediction — Production</div>

<div align="center">

![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?logo=github-actions)
![Deploy](https://img.shields.io/badge/Deploy-Render-46E3B7?logo=render)
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136.3-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58.0-FF4B4B?logo=streamlit)

> Branche de mise en production du modèle de prédiction du churn bancaire.  
> Déploiement automatisé via GitHub Actions → Render.
</div>


## Liens

| Service | URL |
|---|---|
| **API FastAPI** | `https://bank-customer-churn-prediction-ic06.onrender.com` |
| **Dashboard Streamlit** | `https://bank-churn-dashboard-pzop.onrender.com/` |

---

## Pipeline CI/CD

```
git push origin production
        ↓
GitHub Actions — Job : test
  ├── pip install -r requirements.txt
  ├── vérification import API
  ├── vérification import feature engineering
  └── pytest tests/ (7 tests)
        ↓ (si OK)
GitHub Actions — Job : deploy
  ├── curl → Deploy Hook API (Render)
  └── curl → Deploy Hook Dashboard (Render)
        ↓
Render redéploie les deux services automatiquement
```

> Si les tests échouent, le déploiement est **bloqué automatiquement**.

---

## Structure

```
production/
│
├── .github/
│   └── workflows/
│       └── deploy.yml            # Pipeline CI/CD
│
├── modele/
│   └── bank_churn_model.joblib   # Modèle sérialisé (GradientBoosting)
│
├── source/
│   ├── main.py                   # API REST FastAPI
│   ├── dashboard.py              # Dashboard Streamlit
│   └── feature_engineering.py    # Pipeline de transformation
│
├── tests/
│   ├── __init__.py
│   └── test_api.py               # 7 tests pytest sur les endpoints
│
├── requirements.txt
└── README.md
```

---

## Lancer en local

**1. Installer les dépendances**

```bash
pip install -r requirements.txt
```

**2. Lancer l'API FastAPI**

```bash
uvicorn source.main:app --reload --port 8000
```

**3. Lancer le Dashboard Streamlit**

```bash
streamlit run source/dashboard.py
# Interface interactive
```

---

## Tests

```bash
pip install pytest httpx
pytest tests/ -v
```

| Test | Description |
|---|---|
| `test_home` | Route racine répond 200 |
| `test_prediction_format` | Réponse contient les 3 champs attendus |
| `test_probabilite_valide` | Probabilité entre 0 et 1 |
| `test_prediction_binaire` | Prédiction vaut 0 ou 1 |
| `test_client_churn_proba_elevee` | Client à risque → proba > 0.5 |
| `test_donnees_manquantes` | Données incomplètes → erreur 422 |
| `test_status_coherent` | Message status cohérent avec la probabilité |

---

## Exemple d'appel API

```bash
curl -X POST "https://bank-churn-api.onrender.com/predire" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Réponse attendue :**

```json
{
  "churn_prediction": 0,
  "churn_probability": 0.1823,
  "status": "Fidélité stable : Faible probabilité de départ."
}
```

---

## Stack technique

| Catégorie | Technologies |
|---|---|
| **Langage** | Python 3.12 |
| **API REST** | FastAPI, Pydantic, Uvicorn |
| **Dashboard** | Streamlit |
| **ML** | Scikit-Learn 1.9, Joblib |
| **Explicabilité** | SHAP |
| **Tests** | Pytest, HTTPX |
| **CI/CD** | GitHub Actions |
| **Hébergement** | Render |

---

## Branche principale

Le code de recherche, l'EDA et la modélisation sont sur la branche [`main`](https://github.com/keinlarry/Bank-Customer-Churn-Prediction/blob/main/notebook/Bank_Customer_Churn.ipynb).