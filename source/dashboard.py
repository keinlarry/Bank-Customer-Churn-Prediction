import streamlit as st
import requests
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
from feature_engineering import FeatureEngineering


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# URL Render en production, localhost en local
API_URL = os.getenv("API_URL", "https://bank-customer-churn-prediction-ic06.onrender.com/predire")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'modele', 'bank_churn_model.joblib')

st.set_page_config(
    page_title="ABC Bank — Détecteur de Churn",
    page_icon="🏦",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DU MODÈLE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# TITRE
# ─────────────────────────────────────────────────────────────────────────────
st.title("🏦 ABC Multinational Bank — Détecteur de Churn Client")
st.markdown("**Interface métier** — Évaluation individuelle du risque de départ client")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR : FORMULAIRE CLIENT
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📋 Données du client")

    customer_name    = st.text_input("Nom du client", value="M. Villani")
    credit_score     = st.slider("Score de crédit", 300, 850, 650)
    country          = st.selectbox("Pays", ["France", "Germany", "Spain"])
    gender           = st.selectbox("Sexe", ["Male", "Female"])
    age              = st.slider("Âge", 18, 92, 42)
    tenure           = st.slider("Ancienneté (années)", 0, 10, 3)
    balance          = st.number_input("Solde du compte (€)", 0.0, 300000.0, 75000.0, step=1000.0)
    products_number  = st.selectbox("Nombre de produits", [1, 2, 3, 4])
    credit_card      = st.radio("Carte de crédit ?", [1, 0], format_func=lambda x: "Oui" if x == 1 else "Non")
    active_member    = st.radio("Membre actif ?", [1, 0], format_func=lambda x: "Oui" if x == 1 else "Non")
    estimated_salary = st.number_input("Salaire estimé (€/an)", 0.0, 300000.0, 100000.0, step=1000.0)

    predict_btn = st.button("🔍 Analyser ce client", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAYLOAD
# ─────────────────────────────────────────────────────────────────────────────
payload = {
    "credit_score":     credit_score,
    "country":          country,
    "gender":           gender,
    "age":              age,
    "tenure":           tenure,
    "balance":          balance,
    "products_number":  products_number,
    "credit_card":      credit_card,
    "active_member":    active_member,
    "estimated_salary": estimated_salary,
}

# ─────────────────────────────────────────────────────────────────────────────
# RÉSULTATS
# ─────────────────────────────────────────────────────────────────────────────
if predict_btn:
    col1, col2 = st.columns([1, 2])

    # ── COLONNE GAUCHE : Score API ────────────────────────────────────────────
    with col1:
        st.subheader(f"Score de risque — {customer_name}")
        try:
            response = requests.post(
                API_URL,
                json=payload,
                timeout=10,
                headers={"ngrok-skip-browser-warning": "true"}
                )
            response.raise_for_status()
            result = response.json()

            churn_proba = result["churn_probability"]
            churn_pred  = result["churn_prediction"]
            status_msg  = result["status"]

            # Métrique principale
            st.metric(label="Probabilité de churn", value=f"{churn_proba * 100:.1f} %")
            st.progress(churn_proba)

            # Badge coloré
            if churn_pred == 1:
                st.error(f"🚨 {status_msg}")
            elif churn_proba > 0.4:
                st.warning(f"⚠️ {status_msg}")
            else:
                st.success(f"✅ {status_msg}")

            # Fiche récap
            st.markdown("---")
            st.markdown("**Profil du client :**")
            st.markdown(f"""
- Pays : **{country}** | Sexe : **{gender}** | Âge : **{age} ans**
- Ancienneté : **{tenure} ans** | Produits : **{products_number}**
- Solde : **{balance:,.0f} €** | Actif : **{"Oui" if active_member else "Non"}**
            """)

        except requests.exceptions.ConnectionError:
            st.error("Impossible de joindre l'API FastAPI (port 8000).")
            st.info("Lancez d'abord : `uvicorn main:app --reload --port 8000`")
            st.stop()
        except Exception as e:
            st.error(f"Erreur API : {e}")
            st.stop()

    # ── COLONNE DROITE : SHAP LOCAL ───────────────────────────────────────────
    with col2:
        st.subheader("Pourquoi ce score ? — Explication individuelle (SHAP)")
        st.caption(
            f"Facteurs ayant conduit le score de **{customer_name}** "
            f"(probabilité : {churn_proba*100:.1f}%) vers le churn ou la fidélité"
        )

        try:
            pipeline     = load_model()
            preprocessor = pipeline.named_steps['preprocessing']   # FeatureEngineering
            scaler       = pipeline.named_steps['scaler']           # StandardScaler
            model_gb     = pipeline.named_steps['model']            # GradientBoostingClassifier


            # On ajoute un customer_id fictif car FeatureEngineering le supprime
            client_raw = pd.DataFrame([{"customer_id": 0, **payload}])

            # Appliquons FeatureEngineering
            client_preprocessed = preprocessor.transform(client_raw)

            # Appliquons le StandardScaler
            client_scaled_np  = scaler.transform(client_preprocessed)
            feature_names     = list(client_preprocessed.columns)


            # Calculons les valeurs SHAP pour CE CLIENT UNIQUEMENT
            explainer_local   = shap.TreeExplainer(model_gb)
            shap_values_local = explainer_local.shap_values(client_scaled_np)

            # Normalisation de la forme des shap_values
            if isinstance(shap_values_local, list):
                shap_client = shap_values_local[1][0]
            elif shap_values_local.ndim == 3:
                shap_client = shap_values_local[1][0]
            elif shap_values_local.ndim == 2:
                shap_client = shap_values_local[0]
            else:
                shap_client = shap_values_local

            shap_client  = np.array(shap_client, dtype=np.float64).flatten()
            feature_vals = np.array(client_scaled_np[0], dtype=np.float64).flatten()

            expected_val = explainer_local.expected_value
            if isinstance(expected_val, (list, np.ndarray)):
                expected_val = float(expected_val[1] if len(expected_val) > 1 else expected_val[0])
            else:
                expected_val = float(expected_val)

            # Waterfall Plot
            shap_exp = shap.Explanation(
                values=shap_client,
                base_values=expected_val,
                data=feature_vals,
                feature_names=feature_names
            )


            fig_wf, _ = plt.subplots(figsize=(9, 5))
            shap.plots.waterfall(shap_exp, max_display=10, show=False)
            st.pyplot(plt.gcf(), clear_figure=True)

            # Tableau des contributions (valeurs non-scalées pour lisibilité métier)
            st.markdown("**Détail des contributions :**")
            contrib_df = pd.DataFrame({
                "Variable":          feature_names,
                "Valeur (client)":   [client_preprocessed.iloc[0][f] for f in feature_names],
                "Impact SHAP":       shap_client,
            }).sort_values("Impact SHAP", key=abs, ascending=False)

            def color_shap(val):
                return "background-color: #ffcccc" if val > 0 else "background-color: #ccffcc"

            st.dataframe(
                contrib_df.style
                    .map(color_shap, subset=["Impact SHAP"])
                    .format({"Valeur (client)": "{:.3f}", "Impact SHAP": "{:+.4f}"}),
                use_container_width=True,
                hide_index=True
            )

            st.info(
                "🔴 Impact SHAP **positif** = pousse vers le **churn** | "
                "🟢 Impact SHAP **négatif** = pousse vers la **fidélité**"
            )

        except FileNotFoundError:
            st.warning("Modèle `bank_churn_model.joblib` introuvable dans le répertoire courant.")
        except Exception as e:
            st.error(f"Erreur SHAP : {e}")

# ─────────────────────────────────────────────────────────────────────────────
# PIED DE PAGE
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption("ABC Multinational Bank © 2026 — Outil interne confidentiel | Modèle : GradientBoosting optimisé (RandomizedSearchCV)")