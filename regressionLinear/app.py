from fastapi import FastAPI
import pandas as pd
import  joblib

# Charger le modèle
model = joblib.load("model.pkl")

# Créer l'application FastAPI
app = FastAPI()

# Route de test
@app.get("/")
def home():
    return {"message": "API de prédiction en ligne 🚀"}

# Endpoint pour faire une prédiction
@app.post("/predict/")
def predict(data: dict):
    try:
        df = pd.DataFrame(data)  # Convertit la liste de dictionnaires en DataFrame
        prediction = model.predict(df)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        return {"error": str(e)}

# Pour tester : envoyer un JSON {"features": [5.1, 3.5, 1.4, 0.2]}
