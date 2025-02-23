import streamlit as st
import numpy as np
import joblib  # Importer joblib pour charger les modèles pickle
import cv2
from collections import Counter
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

# Cache le modèle pour éviter de le recharger à chaque interaction
@st.cache_resource
def load_model_text_generation():
    return pipeline("text-generation", model="distilgpt2")

@st.cache_resource
def load_model_text_translate():
    return pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")


# -----------------------
# 1. Charger les modèles sauvegardés avec joblib (.pkl)
# -----------------------
model_paths = ["cnn_pneumonia_model.pkl", "augmented_data_cnn_pneumonia_model.pkl", "resNet50_pneumonia_model.pkl"] # "augmented_data_cnn_pneumonia_model.pkl"]  # Mets ici les bons chemins de tes modèles
models = [joblib.load(path) for path in model_paths]  # Charger les modèles avec joblib

# -----------------------
# 2. Fonction pour charger et prétraiter l'image
# -----------------------
def preprocess_image(image, img_size=224):
    """Prétraitement de l'image pour correspondre aux modèles."""
    #image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = cv2.resize(image, (img_size, img_size))  # Redimensionner
    image = image / img_size # Normalisation
    image = np.expand_dims(image, axis=0)  # Ajouter une dimension batch
    return image

# -----------------------
# 3. Fonction pour faire la prédiction avec Majority Voting
# -----------------------
def predict_with_majority_voting(image):
    """Fait une prédiction avec les 3 modèles et applique le Majority Voting."""
    predictions = []
    prob_predictions = []
    for model in models:
        prob_pred = model.predict(image)  # Prédiction avec le modèle joblib
        prob_predictions.append(prob_pred)
        class_pred = int(prob_pred >= 0.99)  # 0 = Normal, 1 = Pneumonie
        predictions.append(class_pred)
    
    # Majority Voting : la classe la plus prédite est choisie
    final_prediction = Counter(predictions).most_common(1)[0][0]
    
    return final_prediction, prob_predictions

# -----------------------
# 4. Interface Streamlit
# -----------------------
st.set_page_config(page_title="Détection de Pneumonie", layout="centered")

st.title("🩺 Détection de Pneumonie via Radiographie Pulmonaire")
st.write("Téléchargez une image de radiographie pour voir si elle présente une pneumonie.")

uploaded_file = st.file_uploader("📤 Uploadez votre image (format JPG ou PNG)", type=["jpg", "png"])

if uploaded_file is not None:
    # Lire l'image en tant que tableau numpy
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Afficher l'image
    st.image(image, caption="📷 Image chargée", use_container_width=True)

    # Prétraitement
    preprocessed_image = preprocess_image(image)

    # Prédiction avec Majority Voting
    result, probs = predict_with_majority_voting(preprocessed_image)

    # Afficher le résultat
    if result == 1:
        st.error(f"⚠️ **Pneumonie détectée .**")
        categorie = "Pneumonia detected"
    else:
        st.success(f"✅ **Aucune pneumonie détectée.**")
        categorie = "No signs of pneunomia"

    
    st.write(f"🎯 **Certitude du modèle :**  {np.mean(probs)*100:.2f}%")


    # Générer une recommandation avec Transformers
    st.subheader("🤖 Recommandation IA")
    text_generator = load_model_text_generation()
    text_translate = load_model_text_translate()
    recommendation = text_generator(
        f"A patient whose AI model diagnosis indicates {categorie} with {np.mean(probs)*100:.2f}% confidence  should", 
        max_length=100, 
        num_return_sequences=1
    )[0]["generated_text"]
     
    recommendation_fr = text_translate(recommendation)[0]['translation_text']

    st.write(f"💡 **Conseil IA :** {recommendation_fr}")


st.markdown("---")
st.write("👨‍⚕️ Cette application utilise **3 modèles: CNN, CNN avec augmentation d'image, Fine-Tunning ResNet50** avec un **Majority Voting** pour la détection de la pneumonie.")
