# Utiliser une image Python
FROM python:3.12

# Définir le répertoire de travail
WORKDIR /competetion

# Copier les fichiers nécessaires
COPY ../requirements.txt ./
COPY app.py ./
COPY model.pkl .


# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port de l'API
EXPOSE 8000

# Lancer FastAPI avec Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
