import streamlit as st
import os
from stable_baselines3 import SAC
from energy_env import EnergyMarketEnv


def evaluate_model(model, env, episodes=1000):
    total_reward = 0
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action, _ = model.predict(state, deterministic=True)
            state, reward, done, _ = env.step(action)
            total_reward += reward
    return total_reward / episodes

# Monatswerte
month_params = {
    "January": [0.127, 1.43, 1.035, 40.712, 12.693],
    "February": [0.135, 1.499, 0.899, 44.357, 11.300],
    "March": [0.143, 1.598, 1.150, 39.112, 18.475],
    "April": [0.132, 1.712, 1.114, 38.993, 17.593],
    "May": [0.129, 1.677, 1.441, 34.981, 16.075],
    "June": [0.150, 1.632, 1.244, 30.401, 17.036],
    "July": [0.148, 1.653, 1.700, 38.202, 16.199],
    "August": [0.165, 1.553, 1.433, 35.824, 15.015]
}

month_name_mapping = {
    "January": "jan",
    "February": "feb",
    "March": "mar",
    "April": "apr",
    "May": "may",
    "June": "jun",
    "July": "jul",
    "August": "aug"
}
st.title("Energy Market Simulation")

# Nutzer wählt das Trainingsmodell
train_month = st.selectbox("Wähle den Trainingsmonat für das Modell:", list(month_params.keys()))
# Nutzer wählt die Testumgebung
test_month = st.selectbox("Wähle den Testmonat für die Umgebung:", list(month_params.keys()))
# Wähle die Länge der Simulation 
n_episodes = st.slider("Wähle die Länge der Simulation:", 1, 10000, 1000)


# Konvertiere die Monatsnamen für den Dateinamen
train_month_short = month_name_mapping[train_month]
test_month_short = month_name_mapping[test_month]

efficiency = st.selectbox("Wähle die Effizienz:", [0.6, 0.7, 0.8, 0.9, 1.0])
storage_level_max = st.selectbox("Wähle die maximale Speicherkapazität:", [0, 1.25, 2.5, 3.75, 5, 6.25, 7.5])
train_params = month_params[train_month]
test_params = month_params[test_month]

# Modellpfad erstellen
model_filename = f"SAC_{train_month_short}_{storage_level_max}_{efficiency}_model"
model_path = os.path.join("models", model_filename)

env = EnergyMarketEnv(storage_level_max, efficiency, *test_params)

if os.path.exists(model_path):
    model = SAC.load(model_path, env=env)
    st.success(f"Modell für {train_month} geladen und auf {test_month} getestet!")
else:
    st.warning("Kein Modell gefunden. Starte Training...")
    model = SAC("MlpPolicy", env, learning_rate=3e-3)
    model.learn(total_timesteps=10000)
    model.save(model_path)
    st.success(f"Neues Modell für {train_month} trainiert und gespeichert!")

# Modellbewertung
avg_profit = evaluate_model(model, env, episodes=n_episodes)
st.write(f"Durchschnittliche Belohnung für {test_month} über {n_episodes} Episoden: {avg_profit:.2f}€")