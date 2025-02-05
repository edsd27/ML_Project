import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Génération d'un DataFrame simulé avec 100 produits, villes et magasins
import numpy as np

np.random.seed(42)
num_products = 100
df = pd.DataFrame({
    'store': [f'Store {i}' for i in range(1, num_products+1)],
    'city': [f'City {i%10}' for i in range(1, num_products+1)],
    'product': [f'Product {i}' for i in range(1, num_products+1)],
    'sales': np.random.randint(10_000, 5_000_000, num_products)
})

# Fonction pour formater les ventes en K/M
def format_sales(value):
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M€"
    elif value >= 1_000:
        return f"{value / 1_000:.1f}K€"
    else:
        return f"{value:.2f}€"

df["formatted_sales"] = df["sales"].apply(format_sales)

# Sélectionner le nombre de valeurs à afficher
top_n = st.slider("Nombre d'éléments à afficher :", min_value=5, max_value=50, value=10, step=5)

# Trier et filtrer les données (Top N)
df_top_products = df.sort_values(by="sales", ascending=False).head(top_n)
df_top_stores = df.groupby("store")["sales"].sum().reset_index().sort_values(by="sales", ascending=False).head(top_n)
df_top_cities = df.groupby("city")["sales"].sum().reset_index().sort_values(by="sales", ascending=False).head(top_n)

# Création des subplots
fig = make_subplots(
    rows=3, cols=1,
    subplot_titles=("Top Products", "Top Stores", "Top Cities"),
    horizontal_spacing=0.15,
    vertical_spacing=0.3
)

# Bar plot 1 : Sales per Product (Top N)
fig.add_trace(
    go.Bar(x=df_top_products['product'], y=df_top_products['sales'], text=df_top_products["formatted_sales"],
           name="Sales per Product", textposition='outside'),
    row=1, col=1
)

# Bar plot 2 : Sales per Store (Top N)
fig.add_trace(
    go.Bar(x=df_top_stores['store'], y=df_top_stores['sales'], text=df_top_stores["sales"].apply(format_sales),
           name="Sales per Store", textposition='outside'),
    row=2, col=1
)

# Bar plot 3 : Sales per City (Top N)
fig.add_trace(
    go.Bar(x=df_top_cities['city'], y=df_top_cities['sales'], text=df_top_cities["sales"].apply(format_sales),
           name="Sales per City", textposition='outside'),
    row=3, col=1
)

# Mise en forme pour améliorer la lisibilité
fig.update_layout(
    title_text="Top Sales Analysis",
    showlegend=False,
    height=1200,
    width=1200
)

# Rotation des labels pour une meilleure lisibilité
fig.update_xaxes(tickangle=45)  # Pivoter les labels

# Affichage dans Streamlit
st.title("Comparative Sales Analysis (Top N)")
st.plotly_chart(fig)
