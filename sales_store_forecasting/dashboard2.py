import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------
# 💾 1. DATENLADEN UND VORBEREITUNG
# -----------------------------------
st.title("📊 Sales Analytics Dashboard")

@st.cache_data
def load_data():
    df_train = pd.read_csv('train.csv', parse_dates=['date'])
    df_oil = pd.read_csv('oil.csv', parse_dates=['date'])
    df_stores = pd.read_csv('stores.csv')
    df_transactions = pd.read_csv('transactions.csv', parse_dates=['date'])
    df_holidays = pd.read_csv('holidays_events.csv', parse_dates=['date'])

    df = df_train.copy()
    df = df.merge(df_oil, on='date', how='left').fillna({'dcoilwtico': 0})
    df = df.merge(df_stores, on='store_nbr', how='left').rename(columns={'type': 'type_store'})
    df = df.merge(df_transactions, on=['date', 'store_nbr'], how='left').fillna({'transactions': 0})
    df = df.merge(df_holidays[['date', 'type']], on='date', how='left').fillna({'type': 'Normal Day'})

    return df


def format_sales(value):
    if value >= 1_000_000:  # Million
        return f"{value / 1_000_000:.2f}M€"
    elif value >= 1_000:  # Millier
        return f"{value / 1_000:.1f}K€"
    else:
        return f"{value:.2f}€"  # Valeur brute

df = load_data()

# -----------------------------------
# 📊 2. INTERAKTIVE FILTER
# -----------------------------------
st.sidebar.header("🔍 Filtere deine Daten")

state_selected = st.sidebar.multiselect("🌎 Staat wählen:", df['state'].unique(), default=['Pichincha'])
filtered_cities = df[df['state'].isin(state_selected)]['city'].unique()
city_selected = st.sidebar.multiselect("🏙️ Stadt wählen:", filtered_cities, default=filtered_cities)
filtered_stores = df[(df['state'].isin(state_selected)) & (df['city'].isin(city_selected))]['store_nbr'].unique()
store_selected = st.sidebar.multiselect("🏬 Store wählen:", filtered_stores, default=filtered_stores)

families_selected = st.sidebar.multiselect("📦 Produktfamilien:", df['family'].unique(), default=df['family'].unique())

date_range = st.sidebar.slider(
    "📅 Zeitraum wählen:",
    min_value=df['date'].min().date(),
    max_value=df['date'].max().date(),
    value=(df['date'].min().date(), df['date'].max().date())
)

df_filtered = df[
    (df['state'].isin(state_selected)) &
    (df['city'].isin(city_selected)) &
    (df['store_nbr'].isin(store_selected)) &
    (df['family'].isin(families_selected)) &
    (df['date'] >= pd.to_datetime(date_range[0])) & (df['date'] <= pd.to_datetime(date_range[1]))
]

# -----------------------------------
# 🔢 3. KPIs & KENNZAHLEN
# -----------------------------------
st.subheader("📈 Wichtige KPIs")

total_sales = df_filtered['sales'].sum()
promo_impact = df_filtered[df_filtered['onpromotion'] >= 1]['sales'].sum() / total_sales * 100 if total_sales else 0
num_unique_stores = df_filtered['store_nbr'].nunique()
top_family = df_filtered.groupby('family')['sales'].sum().idxmax() if not df_filtered.empty else "N/A"
top_holiday = df_filtered.query('type != "Normal Day"').groupby('type')['sales'].sum().idxmax() if not df_filtered.empty else "N/A"

st.metric("💰 Gesamtumsatz", f"{total_sales:,.2f}€")
st.metric("🏬 Anzahl Stores", num_unique_stores)
st.metric("🔥 Meistverkaufte Produktfamilie", top_family)
st.metric("🎉 Höchster Umsatz an Feiertag", top_holiday)
st.metric("📢 Promo-Impact", f"{promo_impact:.2f}%")

# -----------------------------------
# 📊 4. VISUALISIERUNGEN
# -----------------------------------

# 🏷️ Umsatzverteilung nach Produktfamilie
st.subheader("📦 Umsatz nach Produktfamilie")
sales_by_family = df_filtered.groupby('family')['sales'].sum().reset_index()
fig = px.bar(
    sales_by_family,
    x='family',
    y='sales',
    title="Umsatz nach Produktfamilie",
    text=sales_by_family['sales'].apply(format_sales),
    labels={'sales': 'Umsatz (€)'},
    color='family',
    color_discrete_sequence=px.colors.qualitative.Set2
)
fig.update_traces(textposition='outside')
st.plotly_chart(fig)

# Visualization: Sales per store
sales_by_store = df_filtered.groupby('store_nbr')['sales'].sum().reset_index()
fig = px.bar(sales_by_store,
    x=[f'{i+1}' for i in sales_by_store['store_nbr'].values],
    y=sales_by_store['sales'],
    text=sales_by_store['sales'].apply(format_sales),
    title="Umsatz nach Filialen",
    labels={'x': 'Filialen', 'y': 'Umsätze (€)'},
    color='store_nbr',
    color_discrete_sequence=px.colors.qualitative.Set2
)
fig.update_traces(textposition='outside')
st.plotly_chart(fig)


# 📍 Umsatz nach Stadt
st.subheader("🏙️ Umsatz nach Stadt")
sales_by_city = df_filtered.groupby('city')['sales'].sum().reset_index()
fig = px.pie(
    sales_by_city, names='city', values='sales', title="Umsatzanteil pro Stadt"
)
st.plotly_chart(fig)

# 📍 Umsatz nach Regionen
st.subheader("🏙️ Umsatz nach Regionen")
sales_by_city = df_filtered.groupby('state')['sales'].sum().reset_index()
fig = px.pie(
    sales_by_city, names='state', values='sales', title="Umsatzanteil pro Region"
)
st.plotly_chart(fig)

# 📅 Saisonale Umsatztrends (gleitender Durchschnitt)
st.subheader("📊 Saisonale Umsatztrends")
df_filtered['year_month'] = df_filtered['date'].dt.to_period('M')
df_trend = df_filtered.groupby('year_month')['sales'].sum().reset_index()
df_trend['year_month'] = df_trend['year_month'].astype(str)

fig = px.line(df_trend, x='year_month', y='sales', title="Umsatztrend (Gleitender Durchschnitt)")
st.plotly_chart(fig)

# 📈 Korrelation zwischen Variablen
st.subheader("📊 Korrelation zwischen Faktoren")
features = st.multiselect("Wähle Variablen für die Korrelation:", ['transactions', 'sales', 'dcoilwtico', 'onpromotion'], default=['transactions', 'sales'])

if len(features) >= 2:
    corr_matrix = df_filtered[features].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale="RdBu",
        text=corr_matrix.values,
        texttemplate="%{text:.2f}"
    ))
    fig.update_layout(title="Korrelationsmatrix")
    st.plotly_chart(fig)

# -----------------------------------
# 📊 5. STATISTISCHE ANALYSEN
# -----------------------------------
st.subheader("📊 Statistische Analysen")

feature = st.selectbox('Wähle eine kategoriale Variable für den Test:', ['city', 'state', 'family', 'store_nbr', 'cluster'])
groups = [df_filtered[df_filtered[feature] == g]['sales'] for g in df_filtered[feature].unique()]

# ANOVA Test
if len(groups) > 1:
    from scipy import stats
    f_stat, p_value = stats.f_oneway(*groups)
    st.write(f"**ANOVA-Test:** F = {f_stat:.2f}, p = {p_value:.4f}")

# Post-hoc Test falls signifikant
#if p_value < 0.05:
#    import scikit_posthocs as sp
#    posthoc = sp.posthoc_dunn(df_filtered, val_col='sales', group_col=feature, p_adjust='bonferroni')
#    st.write(posthoc)

# -----------------------------------
# 📌 FAZIT
# -----------------------------------
st.subheader("📌 Fazit")
st.write("""
🔹 Dieses interaktive Dashboard bietet eine detaillierte Analyse der Verkaufsdaten.  
🔹 Mit interaktiven Filtern können spezifische Stores, Produkte und Zeiträume ausgewählt werden.  
🔹 Statistische Tests helfen, signifikante Zusammenhänge zu erkennen.  
""")

# 👌 Fertig! Dein Dashboard ist jetzt optimiert 🚀
