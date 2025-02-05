import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats
import scikit_posthocs as sp
import statsmodels.api as sm
import plotly.graph_objects as go

def format_sales(value):
    if value >= 1_000_000:  # Million
        return f"{value / 1_000_000:.2f}M€"
    elif value >= 1_000:  # Millier
        return f"{value / 1_000:.1f}K€"
    else:
        return f"{value:.2f}€"  # Valeur brute

df_train  =  pd.read_csv('train.csv', parse_dates=['date'])
df_oil = pd.read_csv('oil.csv', parse_dates=['date'])
df_sample_submission = pd.read_csv('sample_submission.csv')
df_stores = pd.read_csv('stores.csv')
df_test = pd.read_csv('test.csv', parse_dates=['date'])
df_transaction = pd.read_csv('transactions.csv', parse_dates=['date'])
df_holidays_events = pd.read_csv('holidays_events.csv', parse_dates=['date'])

df = df_train.copy()
# Merge train data with oil prices
df = pd.merge(df, df_oil, on='date', how='left')
df['dcoilwtico'] = df['dcoilwtico'].fillna(0)

# Merge with store information
df = pd.merge(df, df_stores, on='store_nbr', how='left')
df.rename(columns={'type': 'type_store'},inplace=True)

df = pd.merge(df, df_transaction, on=['date', 'store_nbr'], how='left')
df.transactions = df.transactions.fillna(0)

df = pd.merge(df, df_holidays_events[['date', 'type']], on='date', how='left')
df.fillna('Normal Day', inplace=True)


# Create interactive dashboard
st.title("📊 Sales Dashboard")

# Interactive Filters
# Sélection des États en premier
state_selected = st.multiselect("Select state:", df['state'].unique(),default=['Pichincha'])

# Filtrer les villes disponibles en fonction des États sélectionnés
filtered_cities = df[df['state'].isin(state_selected)]['city'].unique()

#selectioner les villes d
city_selected = st.multiselect("Select city:", filtered_cities, default=filtered_cities)

# Filtrer les magasins disponibles en fonction des États et des Villes sélectionnés
filtered_stores = df[(df['state'].isin(state_selected)) & (df['city'].isin(city_selected))]['store_nbr'].unique()

# Sélection des magasins
store_selected = st.multiselect("Select a Store:", filtered_stores, default=filtered_stores)

df_filtered = df[
    (df['state'].isin(state_selected)) &
    (df['city'].isin(city_selected)) &
    (df['store_nbr'].isin(store_selected))
]


families_selected = st.multiselect("Select Product Families:", df_filtered['family'].unique(), default=df_filtered['family'].unique())
df_filtered = df_filtered[df_filtered['family'].isin(families_selected)]

# Using Streamlit to create a date range slider
date_range = st.slider(
    "Select Date Range:",
    min_value=df['date'].min().date(),  # min_value should be a datetime.date object
    max_value=df['date'].max().date(),  # max_value should be a datetime.date object
    value=(df['date'].min().date(), df['date'].max().date())  # Default value as the full date range
)

# You can use the selected date range here
df_filtered = df_filtered[(df_filtered['date'] >= pd.to_datetime(date_range[0])) & (df_filtered['date'] <= pd.to_datetime(date_range[1]))]

st.write(df_filtered)

# KPIs
num_unique_families = df_filtered['family'].nunique()
num_unique_stores = df_filtered['store_nbr'].nunique()
num_unique_clusters = df_filtered['cluster'].nunique()
total_sales = df_filtered['sales'].sum()
top_family = df_filtered.groupby('family')['sales'].sum().idxmax()
top_holiday = df_filtered.query('type != "Normal Day"').groupby('type')['sales'].sum().idxmax()
promo_impact = df_filtered[df_filtered['onpromotion'] >= 1]['sales'].sum() / df_filtered['sales'].sum() * 100

# Growth Metrics
df_filtered['year'] = df_filtered['date'].dt.year
sales_by_year = df_filtered.groupby('year')['sales'].sum().reset_index()

# Streamlit Dashboard
st.subheader("Summary & KPIs")

# KPIs Display
st.write(f"**🛍️ Total Product Families:** {num_unique_families}")
st.write(f"**🏬 Total Stores:** {num_unique_stores}")
st.write(f"**📊 Total Clusters:** {num_unique_clusters}")
st.write(f"**💰 Total Sales :** {format_sales(total_sales)}")
st.write(f"**🔥 Top Selling Product Family:** {top_family}")
st.write(f"**🎉 Top Holiday with Most Sales:** {top_holiday}")
st.write(f"**📢 Impact of Promotions on Sales:** {promo_impact:.2f}%")


# Visualization: Sales per Family
st.subheader("Sales Analysis")
sales_by_family = df_filtered.groupby('family')['sales'].sum().reset_index()
fig = px.bar(sales_by_family,
    x=sales_by_family['family'],
    y=sales_by_family['sales'],
    text=sales_by_family['sales'].apply(format_sales),
    title="Total Sales by Product Family",
    labels={'x': 'Product Family', 'y': 'Sales (€)'},
    color='family',
    color_discrete_sequence=px.colors.qualitative.Set2
)
fig.update_traces(textposition='outside')

fig.update_xaxes(tickangle=45)

fig.update_layout(
    showlegend=False,
    height=600,
    width=1200
)

st.plotly_chart(fig)

# Visualization: Sales per store
top_n = st.slider("Display top sales by store:", min_value=5, max_value=df_filtered.store_nbr.nunique(), value=5, step=2)
sales_by_store = df_filtered.groupby('store_nbr')['sales'].sum().reset_index().sort_values(by='sales', ascending=False).head(top_n)
fig = px.bar(sales_by_store,
    x=[f'store {i+1}' for i in sales_by_store['store_nbr'].values],
    y=sales_by_store['sales'],
    text=sales_by_store['sales'].apply(format_sales),
    title="Total Sales by store",
    labels={'x': 'stores', 'y': 'Sales (€)'},
    color='store_nbr',
    color_discrete_sequence=px.colors.qualitative.Set2
)
fig.update_traces(textposition='outside')
st.plotly_chart(fig)

# Visualization: Sales per city
sales_by_city = df_filtered.groupby('city')['sales'].sum().reset_index()
fig = px.pie(sales_by_city,names='city',values='sales',title="Rate Sales by cities",color='city',color_discrete_sequence=px.colors.qualitative.Set2)
st.plotly_chart(fig)

# Visualization: Sales by Holiday Type
sales_by_holiday_type = df_filtered.query('type!="Normal Day"').groupby('type')['sales'].sum().reset_index()
fig = px.bar(sales_by_holiday_type,
        x='type',
        y='sales',
        text=sales_by_holiday_type['sales'].apply(format_sales),
        title="Total Sales by diverse type of holiday",
        labels={'x': 'type', 'y': 'Sales (€)'},
        color='type',
        color_discrete_sequence=px.colors.qualitative.Set2
)
st.plotly_chart(fig)

# Sales by Month
trend_selected = st.selectbox("Select seasonality:", ['year', 'month', 'week' , 'day'])
st.write(f"Seasonality: Sales by {trend_selected}")
if trend_selected == 'month':
    df_filtered['month'] = df_filtered['date'].dt.strftime('%b')
elif trend_selected == 'week':
    df_filtered['week'] = df_filtered['date'].dt.strftime('%w')
elif trend_selected == 'day':
    df_filtered['day'] = df_filtered['date'].dt.strftime('%d')
elif trend_selected == 'year':
    df_filtered['year'] = df_filtered['date'].dt.year

sales_by_trend = df_filtered.groupby(trend_selected)['sales'].sum().reset_index()
fig = px.bar(sales_by_trend,
        x=trend_selected,
        y='sales',
        text=sales_by_trend['sales'].apply(format_sales),
        title=f"Total Sales by {trend_selected}",
        labels={'x': trend_selected, 'y': 'Sales (€)'},
        color=trend_selected,
        color_discrete_sequence=px.colors.qualitative.Set2
)
fig.update_traces(textposition='outside')
st.plotly_chart(fig)

#scatter Plot to visualise each product family by store
select_store_d = st.selectbox('select store ',df.store_nbr.unique())

family_filtered_d = df[df['store_nbr']==select_store_d]['family'].unique()
select_family_d = st.selectbox('select family product ', family_filtered_d)

filtered_s_f = df[(df['store_nbr']==select_store_d) & (df['family']==select_family_d)]
fig =  px.line(filtered_s_f,
               x='date',
               y='sales',
               #text=filtered_s_f['sales'].apply(format_sales),
               #labels={'x': 'date', 'y': 'Sales'},
               title=f'Distribution of Sales by Product Family {select_family_d} in store {select_store_d} over the time')
st.plotly_chart(fig)

st.subheader("Multivariate analysis")
st.write(f" Correlation between different factors for the sale of product family {select_family_d} in {select_store_d} ")
feature_selected = st.multiselect("Select at least two factors:", ['transactions', 'sales', 'dcoilwtico', 'onpromotion'], ['transactions', 'sales'])

corr_matrix = filtered_s_f[feature_selected].corr()
fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale="RdBu",
        text=corr_matrix.values,
        texttemplate="%{text:.2f}"
    ))
fig.update_layout(title="correlation heatmap")

st.plotly_chart(fig)


fig = px.box(df_filtered, x=[f'store {i}' for i in df_filtered.store_nbr.values],
              y = 'sales', title='Box plot stores',labels={'x':'stores','y': 'Sales (€)'}, color='store_nbr', color_discrete_sequence=px.colors.qualitative.Set2)
st.plotly_chart(fig)


fig = px.box(df_filtered, x= df_filtered.family.values,
              y = 'sales', title='Box plot for family product ',labels={'x':'family product','y': 'Sales (€)'}, color='family', color_discrete_sequence=px.colors.qualitative.Set2)
st.plotly_chart(fig)

feature = st.selectbox('select a categorical variable for the Test', ['city', 'state', 'family', 'store_nbr', 'cluster'])

fig =px.bar(x =df[feature].value_counts().index, y =df[feature].value_counts().values, labels={'x':f'{feature}', 'y': 'frequency'},title='Histogram of {feature}')# color=feature, color_discrete_sequence=px.colors.qualitative.Set2)
st.plotly_chart(fig)
groups = [df[df[feature] == g]['sales'] for g in df[feature].unique()]
st.write(f'rate of sale by each {feature}')
fig = px.pie(values=[g.sum() for g in groups], names=[f for f in df[feature].unique()])
st.plotly_chart(fig)
#st.write(f"Test of normal distribution for {feature}")
##model = smf.ols(f'sales ~ C({feature})', df[feature]).fit()
#residus = model.resid
# test of normal of shapiro-Wilk
#shapiro_test = stats.shapiro(residus)
#st.write(f" shapiro-Wilk: p_value = {shapiro_test}")
#sm.qqplot(residus, line='s')
#plt.show()
st.write('Test ANOVA 📊')
f_stat, p_value = stats.f_oneway(*groups)
st.write( f"Test ANOVA: F = {f_stat:.2f}, p-value = {p_value:.4f}")
st.write('Test Kruskal 📊')
k_stat , p_value = stats.kruskal(*groups)
st.write(f"Test Kruskal: F = {k_stat:.2f}, p-value = {p_value:.4f}")
if p_value < 0.05:
    posthoc = sp.posthoc_dunn(df, val_col='sales', group_col=feature, p_adjust='bonferroni')
    st.write(posthoc)

st.write("This Dashboard is a powerful tool to analyse KPI for sales Explore historical sales data to uncover patterns and trend Identify key factors affecting sales (holidays, promotions, store types, etc. Develop a powerful predictive model using machine learning")
