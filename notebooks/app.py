import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
import re
from store_sales import *
import streamlit as st



data = pd.read_csv('../data/train.csv')
holidays_events_data = pd.read_csv('../data/holidays_events.csv')
oil_data = pd.read_csv('../data/oil.csv')
stores_data = pd.read_csv('../data/stores.csv')

X = data.drop(['sales'], axis=1)
y = data['sales']
X_train, X_valid_test, y_train, y_valid_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test, shuffle=False, test_size=0.5, random_state=42)
train_data = pd.concat([X_train, y_train], axis=1)
train_data = prepare_data(train_data, holidays_events_data, oil_data, stores_data)



item_families_sales_summary = train_data.groupby('item_family')['item_sales'].sum().reset_index().sort_values(by='item_sales', ascending=False)
fig1 = px.bar(
    item_families_sales_summary,
    x='item_family',
    y='item_sales',
    title="Most Sellable Items' Families",
    labels={'item_family': "Items' Families", 'item_sales': 'Total Sales'},
    color_discrete_sequence=['skyblue']
)
fig1.update_layout(xaxis_tickangle=-90)
st.title("Total sales analysis")
st.plotly_chart(fig1)


store_type_sales_summary = train_data.groupby('store_type')['item_sales'].mean().reset_index().sort_values(by='item_sales', ascending=False)
fig2 = px.bar(
    store_type_sales_summary,
    x='store_type',
    y='item_sales',
    title="Store Types' Average Sales",
    labels={'store_type': 'Store Type', 'item_sales': 'Average Sales'},
    color='store_type',
    color_continuous_scale='viridis'
)
fig2.update_layout(showlegend=False)
st.title("Sales analysis by Store Type")
st.plotly_chart(fig2)



store_cluster_sales_summary = train_data.groupby('store_cluster')['item_sales'].mean().reset_index().sort_values(by='item_sales', ascending=False)
store_cluster_sales_summary['store_cluster'] = pd.Categorical(store_cluster_sales_summary['store_cluster'], categories=store_cluster_sales_summary['store_cluster'], ordered=True)
fig3 = px.bar(
    store_cluster_sales_summary,
    x='store_cluster',
    y='item_sales',
    title="Store clusters' average sales",
    labels={'store_cluster' : 'Store Cluster', 'item_sales' : 'Average Sales'},
    color='store_cluster',
    color_continuous_scale='viridis'
)
fig3.update_layout(showlegend=False)
st.title("Sales analysis by Store Cluster")
st.plotly_chart(fig3)



cities_sales_summary = train_data.groupby('city')['item_sales'].mean().reset_index().sort_values(by='item_sales', ascending=False)
fig4 = px.bar(
    cities_sales_summary,
    x='city',
    y='item_sales',
    title="Cities' average sales",
    labels={'city' : 'City', 'item_sales' : 'Average Sales'},
    color='city',
    color_continuous_scale='viridis'
)
fig4.update_layout(showlegend=False)
st.title("Sales analysis by City")
st.plotly_chart(fig4)



states_sales_summary = train_data.groupby('state')['item_sales'].mean().reset_index().sort_values(by='item_sales', ascending=False)
fig5 = px.bar(
    states_sales_summary,
    x='state',
    y='item_sales',
    title="States' average sales",
    labels={'state' : 'State', 'item_sales' : 'Average Sales'},
    color='state',
    color_continuous_scale='viridis'
)
fig5.update_layout(showlegend=False)
st.title("Sales analysis by State")
st.plotly_chart(fig5)


day_types_sales_summary = train_data.groupby('day_type')['item_sales'].mean().reset_index().sort_values(by='item_sales', ascending=False)
fig6 = px.bar(
    day_types_sales_summary,
    x='day_type',
    y='item_sales',
    title="Day types' average sales",
    labels={'day_type' : 'Day type', 'item_sales' : 'Average Sales'},
    color='day_type',
    color_continuous_scale='viridis'
)
fig6.update_layout(showlegend=False)
st.title("Sales analysis by Day Type")
st.plotly_chart(fig6)



holiday_status_sales_summary = train_data.groupby('holiday_status')['item_sales'].mean().reset_index().sort_values(by='item_sales', ascending=False)
fig7 = px.bar(
    holiday_status_sales_summary,
    x='holiday_status',
    y='item_sales',
    title="Holiday status' average sales",
    labels={'holiday_status' : 'Holiday Status', 'item_sales' : 'Average Sales'},
    color='holiday_status',
    color_continuous_scale='viridis'
)
fig7.update_layout(showlegend=False)
st.title("Sales analysis by Holiday Status")
st.plotly_chart(fig7)



holiday_description_sales_summary = train_data.groupby('holiday_description')['item_sales'].mean().reset_index()
filtered_summary = holiday_description_sales_summary[holiday_description_sales_summary['holiday_description'].str.startswith('Terremoto Manabi')]

def extract_number(description):
    match = re.search(r'\d+', description)
    if match:
        return int(match.group())
    else:
        return 0  

filtered_summary['n'] = filtered_summary['holiday_description'].apply(extract_number)
filtered_summary = filtered_summary.sort_values(by='n')

fig = go.Figure()
fig.add_trace(go.Bar(x=filtered_summary['holiday_description'], y=filtered_summary['item_sales'],
                     marker_color='lightblue', name='holiday_description'))
fig.update_xaxes(title='Holiday Description', tickangle=90)
fig.update_yaxes(title='Average Sales')
fig.update_layout(title="Average Sales of the days after the catastrophe",
                  height=600)
st.title("Sales analysis after the Earthquake on April 16, 2016")
st.plotly_chart(fig)
