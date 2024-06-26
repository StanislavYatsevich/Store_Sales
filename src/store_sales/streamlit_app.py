import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import re
import streamlit as st
from store_sales import DATA_FOR_STREAMLIT_PATH
from pathlib import Path

train_data = pd.read_csv(Path(DATA_FOR_STREAMLIT_PATH))

item_families_sales_summary = (
    train_data.groupby("item_family")["item_sales"]
    .sum()
    .reset_index()
    .sort_values(by="item_sales", ascending=False)
)
store_type_sales_summary = (
    train_data.groupby("store_type")["item_sales"]
    .mean()
    .reset_index()
    .sort_values(by="item_sales", ascending=False)
)
store_cluster_sales_summary = (
    train_data.groupby("store_cluster")["item_sales"]
    .mean()
    .reset_index()
    .sort_values(by="item_sales", ascending=False)
)
store_cluster_sales_summary["store_cluster"] = pd.Categorical(
    store_cluster_sales_summary["store_cluster"],
    categories=store_cluster_sales_summary["store_cluster"],
    ordered=True,
)
cities_sales_summary = (
    train_data.groupby("city")["item_sales"]
    .mean()
    .reset_index()
    .sort_values(by="item_sales", ascending=False)
)
states_sales_summary = (
    train_data.groupby("state")["item_sales"]
    .mean()
    .reset_index()
    .sort_values(by="item_sales", ascending=False)
)
day_types_sales_summary = (
    train_data.groupby("day_type")["item_sales"]
    .mean()
    .reset_index()
    .sort_values(by="item_sales", ascending=False)
)
holiday_status_sales_summary = (
    train_data.groupby("holiday_status")["item_sales"]
    .mean()
    .reset_index()
    .sort_values(by="item_sales", ascending=False)
)

holiday_description_sales_summary = (
    train_data.groupby("holiday_description")["item_sales"].mean().reset_index()
)
filtered_summary = holiday_description_sales_summary[
    holiday_description_sales_summary["holiday_description"].str.startswith(
        "Terremoto Manabi"
    )
]


def extract_number(description):
    match = re.search(r"\d+", description)
    if match:
        return int(match.group())
    else:
        return 0


filtered_summary["n"] = filtered_summary["holiday_description"].apply(extract_number)
filtered_summary = filtered_summary.sort_values(by="n")


def create_charts(
    item_families=None,
    store_types=None,
    store_clusters=None,
    cities=None,
    states=None,
    day_types=None,
    holiday_statuses=None,
    holiday_descriptions=None,
):
    if item_families:
        item_families_sales_summary_filtered = item_families_sales_summary[
            item_families_sales_summary["item_family"].isin(item_families)
        ]
    else:
        item_families_sales_summary_filtered = item_families_sales_summary
    fig1 = px.bar(
        item_families_sales_summary_filtered,
        x="item_family",
        y="item_sales",
        title="Most Sellable Items' Families",
        labels={"item_family": "Items' Families", "item_sales": "Total Sales"},
        color_discrete_sequence=["skyblue"],
    )
    fig1.update_layout(xaxis_tickangle=-90)

    if store_types:
        store_type_sales_summary_filtered = store_type_sales_summary[
            store_type_sales_summary["store_type"].isin(store_types)
        ]
    else:
        store_type_sales_summary_filtered = store_type_sales_summary
    fig2 = px.bar(
        store_type_sales_summary_filtered,
        x="store_type",
        y="item_sales",
        title="Store Types' Average Sales",
        labels={"store_type": "Store Type", "item_sales": "Average Sales"},
        color="store_type",
        color_continuous_scale="viridis",
    )
    fig2.update_layout(showlegend=False)

    if store_clusters:
        store_cluster_sales_summary_filtered = store_cluster_sales_summary[
            store_cluster_sales_summary["store_cluster"].isin(store_clusters)
        ]
    else:
        store_cluster_sales_summary_filtered = store_cluster_sales_summary
    fig3 = px.bar(
        store_cluster_sales_summary_filtered,
        x="store_cluster",
        y="item_sales",
        title="Store clusters' average sales",
        labels={"store_cluster": "Store Cluster", "item_sales": "Average Sales"},
        color="store_cluster",
        color_continuous_scale="viridis",
    )
    fig3.update_layout(showlegend=False)

    if cities:
        cities_sales_summary_filtered = cities_sales_summary[
            cities_sales_summary["city"].isin(cities)
        ]
    else:
        cities_sales_summary_filtered = cities_sales_summary
    fig4 = px.bar(
        cities_sales_summary_filtered,
        x="city",
        y="item_sales",
        title="Cities' average sales",
        labels={"city": "City", "item_sales": "Average Sales"},
        color="city",
        color_continuous_scale="viridis",
    )
    fig4.update_layout(showlegend=False)

    if states:
        states_sales_summary_filtered = states_sales_summary[
            states_sales_summary["state"].isin(states)
        ]
    else:
        states_sales_summary_filtered = states_sales_summary
    fig5 = px.bar(
        states_sales_summary_filtered,
        x="state",
        y="item_sales",
        title="States' average sales",
        labels={"state": "State", "item_sales": "Average Sales"},
        color="state",
        color_continuous_scale="viridis",
    )
    fig5.update_layout(showlegend=False)

    if day_types:
        day_types_sales_summary_filtered = day_types_sales_summary[
            day_types_sales_summary["day_type"].isin(day_types)
        ]
    else:
        day_types_sales_summary_filtered = day_types_sales_summary
    fig6 = px.bar(
        day_types_sales_summary_filtered,
        x="day_type",
        y="item_sales",
        title="Day types' average sales",
        labels={"day_type": "Day type", "item_sales": "Average Sales"},
        color="day_type",
        color_continuous_scale="viridis",
    )
    fig6.update_layout(showlegend=False)

    if holiday_statuses:
        holiday_status_sales_summary_filtered = holiday_status_sales_summary[
            holiday_status_sales_summary["holiday_status"].isin(holiday_statuses)
        ]
    else:
        holiday_status_sales_summary_filtered = holiday_status_sales_summary
    fig7 = px.bar(
        holiday_status_sales_summary_filtered,
        x="holiday_status",
        y="item_sales",
        title="Holiday status' average sales",
        labels={"holiday_status": "Holiday Status", "item_sales": "Average Sales"},
        color="holiday_status",
        color_continuous_scale="viridis",
    )
    fig7.update_layout(showlegend=False)

    if holiday_descriptions:
        filtered_summary_filtered = filtered_summary[
            filtered_summary["holiday_description"].isin(holiday_descriptions)
        ]
    else:
        filtered_summary_filtered = filtered_summary
    fig8 = go.Figure()
    fig8.add_trace(
        go.Bar(
            x=filtered_summary_filtered["holiday_description"],
            y=filtered_summary_filtered["item_sales"],
            marker_color="lightblue",
            name="holiday_description",
        )
    )
    fig8.update_xaxes(title="Holiday Description", tickangle=90)
    fig8.update_yaxes(title="Average Sales")
    fig8.update_layout(
        title="Average Sales of the days after the catastrophe", height=600
    )

    return fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8


st.title("Sales Analysis Dashboard")


item_families_plot_name = "Most Sellable Items Families"
store_types_plot_name = "Store Types Average Sales"
store_clusters_plot_name = "Store Clusters Average Sales"
cities_plot_name = "Cities Average Sales"
states_plot_name = "States Average Sales"
day_types_plot_name = "Day Types Average Sales"
holiday_status_plot_name = "Holiday Status Average Sales"
earthquake_plot_name = "Sales after Earthquake"


option = st.selectbox(
    "Select the chart to display",
    (
        item_families_plot_name,
        store_types_plot_name,
        store_clusters_plot_name,
        cities_plot_name,
        states_plot_name,
        day_types_plot_name,
        holiday_status_plot_name,
        earthquake_plot_name,
    ),
)


if option == item_families_plot_name:
    selected_item_families = st.multiselect(
        "Select item families to display",
        item_families_sales_summary["item_family"].unique().tolist(),
        default=item_families_sales_summary["item_family"].unique().tolist(),
    )
else:
    selected_item_families = None


if option == store_types_plot_name:
    selected_store_types = st.multiselect(
        "Select store types to display",
        store_type_sales_summary["store_type"].unique().tolist(),
        default=store_type_sales_summary["store_type"].unique().tolist(),
    )
else:
    selected_store_types = None


if option == store_clusters_plot_name:
    selected_store_clusters = st.multiselect(
        "Select store clusters to display",
        store_cluster_sales_summary["store_cluster"].unique().tolist(),
        default=store_cluster_sales_summary["store_cluster"].unique().tolist(),
    )
else:
    selected_store_clusters = None


if option == cities_plot_name:
    selected_cities = st.multiselect(
        "Select cities to display",
        cities_sales_summary["city"].unique().tolist(),
        default=cities_sales_summary["city"].unique().tolist(),
    )
else:
    selected_cities = None


if option == states_plot_name:
    selected_states = st.multiselect(
        "Select states to display",
        states_sales_summary["state"].unique().tolist(),
        default=states_sales_summary["state"].unique().tolist(),
    )
else:
    selected_states = None


if option == day_types_plot_name:
    selected_day_types = st.multiselect(
        "Select day types to display",
        day_types_sales_summary["day_type"].unique().tolist(),
        default=day_types_sales_summary["day_type"].unique().tolist(),
    )
else:
    selected_day_types = None


if option == holiday_status_plot_name:
    selected_holiday_statuses = st.multiselect(
        "Select holiday statuses to display",
        holiday_status_sales_summary["holiday_status"].unique().tolist(),
        default=holiday_status_sales_summary["holiday_status"].unique().tolist(),
    )
else:
    selected_holiday_statuses = None


if option == earthquake_plot_name:
    selected_holidays = st.multiselect(
        "Select days after the earthquake to display",
        filtered_summary["holiday_description"].unique().tolist(),
        default=filtered_summary["holiday_description"].unique().tolist(),
    )
else:
    selected_holidays = None


fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8 = create_charts(
    selected_item_families,
    selected_store_types,
    selected_store_clusters,
    selected_cities,
    selected_states,
    selected_day_types,
    selected_holiday_statuses,
    selected_holidays,
)

options_to_plots = {
    item_families_plot_name: fig1,
    store_types_plot_name: fig2,
    store_clusters_plot_name: fig3,
    cities_plot_name: fig4,
    states_plot_name: fig5,
    day_types_plot_name: fig6,
    holiday_status_plot_name: fig7,
    earthquake_plot_name: fig8,
}

st.plotly_chart(options_to_plots[option])
