"""
charts.py — every chart on the dashboard, built by a function that RETURNS it.

Returning the chart (instead of drawing it on the page) keeps the dashboard short
and lets the tests look inside each one: how many cells, traces or markers it has.

    seaborn / matplotlib (Lesson 6-1)   make_heatmap, make_violin   -> Figure
    Folium (Lesson 6-2)                 make_shop_map               -> folium.Map
    Plotly (Lesson 6-3)                 make_trend                  -> plotly Figure
    choropleth (Lesson 6-4)             make_choropleth             -> plotly Figure
"""

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.figure import Figure

from .settings import REASON_COLORS, REASONS, SYRACUSE


def make_heatmap(table: pd.DataFrame, title: str) -> Figure:
    """A seaborn heatmap of `table` (busy_table or reason_table), one cell per value.

    How to build it:

    - `fig, ax = plt.subplots(figsize=(11, 3.6))`
    - `sns.heatmap(table, ax=ax, cmap="YlOrBr", linewidths=0.5)`
    - Title it, label the x axis "Hour of the day", blank the y label, return `fig`.
    """
    fig, ax = plt.subplots(figsize=(11, 3.6))
    sns.heatmap(table, ax=ax, cmap="YlOrBr", linewidths=0.5)
    ax.set_title(title)
    ax.set_xlabel("Hour of the day")
    ax.set_ylabel("")
    return fig


def make_violin(frame: pd.DataFrame, by: str = "shop") -> Figure:
    """Violins of order `total`, one per value of the column `by` — how much
    people spend at each shop (`by="shop"`), or on each kind of visit
    (`by="reason"`).

    `sns.violinplot(data=frame, x=by, y="total", order=..., ax=ax, cut=0)`;
    `cut=0` keeps each violin inside the real range of totals. For reasons, put
    them in settings.REASONS order (leave out any with no visits); otherwise sort
    the values. Title it "What an order comes to, by <by>", label the y axis
    "Order total ($)", return the Figure.
    """
    fig, ax = plt.subplots(figsize=(11, 4))
    if by == "reason":
        order = [reason for reason in REASONS if reason in set(frame[by])]
    else:
        order = sorted(frame[by].dropna().unique())
    sns.violinplot(
        data=frame,
        x=by,
        y="total",
        order=order,
        ax=ax,
        cut=0,
    )
    ax.set_title(f"What an order comes to, by {by}")
    ax.set_xlabel(by)
    ax.set_ylabel("Order total ($)")
    return fig


def make_trend(monthly: pd.DataFrame, by: str = "shop",
               value: str = "sales") -> go.Figure:
    """An interactive line chart of `value` by month, one line per value of `by`.

        make_trend(monthly_sales(orders))                         sales per shop
        make_trend(monthly_sales(visits, "reason"), "reason", "orders")
                                                         visits per visit reason

    `px.line(monthly, x="month", y=value, color=by, markers=True, title=...)`;
    hovering shows each month's numbers. For reasons, also pass
    `color_discrete_map=settings.REASON_COLORS` so each reason keeps its color.
    """
    kwargs = {}
    if by == "reason":
        kwargs["color_discrete_map"] = REASON_COLORS
    title = f"{value.title()} by {by}"
    return px.line(
        monthly,
        x="month",
        y=value,
        color=by,
        markers=True,
        title=title,
        **kwargs,
    )


def make_shop_map(shops: pd.DataFrame, orders: pd.DataFrame) -> folium.Map:
    """A Folium map of our shops and the rivals.

    - Center it on settings.SYRACUSE, `zoom_start=12` (the default tiles are
      OpenStreetMap's).
    - Each of *our* shops: a green `folium.CircleMarker` whose radius grows with
      the shop's revenue (sum of `total` in `orders`): `6 + 18 * revenue / the
      largest revenue`, with a popup showing the name and revenue.
    - Each rival: a red `folium.Marker` (`icon=folium.Icon(color="red")`) with its
      name as the popup.
    - `tooltip=` the shop's name on every marker.
    """
    fmap = folium.Map(location=list(SYRACUSE), zoom_start=12)
    revenue = (
        orders.groupby("shop")["total"].sum()
        if "shop" in orders.columns
        else pd.Series(dtype=float)
    )
    largest = revenue.max() if len(revenue) else 0
    for shop in shops.itertuples():
        popup_name = shop.shop
        if shop.kind == "ours":
            shop_revenue = revenue.get(shop.shop, 0)
            radius = 6 + 18 * shop_revenue / largest if largest else 6
            popup = f"{shop.shop}: ${shop_revenue:,.2f}"
            folium.CircleMarker(
                location=[shop.lat, shop.lon],
                radius=radius,
                color="green",
                fill=True,
                fill_color="green",
                popup=popup,
                tooltip=popup_name,
            ).add_to(fmap)
        else:
            folium.Marker(
                location=[shop.lat, shop.lon],
                icon=folium.Icon(color="red"),
                popup=popup_name,
                tooltip=popup_name,
            ).add_to(fmap)
    return fmap


def make_choropleth(zips: gpd.GeoDataFrame, table: pd.DataFrame, column: str,
                    title: str, shops: pd.DataFrame | None = None) -> go.Figure:
    """A choropleth map: each ZIP filled by `table[column]` (ZIPs missing from
    `table` count as 0).

    How to build it:

    - values <- `zips[["zip"]]` merged with `table[["zip", column]]` (how="left"),
      missing filled with 0.
    - `px.choropleth_map(values, geojson=zips.__geo_interface__,
      locations="zip", featureidkey="properties.zip", color=column,
      color_continuous_scale="Oranges", map_style="open-street-map",
      center={"lat": 43.06, "lon": -76.14}, zoom=9.5, opacity=0.75, title=title)`
      — `featureidkey` tells Plotly which property of each polygon matches the
      `locations` column.
    - With `shops`, add them on top: `fig.add_scattermap(lat=..., lon=...,
      text=shops["shop"], mode="markers", marker={"size": 11, "color": ...},
      name="shops")` — green for ours, red for rivals.
    """
    values = zips[["zip"]].merge(
        table[["zip", column]],
        on="zip",
        how="left",
    )
    values[column] = values[column].fillna(0)
    fig = px.choropleth_map(
        values,
        geojson=zips.__geo_interface__,
        locations="zip",
        featureidkey="properties.zip",
        color=column,
        color_continuous_scale="Oranges",
        map_style="open-street-map",
        center={"lat": 43.06, "lon": -76.14},
        zoom=9.5,
        opacity=0.75,
        title=title,
    )
    if shops is not None:
        colors = ["green" if kind == "ours" else "red"
                  for kind in shops["kind"]]
        fig.add_scattermap(
            lat=shops["lat"],
            lon=shops["lon"],
            text=shops["shop"],
            mode="markers",
            marker={"size": 11, "color": colors},
            name="shops",
        )
    return fig


def close(fig: Figure) -> None:
    """Free a matplotlib Figure once Streamlit has drawn it.  *(Provided.)*"""
    plt.close(fig)
