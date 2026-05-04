import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd

from scr.data_loader import load_data
from scr.features import add_features
from scr.model import train_model

# ---------------------------------------------------------------
# Configurazione Streamlit
# ---------------------------------------------------------------
st.set_page_config(page_title="Wine Quality ML", layout="wide")
st.title("Wine Quality Analysis and Machine Learning")

# ---------------------------------------------------------------
# Load e Features
# ---------------------------------------------------------------
df = load_data()
df = add_features(df)

# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
st.sidebar.title("Filtri")

show_red = st.sidebar.checkbox("Vino Rosso", value=True)
show_white = st.sidebar.checkbox("Vino Bianco", value=True)

min_quality = st.sidebar.slider("Qualità minima", 0, 10, 0)

# ---------------------------------------------------------------
# Filtri
# ---------------------------------------------------------------
df_filtered = df.copy()

if show_red and not show_white:
    df_filtered = df_filtered[df_filtered["type_white"] == 0]

elif show_white and not show_red:
    df_filtered = df_filtered[df_filtered["type_white"] == 1]

elif not show_red and not show_white:
    st.error("Seleziona almeno un vino.")
    st.stop()

df_filtered = df_filtered[df_filtered["quality"] >= min_quality]

if df_filtered.empty:
    st.error("Nessun dato disponibile con i filtri attuali")
    st.stop()

st.info(f"Campioni disponibili: {len(df_filtered)}")

# ---------------------------------------------------------------
# Data preview
# ---------------------------------------------------------------
st.subheader("Dataset")

df_show = df_filtered.copy()
df_show["type"] = df_show[["type_red", "type_white"]].idxmax(axis=1)
df_show["type"] = df_show["type"].str.replace("type_", "")

df_show = df_show.drop(columns=["type_red", "type_white"])

st.dataframe(df_show.head())

# ---------------------------------------------------------------
# EDA
# ---------------------------------------------------------------
st.subheader("Distribuzione qualità")

fig = px.histogram(
    df_show,
    x="quality",
    color="type",
    barmode="overlay",
    title="Distribuzione qualità del vino",
    color_discrete_map={
        "red": "#8B0000",
        "white": "#FFD700"
    }
)

fig.update_layout(
    title_font_size=22,
    title_x=0.5,
    xaxis_title="Qualità del vino",
    yaxis_title="Numero di campioni",
    bargap=0.1,
    template="plotly_white"
)

fig.update_traces(
    marker_line_width=1,
    marker_line_color="black"
)

st.plotly_chart(fig)

# Distribuzione per tipo
st.subheader("Distribuzione per tipo")

type_counts = df_show["type"].value_counts()

fig_type = px.bar(
    x=type_counts.index,
    y=type_counts.values,
    labels={"x": "Tipo", "y": "Numero"},
    title="Distribuzione per tipo"
)

st.plotly_chart(fig_type)

# ---------------------------------------------------------------
# MACHINE LEARNING
# ---------------------------------------------------------------
st.subheader("Modello Random Forest")

test_size = st.slider("Test size", 0.1, 0.4, 0.2)

@st.cache_data
def get_model(df, test_size):
    return train_model(df, test_size)

if st.button("Allena modello"):
    results = get_model(df_filtered, test_size)

    st.write("Accuracy")
    st.write(results["accuracy"])

    st.text(results["report"])

# ---------------------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------------------
st.subheader("Feature importance")

if st.button("Mostra feature importance"):
    results = train_model(df_filtered, test_size)

    model = results["model"]
    X_test = results["X_test"]

    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance": importances
    }).sort_values("importance", ascending=True)

    fig2 = plt.figure()
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.title("Feature Importance")

    st.pyplot(fig2)