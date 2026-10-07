import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered",
)

# ---------------------------------------------------------------------------
# LOAD MODEL (cached so it only loads once, not on every interaction)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    bundle = joblib.load("iris_best_model.pkl")
    return bundle["model"], bundle["scaler"], bundle["label_encoder"]

model, scaler, le = load_model()

# Reference dataset just for slider ranges + the context scatter plot
@st.cache_data
def load_reference_data():
    return sns.load_dataset("iris")

iris_df = load_reference_data()

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.title("🌸 Iris Flower Species Classifier")
st.write(
    "Adjust the sepal and petal measurements in the sidebar and the model "
    "will predict the Iris species in real time."
)

# ---------------------------------------------------------------------------
# SIDEBAR: INPUT SLIDERS
# ---------------------------------------------------------------------------
st.sidebar.header("Input Measurements (cm)")

def slider_for(col_name, label):
    return st.sidebar.slider(
        label,
        float(iris_df[col_name].min()),
        float(iris_df[col_name].max()),
        float(iris_df[col_name].mean()),
        step=0.1,
    )

sepal_length = slider_for("sepal_length", "Sepal Length")
sepal_width = slider_for("sepal_width", "Sepal Width")
petal_length = slider_for("petal_length", "Petal Length")
petal_width = slider_for("petal_width", "Petal Width")

# ---------------------------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------------------------
input_df = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]],
    columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
)

input_scaled = scaler.transform(input_df)

# predict() may return integer codes or string labels depending on how the
# model was trained -- handle both cases so the app never crashes.
raw_pred = model.predict(input_scaled)[0]
if isinstance(raw_pred, (int, np.integer)):
    pred_species = le.inverse_transform([raw_pred])[0]
else:
    pred_species = raw_pred

proba = None
if hasattr(model, "predict_proba"):
    proba = model.predict_proba(input_scaled)[0]
    proba_df = pd.DataFrame(
        {"species": le.classes_, "probability": proba}
    ).sort_values("probability", ascending=False)

# ---------------------------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Your Input")
    st.table(input_df.T.rename(columns={0: "value (cm)"}))

with col2:
    st.subheader("Prediction")
    species_emoji = {"setosa": "🌼", "versicolor": "🌸", "virginica": "🌺"}
    emoji = species_emoji.get(pred_species, "🌷")
    st.markdown(f"### {emoji} **{pred_species.capitalize()}**")
    if proba is not None:
        st.metric("Confidence", f"{proba.max():.1%}")

if proba is not None:
    st.subheader("Prediction Probabilities")
    st.bar_chart(proba_df.set_index("species"))

# ---------------------------------------------------------------------------
# CONTEXT PLOT
# ---------------------------------------------------------------------------
st.subheader("Where your sample falls (Petal Length vs Petal Width)")
fig, ax = plt.subplots(figsize=(6, 4))
sns.scatterplot(
    data=iris_df, x="petal_length", y="petal_width", hue="species", ax=ax, alpha=0.6
)
ax.scatter(petal_length, petal_width, color="black", s=200, marker="X", label="Your input")
ax.legend()
st.pyplot(fig)

st.subheader("Where your sample falls (sepal Length vs sepal Width)")
fig, ax = plt.subplots(figsize=(6, 4))
sns.scatterplot(
    data=iris_df, x="sepal_length", y="sepal_width", hue="species", ax=ax, alpha=0.6
)
ax.scatter(sepal_length, sepal_width, color="black", s=200, marker="X", label="Your input")
ax.legend()
st.pyplot(fig)

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
with st.expander("About this app"):
    st.write(
        """
        - **Dataset**: seaborn's built-in `iris` dataset (150 samples, 3 species)
        - **Model**: trained and tuned offline with scikit-learn
          (KNN / Logistic Regression, selected via cross-validation)
        - **Preprocessing**: inputs are scaled with the same `StandardScaler`
          used at training time before being passed to the model
        """
    )