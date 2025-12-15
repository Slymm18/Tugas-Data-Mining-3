import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Clustering Postingan Facebook",
    layout="centered"
)

st.title("📊 Clustering Postingan Facebook")
st.write(
    "Aplikasi ini menggunakan **K-Means Clustering (k = 2)** "
    "untuk mengelompokkan postingan Facebook berdasarkan tingkat engagement."
)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_models():
    scaler = joblib.load("scaler_fb.pkl")
    kmeans = joblib.load("kmeans_fb_k2.pkl")
    pca = joblib.load("pca_fb.pkl")
    return scaler, kmeans, pca

scaler, kmeans, pca = load_models()

# =========================
# SIDEBAR INPUT USER
# =========================
st.sidebar.header("📝 Input Data Postingan")

num_reactions = st.sidebar.number_input("Total Reactions", min_value=0, value=50)
num_comments  = st.sidebar.number_input("Total Comments", min_value=0, value=5)
num_shares    = st.sidebar.number_input("Total Shares", min_value=0, value=3)
num_likes     = st.sidebar.number_input("Likes", min_value=0, value=40)
num_loves     = st.sidebar.number_input("Loves", min_value=0, value=5)
num_wows      = st.sidebar.number_input("Wows", min_value=0, value=1)
num_hahas     = st.sidebar.number_input("Hahas", min_value=0, value=2)
num_sads      = st.sidebar.number_input("Sads", min_value=0, value=0)
num_angrys    = st.sidebar.number_input("Angrys", min_value=0, value=0)

# =========================
# DATAFRAME INPUT
# =========================
input_data = pd.DataFrame([{
    "num_reactions": num_reactions,
    "num_comments": num_comments,
    "num_shares": num_shares,
    "num_likes": num_likes,
    "num_loves": num_loves,
    "num_wows": num_wows,
    "num_hahas": num_hahas,
    "num_sads": num_sads,
    "num_angrys": num_angrys
}])

st.subheader("📌 Data Input")
st.dataframe(input_data)

# =========================
# PREDIKSI CLUSTER
# =========================
if st.button("🔍 Prediksi Cluster"):

    # Scaling
    input_scaled = scaler.transform(input_data)

    # Prediksi cluster
    cluster = kmeans.predict(input_scaled)[0]

    # PCA transform
    input_pca = pca.transform(input_scaled)

    # =========================
    # HASIL CLUSTER
    # =========================
    st.subheader("📊 Hasil Clustering")

    if cluster == 0:
        st.success("🟣 **Cluster 0: Engagement Rendah**")
        st.write("""
        Postingan ini tergolong **engagement rendah**, ditandai dengan:
        - Jumlah like, komentar, dan share relatif kecil
        - Interaksi pengguna masih terbatas
        """)
    else:
        st.success("🟡 **Cluster 1: Engagement Tinggi**")
        st.write("""
        Postingan ini tergolong **engagement tinggi**, ditandai dengan:
        - Banyak reaksi dan share
        - Interaksi pengguna aktif
        - Potensi viral lebih besar
        """)

    # =========================
    # VISUALISASI PCA
    # =========================
    st.subheader("📈 Visualisasi PCA (2D)")

    fig, ax = plt.subplots(figsize=(7, 5))

    # Plot centroid
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    ax.scatter(
        centroids_pca[:, 0],
        centroids_pca[:, 1],
        marker="X",
        s=200
    )

    # Plot data user
    ax.scatter(
        input_pca[0, 0],
        input_pca[0, 1],
        s=150
    )

    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("Posisi Data Input pada Ruang PCA")

    st.pyplot(fig)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption(
    "Model: K-Means (k=2) | "
    "Preprocessing: StandardScaler | "
    "Visualisasi: PCA"
)
