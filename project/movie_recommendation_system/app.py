import streamlit as st, pickle
import requests
import os
from dotenv import load_dotenv

load_dotenv()

top5 = pickle.load(open("top5.pkl", "rb")) 
movies = pickle.load(open("movies.pkl", "rb"))

API_KEY = os.getenv("API_KEY")   # on Streamlit Cloud: st.secrets["API_KEY"]

PLACEHOLDER = "https://via.placeholder.com/500x750?text=No+Poster"

@st.cache_data(show_spinner=False)
def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        r = requests.get(url, params={"api_key": API_KEY}, timeout=5)
        r.raise_for_status()
        path = r.json().get("poster_path")
        if path:
            return "https://image.tmdb.org/t/p/w500" + path
    except requests.RequestException:
        pass
    return PLACEHOLDER

st.title("Movie Recommendation System")

choice = st.selectbox("Pick a movie", movies["title"].values)

if st.button("Recommend"):
    i = movies[movies["title"] == choice].index[0]
    cols = st.columns(5)
    for col, j in zip(cols, top5[i]):
        row = movies.iloc[j]
        with col:
            st.image(fetch_poster(int(row["id"])))
            st.caption(row["title"])