import requests

def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        r = requests.get(url, params={"api_key": "8265bd1679663a7ea12ac168da84d2e8"}, timeout=5)
        r.raise_for_status()
        print(r.json().get("overview"))
        path = r.json().get("poster_path")
        if path:
            return "https://image.tmdb.org/t/p/w500" + path
    except requests.RequestException:
        pass
    return "image not fount"

print(fetch_poster(19995))