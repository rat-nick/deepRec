import requests
import urllib.request
from config import *
from DAL import DAL

POSTER_SIZE = "w185"
BASE_URL = "http://image.tmdb.org/t/p/"


def getPoster(tmdbID):
    res = requests.get(f"{TMDB_URL}/movie/{tmdbID}?api_key={TMDB_API_KEY}")
    imgURL = res.json()["poster_path"]
    # x = requests.get(f"{BASE_URL}/{POSTER_SIZE}/{imgURL}")
    # print(x.content)
    urllib.request.urlretrieve(
        f"{BASE_URL}/{POSTER_SIZE}/{imgURL}", f"./posters/{tmdbID}.jpg"
    )


if __name__ == "__main__":
    dal = DAL()
    for id in dal.movies["tmdbId"]:
        try:
            getPoster(id)
        except:
            pass
