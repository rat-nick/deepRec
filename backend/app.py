from __future__ import absolute_import

from pprint import pprint

from flask import Flask, request
from flask_cors import CORS, cross_origin

from .DataAccess import DataAccess
from .Recommender import Recommender

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"


print("Loading data access...")
data_access = DataAccess()
print("Loading recommender...")
recommender = Recommender()


@app.route("/recommend", methods=["POST"])
@cross_origin()
def recommend():
    req = request.get_json()["userPreference"]
    print(req)
    req = list(map(lambda x: str(x["id"]), req))
    req = data_access.raw2inner(req)

    res = recommender.recommend(req, 20)
    res = list(map(lambda x: x[0], res))
    res = data_access.inner2raw(res)
    return data_access.attachInfo(res)


@app.route("/", methods=["GET"])
@cross_origin()
def all_items():
    return data_access.all_items()[:50]


@app.route("/sample", methods=["GET"])
@cross_origin()
def random_items():
    return data_access.sample_items(20)


@app.route("/search", methods=["GET"])
def search():
    term = request.args.get("term")
    return data_access.search(term)
