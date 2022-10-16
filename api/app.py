from flask import Flask, request
from flask_cors import CORS, cross_origin
from surprise import Dataset, Reader
from surprise.dataset import DatasetAutoFolds

from ..recommender.algo.mySVD import mySVD
from ..recommender.algo.RBMAlgorithm import RBMAlgorithm
from ..recommender.RecommenderEngine import RecommenderEngine
from .DAL import *

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"


dal = DAL()
print("Initialized in memory storage...")
dataset = DatasetAutoFolds(df=dal.ratings, reader=Reader(rating_scale=(1, 10)))

print("Fitting the algorithm...")
algo = RBMAlgorithm(dataset, verbose=True)
print("Fitting completed!")
print("Initializing recommender engine...")
receng = RecommenderEngine(algo, dataset.build_full_trainset())
print("Recommender engine initialized...")


@app.route("/allItems")
@cross_origin()
def getAllItems():
    return dal.movies.head(100).to_json(orient="records")
    # return "All Items"


@app.route("/search/<term>")
@cross_origin()
def searchItems(term):
    return dal.searchItems(term).to_json(orient="records")


@app.route("/recommend", methods=["POST"])
@cross_origin()
def recommend():
    req = request.get_json()
    print(req)
    req = map(req, lambda x: (x["id"], x["rating"]))
    res = receng.getRecommendations(req)
    print(res)
    return None
