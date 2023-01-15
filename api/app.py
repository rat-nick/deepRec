from __future__ import absolute_import

from flask import Flask, request
from flask_cors import CORS, cross_origin
from surprise import Dataset, Reader
from surprise.dataset import DatasetAutoFolds

from deepRec.recommender.algo.RBMAlgorithm import RBMAlgorithm
from deepRec.recommender.RecommenderEngine import RecommenderEngine
from .DAL import *
import pprint

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"


dal = DAL()
print("Initialized in memory storage...")
dataset = DatasetAutoFolds(df=dal.ratings, reader=Reader(rating_scale=(1, 10)))

print("Loading RBM model...")
rbmAlgo = RBMAlgorithm(model_path="models/rbm.pt", model=None, dataset=dataset)
print("Loaded RBM model!")

print("Initializing recommender engine...")
receng = RecommenderEngine(rbmAlgo, dataset.build_full_trainset())
print("Recommender engine initialized...")


@app.route("/allItems")
@cross_origin()
def getAllItems():
    return dal.movies.sample(100).to_json(orient="records")
    # return "All Items"


@app.route("/search/<term>")
@cross_origin()
def searchItems(term):
    print(term)
    return dal.searchItems(term).to_json(orient="records")


@app.route("/recommend", methods=["POST"])
@cross_origin()
def recommend():
    # print(request)
    req = request.get_json()
    pprint.pprint(req)
    req = list(map(lambda x: (x["id"], x["rating"]), req))
    # print(req)
    recs = receng.getRecommendations(req)
    pprint.pp(recs)
    recsIDs = [ids for ids, _ in recs]
    # print(recsIDs)
    return dal.getMoviesWithIDs(recsIDs).to_json(orient="records")
