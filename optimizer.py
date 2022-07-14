from surprise import Dataset
from surprise.model_selection.search import GridSearchCV
from RBMAlgorithm import RBMAlgorithm

params = {
    "n_hidden": [50, 100, 200, 300, 400],
    "learning_rate": [0.1, 0.05, 0.01, 0.005, 0.001],
    "batch_size": [1, 10, 20],
    "l1": [0.1, 0.01, 0.001, 0.0001],
    "l2": [0.01, 0.001, 0.0001, 0.00001],
    "early_stopping": [True],
}

data = Dataset.load_builtin()

gs = GridSearchCV(
    RBMAlgorithm,
    param_grid=params,
    measures=["rmse", "mae"],
    cv=3,
    n_jobs=6,
    joblib_verbose=2,
)

gs.fit(data)
print(gs.best_score["rmse"])
print(gs.best_params["rmse"])
