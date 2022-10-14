from surprise import Dataset
from surprise.model_selection.search import GridSearchCV
from RBMAlgorithm import RBMAlgorithm

params = {
    "n_hidden": [50, 100],
    "learning_rate": [0.1, 0.05],
    "batch_size": [1, 10],
    "l1": [0.1],
    "l2": [0.01],
    "early_stopping": [True],
}

data = Dataset.load_builtin()

gs = GridSearchCV(
    RBMAlgorithm,
    param_grid=params,
    measures=["rmse", "mae"],
    cv=3,
    n_jobs=-1,
    joblib_verbose=3,
)
print("Start search...")
gs.fit(data)

print(gs.best_score["rmse"])
print(gs.best_params["rmse"])
