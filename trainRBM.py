from recommender.algo.RBMAlgorithm import RBMAlgorithm
from recommender.model.RBM import RBM
from data.dataset import MyDataset

dataset = MyDataset(
    data_dir="ml-10m",
    ratings_path="ratings.dat",
    ratings_sep="::",
    items_path="movies.dat",
    items_sep="::",
)
print("Succesfully read data!")
dataset.trainTestValidationSplit()
print("Performed data split!")
print("Enter batch size:")
bs = 20
rbm = RBM(
    n_visible=dataset.nItems,
    ratings=10,
    verbose=True,
    device="cuda",
    n_hidden=100,
    batch_size=bs,
    learning_rate=1e-7,
    # momentum=0.3,
)

rbm.fit(dataset)
