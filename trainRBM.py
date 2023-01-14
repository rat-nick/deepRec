from recommender.model.RBM import RBM
from data.dataset import MyDataset

dataset = MyDataset("ml-1m")
print("Succesfully read data!")
dataset.trainTestValidationSplit()
print("Performed data split!")
print("Enter batch size:")
bs = 100
rbm = RBM(
    n_visible=dataset.nItems,
    ratings=5,
    verbose=True,
    device="cuda",
    n_hidden=100,
    batch_size=bs,
    learning_rate=1e-3,
    early_stopping=True,
)

rbm.fit(dataset)
rbm.fit(dataset, t=3)
rbm.fit(dataset, t=7)
rbm.fit(dataset, t=11)
