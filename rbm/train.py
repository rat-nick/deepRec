import matplotlib.pyplot as plt

from data import dataset

from . import model, optimizer, params

ds = dataset.MyDataset("ml-100k")
ds.trainTestValidationSplit()

rbm = model.Model((ds.nItems, 5), (100,))
rbm.summarize()

hyperParams = params.HyperParams(
    batch_size=10, lr=1e-3, early_stopping=True, max_epochs=1000, patience=5
)

opt = optimizer.Optimizer(hyperParams, rbm, ds, True)
opt.fit()

plt.title("RBM RMSE loss")
plt.plot(rbm.get_buffer("valid_rmse")[: rbm.current_epoch])
plt.plot(rbm.get_buffer("train_rmse")[: rbm.current_epoch])
plt.legend(["valid rmse", "train rmse"])

plt.savefig("RMSE.png")
input()
