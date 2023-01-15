import matplotlib.pyplot as plt

from data import dataset

from . import model, optimizer, params

ds = dataset.MyDataset("ml-1m")
ds.trainTestValidationSplit()

rbm = model.Model((ds.nItems, 5), (200,), dev="cuda")
rbm.summarize()

hyperParams = params.HyperParams(
    batch_size=100, lr=1e-3, early_stopping=True, max_epochs=1000, patience=10
)

opt = optimizer.Optimizer(hyperParams, rbm, ds, True)
opt.fit()
opt.t = 3
opt.fit()
opt.t = 5
opt.fit()
opt.t = 7
opt.fit()

plt.title("RBM RMSE loss")
plt.plot(rbm.get_buffer("valid_rmse").to("cpu")[: rbm.current_epoch])
plt.plot(rbm.get_buffer("train_rmse").to("cpu")[: rbm.current_epoch])
plt.legend(["valid rmse", "train rmse"])

plt.savefig("RMSE.png")
input()
