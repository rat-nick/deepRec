import matplotlib.pyplot as plt

from data import dataset

from . import model, optimizer, params

ds = dataset.MyDataset("ml-1m")
ds.trainTestValidationSplit()

rbm = model.Model((ds.nItems, 5), (200,), device="cuda")
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
plt.plot(rbm.get_buffer("train_rmse").to("cpu")[: rbm.current_epoch])
plt.plot(rbm.get_buffer("valid_rmse").to("cpu")[: rbm.current_epoch])
plt.legend(["train", "valid"])

plt.savefig("RMSE.png")

n = 0
rmse = mae = 0
for minibatch in ds.batches(ds.testData, 1):
    rbm.eval()
    _1, _2 = opt.batch_error(minibatch)
    rmse += _1
    mae += _2
    n += 1

print("Testing loss: RMSE : %.5f \t MAE : %.5f " % (rmse / n, mae / n))
