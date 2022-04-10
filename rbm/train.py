from sklearn.model_selection import train_test_split
import torch
import config

from rbm import RBM
from dataset import Dataset

ds = Dataset()
trainset, testset = train_test_split(ds.getDatasetAsTensor(), train_size=0.8)
# print(trainset.shape, testset.shape)
# torch.device("cuda")
rbm = RBM(visible_units=trainset.shape[1], hidden_units=50, learning_rate=0.01)

# Train the RBM
# First for loop - go through every single epoch
for epoch in range(1, config.epochs + 1):

    # Second for loop - go through every single user
    # Lower bound is 0, upper bound is (nb_users - batch_size_), batch_size_ is the step of each batch (512)
    # The 1st batch is for user with ID = 0 to user with ID = 511
    for user in range(len(trainset)):

        # At the beginning, v0 = vk. Then we update vk
        badSample = trainset[user]
        goodSample = trainset[user]

        # Third for loop - perform contrastive divergence
        for k in range(3):
            _, hk = rbm.sample_h(badSample)  # forward pass
            _, badSample = rbm.sample_v(hk)  # backward pass

        # Calculate the loss using contrastive divergence
        rbm.train(goodSample, badSample)
        print(f"user {user} completed")
    print("Epoch finished")

    print(
        "Epoch: "
        + str(epoch)
        + "- RMSE Reconstruction Error: "
        + str(train_recon_error.data.numpy() / s)
    )
    wandb.log({"Train RMSE": train_recon_error.data.numpy() / s})

# Plot the RMSE reconstruction error with respect to increasing number of epochs
plt.plot(reconerr)
plt.ylabel("Training Data RMSE Reconstruction Error")
plt.xlabel("Epoch")
plt.savefig("pics/result.png")

# Evaluate the RBM on test set
test_recon_error = (
    0  # RMSE reconstruction error initialized to 0 at the beginning of training
)
s = 0.0  # a counter (float type)

# for loop - go through every single user
for user in range(config.nb_users):
    v = training_set[
        user : user + 1
    ]  # training set inputs are used to activate neurons of my RBM
    vt = test_set[user : user + 1]  # target

    if len(vt[vt >= 0]) > 0:
        _, h = rbm.sample_h(v)
        _, v = rbm.sample_v(h)

        # Update test RMSE reconstruction error
        test_recon_error += torch.sqrt(torch.mean((vt[vt >= 0] - v[vt >= 0]) ** 2))
        s += 1.0

# Display and log the RMSE metrics
print("RMSE Reconstruction error:  " + str(test_recon_error.data.numpy() / s))
wandb.log({"Test RMSE": test_recon_error.data.numpy() / s})
