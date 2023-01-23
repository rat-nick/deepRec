import torch
from surprise import Dataset, Reader


rd = Reader(line_format="user item rating timestamp", sep=",")
ds = Dataset(rd)
ds = ds.load_from_file("data/clean/ratings.csv", rd)
ts = ds.build_full_trainset()


n_users = ts.n_users
n_items = ts.n_items


tensor = torch.zeros(size=(n_users, n_items))

for u, i, r in ts.all_ratings():
    tensor[u][i] = r

print(tensor.count_nonzero().item() / (tensor.shape[0] * tensor.shape[1]))

torch.save(tensor.to("cpu"), "vae/sparse.pt")