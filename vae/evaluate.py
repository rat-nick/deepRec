from .model import Model
from .dataset import Testset
from torch.utils.data import DataLoader
from metrics.tensor import ndcg, recall, precision
import numpy as np

model = Model(
    input_size=3416,
    latent_size=200,
    decoder_layers=[600],
    encoder_layers=[600],
    path="./vae/vae.pt",
)
model.eval()

r50_lst = []
r20_lst = []
p5_lst = []
p10_lst = []
ndcg_lst = []


for i in range(1, 6):
    testset = Testset(f"data/folds/{i}/test.csv")
    loader = dataloader = DataLoader(testset, batch_size=1, shuffle=False)
    r50 = 0
    r20 = 0
    p5 = 0
    p10 = 0
    n100 = 0
    for fi, ho in loader:
        rec = model(fi)
        n100 += ndcg(rec, ho, k=100)
        r50 += recall(rec, ho, 50)
        r20 += recall(rec, ho, 20)
        p10 += precision(rec, ho, 10)
        p5 += precision(rec, ho, 5)
    ndcg_lst += [n100 / len(loader)]
    r50_lst += [r50 / len(loader)]
    r20_lst = [r20 / len(loader)]
    p10_lst = [p10 / len(loader)]
    p5_lst = [p5 / len(loader)]
print("ndcg:\t%.4f ± %.4f" % (np.mean(ndcg_lst), np.std(ndcg_lst)))
print("r50:\t%.4f ± %.4f" % (np.mean(r50_lst), np.std(r50_lst)))
print("r20:\t%.4f ± %.4f" % (np.mean(r20_lst), np.std(r20_lst)))
print("p10:\t%.4f ± %.4f" % (np.mean(p10_lst), np.std(p10_lst)))
print("p5:\t%.4f ± %.4f" % (np.mean(p5_lst), np.std(p5_lst)))
