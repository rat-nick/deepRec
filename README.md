# deepRec

Using deep learning models for building recommendation systems.

## Dataset

We used the [ml-1m](https://files.grouplens.org/datasets/movielens/ml-1m.zip) dataset for explicit and implicit ratings.

## Models

- RBM (Restricted Boltzmann Machine) 
- VAE (Variational Autoencoder)

## Usage

### Training models

From the root directory run 

```
python -m rbm.train
```

to train an RBM model and


```
python -m vae.train
```

to train a VAE model.
