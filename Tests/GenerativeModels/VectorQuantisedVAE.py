"""
Vector-quantised variational autoencoder
===========================================

This script trains a vector-quantised variational autoencoder (VQ-VAE) on the 8 x 8 digits of scikit-learn. Unlike
a variational autoencoder, which maps an image to a continuous latent vector, a VQ-VAE describes every image with
a few discrete codes, like words from a learned vocabulary. Every code refers to a vector in a learned codebook,
and the decoder turns the code vectors back into an image.

New images are generated in two stages: a second model learns which combinations of codes occur in the data, new
codes are sampled from it, and the decoder turns them into images.
"""
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

from DLL.DeepLearning.Model import Model
from DLL.DeepLearning.Layers import Dense
from DLL.DeepLearning.Layers.Activations import ReLU, SoftMax
from DLL.DeepLearning.Optimisers import ADAM
from DLL.DeepLearning.Losses import CCE


torch.manual_seed(0)

# %%
# Data
# ----

digits = load_digits()
images = torch.tensor(digits.data, dtype=torch.float32) / 16
n_pixels = images.shape[1]

# %%
# The model
# ---------
#
# The encoder maps an image to 4 latent vectors of dimension 8. Each of them is replaced by the nearest of the 32
# vectors in the codebook, so an image is described by 4 codes, each of which is a number between 0 and 31. The
# decoder maps the 4 code vectors back to the pixels.

n_latents, latent_dim, codebook_size = 4, 8, 32

encoder = Model(n_pixels)
encoder.add(Dense(256, activation=ReLU()))
encoder.add(Dense(128, activation=ReLU()))
encoder.add(Dense(n_latents * latent_dim))
encoder.compile(optimiser=ADAM(learning_rate=0.001))

decoder = Model(n_latents * latent_dim)
decoder.add(Dense(128, activation=ReLU()))
decoder.add(Dense(256, activation=ReLU()))
decoder.add(Dense(n_pixels))
decoder.compile(optimiser=ADAM(learning_rate=0.001))

codebook = torch.randn(codebook_size, latent_dim)


def quantise(z):
    # z has shape (n_samples, n_latents, latent_dim); find the nearest codebook vector for every latent vector
    distances = ((z.unsqueeze(2) - codebook) ** 2).sum(dim=3)
    codes = distances.argmin(dim=2)
    return codes, codebook[codes]


# %%
# Training
# --------
#
# Choosing the nearest codebook vector has no gradient, so the gradient of the decoder input is copied to the
# encoder output as it is (the straight-through estimator). The loss is the reconstruction error plus a
# commitment term :math:`\beta\|z_e - \text{sg}(e)\|^2`, which keeps the encoder outputs :math:`z_e` close to
# their codebook vectors :math:`e`. Here :math:`\text{sg}` stops the gradient.
#
# The codebook itself is not updated with gradients. Each codebook vector is moved towards the mean of the encoder
# outputs assigned to it with an exponential moving average. Codebook vectors that are not used in a batch are
# moved onto random encoder outputs, so that no code stays unused.

n_epochs = 200
batch_size = 128
beta = 0.25
decay = 0.9
history = {"reconstruction": [], "codes used": []}
encoder.train()
decoder.train()
for epoch in range(n_epochs):
    reconstruction_total = 0
    for batch in torch.randperm(len(images)).split(batch_size):
        x = images[batch]
        n = len(x)
        encoder.optimiser.zero_grad()
        decoder.optimiser.zero_grad()

        z_e = encoder.predict(x).reshape(n, n_latents, latent_dim)
        codes, z_q = quantise(z_e)
        x_hat = decoder.predict(z_q.reshape(n, -1))

        reconstruction = ((x_hat - x) ** 2).sum(dim=1).mean()
        reconstruction_total += reconstruction.item() * n
        dL_dz_q = decoder.backward(2 * (x_hat - x) / n)
        dL_dz_e = dL_dz_q + 2 * beta * (z_e - z_q).reshape(n, -1) / n  # straight-through + commitment
        encoder.backward(dL_dz_e)
        encoder.optimiser.update_parameters()
        decoder.optimiser.update_parameters()

        # move every codebook vector towards the mean of its assigned encoder outputs
        flat_z, flat_codes = z_e.reshape(-1, latent_dim), codes.flatten()
        counts = torch.bincount(flat_codes, minlength=codebook_size)
        sums = torch.zeros(codebook_size, latent_dim).index_add_(0, flat_codes, flat_z)
        used = counts > 0
        codebook[used] = decay * codebook[used] + (1 - decay) * sums[used] / counts[used, None]
        unused = torch.nonzero(~used).flatten()
        codebook[unused] = flat_z[torch.randint(0, len(flat_z), (len(unused),))]

    all_codes = quantise(encoder.predict(images).reshape(-1, n_latents, latent_dim))[0]
    history["reconstruction"].append(reconstruction_total / len(images))
    history["codes used"].append(len(torch.unique(all_codes)))
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}: reconstruction error {history['reconstruction'][-1]:.3f}, codes used {history['codes used'][-1]} / {codebook_size}")
encoder.eval()
decoder.eval()

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(history["reconstruction"])
ax[0].set_title("Reconstruction error")
ax[1].plot(history["codes used"])
ax[1].set_title("Number of codes in use")
for a in ax:
    a.set_xlabel("Epoch")
    a.grid(alpha=0.3)
plt.tight_layout()

# %%
# Reconstructions
# ---------------
#
# Original images and their reconstructions from only 4 codes. The numbers are the codes of each image.

codes, z_q = quantise(encoder.predict(images).reshape(-1, n_latents, latent_dim))
reconstructions = decoder.predict(z_q.reshape(len(images), -1))
fig, axes = plt.subplots(2, 12, figsize=(12, 2.8))
for i in range(12):
    axes[0, i].imshow(images[i].reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
    axes[1, i].imshow(reconstructions[i].reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
    axes[1, i].set_xlabel(" ".join(str(c) for c in codes[i].tolist()), fontsize=7)
for ax in axes.ravel():
    ax.set_xticks([])
    ax.set_yticks([])
axes[0, 0].set_ylabel("Original")
axes[1, 0].set_ylabel("Decoded")
plt.tight_layout()

# %%
# A prior over the codes
# ----------------------
#
# To generate new images, the distribution of the 4 codes is learned with a small autoregressive model: the
# probability of every code given the codes before it. The model gets the previous codes as one-hot vectors (with
# zeros for the codes not yet chosen) together with a one-hot vector of the position, and predicts the
# distribution of the next code.

one_hot = torch.eye(codebook_size)


def prior_input(previous_codes, position):
    n = len(previous_codes)
    context = torch.zeros(n, n_latents - 1, codebook_size)
    if position > 0:
        context[:, :position] = one_hot[previous_codes[:, :position]]
    return torch.cat([context.reshape(n, -1), torch.eye(n_latents)[position].expand(n, n_latents)], dim=1)


prior_X = torch.cat([prior_input(codes, position) for position in range(n_latents)])
prior_Y = torch.cat([one_hot[codes[:, position]] for position in range(n_latents)])
prior = Model(prior_X.shape[1])
prior.add(Dense(128, activation=ReLU()))
prior.add(Dense(codebook_size, activation=SoftMax()))
prior.compile(optimiser=ADAM(learning_rate=0.003), loss=CCE())
prior.fit(prior_X, prior_Y, epochs=100, batch_size=128)

# %%
# Generating new digits
# ---------------------
#
# The codes are sampled one position at a time from the prior and decoded into images.

n_samples = 40
sampled = torch.zeros(n_samples, n_latents, dtype=torch.long)
for position in range(n_latents):
    probabilities = prior.predict(prior_input(sampled, position))
    sampled[:, position] = torch.multinomial(probabilities, 1).squeeze(1)
samples = decoder.predict(codebook[sampled].reshape(n_samples, -1))

fig, axes = plt.subplots(4, 10, figsize=(10, 4))
for sample, ax in zip(samples, axes.ravel()):
    ax.imshow(sample.reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()
