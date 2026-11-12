"""
Convolutional generator for larger images
============================================

This script trains a variational autoencoder with convolutional networks on the 28 x 28 images of MNIST. The
decoder is a convolutional generator: a dense layer turns the latent vector into small 7 x 7 feature maps, and
:class:`~DLL.DeepLearning.Layers.Upsampling2D` layers followed by convolutions double their size twice to produce
the full image. The encoder works the other way round and halves the size of the image with max pooling.

Only the first 20 000 training images are used, and the model is trained for 1500 steps, which takes about a
minute and a half on a CPU.
"""
import torch
import matplotlib.pyplot as plt
from torchvision import datasets

from DLL.DeepLearning.Model import Model
from DLL.DeepLearning.Layers import Dense, Conv2D, Flatten, Reshape, MaxPooling2D, Upsampling2D
from DLL.DeepLearning.Layers.Activations import ReLU, Sigmoid, LeakyReLU
from DLL.DeepLearning.Optimisers import ADAM


torch.manual_seed(0)

# %%
# Data
# ----
#
# The pixels are scaled to [0, 1]. The test images are only used to show reconstructions of unseen digits.

train_images = (datasets.MNIST(root="./data", train=True, download=True).data[:20000].float() / 255).unsqueeze(1)
test_images = (datasets.MNIST(root="./data", train=False, download=True).data[:1000].float() / 255).unsqueeze(1)

# %%
# The model
# ---------
#
# The encoder halves the image twice with convolutions and max pooling (28 x 28, 14 x 14, 7 x 7) and outputs the
# mean and the log-variance of a 16-dimensional latent distribution. The decoder is the convolutional generator:
#
# - a dense layer maps the latent vector to 32 feature maps of size 7 x 7,
# - :class:`~DLL.DeepLearning.Layers.Upsampling2D` repeats every pixel to get 14 x 14 feature maps, and a
#   convolution refines them,
# - a second upsampling and convolution give 28 x 28 feature maps,
# - a final convolution with a sigmoid outputs the pixel probabilities.
#
# Repeating pixels and then convolving is a simple alternative to transposed convolutions, and it avoids the
# checkerboard patterns that transposed convolutions sometimes produce.

latent_dim = 16

encoder = Model((1, 28, 28))
encoder.add(Conv2D(kernel_size=3, output_depth=16, activation=LeakyReLU(0.2)))
encoder.add(MaxPooling2D(pool_size=2))
encoder.add(Conv2D(kernel_size=3, output_depth=32, activation=LeakyReLU(0.2)))
encoder.add(MaxPooling2D(pool_size=2))
encoder.add(Flatten())
encoder.add(Dense(2 * latent_dim))
encoder.compile(optimiser=ADAM(learning_rate=0.001))

decoder = Model(latent_dim)
decoder.add(Dense(32 * 7 * 7, activation=ReLU()))
decoder.add(Reshape((32, 7, 7)))
decoder.add(Upsampling2D(scale_factor=2))
decoder.add(Conv2D(kernel_size=3, output_depth=32, activation=ReLU()))
decoder.add(Upsampling2D(scale_factor=2))
decoder.add(Conv2D(kernel_size=3, output_depth=16, activation=ReLU()))
decoder.add(Conv2D(kernel_size=3, output_depth=1, activation=Sigmoid()))
decoder.compile(optimiser=ADAM(learning_rate=0.001))
decoder.summary()

# %%
# Training
# --------
#
# The training loop is the same as in the variational autoencoder example: the reconstruction error is the binary
# cross entropy summed over the pixels, the KL divergence keeps the latent distributions close to the standard
# normal prior, and the gradient flows from the decoder through the sampling step into the encoder.

n_steps = 1500
batch_size = 64
eps = 1e-6
history = {"reconstruction": [], "kl": []}
encoder.train()
decoder.train()
for step in range(n_steps):
    x = train_images[torch.randint(0, len(train_images), (batch_size,))]
    encoder.optimiser.zero_grad()
    decoder.optimiser.zero_grad()

    mean, log_var = encoder.predict(x).chunk(2, dim=1)
    std = torch.exp(0.5 * log_var)
    noise = torch.randn_like(std)
    z = mean + std * noise
    x_hat = decoder.predict(z).clamp(eps, 1 - eps)

    history["reconstruction"].append(-(x * torch.log(x_hat) + (1 - x) * torch.log(1 - x_hat)).sum().item() / batch_size)
    history["kl"].append(-0.5 * (1 + log_var - mean ** 2 - log_var.exp()).sum().item() / batch_size)

    dL_dz = decoder.backward((x_hat - x) / (x_hat * (1 - x_hat)) / batch_size)
    dL_dmean = dL_dz + mean / batch_size
    dL_dlog_var = dL_dz * 0.5 * std * noise + 0.5 * (log_var.exp() - 1) / batch_size
    encoder.backward(torch.cat([dL_dmean, dL_dlog_var], dim=1))
    encoder.optimiser.update_parameters()
    decoder.optimiser.update_parameters()
    if (step + 1) % 250 == 0:
        print(f"Step {step + 1}: reconstruction {sum(history['reconstruction'][-250:]) / 250:.1f}, KL divergence {sum(history['kl'][-250:]) / 250:.1f}")
encoder.eval()
decoder.eval()

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(torch.tensor(history["reconstruction"]).reshape(-1, 25).mean(dim=1))
ax[0].set_title("Reconstruction error")
ax[1].plot(torch.tensor(history["kl"]).reshape(-1, 25).mean(dim=1))
ax[1].set_title("KL divergence")
for a in ax:
    a.set_xlabel("Training step / 25")
    a.grid(alpha=0.3)
plt.tight_layout()

# %%
# Reconstructions of test images
# ------------------------------
#
# Test images (top) and their reconstructions from the latent means (bottom).

examples = test_images[:10]
reconstructions = decoder.predict(encoder.predict(examples)[:, :latent_dim])
fig, axes = plt.subplots(2, 10, figsize=(10, 2.2))
for i in range(10):
    axes[0, i].imshow(examples[i, 0], cmap="gray_r", vmin=0, vmax=1)
    axes[1, i].imshow(reconstructions[i, 0], cmap="gray_r", vmin=0, vmax=1)
for ax in axes.ravel():
    ax.set_xticks([])
    ax.set_yticks([])
axes[0, 0].set_ylabel("Original")
axes[1, 0].set_ylabel("Decoded")
plt.tight_layout()

# %%
# Generating new digits
# ---------------------
#
# Latent vectors sampled from the standard normal prior and turned into images by the convolutional generator.
# The images are blurry, which is typical of variational autoencoders, but most of them are recognisable digits.

samples = decoder.predict(torch.randn(40, latent_dim))
fig, axes = plt.subplots(4, 10, figsize=(10, 4))
for sample, ax in zip(samples, axes.ravel()):
    ax.imshow(sample[0], cmap="gray_r", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()

# %%
# Interpolating between digits
# ----------------------------
#
# Moving along a straight line between the latent means of two test images turns one digit smoothly into the
# other.

pairs = [(0, 1), (2, 3), (4, 7)]
n_interpolation = 10
fig, axes = plt.subplots(len(pairs), n_interpolation, figsize=(n_interpolation, len(pairs) * 1.1))
for row, (i, j) in enumerate(pairs):
    start, end = encoder.predict(test_images[[i, j]])[:, :latent_dim]
    weights = torch.linspace(0, 1, n_interpolation).unsqueeze(1)
    path = decoder.predict((1 - weights) * start + weights * end)
    for column in range(n_interpolation):
        axes[row, column].imshow(path[column, 0], cmap="gray_r", vmin=0, vmax=1)
        axes[row, column].set_xticks([])
        axes[row, column].set_yticks([])
plt.tight_layout()
plt.show()
