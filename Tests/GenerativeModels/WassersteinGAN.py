"""
Wasserstein GAN
==================

This script compares a standard generative adversarial network (GAN) with a Wasserstein GAN (WGAN). In a WGAN,
the discriminator is replaced by a critic, which outputs an unbounded score instead of a probability. The critic
estimates the Wasserstein distance between the real and the generated data, and the generator is trained to
reduce it. This gives the generator useful gradients even when the critic is trained well, which makes the
training more stable than with a standard GAN.

The models are first compared on a two-dimensional dataset of 10 Gaussian blobs, where it is easy to see whether
the generator covers all of them. Then a WGAN is trained on the 8 x 8 digits of scikit-learn.
"""
import math

import torch
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

from DLL.DeepLearning.Model import Model
from DLL.DeepLearning.Layers import Dense
from DLL.DeepLearning.Layers.Activations import ReLU, Tanh, Sigmoid
from DLL.DeepLearning.Optimisers import ADAM, RMSPROP


torch.manual_seed(0)

# %%
# Ten Gaussian blobs
# ------------------
#
# The data is a mixture of 10 Gaussian distributions with a standard deviation of 0.1, placed on a circle of
# radius 2. A generator that only produces some of the blobs suffers from mode collapse, a common failure of GANs.

n_blobs, blob_std = 10, 0.1
angles = torch.arange(n_blobs) * 2 * math.pi / n_blobs
centers = 2 * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)


def sample_blobs(n):
    return centers[torch.randint(0, n_blobs, (n,))] + blob_std * torch.randn(n, 2)


def evaluate(generator, n=2000):
    # a blob counts as covered if at least 1 % of the samples are within 3 standard deviations of its center
    samples = generator.predict(torch.randn(n, 2))
    nearest = torch.cdist(samples, centers).min(dim=1)
    close = nearest.values < 3 * blob_std
    counts = torch.bincount(nearest.indices[close], minlength=n_blobs)
    return int((counts >= n / 100).sum()), close.float().mean().item()


# %%
# Training both models
# --------------------
#
# Both models use the same small dense networks, and the discriminator or critic is updated 5 times for every
# update of the generator. The two models differ in three ways:
#
# - The GAN discriminator outputs a probability and is trained with the binary cross entropy. The WGAN critic
#   outputs a score and is trained to maximise :math:`E[D(x_{real})] - E[D(x_{generated})]`, an estimate of the
#   Wasserstein distance.
# - The WGAN generator minimises :math:`-E[D(G(z))]`, while the GAN generator minimises :math:`-E[\log D(G(z))]`.
# - The critic has to be Lipschitz continuous for the estimate to be valid. As in the original WGAN, this is
#   enforced by clipping the parameters of the critic to a small interval after every update.


def build_networks(wasserstein):
    generator = Model(2)
    generator.add(Dense(64, activation=ReLU()))
    generator.add(Dense(64, activation=ReLU()))
    generator.add(Dense(2))
    critic = Model(2)
    critic.add(Dense(64, activation=ReLU()))
    critic.add(Dense(64, activation=ReLU()))
    critic.add(Dense(0) if wasserstein else Dense(0, activation=Sigmoid()))
    if wasserstein:
        generator.compile(optimiser=RMSPROP(learning_rate=0.001))
        critic.compile(optimiser=RMSPROP(learning_rate=0.001))
    else:
        generator.compile(optimiser=ADAM(learning_rate=0.001, beta1=0.5))
        critic.compile(optimiser=ADAM(learning_rate=0.001, beta1=0.5))
    return generator, critic


def train(wasserstein, n_steps=3000, n_critic=5, batch_size=128, clip=0.1, eps=1e-6):
    generator, critic = build_networks(wasserstein)
    trace = []
    for step in range(n_steps):
        for _ in range(n_critic):
            real = sample_blobs(batch_size)
            fake = generator.predict(torch.randn(batch_size, 2))
            critic.optimiser.zero_grad()
            output = critic.predict(torch.cat([real, fake]))
            if wasserstein:
                # gradient of E[D(fake)] - E[D(real)]
                critic.backward(torch.cat([-torch.ones(batch_size), torch.ones(batch_size)]) / batch_size)
            else:
                labels = torch.cat([torch.ones(batch_size), torch.zeros(batch_size)])
                p = output.clamp(eps, 1 - eps)
                critic.backward((p - labels) / (p * (1 - p)) / len(p))
            critic.optimiser.update_parameters()
            if wasserstein:
                for parameter in critic.optimiser.model_parameters:
                    parameter.clamp_(-clip, clip)

        generator.optimiser.zero_grad()
        output = critic.predict(generator.predict(torch.randn(batch_size, 2)))
        if wasserstein:
            gradient = -torch.ones(batch_size) / batch_size
        else:
            gradient = -1 / (output.clamp(eps, 1) * batch_size)
        generator.backward(critic.backward(gradient))
        generator.optimiser.update_parameters()
        if (step + 1) % 250 == 0:
            trace.append((step + 1, *evaluate(generator)))
    return generator, trace


gan, gan_trace = train(wasserstein=False)
wgan, wgan_trace = train(wasserstein=True)
for name, trace in (("GAN", gan_trace), ("WGAN", wgan_trace)):
    print(f"{name}: covers {trace[-1][1]} of the {n_blobs} blobs, {100 * trace[-1][2]:.0f} % of the samples are close to a blob")

# %%
# Results on the blobs
# --------------------
#
# The generated samples of both models on top of the real data, and the number of blobs covered during the
# training. With 5 discriminator updates per generator update, the discriminator of the standard GAN becomes too
# strong, its gradients become less informative, and the generator tends to drop some of the blobs. The WGAN
# critic can be trained this much without that problem.

real = sample_blobs(2000)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (name, generator) in zip(axes[:2], (("Standard GAN", gan), ("Wasserstein GAN", wgan))):
    samples = generator.predict(torch.randn(2000, 2))
    ax.scatter(real[:, 0], real[:, 1], s=3, color="lightgray", label="real data")
    ax.scatter(samples[:, 0], samples[:, 1], s=3, label="generated")
    ax.set_title(name)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", markerscale=4)
for name, trace in (("standard GAN", gan_trace), ("Wasserstein GAN", wgan_trace)):
    axes[2].plot([step for step, _, _ in trace], [covered for _, covered, _ in trace], marker="o", label=name)
axes[2].set_xlabel("Generator update")
axes[2].set_ylabel("Blobs covered")
axes[2].set_ylim(0, n_blobs + 0.5)
axes[2].grid(alpha=0.3)
axes[2].legend()
plt.tight_layout()

# %%
# A Wasserstein GAN for images
# ----------------------------
#
# The same training procedure is used for the digits, with the same dense networks as in the standard GAN
# example. The estimate of the Wasserstein distance, :math:`E[D(x_{real})] - E[D(x_{generated})]`, decreases as
# the generated images improve. This is a useful property: the discriminator loss of a standard GAN says little
# about the quality of the images.

images = torch.tensor(load_digits().data, dtype=torch.float32) / 8 - 1
noise_dim = 16

generator = Model(noise_dim)
generator.add(Dense(128, activation=ReLU()))
generator.add(Dense(256, activation=ReLU()))
generator.add(Dense(64, activation=Tanh()))
generator.compile(optimiser=RMSPROP(learning_rate=0.001))
critic = Model(64)
critic.add(Dense(256, activation=ReLU()))
critic.add(Dense(128, activation=ReLU()))
critic.add(Dense(0))
critic.compile(optimiser=RMSPROP(learning_rate=0.001))

n_steps, n_critic, batch_size, clip = 2500, 5, 64, 0.05
distance_estimates = []
for step in range(n_steps):
    for _ in range(n_critic):
        real = images[torch.randint(0, len(images), (batch_size,))]
        fake = generator.predict(torch.randn(batch_size, noise_dim))
        critic.optimiser.zero_grad()
        output = critic.predict(torch.cat([real, fake]))
        critic.backward(torch.cat([-torch.ones(batch_size), torch.ones(batch_size)]) / batch_size)
        critic.optimiser.update_parameters()
        for parameter in critic.optimiser.model_parameters:
            parameter.clamp_(-clip, clip)
    distance_estimates.append((output[:batch_size].mean() - output[batch_size:].mean()).item())

    generator.optimiser.zero_grad()
    output = critic.predict(generator.predict(torch.randn(batch_size, noise_dim)))
    generator.backward(critic.backward(-torch.ones(batch_size) / batch_size))
    generator.optimiser.update_parameters()
    if (step + 1) % 500 == 0:
        print(f"Step {step + 1}: estimated Wasserstein distance {sum(distance_estimates[-500:]) / 500:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [1, 1.4]})
ax[0].plot(torch.tensor(distance_estimates).reshape(-1, 50).mean(dim=1))
ax[0].set_xlabel("Generator update / 50")
ax[0].set_ylabel("Estimated Wasserstein distance")
ax[0].grid(alpha=0.3)
samples = generator.predict(torch.randn(40, noise_dim))
canvas = -torch.ones(4 * 9 + 1, 10 * 9 + 1)  # -1 is a white pixel
for i, sample in enumerate(samples):
    row, column = divmod(i, 10)
    canvas[1 + 9 * row:9 + 9 * row, 1 + 9 * column:9 + 9 * column] = sample.reshape(8, 8)
ax[1].imshow(canvas, cmap="gray_r", vmin=-1, vmax=1)
ax[1].set_title("Digits generated by the WGAN")
ax[1].axis("off")
plt.tight_layout()
plt.show()
