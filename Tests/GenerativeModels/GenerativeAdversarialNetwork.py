"""
Generative adversarial network
=================================

This script trains a generative adversarial network (GAN) on the 8 x 8 digits of scikit-learn. A generator turns
random noise into images, and a discriminator tries to tell the generated images from real ones. The two networks
are trained against each other: the discriminator learns to spot fakes, and the generator learns to fool the
discriminator.

Both networks are :class:`~DLL.DeepLearning.Model.Model` objects, and the adversarial training loop is written by
hand. At the end, a conditional GAN is trained, which can generate a chosen digit.
"""
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

from DLL.DeepLearning.Model import Model
from DLL.DeepLearning.Layers import Dense
from DLL.DeepLearning.Layers.Activations import ReLU, Tanh, Sigmoid
from DLL.DeepLearning.Optimisers import ADAM


torch.manual_seed(0)

# %%
# Data
# ----
#
# The images are scaled to [-1, 1], which matches the range of the generator's tanh output.

digits = load_digits()
images = torch.tensor(digits.data, dtype=torch.float32) / 8 - 1
n_pixels = images.shape[1]
noise_dim = 16

# %%
# The networks
# ------------
#
# The generator maps 16 random numbers to the 64 pixels of an image. The discriminator maps an image to the
# probability that it is a real image. Both use the Adam parameters that are common for GANs.

generator = Model(noise_dim)
generator.add(Dense(128, activation=ReLU()))
generator.add(Dense(256, activation=ReLU()))
generator.add(Dense(n_pixels, activation=Tanh()))
generator.compile(optimiser=ADAM(learning_rate=0.0005, beta1=0.5))

discriminator = Model(n_pixels)
discriminator.add(Dense(256, activation=ReLU()))
discriminator.add(Dense(128, activation=ReLU()))
discriminator.add(Dense(0, activation=Sigmoid()))
discriminator.compile(optimiser=ADAM(learning_rate=0.0005, beta1=0.5))

# %%
# Training
# --------
#
# Every step has two parts:
#
# 1. The discriminator gets a batch of real and generated images and is trained to output 1 for the real and 0
#    for the generated images with the binary cross entropy. The real and generated images go through the
#    discriminator as one batch, because a model remembers only its latest forward pass for the backward pass.
# 2. The generator is trained to make the discriminator output 1 for new generated images, i.e. it minimises
#    :math:`-\log D(G(z))`. The gradient flows through the discriminator, which returns the gradient with respect
#    to its input, and from there into the generator. Only the generator's parameters are updated in this part.

n_steps = 6000
batch_size = 64
eps = 1e-6
fixed_noise = torch.randn(40, noise_dim)
snapshots = {}
history = {"discriminator": [], "generator": [], "real": [], "fake": []}
generator.train()
discriminator.train()
for step in range(n_steps):
    # 1. discriminator step
    real = images[torch.randint(0, len(images), (batch_size,))]
    fake = generator.predict(torch.randn(batch_size, noise_dim))
    labels = torch.cat([torch.ones(batch_size), torch.zeros(batch_size)])
    discriminator.optimiser.zero_grad()
    p = discriminator.predict(torch.cat([real, fake])).clamp(eps, 1 - eps)
    discriminator_loss = -(labels * torch.log(p) + (1 - labels) * torch.log(1 - p)).mean()
    discriminator.backward((p - labels) / (p * (1 - p)) / len(p))
    discriminator.optimiser.update_parameters()

    # 2. generator step
    generator.optimiser.zero_grad()
    fake = generator.predict(torch.randn(batch_size, noise_dim))
    p_fake = discriminator.predict(fake).clamp(eps, 1 - eps)
    generator_loss = -torch.log(p_fake).mean()
    dL_dfake = discriminator.backward(-1 / (p_fake * batch_size))
    generator.backward(dL_dfake)
    generator.optimiser.update_parameters()

    history["discriminator"].append(discriminator_loss.item())
    history["generator"].append(generator_loss.item())
    history["real"].append(p[:batch_size].mean().item())
    history["fake"].append(p[batch_size:].mean().item())
    if (step + 1) in (200, 1000, 3000, n_steps):
        generator.eval()
        snapshots[step + 1] = generator.predict(fixed_noise[:10])
        generator.train()
    if (step + 1) % 1000 == 0:
        print(f"Step {step + 1}: discriminator loss {sum(history['discriminator'][-1000:]) / 1000:.3f}, generator loss {sum(history['generator'][-1000:]) / 1000:.3f}")
generator.eval()
discriminator.eval()


def smooth(values, window=100):
    return torch.tensor(values).reshape(-1, window).mean(dim=1)


fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(smooth(history["discriminator"]), label="discriminator")
ax[0].plot(smooth(history["generator"]), label="generator")
ax[0].set_ylabel("Loss")
ax[1].plot(smooth(history["real"]), label="real images")
ax[1].plot(smooth(history["fake"]), label="generated images")
ax[1].set_ylabel("Mean output of the discriminator")
for a in ax:
    a.set_xlabel("Training step / 100")
    a.grid(alpha=0.3)
    a.legend()
plt.tight_layout()

# %%
# The losses do not decrease steadily like in supervised learning, because the two networks compete: when one of
# them improves, the task of the other gets harder. The discriminator should not win completely, which would
# show as its output for the generated images going to zero.
#
# Training progress
# -----------------
#
# The same 10 noise vectors decoded by the generator at different stages of the training.

fig, axes = plt.subplots(len(snapshots), 10, figsize=(10, 1.1 * len(snapshots)))
for row, (step, samples) in enumerate(snapshots.items()):
    for column in range(10):
        axes[row, column].imshow(samples[column].reshape(8, 8), cmap="gray_r", vmin=-1, vmax=1)
        axes[row, column].set_xticks([])
        axes[row, column].set_yticks([])
    axes[row, 0].set_ylabel(f"step {step}", rotation=0, labelpad=30, fontsize=9)
plt.tight_layout()

# %%
# Generated digits
# ----------------
#
# Images generated from 40 random noise vectors after the training.

samples = generator.predict(fixed_noise)
fig, axes = plt.subplots(4, 10, figsize=(10, 4))
for sample, ax in zip(samples, axes.ravel()):
    ax.imshow(sample.reshape(8, 8), cmap="gray_r", vmin=-1, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()

# %%
# Conditional GAN
# ---------------
#
# In a conditional GAN, both networks also get the label of the image as a one-hot vector. The generator learns
# to draw the requested digit, and the discriminator learns to reject images that do not match their label. The
# training loop is the same as above, with the labels added to the inputs.

labels = torch.eye(10)[torch.tensor(digits.target)]

conditional_generator = Model(noise_dim + 10)
conditional_generator.add(Dense(128, activation=ReLU()))
conditional_generator.add(Dense(256, activation=ReLU()))
conditional_generator.add(Dense(n_pixels, activation=Tanh()))
conditional_generator.compile(optimiser=ADAM(learning_rate=0.0005, beta1=0.5))

conditional_discriminator = Model(n_pixels + 10)
conditional_discriminator.add(Dense(256, activation=ReLU()))
conditional_discriminator.add(Dense(128, activation=ReLU()))
conditional_discriminator.add(Dense(0, activation=Sigmoid()))
conditional_discriminator.compile(optimiser=ADAM(learning_rate=0.0005, beta1=0.5))

conditional_generator.train()
conditional_discriminator.train()
for step in range(n_steps):
    indices = torch.randint(0, len(images), (batch_size,))
    real, real_labels = images[indices], labels[indices]
    fake_labels = torch.eye(10)[torch.randint(0, 10, (batch_size,))]
    fake = conditional_generator.predict(torch.cat([torch.randn(batch_size, noise_dim), fake_labels], dim=1))
    targets = torch.cat([torch.ones(batch_size), torch.zeros(batch_size)])
    conditional_discriminator.optimiser.zero_grad()
    p = conditional_discriminator.predict(torch.cat([torch.cat([real, real_labels], dim=1), torch.cat([fake, fake_labels], dim=1)])).clamp(eps, 1 - eps)
    conditional_discriminator.backward((p - targets) / (p * (1 - p)) / len(p))
    conditional_discriminator.optimiser.update_parameters()

    conditional_generator.optimiser.zero_grad()
    fake_labels = torch.eye(10)[torch.randint(0, 10, (batch_size,))]
    fake = conditional_generator.predict(torch.cat([torch.randn(batch_size, noise_dim), fake_labels], dim=1))
    p_fake = conditional_discriminator.predict(torch.cat([fake, fake_labels], dim=1)).clamp(eps, 1 - eps)
    # the discriminator returns the gradient with respect to its whole input; the first columns belong to the image
    dL_dfake = conditional_discriminator.backward(-1 / (p_fake * batch_size))[:, :n_pixels]
    conditional_generator.backward(dL_dfake)
    conditional_generator.optimiser.update_parameters()
    if (step + 1) % 2000 == 0:
        print(f"Conditional GAN, step {step + 1}")
conditional_generator.eval()

# %%
# Every row is generated for the digit on its left from different noise.

n_per_digit = 10
y = torch.eye(10).repeat_interleave(n_per_digit, dim=0)
generated = conditional_generator.predict(torch.cat([torch.randn(len(y), noise_dim), y], dim=1))
fig, axes = plt.subplots(10, n_per_digit, figsize=(n_per_digit, 10))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(generated[i].reshape(8, 8), cmap="gray_r", vmin=-1, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
for digit in range(10):
    axes[digit, 0].set_ylabel(str(digit), rotation=0, labelpad=12, fontsize=12)
fig.suptitle("Digits generated by the conditional GAN")
plt.tight_layout()
plt.show()
