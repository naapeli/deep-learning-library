"""
Masked diffusion language model
==================================

This script trains a small masked diffusion language model. Diffusion models for images add noise to the data and
learn to remove it. For text, the noise is masking: some of the characters are replaced by a special mask token,
and the model learns to fill them in. Unlike an autoregressive language model, which writes text from left to
right, a masked diffusion model starts from a fully masked sequence and reveals the characters in any order,
looking at the context on both sides.

The model is built from :class:`~DLL.DeepLearning.Layers.MultiHeadAttention` without the causal mask, so every
position can attend to every other position. The text is the documentation of the Python language that is shipped
with Python itself in the ``pydoc_data`` module, so nothing has to be downloaded.
"""
import random
import re

import torch
import matplotlib.pyplot as plt
from pydoc_data.topics import topics

from DLL.DeepLearning.Model import Model
from DLL.DeepLearning.Layers import MultiHeadAttention, Dense, Add, Identity
from DLL.DeepLearning.Layers.Regularisation import LayerNorm
from DLL.DeepLearning.Layers.Activations import ReLU, SoftMax
from DLL.DeepLearning.Optimisers import ADAM


torch.manual_seed(0)
random.seed(0)

# %%
# Data
# ----
#
# The documentation consists of about 500 000 characters. To keep the task small, the text is converted to lower
# case and only letters, spaces, full stops and commas are kept. The paragraphs are split randomly into a training
# set (95 %) and a validation set (5 %) and cut into windows of 32 characters.

paragraphs = []
for topic in topics.values():
    for paragraph in topic.split("\n\n"):
        paragraph = re.sub(r"[^a-z .,]", "", " ".join(paragraph.lower().split()))
        paragraph = " ".join(paragraph.split())
        if len(paragraph) > 40:
            paragraphs.append(paragraph)
random.shuffle(paragraphs)
n_train = int(0.95 * len(paragraphs))
train_text = " ".join(paragraphs[:n_train])
val_text = " ".join(paragraphs[n_train:])

chars = sorted(set(train_text + val_text))
char_to_index = {char: i for i, char in enumerate(chars)}
vocab_size = len(chars)
mask_index = vocab_size  # the mask token gets its own index after the characters
seq_len = 32


def windows(text, stride):
    ids = torch.tensor([char_to_index[char] for char in text])
    return torch.stack([ids[start:start + seq_len] for start in range(0, len(ids) - seq_len, stride)])


train_ids = windows(train_text, stride=7)
val_ids = windows(val_text, stride=16)[:600]
print(f"{vocab_size} different characters, {len(train_ids)} training and {len(val_ids)} validation windows")


def encode(ids):
    # one-hot encoding of the characters and the mask token, followed by a one-hot encoding of the position
    tokens = torch.eye(vocab_size + 1)[ids]
    positions = torch.eye(seq_len).expand(len(ids), seq_len, seq_len)
    return torch.cat([tokens, positions], dim=2)


# %%
# The model
# ---------
#
# Two blocks of multi-head attention (without the causal mask) and a feed-forward layer, both with residual
# connections and layer normalisation. The model outputs a probability distribution over the characters at every
# position.

embedding_dim = 64
model = Model((seq_len, vocab_size + 1 + seq_len))
model.add(Dense((seq_len, embedding_dim)))
for _ in range(2):
    model.add(Add(MultiHeadAttention((seq_len, embedding_dim), n_heads=4, use_mask=False), Identity()))
    model.add(LayerNorm())
    model.add(Add(Dense((seq_len, embedding_dim), activation=ReLU()), Identity()))
    model.add(LayerNorm())
model.add(Dense((seq_len, vocab_size), activation=SoftMax()))
model.compile(optimiser=ADAM(learning_rate=0.003))
model.summary()

# %%
# Training
# --------
#
# For every window, a masking rate :math:`t` is drawn uniformly from (0, 1), and every character is replaced by
# the mask token with probability :math:`t`. The model is trained to predict the original characters at the masked
# positions with the cross entropy, averaged over the masked positions. Training on all masking rates teaches the
# model to fill in anything from a single missing character to a fully masked window.


def mask(ids, rate):
    masked = torch.rand(ids.shape) < rate
    return torch.where(masked, mask_index, ids), masked


def masked_cross_entropy(probabilities, ids, masked):
    target_probabilities = probabilities.gather(2, ids.unsqueeze(2)).squeeze(2)
    return -(torch.log(target_probabilities + 1e-9) * masked).sum() / masked.sum()


generator = torch.Generator().manual_seed(1)
val_rates = torch.rand(len(val_ids), 1, generator=generator)
val_masked = torch.rand(val_ids.shape, generator=generator) < val_rates
val_inputs = encode(torch.where(val_masked, mask_index, val_ids))

n_steps = 6000
batch_size = 64
history = {"step": [], "train": [], "val": []}
running_loss = 0
model.train()
for step in range(n_steps):
    ids = train_ids[torch.randint(0, len(train_ids), (batch_size,))]
    rates = torch.rand(batch_size, 1)
    inputs, masked = mask(ids, rates)
    model.optimiser.zero_grad()
    probabilities = model.predict(encode(inputs))
    running_loss += masked_cross_entropy(probabilities, ids, masked).item()
    one_hot_targets = torch.eye(vocab_size)[ids]
    gradient = -one_hot_targets * masked.unsqueeze(2) / (probabilities + 1e-9) / masked.sum()
    model.backward(gradient)
    model.optimiser.update_parameters()
    if (step + 1) % 250 == 0:
        model.eval()
        history["step"].append(step + 1)
        history["train"].append(running_loss / 250)
        history["val"].append(masked_cross_entropy(model.predict(val_inputs), val_ids, val_masked).item())
        running_loss = 0
        model.train()
    if (step + 1) % 1000 == 0:
        print(f"Step {step + 1}: training loss {history['train'][-1]:.3f}, validation loss {history['val'][-1]:.3f}")
model.eval()

plt.figure(figsize=(8, 4))
plt.plot(history["step"], history["train"], label="training")
plt.plot(history["step"], history["val"], label="validation")
plt.xlabel("Training step")
plt.ylabel("Cross entropy of the masked characters")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# %%
# How much context helps
# ----------------------
#
# The more of a window is masked, the less context the model has. The validation accuracy of the masked
# characters as a function of the masking rate shows how the model uses the characters on both sides.

rates = torch.linspace(0.1, 1.0, 10)
accuracies = []
for rate in rates:
    inputs, masked = mask(val_ids, rate)
    predictions = model.predict(encode(inputs)).argmax(dim=2)
    accuracies.append(((predictions == val_ids) & masked).sum().item() / masked.sum().item())
plt.figure(figsize=(8, 4))
plt.plot(rates, accuracies, marker="o")
plt.xlabel("Fraction of the characters masked")
plt.ylabel("Accuracy of the masked characters")
plt.grid(alpha=0.3)
plt.tight_layout()

# %%
# Generating text
# ---------------
#
# Generation starts from a fully masked window. At each of 16 steps, the model predicts every masked position,
# a character is sampled for each of them, and the two most confident predictions are revealed. The figure shows
# how the text appears; masked positions are drawn as underscores. The model usually reveals the spaces first,
# which fixes the lengths of the words, and then fills in common words. The model is small and trained for about
# a minute, so the generated text consists of English-like words rather than sentences. Characters that are
# given in advance stay fixed, so the same procedure also fills in gaps in a given text.


def generate(ids, n_steps=16, temperature=0.5):
    ids = ids.clone()
    states = [ids.clone()]
    for step in range(n_steps):
        masked = ids == mask_index
        n_masked = masked.sum(dim=1)
        if n_masked.max() == 0:
            break
        probabilities = model.predict(encode(ids)) ** (1 / temperature)
        probabilities = probabilities / probabilities.sum(dim=2, keepdim=True)
        sampled = torch.multinomial(probabilities.reshape(-1, vocab_size), 1).reshape(ids.shape)
        confidence = probabilities.gather(2, sampled.unsqueeze(2)).squeeze(2).masked_fill(~masked, -1)
        n_reveal = torch.ceil(n_masked / (n_steps - step)).long()
        for row in range(len(ids)):
            reveal = confidence[row].topk(int(n_reveal[row])).indices
            ids[row, reveal] = sampled[row, reveal]
        states.append(ids.clone())
    return states


def to_text(ids):
    return "".join("_" if i == mask_index else chars[i] for i in ids.tolist())


states = generate(torch.full((1, seq_len), mask_index))
fig, ax = plt.subplots(figsize=(8, 5))
for step, state in enumerate(states):
    ax.text(0, -step, f"step {step:2d}   " + to_text(state[0]), family="monospace", fontsize=10, va="center")
ax.set_xlim(0, 1)
ax.set_ylim(-len(states), 1)
ax.axis("off")
ax.set_title("Unmasking a fully masked window")
plt.tight_layout()

print("Generated windows:")
for state in generate(torch.full((8, seq_len), mask_index))[-1]:
    print("  " + to_text(state))

# %%
# Filling in gaps
# ---------------
#
# The window is given except for a few characters inside a word, and the model fills them in using the context
# on both sides. A lower temperature of 0.3 makes the model choose its most likely characters. Longer gaps are
# much harder for a model of this size.

prompts = ["the val__ of the expression is th", "a functi__ is defined with the d", "the ret___ statement is used to "]
for prompt in prompts:
    ids = torch.tensor([[mask_index if char == "_" else char_to_index[char] for char in prompt.ljust(seq_len)[:seq_len]]])
    print(f"{prompt.ljust(seq_len)[:seq_len]}  ->  {to_text(generate(ids, temperature=0.3)[-1][0])}")
plt.show()
