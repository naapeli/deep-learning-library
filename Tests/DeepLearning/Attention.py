"""
Character-level language model with attention
================================================

This script trains a tiny language model with
:class:`~DLL.DeepLearning.Layers.MultiHeadAttention`. The model reads 24 characters at a time and predicts the
next character at every position, like a (very) small GPT. A causal mask makes sure that each position only
attends to itself and the characters before it.

The text is the Zen of Python, which is shipped with Python itself, so nothing has to be downloaded. The text
is tiny, so the model mostly memorises it, but it is enough to show what attention adds: the same network
without the attention layer can only look at the current character.
"""
import codecs
import this  # importing this prints the Zen of Python

import torch
import matplotlib.pyplot as plt

from DLL.DeepLearning.Model import Model
from DLL.DeepLearning.Layers import MultiHeadAttention, Dense, Add, Identity
from DLL.DeepLearning.Layers.Regularisation import LayerNorm
from DLL.DeepLearning.Layers.Activations import ReLU, SoftMax
from DLL.DeepLearning.Optimisers import ADAM
from DLL.DeepLearning.Losses import CCE


torch.manual_seed(0)

# %%
# Data
# ----
#
# Every character is a token. A training sample is a window of 24 consecutive characters, and the target is the
# same window shifted by one character. Each token is encoded as a one-hot vector of the character, followed by
# a one-hot vector of its position in the window. Attention itself does not know the order of the tokens, so the
# position has to be part of the input.

text = codecs.decode(this.s, "rot13")
chars = sorted(set(text))
char_to_index = {char: i for i, char in enumerate(chars)}
encoded_text = torch.tensor([char_to_index[char] for char in text])
vocab_size = len(chars)
seq_len = 24

starts = torch.arange(len(encoded_text) - seq_len)
input_ids = torch.stack([encoded_text[start:start + seq_len] for start in starts])
target_ids = torch.stack([encoded_text[start + 1:start + seq_len + 1] for start in starts])


def encode(ids):
    one_hot_chars = torch.eye(vocab_size)[ids]
    positions = torch.eye(seq_len).expand(len(ids), seq_len, seq_len)
    return torch.cat([one_hot_chars, positions], dim=2)


X = encode(input_ids)
Y = torch.eye(vocab_size)[target_ids]
print(f"{len(text)} characters, {vocab_size} different characters, {len(X)} windows of {seq_len} characters")

# %%
# The models
# ----------
#
# The language model embeds every token into 48 features with a dense layer. Then follows a block of masked
# multi-head attention with 4 heads and a residual connection, and a residual feed-forward block, both followed
# by layer normalisation. The last dense layer predicts a probability distribution over the next character at
# every position.
#
# The baseline is the same network without the attention block. Every position is then processed on its own,
# so the baseline can only learn which character usually follows the current one.

embedding_dim = 48


def build_model(use_attention):
    model = Model((seq_len, vocab_size + seq_len))
    model.add(Dense((seq_len, embedding_dim)))
    if use_attention:
        model.add(Add(attention, Identity()))
        model.add(LayerNorm())
    model.add(Add(Dense((seq_len, embedding_dim), activation=ReLU()), Identity()))
    model.add(LayerNorm())
    model.add(Dense((seq_len, vocab_size), activation=SoftMax()))
    model.compile(optimiser=ADAM(learning_rate=0.003), loss=CCE(), metrics=["loss", "accuracy"])
    return model


attention = MultiHeadAttention((seq_len, embedding_dim), n_heads=4, use_mask=True)
model = build_model(use_attention=True)
baseline = build_model(use_attention=False)
model.summary()

# %%
# Training
# --------
#
# Both models are trained for 60 epochs. The loss is the cross entropy of the next character, averaged over all
# positions, and the accuracy is the fraction of positions where the most likely character is the correct one.

history = model.fit(X, Y, epochs=60, batch_size=64)
history_baseline = baseline.fit(X, Y, epochs=60, batch_size=64)
print(f"With attention: loss {history['loss'][-1]:.3f}, accuracy {history['accuracy'][-1]:.3f}")
print(f"Without attention: loss {history_baseline['loss'][-1]:.3f}, accuracy {history_baseline['accuracy'][-1]:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(history["loss"], label="with attention")
ax[0].plot(history_baseline["loss"], label="without attention")
ax[0].set_ylabel("Cross entropy per character")
ax[1].plot(history["accuracy"], label="with attention")
ax[1].plot(history_baseline["accuracy"], label="without attention")
ax[1].set_ylabel("Next character accuracy")
for a in ax:
    a.set_xlabel("Epoch")
    a.grid(alpha=0.3)
    a.legend()
plt.tight_layout()

# %%
# Generating text
# ---------------
#
# The model generates text one character at a time. The window ends with the latest characters, the predicted
# distribution at its last position is sharpened with a temperature of 0.5, and the next character is sampled
# from it.


def generate(model, prompt, n_chars, temperature=0.5):
    ids = [char_to_index[char] for char in prompt]
    for _ in range(n_chars):
        window = torch.tensor(ids[-seq_len:]).unsqueeze(0)
        probabilities = model.predict(encode(window))[0, -1] ** (1 / temperature)
        ids.append(torch.multinomial(probabilities / probabilities.sum(), 1).item())
    return "".join(chars[i] for i in ids)


prompt = "Beautiful is better than"
print("With attention:\n" + generate(model, prompt, 150))
print("\nWithout attention:\n" + generate(baseline, prompt, 150))

# %%
# Attention weights
# -----------------
#
# Every head assigns weights to the current and the previous characters, and the weights in each row sum to one.
# The causal mask keeps everything above the diagonal at zero. The heads learn different patterns, for instance
# looking at the previous character or at the start of the current word. Spaces are shown as ␣.

sample = "Simple is better than co"
model.predict(encode(torch.tensor([char_to_index[char] for char in sample]).unsqueeze(0)))
weights = attention.attention_scores[0]
labels = ["␣" if char == " " else char for char in sample]

fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
for head, ax in enumerate(axes):
    ax.imshow(weights[head], cmap="Blues", vmin=0, vmax=1)
    ax.set_title(f"Head {head + 1}")
    ax.set_xticks(range(seq_len), labels, fontsize=7)
    ax.set_yticks(range(seq_len), labels, fontsize=7)
    ax.set_xlabel("Attended character")
axes[0].set_ylabel("Current character")
plt.tight_layout()
plt.show()
