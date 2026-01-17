"""
MNIST Image classification
==================================

This script implements a model to classify the MNIST dataset. The model mainly consists of 
convolutonal layers and pooling layers with a few dense layers at the end. As the script is 
only for demonstration purposes, only 100 first datapoints are used to make the training faster. 
For a full example, change the parameter n to 60000. If n is increased, more epochs may need 
to be added and other hyperparameters tuned.
"""
import torch
from torchvision import datasets
import matplotlib.pyplot as plt

from DLL.DeepLearning.Model import Model
from DLL.DeepLearning.Layers import Dense, Conv2D, Flatten, MaxPooling2D, Reshape, Add, Identity
from DLL.DeepLearning.Layers.Regularisation import Dropout, BatchNorm, GroupNorm, InstanceNorm, LayerNorm
from DLL.DeepLearning.Layers.Activations import ReLU, SoftMax
from DLL.DeepLearning.Losses import CCE
from DLL.DeepLearning.Optimisers import SGD, ADAM
from DLL.DeepLearning.Callbacks import EarlyStopping
from DLL.DeepLearning.Initialisers import Xavier_Normal, Xavier_Uniform, Kaiming_Normal, Kaiming_Uniform
from DLL.Data.Preprocessing import OneHotEncoder, data_split
from DLL.Data.Metrics import accuracy


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

torch.manual_seed(0)
torch.cuda.manual_seed(0)

n = 512  # 60000
train_set = datasets.MNIST(root="./data", train=True, download=True)
test_set = datasets.MNIST(root="./data", train=False, download=True)
train_images = train_set.data[:n]
train_labels = train_set.targets[:n]
test_images = test_set.data[:n]
test_labels = test_set.targets[:n]
train_images = train_images.unsqueeze(1).to(dtype=torch.float32, device=device)
test_images = test_images.unsqueeze(1).to(dtype=torch.float32, device=device)
train_images = train_images / train_images.max()
test_images = test_images / test_images.max()
train_labels = train_labels.to(dtype=torch.float32, device=device)
test_labels = test_labels.to(dtype=torch.float32, device=device)

train_images, train_labels, validation_images, validation_labels, _, _ = data_split(train_images, train_labels, train_split=0.8, validation_split=0.2)

label_encoder = OneHotEncoder()
train_labels = label_encoder.fit_encode(train_labels)
validation_labels = label_encoder.encode(validation_labels)
test_labels = label_encoder.encode(test_labels)
print(train_images.shape, train_labels.shape, validation_images.shape, validation_labels.shape, test_images.shape, test_labels.shape)
print(train_labels[:2])

model = Model((1, 28, 28), device=device)
model.train()

model.add(Conv2D(kernel_size=3, output_depth=32, initialiser=Kaiming_Normal(), activation=ReLU()))
model.add(Add(Conv2D(kernel_size=3, output_depth=32, initialiser=Kaiming_Normal(), activation=ReLU()), Identity()))
model.add(Add(Conv2D(kernel_size=3, output_depth=32, initialiser=Kaiming_Normal(), activation=ReLU()), Identity()))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNorm())
model.add(Add(Conv2D(kernel_size=3, output_depth=32, initialiser=Kaiming_Normal(), activation=ReLU()), Identity()))
model.add(Add(Conv2D(kernel_size=3, output_depth=32, initialiser=Kaiming_Normal(), activation=ReLU()), Identity()))
model.add(Add(Conv2D(kernel_size=3, output_depth=32, initialiser=Kaiming_Normal(), activation=ReLU()), Identity()))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNorm())
model.add(Add(Conv2D(kernel_size=3, output_depth=32, initialiser=Kaiming_Normal(), activation=ReLU()), Identity()))
model.add(Add(Conv2D(kernel_size=3, output_depth=32, initialiser=Kaiming_Normal(), activation=ReLU()), Identity()))
model.add(Add(Conv2D(kernel_size=3, output_depth=32, initialiser=Kaiming_Normal(), activation=ReLU()), Identity()))
model.add(Conv2D(kernel_size=3, output_depth=8, initialiser=Kaiming_Normal(), activation=ReLU()))
model.add(BatchNorm())
model.add(Dropout(p=0.3))
model.add(Flatten())
model.add(Dense(200, activation=ReLU()))
model.add(Dense(10, activation=SoftMax()))

model.compile(optimiser=ADAM(learning_rate=0.001), loss=CCE(), metrics=["loss", "val_loss", "val_accuracy", "accuracy"], callbacks=(EarlyStopping(patience=3),))
model.summary()


history = model.fit(train_images, train_labels, val_data=(validation_images, validation_labels), epochs=25, batch_size=256, verbose=True)

model.eval()
test_pred = model.predict(test_images)
print(accuracy(test_pred, test_labels))

plt.figure(figsize=(8, 6))
plt.subplot(1, 2, 1)
plt.plot(history["val_loss"], label="validation loss")
plt.plot(history["loss"], label="loss")
plt.xlabel("Epoch")
plt.ylabel("Categorical cross entropy")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history["val_accuracy"], label="validation accuracy")
plt.plot(history["accuracy"], label="accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

fig, ax = plt.subplots(2, 2, figsize=(8, 8))
ax = ax.ravel()
for i in range(len(ax)):
    ax[i].imshow(test_images[i].numpy()[0], cmap='gray', vmin=0, vmax=1)
    ax[i].set_title(f"True label: {test_labels[i].argmax()} | Predicted label: {test_pred[i].argmax()}")
plt.show()
