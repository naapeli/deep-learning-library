from Model import Model
from Layers.Dense import Dense
from Layers.Conv2D import Conv2D
from Layers.Flatten import Flatten
from Layers.MaxPooling2D import MaxPooling2D
from Layers.Regularisation.Dropout import Dropout
from Layers.Activations.ReLU import ReLU
from Layers.Activations.SoftMax import SoftMax
from Losses.CCE import cce
from Data.Preprocessing import OneHotEncoder, data_split
from Data.Metrics import accuracy

import torch
import matplotlib.pyplot as plt
import tensorflow as tf


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = torch.from_numpy(train_images).to(dtype=torch.float32).reshape(60000, 1, 28, 28)
train_labels = torch.from_numpy(train_labels).to(dtype=torch.float32)
test_images = torch.from_numpy(test_images).to(dtype=torch.float32).reshape(10000, 1, 28, 28)
test_labels = torch.from_numpy(test_labels).to(dtype=torch.float32)
train_images = train_images / train_images.max()
test_images = test_images / test_images.max()

train_images, train_labels, validation_images, validation_labels, _, _ = data_split(train_images, train_labels, train_split=0.9, validation_split=0.1)

label_encoder = OneHotEncoder()
train_labels = label_encoder.one_hot_encode(train_labels)
validation_labels = label_encoder.one_hot_encode(validation_labels)
test_labels = label_encoder.one_hot_encode(test_labels)
print(train_images.shape, train_labels.shape, validation_images.shape, validation_labels.shape, test_images.shape, test_labels.shape)
print(train_labels[:2])

model = Model((1, 28, 28))
model.add(Conv2D(kernel_size=3, output_depth=32, activation=ReLU()))
model.add(Conv2D(kernel_size=3, output_depth=64, activation=ReLU()))
model.add(MaxPooling2D(pool_size=3))
model.add(Dropout(p=0.5))
model.add(Flatten())
model.add(Dense(250, activation=ReLU()))
model.add(Dense(10, activation=SoftMax()))
model.compile(optimiser=None, loss=cce(), metrics=["loss", "val_loss", "val_accuracy", "accuracy"])
model.summary()
model.fit(train_images, train_labels, val_data=(validation_images[:1000], validation_labels[:1000]), epochs=1, batch_size=64)
