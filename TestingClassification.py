from Model import Model
from Layers.Dense import Dense
from Layers.Regularisation.Dropout import Dropout
from Layers.Regularisation.BatchNormalisation import BatchNorm1d
from Layers.Regularisation.GroupNormalisation import GroupNorm1d
from Layers.Regularisation.InstanceNormalisation import InstanceNorm1d
from Layers.Regularisation.LayerNormalisation import LayerNorm1d
from Layers.Activations.ReLU import ReLU
from Layers.Activations.SoftMax import SoftMax
from Losses.CCE import cce
from Data.Preprocessing import data_split, OneHotEncoder, MinMaxScaler
from Data.Metrics import accuracy

import torch
import matplotlib.pyplot as plt
from sklearn import datasets


iris = datasets.load_iris()

encoder = OneHotEncoder()
scaler = MinMaxScaler()
x = torch.tensor(iris.data, dtype=torch.float32)
scaler.fit(x)
x = scaler.transform(x)
y = encoder.one_hot_encode(torch.tensor(iris.target, dtype=torch.float32))
x_train, y_train, x_val, y_val, x_test, y_test = data_split(x, y, train_split=0.6, validation_split=0.2)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = Model(4, data_type=torch.float32)
model.add(Dense(20, normalisation=BatchNorm1d(), activation=ReLU()))
model.add(Dense(20, normalisation=BatchNorm1d(), activation=ReLU()))
model.add(Dense(20, normalisation=BatchNorm1d(), activation=ReLU()))
model.add(Dense(3, activation=SoftMax()))
model.compile(optimiser=None, loss=cce(), metrics=["loss", "val_loss", "val_accuracy", "accuracy"])
model.summary()

_, ax = plt.subplots()
scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
_ = ax.legend(
    scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
)
plt.show()

errors = model.fit(x_train, y_train, val_data=(x_val, y_val), epochs=100, batch_size=32)
test_predictions = model.predict(x_test)
print(f"Test accuracy: {accuracy(test_predictions, y_test)}")

plt.plot(errors["loss"], label="loss")
plt.plot(errors["val_loss"], label="val_loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Categorical cross entropy")
plt.show()

plt.plot(errors["accuracy"], label="accuracy")
plt.plot(errors["val_accuracy"], label="val_accuracy")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()
