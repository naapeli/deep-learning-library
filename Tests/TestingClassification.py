from src.DLL.DeepLearning.Model import Model
from src.DLL.DeepLearning.Layers import Dense
from src.DLL.DeepLearning.Layers.Regularisation import BatchNorm, GroupNorm, InstanceNorm, LayerNorm, Dropout
from src.DLL.DeepLearning.Layers.Activations import ReLU, SoftMax
from src.DLL.DeepLearning.Losses import cce
from src.DLL.DeepLearning.Optimisers import sgd
from src.DLL.Data.Preprocessing import data_split, OneHotEncoder, MinMaxScaler
from src.DLL.Data.Metrics import accuracy

import torch
import matplotlib.pyplot as plt
from sklearn import datasets


iris = datasets.load_iris()

encoder = OneHotEncoder()
scaler = MinMaxScaler()
x = torch.tensor(iris.data, dtype=torch.float32)
scaler.fit(x)
encoder.fit(torch.tensor(iris.target, dtype=torch.float32))
x = scaler.transform(x)
y = encoder.one_hot_encode(torch.tensor(iris.target, dtype=torch.float32))
x_train, y_train, x_val, y_val, x_test, y_test = data_split(x, y, train_split=0.6, validation_split=0.2)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# can get better results with only batch normalisation
model = Model((4,), data_type=torch.float32)
model.add(Dense((20,), normalisation=BatchNorm(), activation=ReLU()))
model.add(Dense((20,), normalisation=GroupNorm(num_groups=10), activation=ReLU()))
model.add(Dense((20,), normalisation=LayerNorm(), activation=ReLU()))
model.add(Dense((3,), activation=SoftMax()))
model.compile(optimiser=sgd(), loss=cce(), metrics=["loss", "val_loss", "val_accuracy", "accuracy"])
model.summary()

_, ax = plt.subplots()
scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
_ = ax.legend(
    scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
)
plt.show()

errors = model.fit(x_train, y_train, val_data=(x_val, y_val), epochs=100, batch_size=32, verbose=True)
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
