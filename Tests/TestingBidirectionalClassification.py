from src.DLL.DeepLearning.Model import Model
from src.DLL.DeepLearning.Layers.Activations import ReLU, SoftMax
from src.DLL.DeepLearning.Layers import RNN, LSTM, Bidirectional
from src.DLL.DeepLearning.Losses import CCE
from src.DLL.DeepLearning.Optimisers import ADAM
from src.DLL.Data.Preprocessing import data_split, OneHotEncoder, MinMaxScaler
from src.DLL.Data.Metrics import accuracy

import torch
import matplotlib.pyplot as plt
from sklearn import datasets


iris = datasets.load_iris()

encoder = OneHotEncoder()
scaler = MinMaxScaler()
x = torch.tensor(iris.data, dtype=torch.float32)
x = scaler.fit_transform(x).unsqueeze(-1)
y = encoder.fit_encode(torch.tensor(iris.target, dtype=torch.float32))
x_train, y_train, x_val, y_val, x_test, y_test = data_split(x, y, train_split=0.6, validation_split=0.2)
print(x.shape, y.shape)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
x_train = x_train.to(device=device)
y_train = y_train.to(device=device)
x_val = x_val.to(device=device)
y_val = y_val.to(device=device)
x_test = x_test.to(device=device)
y_test = y_test.to(device=device)

model = Model((4, 1), data_type=torch.float32, device=device)
model.add(Bidirectional(LSTM((4, 20), 10, return_last=False, activation=ReLU())))
model.add(RNN((3,), 10, return_last=True, activation=SoftMax()))
model.compile(optimiser=ADAM(), loss=CCE(), metrics=["loss", "val_loss", "val_accuracy", "accuracy"])
model.summary()

_, ax = plt.subplots()
scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
_ = ax.legend(
    scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
)

errors = model.fit(x_train, y_train, val_data=(x_val, y_val), epochs=200, batch_size=32, verbose=True)
test_predictions = model.predict(x_test)
print(f"Test accuracy: {accuracy(test_predictions, y_test)}")

plt.figure()
plt.plot(errors["loss"], label="loss")
plt.plot(errors["val_loss"], label="val_loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Categorical cross entropy")

plt.figure()
plt.plot(errors["accuracy"], label="accuracy")
plt.plot(errors["val_accuracy"], label="val_accuracy")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()
