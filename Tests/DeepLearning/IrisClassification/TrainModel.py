from src.DLL.DeepLearning.Model import Model, save_model
from src.DLL.DeepLearning.Layers import Dense, Identity, Add, LayerList
from src.DLL.DeepLearning.Layers.Regularisation import BatchNorm, GroupNorm, InstanceNorm, LayerNorm, Dropout
from src.DLL.DeepLearning.Layers.Activations import ReLU, SoftMax
from src.DLL.DeepLearning.Losses import CCE
from src.DLL.DeepLearning.Optimisers import SGD
from src.DLL.DeepLearning.Initialisers import Xavier_Normal, Xavier_Uniform, Kaiming_Normal, Kaiming_Uniform
from src.DLL.DeepLearning.Callbacks import EarlyStopping, BackUp, ReduceLROnPlateau
from src.DLL.Data.Preprocessing import data_split, OneHotEncoder, MinMaxScaler
from src.DLL.Data.Metrics import accuracy

import torch
import matplotlib.pyplot as plt
from sklearn import datasets


iris = datasets.load_iris()

encoder = OneHotEncoder()
scaler = MinMaxScaler()
x = torch.tensor(iris.data, dtype=torch.float32)
x = scaler.fit_transform(x)
y = encoder.fit_encode(torch.tensor(iris.target, dtype=torch.float32))
x_train, y_train, x_val, y_val, x_test, y_test = data_split(x, y, train_split=0.6, validation_split=0.2)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
x_train = x_train.to(device=device)
y_train = y_train.to(device=device)
x_val = x_val.to(device=device)
y_val = y_val.to(device=device)
x_test = x_test.to(device=device)
y_test = y_test.to(device=device)

# can get better results with only batch normalisation
model = Model(4, data_type=torch.float32, device=device)
layers = LayerList(
    Dense(20, initialiser=Kaiming_Normal(), normalisation=BatchNorm(), activation=ReLU()),
    Add(Dense(20, initialiser=Kaiming_Normal(), normalisation=GroupNorm(num_groups=10), activation=ReLU()), Identity()),
    Add(Dense(20, initialiser=Kaiming_Normal(), normalisation=LayerNorm(), activation=ReLU()), Identity()),
    Dense(3, initialiser=Xavier_Uniform(), activation=SoftMax())
)
model.add(layers)
callbacks = (
    EarlyStopping(monitor="val_accuracy", mode="max", patience=30, restore_best_model=False, warmup_length=100, verbose=True),
    BackUp(filepath="./Tests/DeepLearning/IrisClassification/model.pkl", frequency=100, verbose=True),
    ReduceLROnPlateau(patience=10, factor=0.9)
)
model.compile(optimiser=SGD(), loss=CCE(), metrics=["loss", "val_loss", "val_accuracy", "accuracy"], callbacks=callbacks)
print(model)  # model.summary()

_, ax = plt.subplots()
scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
_ = ax.legend(
    scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
)
plt.show()

errors = model.fit(x_train, y_train, val_data=(x_val, y_val), epochs=500, batch_size=32, verbose=True)
save_model(model, "./Tests/DeepLearning/IrisClassification/model.pkl")
test_predictions = model.predict(x_test)
print(f"Test accuracy: {accuracy(test_predictions, y_test)}")

plt.semilogy(errors["loss"], label="loss")
plt.semilogy(errors["val_loss"], label="val_loss")
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
