import torch
from sklearn import datasets
from MachineLearning.DecisionTree import DecisionTreeClassifier
from Data.Metrics import accuracy
from Data.Preprocessing import data_split


iris = datasets.load_breast_cancer()

x = torch.tensor(iris.data, dtype=torch.float32)
y = torch.tensor(iris.target, dtype=torch.float32)
print(x.shape, y.shape)
x_train, y_train, _, _, x_test, y_test = data_split(x, y, train_split=0.8, validation_split=0.0)

model = DecisionTreeClassifier(max_depth=10)
model.fit(x_train, y_train)

predictions = model.predict(x_test)
print(accuracy(predictions, y_test))
