import torch
from sklearn import datasets
from Data.Metrics import accuracy
from Data.Preprocessing import data_split
from MachineLearning.DecisionTree import DecisionTree
from MachineLearning.RandomForest import RandomForestClassifier
from sklearn import tree
from sklearn import ensemble
import time


breast_cancer = datasets.load_breast_cancer()

x = torch.tensor(breast_cancer.data, dtype=torch.float32)
y = torch.tensor(breast_cancer.target, dtype=torch.float32)
x_train, y_train, _, _, x_test, y_test = data_split(x, y, train_split=0.8, validation_split=0.0)

model = DecisionTree(max_depth=10)
model.fit(x_train, y_train)
predictions = model.predict(x_test)
print(accuracy(predictions, y_test))

model = tree.DecisionTreeClassifier(max_depth=10, criterion='entropy')
model.fit(x_train.numpy(), y_train.numpy())
predictions = model.predict(x_test)
print(accuracy(torch.tensor(predictions), y_test))

start = time.perf_counter()
model = RandomForestClassifier(n_trees=10, max_depth=10)
model.fit(x_train, y_train)
predictions = model.predict(x_test)
print(accuracy(predictions, y_test))
print(time.perf_counter() - start)

model = ensemble.RandomForestClassifier(n_estimators=10, max_depth=10, criterion='entropy')
model.fit(x_train.numpy(), y_train.numpy())
predictions = model.predict(x_test)
print(accuracy(torch.tensor(predictions), y_test))
