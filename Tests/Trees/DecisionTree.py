"""
Decision tree and random forest classifiers
=============================================

This script evaluates the performance of decision tree and random forest classifiers 
on the Breast Cancer dataset using both DLL (`DLL.MachineLearning.SupervisedLearning.Trees`) 
and scikit-learn.
"""

import torch
from sklearn import datasets
from sklearn import tree
from sklearn import ensemble
import time

from DLL.Data.Metrics import accuracy
from DLL.Data.Preprocessing import data_split
from DLL.MachineLearning.SupervisedLearning.Trees import DecisionTree, RandomForestClassifier


breast_cancer = datasets.load_breast_cancer()

x = torch.tensor(breast_cancer.data, dtype=torch.float32)
y = torch.tensor(breast_cancer.target, dtype=torch.float32)
x_train, y_train, _, _, x_test, y_test = data_split(x, y, train_split=0.8, validation_split=0.0)

start = time.perf_counter()
model = DecisionTree(max_depth=1, ccp_alpha=0.0)
model.fit(x_train, y_train)
predictions = model.predict(x_test)
probas = model.predict_proba(x_test)
print(f"DLL decision tree accuracy: {accuracy(predictions, y_test)}")
print(f"DLL decision tree execution time: {time.perf_counter() - start}")

start = time.perf_counter()
model = tree.DecisionTreeClassifier(max_depth=1, criterion='entropy')
model.fit(x_train.numpy(), y_train.numpy())
predictions = model.predict(x_test)
print(f"SKlearn decision tree accuracy: {accuracy(torch.tensor(predictions), y_test)}")
print(f"SKlearn decision tree execution time: {time.perf_counter() - start}")

start = time.perf_counter()
model = RandomForestClassifier(n_trees=10, max_depth=1)
model.fit(x_train, y_train)
predictions = model.predict(x_test)
probas = model.predict_proba(x_test)
print(f"DLL random forest accuracy: {accuracy(predictions, y_test)}")
print(f"DLL random forest execution time: {time.perf_counter() - start}")

start = time.perf_counter()
model = ensemble.RandomForestClassifier(n_estimators=10, max_depth=1, criterion='entropy')
model.fit(x_train.numpy(), y_train.numpy())
predictions = model.predict(x_test)
print(f"SKlearn random forest accuracy: {accuracy(torch.tensor(predictions), y_test)}")
print(f"SKlearn random forest execution time: {time.perf_counter() - start}")
