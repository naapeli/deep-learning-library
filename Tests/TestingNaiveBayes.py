import torch
from sklearn import datasets

from src.DLL.Data.Preprocessing import data_split
from src.DLL.Data.Metrics import accuracy
from src.DLL.MachineLearning.SupervisedLearning.NaiveBayes.GaussianNaiveBayes import GaussianNaiveBayes
from src.DLL.MachineLearning.SupervisedLearning.NaiveBayes.BernoulliNaiveBayes import BernoulliNaiveBayes
from src.DLL.MachineLearning.SupervisedLearning.NaiveBayes.MultinomialNaiveBayes import MultinomialNaiveBayes


iris = datasets.load_iris()
X = torch.tensor(iris.data, dtype=torch.float32)
y = torch.tensor(iris.target, dtype=torch.float32)

X_train, y_train, X_test, y_test, _, _ = data_split(X, y, train_split=0.8, validation_split=0.2)

model = GaussianNaiveBayes()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(accuracy(predictions, y_test))


y_bernoulli = (y != 0).to(torch.int32)
X_bernoulli = (X > torch.mean(X, dim=0)).to(torch.int32)
X_train, y_train, X_test, y_test, _, _ = data_split(X_bernoulli, y_bernoulli, train_split=0.8, validation_split=0.2)

model = BernoulliNaiveBayes()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(accuracy(predictions, y_test))


n_samples = 2000
n_features = 10
X_multinomial = torch.poisson(torch.ones(size=(n_samples, n_features)) * 3)
y_multinomial = torch.bitwise_or(torch.bitwise_and(X_multinomial[:, 0] >= 5, X_multinomial[:, 1] < 3), (X_multinomial[:, 2] >= 3)).to(torch.float32)
X_train, y_train, X_test, y_test, _, _ = data_split(X_multinomial, y_multinomial, train_split=0.8, validation_split=0.2)

model = MultinomialNaiveBayes()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(accuracy(predictions, y_test))