import torch
import matplotlib.pyplot as plt

# from src.DLL.MachineLearning.SupervisedLearning.LinearModels import LogisticRegression
from sklearn.linear_model import LogisticRegression
from src.DLL.Data.Metrics import accuracy
from src.DLL.Data.Preprocessing import data_split

def lwlr_data1(n):
    n1 = int(n / 3)
    a = torch.empty(n1).normal_(mean=-1.0, std=0.5)
    b = torch.empty(n1).normal_(mean=1.0, std=0.5)
    c = torch.empty(n - 2 * n1).normal_(mean=3.0, std=0.5)
    x = torch.hstack([a, b, c]).reshape(-1, 1)
    y = torch.hstack([torch.zeros(n1), torch.ones(n1), torch.zeros(n - n1 * 2)])
    return x, y

x, y = lwlr_data1(n=300)
x_train, y_train, x_test, y_test, _, _ = data_split(x[:200], y[:200])

plt.figure(figsize=(6, 3))
plt.scatter(x_train, y_train, s=5, c='orange', alpha=0.5, label='train')
plt.scatter(x_test, y_test, marker='+', s=30, c='blue', alpha=0.5, label='test')
plt.legend()
plt.show()


def get_weight(xx, tx, tau):
    distance = torch.sum(torch.square(xx - tx), axis=1)
    w = torch.exp(-distance / (2 * tau * tau))
    return w

# y_prob = []
# for test_point in x_test:
#     weight = get_weight(x_train, test_point, 0.6)
#     model = LogisticRegression()
#     model.fit(x_train, y_train, sample_weight=weight)
#     # model.fit(x_train, y_train, sample_weight=weight, epochs=2000)
#     # y_prob.append(model.predict_proba(test_point.unsqueeze(0)))
#     y_prob.append(torch.from_numpy(model.predict_proba(test_point.unsqueeze(0))[:, 1]))
# y_prob = torch.stack(y_prob).reshape(-1,)
model = LogisticRegression()
# model.fit(x_train, y_train, sample_weight=None, epochs=200)
# y_prob = model.predict_proba(x_test)
model.fit(x_train, y_train, sample_weight=None)
y_prob = model.predict_proba(x_test)[:, 1]

plt.figure(figsize=(6, 3))
plt.scatter(x_train, y_train, s=5, c='orange', label='train')
plt.scatter(x_test, y_test, marker='+', s=30, c='blue', label='test')
print(x_test.shape, y_prob.shape)
plt.scatter(x_test, y_prob, s=5, c='red', label='prediction')
plt.legend()
plt.axhline(y=0.5, ls='--', lw=0.5, c='black')
plt.axvline(x=0, ls='--', lw=0.5, c='black')
plt.axvline(x=2, ls='--', lw=0.5, c='black')
plt.show()

# Measure the accuracy of the test data
y_pred = (y_prob > 0.5).int()
print(f"Accuracy of the test data = {accuracy(y_pred, y_test)}")
