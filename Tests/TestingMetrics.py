import torch
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score as f1, confusion_matrix as conf_mat

from src.DLL.Data.Metrics import accuracy, precision, recall, f1_score, confusion_matrix, calculate_metrics


n = 1000
# ==========================CLASSIFICATION==========================
torch_predicted = (torch.rand(size=(n,)) > 0.7)
torch_true = (torch.rand(size=(n,)) > 0.7)
sk_predicted = torch_predicted.numpy()
sk_true = torch_true.numpy()

print(accuracy(torch_predicted, torch_true), accuracy_score(sk_true, sk_predicted))
print(precision(torch_predicted, torch_true), precision_score(sk_true, sk_predicted))
print(recall(torch_predicted, torch_true), recall_score(sk_true, sk_predicted))
print(f1_score(torch_predicted, torch_true), f1(sk_true, sk_predicted))
print(confusion_matrix(torch_predicted, torch_true), conf_mat(sk_true, sk_predicted))

metrics = ["accuracy", "precision", "recall", "f1_score", "bce", "cce"]
print(calculate_metrics(data=(torch_predicted.to(torch.float32), torch_true.to(torch.float32)), metrics=metrics))


# ==========================REGRESSION==========================
torch_predicted = torch.randn(size=(n,))
torch_true = torch.randn(size=(n,))
sk_predicted = torch_predicted.numpy()
sk_true = torch_true.numpy()

metrics = ["rmse", "mae", "mse", "huber"]
print(calculate_metrics(data=(torch_predicted.to(torch.float32), torch_true.to(torch.float32)), metrics=metrics))