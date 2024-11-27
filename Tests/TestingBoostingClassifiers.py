import torch
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier as sk_AdaBoostClassifier, GradientBoostingClassifier as sk_GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from src.DLL.MachineLearning.SupervisedLearning.Trees import GradientBoostingClassifier, AdaBoostClassifier, XGBoostingClassifier
from src.DLL.Data.Preprocessing import data_split
from src.DLL.Data.Metrics import accuracy, roc_curve, auc

n_classes = 2
X, y = datasets.make_blobs(n_samples=200, n_features=2, cluster_std=3, centers=n_classes, random_state=3)

x_train, y_train, _, _, x_test, y_test = data_split(torch.from_numpy(X).to(dtype=torch.float32), torch.from_numpy(y), train_split=0.7, validation_split=0.0)

model = GradientBoostingClassifier(n_trees=50, max_depth=1, learning_rate=0.5, loss="log_loss")
history = model.fit(x_train, y_train)
y_pred_proba = model.predict_proba(x_test)
y_pred = model.predict(x_test)
print("gradientboost accuracy: ", accuracy(y_pred, y_test))

model2 = AdaBoostClassifier(n_trees=50, max_depth=1)
errors = model2.fit(x_train, y_train)
y_pred_proba2 = model2.predict_proba(x_test)
y_pred2 = model2.predict(x_test)
print("adaboost accuracy: ", accuracy(y_pred2, y_test))

model3 = sk_AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), learning_rate=0.5)
# model3 = sk_GradientBoostingClassifier()
model3.fit(x_train.numpy(), y_train.numpy())
pred3 = torch.from_numpy(model3.predict(x_test.numpy()))
print("sklearn accuracy: ", accuracy(pred3, y_test))

model4 = XGBoostingClassifier(n_trees=50, learning_rate=0.5, reg_lambda=0.01, max_depth=1, loss="log_loss")
history4 = model4.fit(x_train, y_train)
y_pred_proba4 = model4.predict_proba(x_test)
y_pred4 = model4.predict(x_test)
print("XGBoost accuracy: ", accuracy(y_pred4, y_test))

plt.title("Ada boost errors and alphas")
plt.plot(errors, label="errors")
plt.plot(model2.confidences, label="confidences")
plt.legend()

if n_classes == 2:
    plt.figure()
    plt.plot(history["loss"])
    plt.ylabel("Loss")
    plt.xlabel("Tree")
    plt.title("Gradient boosting classifier loss as a function of fitted trees")
    
    plt.figure()
    plt.plot(history4["loss"])
    plt.ylabel("Loss")
    plt.xlabel("Tree")
    plt.title("XGBoost loss as a function of fitted trees")

    thresholds = torch.linspace(0, 1, 100)
    fpr, tpr = roc_curve(y_pred_proba, y_test, thresholds)
    fpr2, tpr2 = roc_curve(y_pred_proba2, y_test, thresholds)
    fpr4, tpr4 = roc_curve(y_pred_proba4, y_test, thresholds)
    fig, ax = plt.subplots(1, 3)
    ax[0].set_title(f"gradient boosting ROC, auc = {auc(fpr, tpr)}")
    ax[0].plot([0, 1], [0, 1])
    ax[0].step(fpr, tpr)
    ax[0].set_xlabel("False positive rate")
    ax[0].set_ylabel("True positive rate")

    ax[1].set_title(f"ada boost ROC, auc = {auc(fpr2, tpr2)}")
    ax[1].plot([0, 1], [0, 1])
    ax[1].step(fpr2, tpr2)
    ax[1].set_xlabel("False positive rate")
    ax[1].set_ylabel("True positive rate")

    ax[2].set_title(f"XG boost ROC, auc = {auc(fpr4, tpr4)}")
    ax[2].plot([0, 1], [0, 1])
    ax[2].step(fpr4, tpr4)
    ax[2].set_xlabel("False positive rate")
    ax[2].set_ylabel("True positive rate")

fig, ax = plt.subplots(1, 3)
ax[0].scatter(x_test[:, 0], x_test[:, 1], c=y_test, alpha=((model.predict(x_test) != y_test) + 0.2) / 1.2)
ax[0].set_title("Gradient boosting")
ax[1].scatter(x_test[:, 0], x_test[:, 1], c=y_test, alpha=((model2.predict(x_test) != y_test) + 0.2) / 1.2)
ax[1].set_title("Adaptive boosting")
ax[2].scatter(x_test[:, 0], x_test[:, 1], c=y_test, alpha=((model4.predict(x_test) != y_test) + 0.2) / 1.2)
ax[2].set_title("XG boosting")
plt.show()
