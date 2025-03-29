import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate synthetic data
X, y = make_classification(n_samples=10_000, n_features=20, n_informative=2, n_redundant=10, random_state=42)

# Split into training and testing sets
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.9, random_state=42)

strategy = "quantile"

# Plot the calibration curves for each model
plt.figure(figsize=(18, 6))

# Subplot 1: Standard models
plt.subplot(1, 3, 1)
for model, label in [
    (LogisticRegression(max_iter=500, solver='lbfgs'), "Logistic Regression"),
    (GaussianNB(), "Gaussian Naive Bayes"),
    (SVC(kernel="linear", probability=True), "Support Vector Classification")
]:
    model.fit(Xtrain, ytrain)
    yprob = model.predict_proba(Xtest)[:, 1]
    accuracy = accuracy_score(ytest, model.predict(Xtest))
    print(f"{label} accuracy: {accuracy}")
    prob_true, prob_pred = calibration_curve(ytest, yprob, n_bins=10, strategy=strategy)
    plt.plot(prob_pred, prob_true, marker="o", label=f"Calibration Curve {label}")

plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.title("Calibration Curve")
plt.legend()
plt.grid(True)

# Subplot 2: Calibrated models with isotonic regression
plt.subplot(1, 3, 2)
for model, label in [
    (CalibratedClassifierCV(LogisticRegression(max_iter=500, solver='lbfgs'), method="isotonic"), "Calibrated Logistic Regression"),
    (CalibratedClassifierCV(GaussianNB(), method="isotonic"), "Calibrated Gaussian Naive Bayes"),
    (CalibratedClassifierCV(SVC(kernel="linear", probability=True), method="isotonic"), "Calibrated SVC")
]:
    model.fit(Xtrain, ytrain)
    yprob = model.predict_proba(Xtest)[:, 1]
    accuracy = accuracy_score(ytest, model.predict(Xtest))
    print(f"{label} accuracy: {accuracy}")
    prob_true, prob_pred = calibration_curve(ytest, yprob, n_bins=10, strategy=strategy)
    plt.plot(prob_pred, prob_true, marker="o", label=f"Calibration Curve {label}")

plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.title("Calibration Curve")
plt.legend()
plt.grid(True)

# Subplot 3: Calibrated models with logistic regression
plt.subplot(1, 3, 3)
for model, label in [
    (CalibratedClassifierCV(LogisticRegression(max_iter=500, solver='lbfgs'), method="sigmoid"), "Calibrated Logistic Regression"),
    (CalibratedClassifierCV(GaussianNB(), method="sigmoid"), "Calibrated Gaussian Naive Bayes"),
    (CalibratedClassifierCV(SVC(kernel="linear", probability=True), method="sigmoid"), "Calibrated SVC")
]:
    model.fit(Xtrain, ytrain)
    yprob = model.predict_proba(Xtest)[:, 1]
    accuracy = accuracy_score(ytest, model.predict(Xtest))
    print(f"{label} accuracy: {accuracy}")
    prob_true, prob_pred = calibration_curve(ytest, yprob, n_bins=10, strategy=strategy)
    plt.plot(prob_pred, prob_true, marker="o", label=f"Calibration Curve {label}")

plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.title("Calibration Curve")
plt.legend()
plt.grid(True)

plt.show()
