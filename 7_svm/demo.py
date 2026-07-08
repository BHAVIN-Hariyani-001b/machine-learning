import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

cancer = load_breast_cancer()
X = cancer.data
y = np.where(cancer.target == 0, -1, 1)   # ✅ Fix 1: convert to {-1, +1}

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, epochs=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.epochs = epochs

    def fit(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        y_ = np.where(y_train <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.epochs):
            indices = np.random.permutation(n_samples)

            for idx in indices:
                x_i = X_train[idx]
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b)

                if condition >= 1:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - y_[idx] * x_i)
                    self.b += self.lr * y_[idx]

    def predict(self, X_test):
        return np.sign(np.dot(X_test, self.w) + self.b)

svm = SVM(learning_rate=0.001, lambda_param=0.01, epochs=1000)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)              # already {-1, +1}
print(accuracy_score(y_test, y_pred))     # ✅ Fix 2: labels now match