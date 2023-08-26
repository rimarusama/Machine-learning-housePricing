import numpy as np
import pandas as pd

def hypothesis(x, theta):
    return np.dot(x, theta)

def error(X, Y, theta):
    m = len(Y)
    y_pred = hypothesis(X, theta)
    squared_error = np.sum((Y - y_pred) ** 2)
    return squared_error / (2 * m)


def train_test_split_custom(X, Y, test_size=0.2, random_state=50):
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_idx = int((1 - test_size) * len(indices))
    train_indices, test_indices = indices[:split_idx], indices[split_idx:]
    X_train, X_test = X[train_indices], X[test_indices]
    Y_train, Y_test = Y[train_indices], Y[test_indices]
    return X_train, X_test, Y_train, Y_test


def gradient(X, Y, theta):
    m = len(Y)
    y_pred = hypothesis(X, theta)
    gradient = np.dot(X.T, (y_pred - Y)) / m
    return gradient

def gradient_update(X, Y, lr=0.3, max_epoch=300):
    m, n = X.shape
    theta = np.zeros((n,))
    error_list = []
    
    for i in range(max_epoch):
        e = error(X, Y, theta)
        error_list.append(e)
        
        grad = gradient(X, Y, theta)
        theta -= lr * grad
    
    return theta, error_list

def r2_score(Y, Y_pred):
    num = np.sum((Y - Y_pred)**2)
    denom = np.sum((Y - np.mean(Y))**2)
    score = 1 - num / denom
    return score * 100


df = pd.read_csv('housing.csv', header=None, delimiter='\s+')
X = df.iloc[:, :13].values
Y = df.iloc[:, 13].values


X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)


X = np.hstack((np.ones((X.shape[0], 1)), X))


X_train, X_test, Y_train, Y_test = train_test_split_custom(X, Y, test_size=0.2)


theta, error_list = gradient_update(X_train, Y_train)


y_pred_train = hypothesis(X_train, theta)
y_pred_test = hypothesis(X_test, theta)


r2_train = r2_score(Y_train, y_pred_train)
r2_test = r2_score(Y_test, y_pred_test)
print("R-squared score on train set:", r2_train)
print("R-squared score on test set:", r2_test)
