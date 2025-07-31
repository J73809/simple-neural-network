import numpy as np

X = np.array([
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 0, 1, 1],
    [1, 1, 0, 0]
])

Y = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 1]
])

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

np.random.seed(42)

w1 = np.random.randn(4, 7)
b1 = np.zeros((1, 7))

w2 = np.random.randn(7, 7)
b2 = np.zeros((1, 7))

w3 = np.random.randn(7, 3)
b3 = np.zeros((1, 3))

lr = 0.08
epochs = 20000

for epoch in range(epochs):
    z1 = X @ w1 + b1
    a1 = relu(z1)

    z2 = a1 @ w2 + b2
    a2 = relu(z2)

    z3 = a2 @ w3 + b3
    a3 = sigmoid(z3)

    loss = np.mean((Y - a3) ** 2)

    d_a3 = (a3 - Y) * sigmoid_derivative(a3)
    d_w3 = a2.T @ d_a3
    d_b3 = np.sum(d_a3, axis=0, keepdims=True)

    d_a2 = (d_a3 @ w3.T) * relu_derivative(a2)
    d_w2 = a1.T @ d_a2
    d_b2 = np.sum(d_a2, axis=0, keepdims=True)

    d_a1 = (d_a2 @ w2.T) * relu_derivative(a1)
    d_w1 = X.T @ d_a1
    d_b1 = np.sum(d_a1, axis=0, keepdims=True)

    w3 -= lr * d_w3
    b3 -= lr * d_b3

    w2 -= lr * d_w2
    b2 -= lr * d_b2

    w1 -= lr * d_w1
    b1 -= lr * d_b1

    if epoch % 1000 == 0 or epoch == epochs - 1:
        predictions = np.round(a3)
        total_outputs = predictions.size
        correct_outputs = np.sum(predictions == Y)
        accuracy = (correct_outputs / total_outputs) * 100
        print(f"Epoch {epoch} - Loss: {loss:.5f} - Accuracy: {accuracy:.2f}%")
