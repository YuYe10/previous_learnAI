import numpy as np


def leaky_relu(x):
    return np.maximum(0, x) + 0.01 * np.minimum(0, x)  # leaky_ReLU激活函数


def softmax(x):
    e_x = np.exp(x - np.max(x))
    sum_exp = np.sum(e_x)
    return e_x / sum_exp


def grad(x, f):
    h = 1e-4
    grad = np.zeros_like(x)

    # noinspection PyTypeChecker
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 还原值
        it.iternext()

    return grad


def calculate_loss(predicted, actual):
    batch_size = predicted.shape[0]
    loss = -np.sum(actual * np.log(predicted + 1e-7)) / batch_size
    return loss


class TwoLayerNN:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化权重矩阵和偏置项
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # 使用He初始化权重矩阵
        self.params = {
            'W1': np.random.randn(hidden_size, input_size) * np.sqrt(2 / input_size),
            'b1': np.zeros((hidden_size, 1)),
            'W2': np.random.randn(output_size, hidden_size) * np.sqrt(2 / hidden_size),
            'b2': np.zeros((output_size, 1))
        }

    def forward(self, x):
        # 前向传播
        W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2']

        # 隐藏层
        Z1 = np.dot(W1, x) + b1
        A1 = leaky_relu(Z1)  # leaky_ReLU激活函数

        # 输出层
        Z2 = np.dot(W2, A1) + b2
        A2 = Z2

        # return A2  # 回归问题
        return softmax(A2)  # 分类问题

    def backward(self, x, t):
        # 反向传播
        batch_size = x.shape[0]
        A2 = self.forward(x)
        dZ2 = A2 - t
        dW2 = np.dot(dZ2, A2.T) / batch_size
        db2 = np.sum(dZ2, axis=1, keepdims=True) / batch_size
        dZ1 = np.dot(self.params['W2'].T, dZ2) * grad(A2, leaky_relu)
        dW1 = np.dot(dZ1, x.T) / batch_size
        db1 = np.sum(dZ1, axis=1, keepdims=True) / batch_size

        gradients = {
            'dW1': dW1,
            'db1': db1,
            'dW2': dW2,
            'db2': db2
        }

        return gradients

    def update_params(self, gradients, learning_rate):
        # 参数更新
        self.params['W1'] -= learning_rate * gradients['dW1']
        self.params['b1'] -= learning_rate * gradients['db1']
        self.params['W2'] -= learning_rate * gradients['dW2']
        self.params['b2'] -= learning_rate * gradients['db2']

    def accuracy(self, x, t):
        y = self.forward(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
