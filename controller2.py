# 线性回归 + Python + 梯度下降法

import numpy as np
import matplotlib.pyplot as plt

x = np.array([[1, 5.56], [2, 5.70], [3, 5.91], [4, 6.40],[5, 6.80],
              [6, 7.05], [7, 8.90], [8, 8.70],[9, 9.00], [10, 9.05]])
m, n = np.shape(x)
x_data = np.ones((m, n))
x_data[:, :-1] = x[:, :-1]
y_data = x[:, -1]

print(x_data)
print(y_data)

m, n = np.shape(x_data)
theta = np.ones(n)

def gradientDescent(iter, x, y, w, alpha):
    x_train = x.transpose()
    for i in range(0, iter):
        pre = np.dot(x, w)
        print('pre=', pre)
        loss = (pre - y)
        gradient = np.dot(x_train, loss) / m
        w = w - alpha * gradient
        cost = 1.0 / 2 * m * np.sum(np.square(np.dot(x, np.transpose(w)) - y))
        # print("第{}次梯度下降损失为: {}".format(i,round(cost,2)))
    return w


result = gradientDescent(1, x_data, y_data, theta, 0.01)
y_pre = np.dot(x_data, result)
print("线性回归模型 w: ", result)

plt.rc('font', family='Arial Unicode MS', size=14)
plt.scatter(x[:, 0], x[:, 1], color='b', label='训练数据')
plt.plot(x[:, 0], y_pre, color='r', label='预测数据')
plt.xlabel('x')
plt.ylabel('y')
plt.title('线性回归预测(梯度下降)')
plt.legend()
plt.show()