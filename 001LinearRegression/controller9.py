import torch
from computation_graph import Scalar, draw_graph
from linear_model import Linear, mse

x = torch.linspace(100, 300, 200)
x = (x - torch.mean(x)) / torch.std(x)
epsilon = torch.randn(x.shape)
y = 10 * x + 5 + epsilon

model = Linear()

batch_size = 20
learning_rate = 0.1
# 梯度积累次数
gradient_accu_iter = 4
# 小批量数据量
micro_size = int(batch_size / gradient_accu_iter)

for t in range(20 * gradient_accu_iter):
    ix = (t * micro_size) % len(x)
    xx = x[ix: ix + micro_size]
    yy = y[ix: ix + micro_size]
    # print('xx =', xx)
    # print('yy =', yy)
    loss = mse([model.error(_x, _y) for _x, _y in zip(xx, yy)])
    # 调整权重
    loss *= 1 / gradient_accu_iter
    loss.backward()
    if (t + 1) % gradient_accu_iter == 0:
        model.a -= learning_rate * model.a.grad
        model.b -= learning_rate * model.b.grad
        model.a.grad = 0.0
        model.b.grad = 0.0
        print(model.string())