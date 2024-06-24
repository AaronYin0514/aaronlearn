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

for t in range(20):
    ix = (t * batch_size) % len(x)
    xx = x[ix: ix + batch_size]
    yy = y[ix: ix + batch_size]
    loss = mse([model.error(_x, _y) for _x, _y in zip(xx, yy)])
    loss.backward()
    model.a -= learning_rate * model.a.grad
    model.b -= learning_rate * model.b.grad
    model.a.grad = 0.0
    model.b.grad = 0.0
    print(model.string())

# 计算图膨胀
model = Linear()
# 定义两组数据
x1 = Scalar(1.5, label='x1', requires_grad=False)
y1 = Scalar(1.0, label='y1', requires_grad=False)
x2 = Scalar(2.0, label='x2', requires_grad=False)
y2 = Scalar(4.0, label='y2', requires_grad=False)
loss = mse([model.error(x1, y1), model.error(x2, y2)])
loss.backward()
graph = draw_graph(loss, 'backward')
graph.render('test.svg', view=True)