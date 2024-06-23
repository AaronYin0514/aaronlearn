'''
随机梯度下降法
'''
import torch

# 产生训练用的数据
x_origin = torch.linspace(100, 300, 200)
print(x_origin.shape)
# 将变量X归一化，否则梯度下降法很容易不稳定
x = (x_origin - torch.mean(x_origin)) / torch.std(x_origin)
print(len(x))
epsilon = torch.randn(x.shape)
y = 10 * x + 5 + epsilon

# 为了使用PyTorch的高层封装函数，我们通过继承Module类来定义函数
class Linear(torch.nn.Module):
  def __init__(self):
    '''
    定义线性回归模型的参数：a, b
    '''
    super().__init__()
    self.a = torch.nn.Parameter(torch.zeros(()))
    self.b = torch.nn.Parameter(torch.zeros(()))

  def forward(self, x):
    """
    根据当前的参数估计值，得到模型的预测结果
    参数
    ----
    x ：torch.tensor，变量x
    返回
    ----
    y_pred ：torch.tensor，模型预测值
    """
    return self.a * x + self.b
  
  def string(self):
    '''
    输出当前模型的结果
    '''
    return f'y = {self.a.item():.2f} * x + {self.b.item():.2f}'
  
'''
# 定义每批次用到的数据量
batch_size = 20
# 定义模型
model = Linear()
# 确定最优化算法
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for t in range(20):
  # 选取当前批次的数据，用于训练模型
  ix = (t * batch_size) % len(x)
  xx = x[ix: ix + batch_size]
  yy = y[ix: ix + batch_size]
  yy_pred = model(xx)
  # 计算当前批次数据的损失
  loss = (yy - yy_pred).pow(2).mean()
  # 将上一次的梯度清零
  optimizer.zero_grad()
  # 计算损失函数的梯度
  loss.backward()
  # 迭代更新模型参数的估计值
  optimizer.step()
  # 注意！loss记录的是模型在当前批次数据上的损失，该数值的波动较大
  print(f'Step {t + 1}, Loss: {loss: .2f}; Result: {model.string()}')
'''

# 定义损失函数
mse = lambda y, y_pred: (y - y_pred).pow(2).mean()

# 在最优化算法的运行过程中，记录模型在批量数据和整体数据上的损失
# 定义每批次用到的数据量
batch_size = 20
# 定义模型
model = Linear()
# 确定最优化算法
learning_rate = 0.1


