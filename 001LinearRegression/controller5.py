import torch

# 产生训练用的数据
x_origin = torch.linspace(100, 300, 200)
# 将变量X归一化，否则梯度下降法很容易不稳定
x = (x_origin - torch.mean(x_origin)) / torch.std(x_origin)
epsilon = torch.randn(x.shape)
y = 10 * x + 5 + epsilon

class Linear(torch.nn.Module):

  def __init__(self):
    '''
    定义线性回归模型的参数：a，b
    '''
    super().__init__()
    self.a = torch.nn.Parameter(torch.zeros(()))
    self.b = torch.nn.Parameter(torch.zeros(()))

  def forward(self, x):
    '''
    根据当前的参数估计值，得到模型的预测结果
    参数
    ----
    x：torch.tensor，变量x
    返回
    ----
    y_pred：torch.tensor，模型预测值
    '''
    return self.a * x + self.b
  
  def string(self):
    '''
    输出当前模型的结果
    '''
    return f'y = {self.a.item():.2f} * x + {self.b.item():.2f}'

learning_rate = 0.1

'''
# 定义模型
model = Linear()
# 确定最优化算法
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for t in range(20):
  # 根据当前的参数估计值，得到模型的预测结果
  # 也就是调用forward函数
  y_pred = model(x)
  # 计算损失函数
  loss = (y - y_pred).pow(2).mean()
  # 将上一次的梯度清零
  optimizer.zero_grad()
  # 计算损失函数的梯度
  loss.backward()
  # 迭代更新模型参数的估计值
  optimizer.step()
  print(f'Step {t + 1}, Loss: {loss: .2f}; Result: {model.string()}')
''' 


# '''
# 利用代码实现PyTorch封装的梯度下降法
model = Linear()
for t in range(20):
  # 根据当前的参数估计值，得到模型的预测结果
  # 也就是调用forward函数
  y_pred = model(x)
  # 计算损失函数
  loss = (y - y_pred).pow(2).mean()
  # 计算损失函数的梯度
  loss.backward()
  with torch.no_grad():
    for param in model.parameters():
      # 迭代更新模型参数的估计值，等同于optimizer.step()
      param -= learning_rate * param.grad
      # 将梯度清零，等同于optimizer.zero_grad()
      param.grad = None
  print(f'Step {t + 1}, Loss: {loss: .2f}; Result: {model.string()}')
# '''

