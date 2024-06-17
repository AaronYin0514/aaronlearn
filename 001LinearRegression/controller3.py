import random
import torch
# from d2l import torch as d2l
from matplotlib import pyplot as plt

def synthetic_data(w, b, num_examples):
  """生成y=Xw+b+噪声"""
  X = torch.normal(0, 1, (num_examples, len(w)))
  y = torch.matmul(X, w) + b
  y += torch.normal(0, 0.01, y.shape)
  return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# plt.title("Image Title")
# plt.scatter(features[:, (1)], labels, label='Image1', c='red')
# plt.show()

def data_iter(batch_size, features, labels):
  num_examples = len(features)
  indices = list(range(num_examples))
  # 这些样本是随机读取的，没有特定的顺序
  random.shuffle(indices)