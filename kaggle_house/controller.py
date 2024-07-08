from datas import train_features, test_features, train_labels
import torch
from d2l import torch as d2l
import numpy as np
import pandas as pd
from models import KaggleHouseModel

loss = torch.nn.MSELoss()

def log_rmse(net, features, labels):
  # 为了在取对数时进一步稳定该值，将小于1的值设置为1
  clipped_preds = torch.clamp(net(features), 1, float('inf'))
  rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
  # print(rmse)
  return rmse.item()

def train(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay, batch_size):
  train_ls, test_ls = [], []
  train_iter = d2l.load_array((train_features, train_labels), batch_size)
  # 这里使用的是Adam优化算法
  optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
  for epoch in range(num_epochs):
    for X, y in train_iter:
      optimizer.zero_grad()
      l = loss(net(X), y)
      l.backward()
      optimizer.step()
    train_ls.append(log_rmse(net, train_features, train_labels))
    if test_labels is not None:
      rmse = log_rmse(net, test_features, test_labels)
      print(rmse)
      test_ls.append(rmse)
    # print(f'第{epoch+1}步，训练log rmse{float(train_ls[-1]):f}，'
    #       f'验证log rmse{float(valid_ls[-1]):f}，')
  return train_ls, test_ls

if __name__ == "__main__":
    print(train_features.shape) # torch.Size([1460, 330])
    print(test_features.shape) # torch.Size([1459, 330])
    print(train_labels.shape) # torch.Size([1460, 1])

    in_features = train_features.shape[1]
    net = KaggleHouseModel(in_features)
    loss = torch.nn.MSELoss()

    num_epochs, lr, weight_decay, batch_size = 100, 5, 0, 64

    train_ls, _ = train(net, train_features, train_labels, None, None, num_epochs, lr, weight_decay, batch_size)
    # d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch', ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    print(f'训练log rmse:{float(train_ls[-1]):f}')
    # 将网络应用于测试集。
    # preds = net(test_features).detach().numpy()
    # 将其重新格式化以导出到Kaggle
    # test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    # submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    # submission.to_csv('submission.csv', index=False)
