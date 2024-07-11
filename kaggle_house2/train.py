from datas import load_datas
import torch
from d2l import torch as d2l
import numpy as np
import pandas as pd
from models import KaggleHouseModel
import argparse

loss = torch.nn.MSELoss()

def log_rmse(net, features, labels):
  # 为了在取对数时进一步稳定该值，将小于1的值设置为1
  clipped_preds = torch.clamp(net(features), 1, float('inf'))
  rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
  # print(rmse)
  return rmse.item()

def train(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay, batch_size, device):
  train_ls, test_ls = [], []
  train_iter = d2l.load_array((train_features, train_labels), batch_size)
  # 这里使用的是Adam优化算法
  optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
  for epoch in range(num_epochs):
    for X, y in train_iter:
      optimizer.zero_grad()
      # X, y = X.to(device), y.to(device)
      l = loss(net(X), y)
      # print(l.item())
      l.backward()
      optimizer.step()
    train_log_rmse = log_rmse(net, train_features, train_labels)
    train_ls.append(train_log_rmse)
    if test_labels is not None:
      test_log_rmse = log_rmse(net, test_features, test_labels)
      print(f'epoch: {epoch + 1}, 训练log_rmse: {float(train_log_rmse):f},  测试log_rmse: {float(test_log_rmse):f}')
      test_ls.append(test_log_rmse)
  return train_ls, test_ls

def k_fold(k, net, X_train, y_train, num_epochs, learning_rate, weigit_decay, batch_size, device):
  train_l_sum, valid_l_sum = 0, 0
  for i in range(k):
    data = get_k_fold_data(k, i, X_train, y_train)
    train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weigit_decay, batch_size, device)
    train_l_sum += train_ls[-1]
    valid_l_sum += valid_ls[-1]
    # if i == 0:
    #   d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls], xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs], legend=['train', 'valid'], yscale='log')
    print(f'折{i+1}，训练log rmse{float(train_ls[-1]):f}，'
          f'验证log rmse{float(valid_ls[-1]):f}，')
  return train_l_sum / k, valid_l_sum / k

def get_k_fold_data(k, i, X, y):
  assert k > 1
  fold_size = X.shape[0] // k
  X_train, y_train = None, None
  for j in range(k):
    idx = slice(j * fold_size, (j + 1) * fold_size)
    X_part, y_part = X[idx, :], y[idx]
    if j == i:
      X_valid, y_valid = X_part, y_part
    elif X_train is None:
      X_train, y_train = X_part, y_part
    else:
      X_train = torch.cat([X_train, X_part], 0)
      y_train = torch.cat([y_train, y_part], 0)
  return X_train, y_train, X_valid, y_valid

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, std=0.01)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练kaggle房价预测')
    parser.add_argument('-k', '--kz', type=bool, help="是否使用k折交叉训练")
    args = parser.parse_args()
    print(args)
    # print(train_features.shape) # torch.Size([1460, 330])
    # device = torch.device('mps')
    device = torch.device('cpu')
    train_features, train_labels, test_features = load_datas(device)

    in_features = train_features.shape[1]
    net = KaggleHouseModel(in_features)
    net.apply(init_weights)
    net.to(device)

    # loss = torch.nn.MSELoss()

    num_epochs, lr, weight_decay, batch_size = 10, 5, 0, 64
    
    if args.kz:
      k = 5
      timer = d2l.Timer()
      train_l, valid_l = k_fold(k, net, train_features, train_labels, num_epochs, lr, weight_decay, batch_size, device)
      print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, ', f'平均验证log rmse: {float(valid_l):f}')
      print(f'{timer.stop():.5f} sec')
    else:
      test_features = train_features[1000:]
      train_features = train_features[:1000]
      test_labels = train_labels[1000:]
      train_labels = train_labels[:1000]
      train(net, train_features, train_labels, test_features, test_labels, num_epochs, lr, weight_decay, batch_size, device)

    # print(f'{timer.stop():.5f} sec')
    # d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch', ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    # print(f'训练log rmse:{float(train_ls[-1]):f}')
    # 将网络应用于测试集。
    # preds = net(test_features).detach().numpy()
    # 将其重新格式化以导出到Kaggle
    # test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    # submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    # submission.to_csv('submission.csv', index=False)
