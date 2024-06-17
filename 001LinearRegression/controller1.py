import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
from matplotlib import font_manager

'''
线性回归模型 sklearn
'''

data = pd.read_csv('./data/simple_example.csv')
train_data = data[:15]
test_data = data[15:]

print(train_data)

# 创建一个线性模型
model = linear_model.LinearRegression()
# 训练模型，估计模型参数
features, labels = ['x'], ['y']
# 训练模型，估计模型参数
model.fit(train_data[features], train_data[labels])

# 均方差，越小越好
error = model.predict(test_data[features]) - test_data[labels]
mse = np.mean(error.values ** 2)
# 决定系数，越接近1越好
score = model.score(test_data[features], test_data[labels])

my_font = font_manager.FontProperties(fname="/System/Library/Fonts/PingFang.ttc")

# 为在Matplotlib中显示中文，设置特殊字体
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams.update({'font.size': 13})
# 创建一个图形框
fig = plt.figure(figsize=(6, 6), dpi=100)
# 在图形框里只画一幅图
ax = fig.add_subplot(111)
ax.set_title('线性回归示例', fontproperties=my_font)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
# 画点图，用蓝色圆点表示原始数据
ax.scatter(data[features], data[labels], color='b', label='真实值: $y = x + \epsilon$')
# 根据截距的正负，打印不同的标签
if model.intercept_ > 0:
    label = f'预测值: $y = {model.coef_.item():.3f}x$ + {model.intercept_.item():.3f}'
else:
    label = f'预测值: $y = {model.coef_.item():.3f}x$ - {abs(model.intercept_.item()):.3f}'
# 画线图，用红色线条表示模型结果
ax.plot(data[features], model.predict(data[features]), color='r', label=label)
# 设置图例的样式
legend = plt.legend(prop=my_font, shadow=True)
legend.get_frame().set_facecolor('#6F93AE')
# 显示均方差和决定系数
ax.text(0.99, 0.01, f'均方差：{mse:.3f}\n决定系数：{score:.3f}',
        style='italic', verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes, color='m', fontsize=16, fontproperties=my_font)
plt.savefig('linear_ml.png', dpi=200)
plt.show()