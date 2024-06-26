# -*- coding: UTF-8 -*-
'''
测试线性回归模型
'''

from graphviz import Digraph
import math
import torch

class Scalar:
    
    def __init__(self, value, prevs=[], op=None, label='', requires_grad=True):
        # 节点的值
        self.value = value
        # 节点的标识（label）和对应的运算（op），用于作图
        self.label = label
        self.op = op
        # 节点的前节点，即当前节点是运算的结果，而前节点是参与运算的量
        self.prevs = prevs
        # 是否需要计算该节点偏导数，即∂loss/∂self（loss表示最后的模型损失）
        self.requires_grad = requires_grad
        # 该节点偏导数，即∂loss/∂self
        self.grad = 0.0
        # 如果该节点的prevs非空，存储所有的∂self/∂prev
        self.grad_wrt = dict()
        # 作图需要，实际上对计算没有作用
        self.back_prop = dict()
        
    def __repr__(self):
        return f'Scalar(value={self.value}, grad={self.grad})'
    
    def __add__(self, other):
        # print('add self =', self)
        # print('add other =', other)
        '''
        定义加法，self + other将触发该函数
        '''
        if not isinstance(other, Scalar):
            other = Scalar(other, requires_grad=False)
        # output = self + other
        output = Scalar(self.value + other.value, [self, other], '+')
        output.requires_grad = self.requires_grad or other.requires_grad
        # 计算偏导数 ∂output/∂self = 1
        output.grad_wrt[self] = 1
        # 计算偏导数 ∂output/∂other = 1
        output.grad_wrt[other] = 1
        return output
    
    def __sub__(self, other):
        '''
        定义减法，self - other将触发该函数
        '''
        if not isinstance(other, Scalar):
            other = Scalar(other, requires_grad=False)
        # output = self - other
        output = Scalar(self.value - other.value, [self, other], '-')
        output.requires_grad = self.requires_grad or other.requires_grad
        # 计算偏导数 ∂output/∂self = 1
        output.grad_wrt[self] = 1
        # 计算偏导数 ∂output/∂other = -1
        output.grad_wrt[other] = -1
        return output
    
    def __mul__(self, other):
        '''
        定义乘法，self * other将触发该函数
        '''
        if not isinstance(other, Scalar):
            value = torch.tensor([other]) if isinstance(other, float) else other 
            other = Scalar(value, requires_grad=False)
        # print('self =', self)
        # print('other =', other)
        # output = self * other
        value = torch.matmul(self.value, other.value)
        output = Scalar(torch.tensor(value), [self, other], '*')
        output.requires_grad = self.requires_grad or other.requires_grad
        # 计算偏导数 ∂output/∂self = other
        output.grad_wrt[self] = other.value
        # 计算偏导数 ∂output/∂other = self
        output.grad_wrt[other] = self.value
        return output
    
    def __pow__(self, other):
        '''
        定义乘方，self**other将触发该函数
        '''
        assert isinstance(other, (int, float))
        # output = self ** other
        output = Scalar(self.value ** other, [self], f'^{other}')
        output.requires_grad = self.requires_grad
        # 计算偏导数 ∂output/∂self = other * self**(other-1)
        output.grad_wrt[self] = other * self.value**(other - 1)
        return output
    
    def sigmoid(self):
        '''
        定义sigmoid
        '''
        s = 1 / (1 + math.exp(-1 * self.value))
        output = Scalar(s, [self], 'sigmoid')
        output.requires_grad = self.requires_grad
        # 计算偏导数 ∂output/∂self = output * (1 - output)
        output.grad_wrt[self] = s * (1 - s)
        return output
    
    def __rsub__(self, other):
        '''
        定义右减法，other - self将触发该函数
        '''
        if not isinstance(other, Scalar):
            other = Scalar(other, requires_grad=False)
        output = Scalar(other.value - self.value, [self, other], '-')
        output.requires_grad = self.requires_grad or other.requires_grad
        # 计算偏导数 ∂output/∂self = -1
        output.grad_wrt[self] = -1
        # 计算偏导数 ∂output/∂other = 1
        output.grad_wrt[other] = 1
        return output
    
    def __radd__(self, other):
        '''
        定义右加法，other + self将触发该函数
        '''
        return self.__add__(other)
    
    def __rmul__(self, other):
        '''
        定义右乘法，other * self将触发该函数
        '''
        return self * other
    
    def backward(self, fn=None):
        '''
        由当前节点出发，求解以当前节点为顶点的计算图中每个节点的偏导数，i.e. ∂self/∂node
        参数
        ----
        fn ：画图函数，如果该变量不等于None，则会返回向后传播每一步的计算的记录
        返回
        ----
        re ：向后传播每一步的计算的记录
        '''
        def _topological_order():
            '''
            利用深度优先算法，返回计算图的拓扑排序（topological sorting）
            '''
            def _add_prevs(node):
                if node not in visited:
                    visited.add(node)
                    for prev in node.prevs:
                        _add_prevs(prev)
                    ordered.append(node)
            ordered, visited = [], set()
            _add_prevs(self)
            return ordered

        def _compute_grad_of_prevs(node):
            '''
            由node节点出发，向后传播
            '''
            # 作图需要，实际上对计算没有作用
            node.back_prop = dict()
            # 得到当前节点在计算图中的梯度。由于一个节点可以在多个计算图中出现，
            # 使用cg_grad记录当前计算图的梯度
            dnode = cg_grad[node]
            # 使用node.grad记录节点的累积梯度
            node.grad += dnode
            for prev in node.prevs:
                # 由于node节点的偏导数已经计算完成，可以向后扩散（反向传播）
                # 需要注意的是，向后扩散到上游节点是累加关系
                grad_spread = dnode * node.grad_wrt[prev]
                cg_grad[prev] = cg_grad.get(prev, 0.0) + grad_spread
                node.back_prop[prev] = node.back_prop.get(prev, 0.0) + grad_spread
        
        # 当前节点的偏导数等于1，因为∂self/∂self = 1。这是反向传播算法的起点
        cg_grad = {self: 1}
        # 为了计算每个节点的偏导数，需要使用拓扑排序的倒序来遍历计算图
        ordered = reversed(_topological_order())
        re = []
        for node in ordered:
            _compute_grad_of_prevs(node)
            # 作图需要，实际上对计算没有作用
            if fn is not None:
                re.append(fn(self, 'backward'))
        return re

def mse(errors):
    '''
    计算均方误差
    '''
    n = len(errors)
    wrt = {}
    value = 0.0
    requires_grad = False
    for item in errors:
        value += item.value ** 2 / n
        wrt[item] = 2 / n * item.value
        requires_grad = requires_grad or item.requires_grad
    output = Scalar(value, errors, 'mse')
    output.requires_grad=requires_grad
    output.grad_wrt = wrt
    return output


class Linear:
    
    def __init__(self):
        '''
        定义线性回归模型的参数：a, b
        '''
        self.a = Scalar(torch.tensor([0.0, 0.0]), label='a')
        self.b = Scalar(torch.tensor(0.0), label='b')

    def forward(self, x):
        '''
        根据当前的参数估计值，得到模型的预测结果
        '''
        return self.a * x + self.b
    
    def error(self, x, y):
        '''
        当前数据的模型误差
        '''
        return y - self.forward(x)

    def string(self):
        '''
        输出当前模型的结果
        '''
        return f'y = {self.a.value} * x + {self.b.value}'

def synthetic_data(w, b, num_examples):
  """生成y=Xw+b+噪声"""
  X = torch.normal(0, 1, (num_examples, len(w)))
  y = torch.matmul(X, w) + b
  # print('y.shape=', y.shape)
  y += torch.normal(0, 0.01, y.shape)
  return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
x, y = synthetic_data(true_w, true_b, 1000)

# x = torch.linspace(100, 300, 200)
# x = (x - torch.mean(x)) / torch.std(x)
# epsilon = torch.randn(x.shape)
# y = 10 * x + 5 + epsilon

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

# for t in range(20):
#     ix = (t * batch_size) % len(x)
#     xx = x[ix: ix + batch_size]
#     yy = y[ix: ix + batch_size]
#     loss = mse([model.error(_x, _y) for _x, _y in zip(xx, yy)])
#     loss.backward()
#     model.a -= learning_rate * model.a.grad
#     model.b -= learning_rate * model.b.grad
#     model.a.grad = 0.0
#     model.b.grad = 0.0
#     print(model.string())
        