'''
计算图
'''
from graphviz import Digraph

# 定义Scalar类，用于展示计算图和Autograd算法的实现细节
class Scalar:
  def __init__(self, value, prevs=[], op=None, label='', requires_grad=True) -> None:
    # 节点的值
    self.value = value
    # 节点的标识（label）和对应的运算（op），用于作图
    self.label = label
    self.op = op
    # 节点的前节点
    self.prevs = prevs
    # 是否需要计算改节点偏导数
    self.requires_grad = requires_grad
    # 存储该节点偏导数
    self.grad = 0.0
    # 如果该节点的prevs非空，则存储所有的
    self.grad_wrt = {}

  def backward(self):
    '''
    从当前节点出发，求解以当前节点为顶点的计算图中每个节点的偏导数，如∂self/∂node
    '''
    def _topological_order():
      '''
      利用深度优先算法，返回计算图的拓扑排序
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
      从node节点出发，向后传播
      '''
      # 得到当前节点在计算图中的梯度。由于一个节点可以在多个计算图中出现，
      # 因此使用cg_grad记录当前计算图的梯度
      dnode = cg_grad[node]
      # 使用node.grad记录节点的累积梯度
      node.grad += dnode
      for prev in node.prevs:
        # 由于node节点的偏导数已经计算完成，因此可以向后传播
        # 需要注意的是，向后传播到上游节点是累加关系
        grad_spread = dnode * node.grad_wrt[prev]
        cg_grad[prev] = cg_grad.get(prev, 0.0) + grad_spread
      
    # 当前节点的偏导数等于1，因为∂self/∂self = 1。这是反向传播算法的起点
    cg_grad = {self: 1}
    # 为了计算每个节点的偏导数，需要使用拓扑排序的倒叙来遍历计算图
    ordered = reversed(_topological_order())
    for node in ordered:
      _compute_grad_of_prevs(node)


  def __repr__(self) -> str:
    return f'{self.value} | {self.op} | {self.label}'
  
  def __add__(self, other):
    '''
    定义加法，self + other将触发该函数
    '''
    if not isinstance(other, Scalar):
      other = Scalar(other, requires_grad=False)
    # output = self + other
    output = Scalar(self.value + other.value, [self, other], '+')
    output.requires_grad = self.requires_grad or other.requires_grad
    # 计算偏导数∂output/∂self = 1
    output.grad_wrt[self] = 1
    # 计算偏导数∂output/∂other
    output.grad_wrt[other] = 1
    return output
  
  def __mul__(self, other):
    '''
    定义乘法，self * other将触发该函数
    '''
    if not isinstance(other, Scalar):
      other = Scalar(other, requires_grad=False)
    # output = self * other
    output = Scalar(self.value * other.value, [self, other], '*')
    output.requires_grad = self.requires_grad or other.requires_grad
    # 计算偏导数∂output/∂self = 1
    output.grad_wrt[self] = other.value
    # 计算偏导数∂output/∂other
    output.grad_wrt[other] = self.value
    return output

def _trace(root):
  # 遍历计算图中的所有点和边
  nodes, edges = set(), set()
  def _build(v):
    if v not in nodes:
      nodes.add(v)
      for prev in v.prevs:
        edges.add((prev, v))
        _build(prev)
  _build(root)
  return nodes, edges


# 定义作图函数，用于画出计算图
def draw_graph(root, direction='forward'):
  '''
  图形化展示以root为顶点的计算图
  参数
  root : Scalar, 计算图的顶点
  direction : str, 向前传播（forward）或者反向传播（backward）
  返回
  ---
  re : Digraph, 计算图
  '''
  nodes, edges = _trace(root)
  rankdir = 'BT' if direction == 'forward' else 'TB'
  graph = Digraph(format='svg', graph_attr={'rankdir': rankdir})
  # 画点
  for node in nodes:
    label = node.label if node.op is None else node.op
    node_attr = f'{{ grad={node.grad:.2f} | value={node.value:.2f} | {label}}}'
    uid = str(id(node))
    graph.node(name=uid, label=node_attr, shape='record')
  # 画边
  for edge in edges:
    id1 = str(id(edge[0]))
    id2 = str(id(edge[1]))
    graph.edge(id1, id2)
  return graph

a = Scalar(1.0, label='a')
b = Scalar(2.0, label='b')
c = Scalar(4.0, label='c')
d = a + b
e = a * c
f = d * e
# print(c)

graph = draw_graph(f)
graph.render('test.svg', view=True)
