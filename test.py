import torch

# a = torch.rand(2, 1)
# print('a=', a)
# b = torch.tensor([[2., 2.]])
# print('b=', b)

# c = a @ b
# print(c)
# c = a.matmul(b)
# print(c)
# c = a.mm(b)
# print(c)
# c = torch.matmul(a, b)
# print(c)
# c = torch.mm(a, b)
# print(c)

# a = torch.Tensor([[1,2,3],[3,4,3]])
# print(a)
# print(a.shape)

# a = torch.tensor([[3,-5,2,1],[1,1,0,-5],[-1,3,1,3],[2,-4,-1,-3]],dtype=torch.float32)
# print(torch.det(a))
# print(a)
# a = torch.tensor([[3,1,-1,2],[-5,1,3,-4],[2,0,1,-1],[1,-5,3,-3]],dtype=torch.float32)
# print(a)

# a = torch.tensor([[1,2],[3,4]],dtype=torch.float32)
# print(torch.det(a))

# a = torch.tensor([[1,3],[2,4]],dtype=torch.float32)
# print(torch.det(a))

# b = x = torch.randn(2, 3)
# print(b)

t = torch.tensor([
    [1, 1, 1, 1],
    [2, 2, 2, 2],
    [3, 3, 3, 3],
], dtype=torch.float32)
t2 = t.reshape(2, -1)

print(t2)
print(t2.shape)


