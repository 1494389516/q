import torch
x=torch.randn(2,3)
b=torch.randn(3,4,requires_grad=True)
print(x.requires_grad)    #是否需要梯度计算
print(b.requires_grad)


