from torch import nn
class MnistNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1=nn.Linear(28*28,128)
        self.hidden2=nn.Linear(128,64)
        self.out=nn.Linear(128,10)

    def forward(self,x):
        x=x.view(-1,28*28)
        x=self.hidden1(x)
        return x
