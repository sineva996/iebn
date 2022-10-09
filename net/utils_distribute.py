from torch import nn
import torch.nn.init as init
from torch import distributed as dist

def get_rank():
    if not dist.is_available():
        return 0

    if not dist.is_initialized():
        return 0

    return dist.get_rank()

def synchronize():
    if not dist.is_available():
        return

    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()

    if world_size == 1:
        return

    dist.barrier()

def get_world_size():
    if not dist.is_available():
        return 1

    if not dist.is_initialized():
        return 1

    return dist.get_world_size()

def reduce_sum(tensor):
    if not dist.is_available():
        return tensor

    if not dist.is_initialized():
        return tensor

    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    return tensor

def gather_grad(params):
    world_size = get_world_size()
    
    if world_size == 1:
        return

    for param in params:
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data.div_(world_size)
  
class ChDb(nn.Module):

    def __init__(self):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.at = nn.Sequential(

            nn.Linear(512, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 512),
            nn.Sigmoid()    
        )

    def forward(self, cd):
        cd = self.gap(cd)
        cd = cd.view(cd.size(0),-1)
        y = self.at(cd)
        out = cd * y

        return out

class SpDb(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.conv_1x3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
        )
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
        )
        self.conv_1x3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(1,3),padding=(0,1)),
            nn.BatchNorm2d(512),
        )
        self.conv_3x1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3,1),padding=(1,0)),
            nn.BatchNorm2d(512),
        )
        self.relu = nn.ReLU()


    def forward(self, sd):
        y = self.conv1x1(sd)
        y = self.relu(self.conv_3x3(y) + self.conv_1x3(y) + self.conv_3x1(y))
        y = y.sum(dim=1,keepdim=True) 
        out = sd * y
        
        return out

class CrossDb(nn.Module):

    def __init__(self):
        super().__init__()
        self.cd = ChDb()
        self.sd = SpDb()
        self.init_weights()


    def init_weights(self):
        for md in self.modules():
            if isinstance(md, nn.Conv2d):
                init.kaiming_normal_(md.weight, mode='fan_out')
                if md.bias is not None:
                    init.constant_(md.bias, 0)
            elif isinstance(md, nn.BatchNorm2d):
                init.constant_(md.weight, 1)
                init.constant_(md.bias, 0)
            elif isinstance(md, nn.Linear):
                init.normal_(md.weight, std=0.001)
                if md.bias is not None:
                    init.constant_(md.bias, 0)
    
    def forward(self, x):
        sd = self.sd(x)
        cd = self.cd(sd)

        return cd