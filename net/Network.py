from torch import nn
from torch.nn import functional as F
import torch
import torch.nn.init as init
from torchvision import models
from net.utils_distribute import CrossDb as cdb

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = resconv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = resconv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, pretrained=False, num_classes=7, drop_rate=0):
        super(ResNet18, self).__init__()
        self.drop_rate = drop_rate
        resnet = models.resnet18(pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, num_classes)


    def forward(self, x):
        x = self.features(x)
        if self.drop_rate > 0:
            x =  nn.Dropout(self.drop_rate)(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)

        return out, out

class ResNet18_IEBN(nn.Module):
    def __init__(self, num_class=7,num_workers=4):
        super(ResNet18_IEBN, self).__init__()
        
        resnet = models.resnet18(True)
        
        checkpoint = torch.load('./net/pre-models/resnet18_msceleb.pth')
        resnet.load_state_dict(checkpoint['state_dict'],strict=True)

        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.num_head = num_workers
        #设置属性
        for i in range(num_workers):
            setattr(self,"db_worker%d" %i, cdb())
        self.sig = nn.Sigmoid()
        self.fc = nn.Linear(512, num_class)#线性层
        self.bn = nn.BatchNorm1d(num_class)

    def forward(self, x):
        #print(x.shape)
        #x=x.unsqueeze(2).unsqueeze(3)
        x = self.features(x)
        workers = []
        for i in range(self.num_head):
            workers.append(getattr(self,"db_worker%d" %i)(x))
        
        workers = torch.stack(workers).permute([1,0,2])
        if workers.size(1)>1:
            workers = F.log_softmax(workers,dim=1)
            
        out = self.fc(workers.sum(dim=1))
        out = self.bn(out)
   
        return out, x, workers

def resconv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)