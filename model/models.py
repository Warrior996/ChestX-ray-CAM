import torch
from torch import nn
import torchvision
# from model.cbam_resnet_nei import *
from model.rsnet import *
from model.se_resnet import se_resnet34, se_resnet101
from model.se_densenet import se_densenet121
from model.densenet import densenet121, densenet161
from model.cbam_densenet import cbam_densenet121, cbam_densenet161
from model.sa_resnet import sa_resnet50
from model.cbam_resnet import cbam_resnet50, cbam_resnet152
from model.resnet_cbam_dca import dca_cbam_resnet50
from model.pyconv import pyconvresnet50
from model.py_resnet import pyconvresnet50
from model.pycbam_resnet import pycbamresnet50
import torchvision

class DenseNet121(nn.Module):
    def __init__(self, N_LABELS, isTrained):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=isTrained)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())
    def forward(self, x):
        x = self.densenet121(x)
        return x

class Se_DenseNet121(nn.Module):
    def __init__(self, N_LABELS, isTrained):
        super(Se_DenseNet121, self).__init__()
        self.se_denseNet121 = se_densenet121(pretrained=isTrained)
        num_ftrs = self.se_denseNet121.classifier.in_features
        self.se_denseNet121.classifier = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())
    def forward(self, x):
        x = self.se_denseNet121(x)
        return x

class CBAM_DenseNet121(nn.Module):
    def __init__(self, N_LABELS, isTrained):
        super(CBAM_DenseNet121, self).__init__()
        self.CBAM_denseNet121 = cbam_densenet121(pretrained=isTrained)
        num_ftrs = self.CBAM_denseNet121.classifier.in_features
        self.CBAM_denseNet121.classifier = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())
    def forward(self, x):
        x = self.CBAM_denseNet121(x)
        return x

class DCA_CBAM_DenseNet121(nn.Module):
    def __init__(self, N_LABELS, isTrained):
        super(DCA_CBAM_DenseNet121, self).__init__()
        self.DCA_CBAM_DenseNet121 = dca_cbam_densenet121(pretrained=isTrained)
        num_ftrs = self.DCA_CBAM_DenseNet121.classifier.in_features
        self.DCA_CBAM_DenseNet121.classifier = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())
    def forward(self, x):
        x = self.DCA_CBAM_DenseNet121(x)
        return x

class SA_DenseNet121(nn.Module):
    def __init__(self, N_LABELS, isTrained):
        super(SA_DenseNet121, self).__init__()
        self.SA_denseNet121 = sa_densenet121(pretrained=isTrained)
        num_ftrs = self.SA_denseNet121.classifier.in_features
        self.SA_denseNet121.classifier = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())
    def forward(self, x):
        x = self.SA_denseNet121(x)
        return x

class DenseNet161(nn.Module):
    def __init__(self, N_LABELS, isTrained):
        super(DenseNet161, self).__init__()
        self.densenet161 = densenet161(pretrained=isTrained)
        num_ftrs = self.densenet161.classifier.in_features
        self.densenet161.classifier = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())
    def forward(self, x):
        x = self.densenet161(x)
        return x

class ResNet50(nn.Module):
    def __init__(self, N_LABELS, isTrained):
        super(ResNet50, self).__init__()
        self.resnet50 = torchvision.models.resnet50(pretrained=isTrained)
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())
    def forward(self, x):
        x = self.resnet50(x)
        return x

class SA_ResNet50(nn.Module):
    def __init__(self, N_LABELS, isTrained):
        super(SA_ResNet50, self).__init__()
        self.sa_resnet50 = sa_resnet50(pretrained=isTrained)
        num_ftrs = self.sa_resnet50.fc.in_features
        self.sa_resnet50.fc = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())
    def forward(self, x):
        x = self.sa_resnet50(x)
        return x

class PYSA_ResNet50(nn.Module):
    def __init__(self, N_LABELS, isTrained):
        super(PYSA_ResNet50, self).__init__()
        self.pysa_resnet50 = pyconvresnet50(pretrained=isTrained)
        num_ftrs = self.pysa_resnet50.fc.in_features
        self.pysa_resnet50.fc = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())
    def forward(self, x):
        x = self.pysa_resnet50(x)
        return x

class PY_ResNet50(nn.Module):
    def __init__(self, N_LABELS, isTrained):
        super(PY_ResNet50, self).__init__()
        self.py_resnet50 = pyconvresnet50(pretrained=isTrained)
        num_ftrs = self.py_resnet50.fc.in_features
        self.py_resnet50.fc = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())
    def forward(self, x):
        x = self.py_resnet50(x)
        return x

class PYCBAM_ResNet50(nn.Module):
    def __init__(self, N_LABELS, isTrained):
        super(PYCBAM_ResNet50, self).__init__()
        self.pycbam_resnet50 = pycbamresnet50(pretrained=isTrained)
        num_ftrs = self.pycbam_resnet50.fc.in_features
        self.pycbam_resnet50.fc = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())
    def forward(self, x):
        x = self.pycbam_resnet50(x)
        return x

class CBAM_ResNet50(nn.Module):
    def __init__(self, N_LABELS, isTrained):
        super(CBAM_ResNet50, self).__init__()
        self.cbam_resnet50 = cbam_resnet50(pretrained=isTrained)
        num_ftrs = self.cbam_resnet50.fc.in_features
        self.cbam_resnet50.fc = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())
    def forward(self, x):
        x = self.cbam_resnet50(x)
        return x

class DCA_CBAM_ResNet50(nn.Module):
    def __init__(self, N_LABELS, isTrained):
        super(DCA_CBAM_ResNet50, self).__init__()
        self.dca_cbam_resnet50 = dca_cbam_resnet50(pretrained=isTrained)
        num_ftrs = self.dca_cbam_resnet50.fc.in_features
        self.dca_cbam_resnet50.fc = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())
    def forward(self, x):
        x = self.dca_cbam_resnet50(x)
        return x

class ResNet34(nn.Module):
    def __init__(self, N_LABELS, isTrained):
        super(ResNet34, self).__init__()
        self.resnet34 = torchvision.models.resnet34(pretrained=isTrained)
        num_ftrs = self.resnet34.fc.in_features
        self.resnet34.fc = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())
    def forward(self, x):
        x = self.resnet34(x)
        return x

class ResNet101(nn.Module):
    def __init__(self, N_LABELS, isTrained):
        super(ResNet101, self).__init__()
        self.resnet101 = torchvision.models.resnet101(pretrained=isTrained)
        num_ftrs = self.resnet101.fc.in_features
        self.resnet101.fc = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())
    def forward(self, x):
        x = self.resnet101(x)
        return x

class ResNet152(nn.Module):
    def __init__(self, N_LABELS, isTrained):
        super(ResNet152, self).__init__()
        self.resnet152 = torchvision.models.resnet152(pretrained=isTrained)
        num_ftrs = self.resnet152.fc.in_features
        self.resnet152.fc = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())
    def forward(self, x):
        x = self.resnet152(x)
        return x

class CBAM_ResNet152(nn.Module):
    def __init__(self, N_LABELS, isTrained):
        super(CBAM_ResNet152, self).__init__()
        self.cbam_resnet152 = cbam_resnet152(pretrained=isTrained)
        num_ftrs = self.cbam_resnet152.fc.in_features
        self.cbam_resnet152.fc = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())
    def forward(self, x):
        x = self.cbam_resnet152(x)
        return x

class Se_ResNet34(nn.Module):
    def __init__(self, N_LABELS):
        super(Se_ResNet34, self).__init__()
        self.se_resnet34 = se_resnet34()
        num_ftrs = self.se_resnet34.fc.in_features
        self.se_resnet34.fc = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())
        print(self.se_resnet34)
    def forward(self, x):
        x = self.se_resnet34(x)
        return x

class Se_ResNet101(nn.Module):
    def __init__(self, N_LABELS):
        super(Se_ResNet101, self).__init__()
        self.se_resnet101 = se_resnet101()
        num_ftrs = self.se_resnet101.fc.in_features
        self.se_resnet101.fc = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())
    def forward(self, x):
        x = self.se_resnet101(x)
        return x

class Cbam_ResNet34(nn.Module):
    def __init__(self, N_LABELS):
        super(Cbam_ResNet34, self).__init__()
        self.cbam_resnet34 = cbam_resnet34()
        num_ftrs = self.cbam_resnet34.fc.in_features
        self.cbam_resnet34.fc = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())
    def forward(self, x):
        x = self.cbam_resnet34(x)
        return x

class Cbam_ResNet101(nn.Module):
    def __init__(self, N_LABELS):
        super(Cbam_ResNet101, self).__init__()
        self.cbam_resnet101 = cbam_resnet101()
        num_ftrs = self.cbam_resnet101.fc.in_features
        self.cbam_resnet101.fc = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())
    def forward(self, x):
        x = self.cbam_resnet101(x)
        return x

class ResNet18(nn.Module):
    def __init__(self, N_LABELS, isTrained):
        super(ResNet18, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=isTrained)
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())
    def forward(self, x):
        x = self.resnet18(x)
        return x

class RsNet34(nn.Module):
    def __init__(self, N_LABELS):
        super(RsNet34, self).__init__()
        self.rsnet = rsnet34()
        num_ftrs = self.rsnet.fc.in_features
        self.rsnet.fc = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())
    def forward(self, x):
        x = self.rsnet(x)
        return x


