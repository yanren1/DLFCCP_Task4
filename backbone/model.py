import torch
import torch.nn as nn
from typing import Callable, List, Optional
import random
import torchvision
from torchvision.models import resnet18, mobilenet_v3_large, efficientnet_b0,resnet101,efficientnet_v2_m


class simpleMLP(nn.Module):
    def __init__(self,
                in_channels: int,
                hidden_channels: List[int],
                norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
                activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
                inplace: Optional[bool] = None,
                bias: bool = True,
                dropout: float = 0.0,
                use_sigmoid = False,
                ):
        super(simpleMLP, self).__init__()
        params = {} if inplace is None else {"inplace": inplace}
        self.use_sigmoid = use_sigmoid
        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout, **params))
        #sigmoid

        # layers.append(torch.nn.Linear(hidden_channels[-1], hidden_channels[-1], bias=bias))
        # layers.append(torch.nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)


    def forward(self, x):
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        if self.use_sigmoid:
            x = torch.sigmoid(x)
        # print(x.shape)
        return x


def MyResnet18(pretrained=True, num_classes = 1000, freeze = False, sigmoid=False):
    model = resnet18(pretrained=pretrained)


    if num_classes != 1000:
        if sigmoid:
            model.fc = nn.Sequential(nn.Linear(model.fc.in_features, num_classes),
                                 nn.Sigmoid())
        else:
            model.fc = nn.Linear(model.fc.in_features, num_classes)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

        for param in model.fc.parameters():
            param.requires_grad = True

    # print(model)
    return model

def MyResnet101(pretrained=True, num_classes = 1000, freeze = False):
    if pretrained:
        model = resnet101(pretrained=pretrained)

        if num_classes != 1000:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model = resnet101(pretrained=pretrained, num_classes=80)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

        for param in model.layer4.parameters():
            param.requires_grad = True
        model.fc.requires_grad = True

    return model

def MyEfficientNet_B0(pretrained=True,num_classes = 1000, freeze = False, sigmoid=False):
    model = efficientnet_b0(pretrained=pretrained)

    if sigmoid:
        model.classifier.add_module('2', nn.Sigmoid())

    if num_classes!=1000:
        model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

        for classifier_layer in model.classifier:
            for param in classifier_layer.parameters():
                param.requires_grad = True

    return model

def MyEfficientnet_v2_m(pretrained=True,num_classes = 1000):
    model = efficientnet_v2_m(pretrained=pretrained)
    if num_classes!=1000:
        model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)

    return model

def MyMobilenet_v3_large(pretrained=True, num_classes = 1000, freeze = False,sigmoid=False):
    model = mobilenet_v3_large(pretrained=pretrained)
    # print(model)
    if sigmoid:
        model.classifier.add_module('4', nn.Sigmoid())
    if num_classes!=1000:
        model.classifier[3] =nn.Linear(in_features=1280, out_features=num_classes, bias=True)
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        for classifier_layer in model.classifier:
            for param in classifier_layer.parameters():
                param.requires_grad = True

    return model

if __name__ == '__main__':
    sample = torch.randn([32, 3, 224, 224])
    model = MyEfficientnet_v2_m()

    # criterion = nn.CrossEntropyLoss().cuda()
    # sample = torch.randn([32, 3, 224, 224]).cuda()
    # target = torch.randint(low=0,high=256,size=[32]).cuda()
    out = model(sample)
    #
    # loss = criterion(sample,target)
    # print(loss)
    # print(target.shape)
    print(out)
