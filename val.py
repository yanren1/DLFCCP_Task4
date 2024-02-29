import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.models import resnet18, mobilenet_v3_large, efficientnet_b0
from torchvision.models import ResNet18_Weights,EfficientNet_B0_Weights, MobileNet_V3_Large_Weights
from torchvision.io import read_image
import os
from PIL import Image
from backbone.model import MyEfficientNet_B0,MyMobilenet_v3_large,MyResnet18,MyResnet101
import numpy as np

def read_label():
    cls_dict = {}
    with open('data/caltech256_label.txt','r') as f:
        lines = f.readlines()
        for l in lines:
            cls_dict[l.split(':')[0]] = l.split(':')[1].strip()
    return cls_dict

def val_pretrain():

    img = read_image("trickyimg/panda-dog.jpg")

    # Step 1: Initialize model with the best available weights


    # weights = ResNet18_Weights.DEFAULT
    # model = resnet18(weights=weights)

    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0()


    # weights = MobileNet_V3_Large_Weights.DEFAULT
    # model = mobilenet_v3_large(weights=weights)

    model.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(img).unsqueeze(0)

    # Step 4: Use the model and print the predicted category
    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    print(f"{category_name}: {100 * score:.1f}%")

def val_custum():
    model = MyEfficientNet_B0(num_classes=257)
    weights_pth = 'model_saved/t1_2/EfficientNet_b0_2024-02-28-02-07-10/final.pt'
    model.load_state_dict(torch.load(weights_pth))
    model.cuda().eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.ColorJitter(),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    img = Image.open('trickyimg/panda-dog.jpg')
    with torch.no_grad():
        img = transform(img).unsqueeze(0)
        img = img.cuda()
        # print(img.shape)
        pred = model(img)
        _, pred = torch.max(pred.data, 1)
        idx = str((pred.item()))

        cls_dict = read_label()
        print(cls_dict[idx])




