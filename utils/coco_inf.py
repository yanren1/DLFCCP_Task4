import torchvision
import torch
from torchvision import models, transforms, datasets
from dataloader.dataloader import CocoDataset
from backbone.model import MyEfficientnet_v2_m
from PIL import Image
import numpy as np

cats_name = np.array(['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
             'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
             'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
             'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
             'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
             'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
             'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
             'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.ColorJitter(),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(degrees=20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def multi_label_inf(img,model,threshhold=0.5):

    # coco = CocoDataset(root='data/coco/val2017', annFile='data/coco/annotations/instances_val2017.json',transform=transform)
    # cats_name = np.array(coco.cats_name)
    # print(cats_name)
    # print(cats_name)

    # model = MyEfficientnet_v2_m(pretrained=False, num_classes=80).cuda()
    # weights_pth = 'best.pt'
    # try:
    #     model.load_state_dict(torch.load(weights_pth))
    #     print('ckpt loaded!')
    # except:
    #     print(f'No {weights_pth}')

    model.eval()
    # img = Image.open('trickyimg/dogandcat.jpg')

    with torch.no_grad():
        img = transform(img).unsqueeze(0)
        img = img.cuda()

        pred = model(img)
        sigmoid_f = torch.nn.Sigmoid()
        idx = (sigmoid_f(pred) >= threshhold)[0].cpu().tolist()

        label = cats_name[idx]
        print(label)
    return ', '.join(label)


