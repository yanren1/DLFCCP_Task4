import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms, models
from torchvision.transforms import transforms

from tqdm import tqdm
import numpy as np
from backbone.ghostnetv2_torch import MyGhostnetv2_0
from backbone.model import MyResnet18,MyEfficientNet_B0,MyMobilenet_v3_large,MyResnet101,MyEfficientnet_v2_m
from dataloader.dataloader import SampleDataset,CocoDataset
from loss.loss import cal_map
from utils.metrics import VOC12mAP

import time
from torch.autograd import Variable
from tqdm import tqdm


def save_model(model_save_pth,model, epoch,train_ce,accuracy,model_type=''):
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    filename = 'model_{}_epoch_{}_train_ce_{:0.2e}_val_Acc_{:0.2e}.pt'.format(model_type,
                                                                                     epoch,
                                                                                     train_ce,
                                                                                     accuracy,)

    filename = os.path.join(model_save_pth,filename)
    torch.save(model.state_dict(), filename)


def train():

    debug = False
    use_pretrain = True
    multi_label = True
    # split train and val set
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=20),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225])
    ])
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.ColorJitter(),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(degrees=20),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225])
    ])

    # train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
    # val_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_val)


    # train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    # val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)

    # dataset = torchvision.datasets.Caltech256(root='./data', transform=transform)
    Train_dataset = CocoDataset(root='data/coco/train2017', annFile='data/coco/annotations/instances_train2017.json',transform=transform_train)
    Val_dataset = CocoDataset(root='data/coco/val2017', annFile='data/coco/annotations/instances_val2017.json',
                               transform=transform_val)

    # train_ratio = 0.9
    # train_size = int(train_ratio * len(dataset))
    # val_size = len(dataset) - train_size
    # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    batch_size = int(64)
    train_loader = torch.utils.data.DataLoader(
        Train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )
    val_loader = DataLoader(Val_dataset, batch_size=batch_size, shuffle=True)

    model_type = ['resnet18','resnet101','EfficientNet_b0','Efficientnet_v2_m','MobilenetV3_Large','ghostnetv2']
    model_type = model_type[3]

    # backbone
    if model_type =='resnet18':
        backbone = MyResnet18(pretrained=True, num_classes=80,freeze=False,sigmoid=True).cuda()
    if model_type =='resnet101':
        backbone = MyResnet101(pretrained=True, num_classes=80,freeze=True).cuda()
    elif model_type =='EfficientNet_b0':
        backbone = MyEfficientNet_B0(pretrained=True, num_classes=80,freeze=True,sigmoid=True).cuda()
    elif model_type =='Efficientnet_v2_m':
        backbone = MyEfficientnet_v2_m(pretrained=True, num_classes=80).cuda()
    elif model_type =='MobilenetV3_Large':
        backbone = MyMobilenet_v3_large(pretrained=True, num_classes=80,freeze=False,sigmoid=True).cuda()
    elif model_type == 'ghostnetv2':
        backbone = MyGhostnetv2_0(num_classes=80).cuda()


    # criterion = nn.CrossEntropyLoss().cuda()
    criterion = nn.BCEWithLogitsLoss().cuda()
    MAP_criterion = VOC12mAP(80)

    # try read pre-train model
    if use_pretrain:
        weights_pth = 'final.pt'
        try:
            backbone.load_state_dict(torch.load(weights_pth))
            print('ckpt loaded!')
        except:
            print(f'No {weights_pth}')
    backbone.cuda()
    # set lr,#epoch, optimizer and scheduler
    lr = 5e-6
    num_epoch = 20
    optimizer = optim.Adam(
        backbone.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5, amsgrad=False)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=1e-8)

    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    model_save_pth = os.path.join('model_saved', model_type + '_'+current_time)
    os.mkdir(model_save_pth)
    writer = SummaryWriter(model_save_pth)
    best_mAP = 0
    # start training
    backbone.train()
    for epoch in range(num_epoch):
        loss_list = []
        train_map_list = []
        for sample, target in tqdm(train_loader):
            backbone.zero_grad()
            # print(sample.shape,target.shape)
            sample, target = sample.cuda(), target.cuda()
            output = backbone(sample)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())

        scheduler.step()

        if (epoch+1) % 1 == 0 or epoch==0:
            # print(f'\r Epoch:{epoch} ce loss = {np.mean(loss_list)} ,lr = {optimizer.param_groups[0]["lr"]}     ', end = ' ')
            tqdm.write(f'\r Epoch:{epoch} ce loss = {np.mean(loss_list)} ,lr = {optimizer.param_groups[0]["lr"]}     ',)
            writer.add_scalar('Training CE Loss', np.mean(loss_list), epoch)
            writer.add_scalar('Learning rate', optimizer.param_groups[0]["lr"], epoch)

        # valing and save
        if (epoch+1) % 1 == 0 or epoch==0:
            # print('Valing.....')
            tqdm.write('Valing.....')
            val_loss_list = []
            backbone.eval()

            if multi_label:
                MAP_criterion.reset()
            else:
                correct = 0
                total = 0

            with torch.no_grad():
                for val_sample, val_target in val_loader:

                    val_sample, val_target = val_sample.cuda(), val_target.cuda()
                    output = backbone(val_sample)
                    val_loss = criterion(output, val_target)

                    if multi_label:
                        output,val_target = output.cpu().numpy(),val_target.cpu().numpy()
                        MAP_criterion.update(output, val_target)

                    else:
                        _, predicted = torch.max(output.data, 1)
                        total += val_target.size(0)
                        correct += (predicted == val_target).sum().item()

                    val_loss_list.append(val_loss.item())

            Train_ce = np.mean(loss_list)
            val_ce = np.mean(val_loss_list)
            if multi_label:
                _, mAP = MAP_criterion.compute()
                writer.add_scalar('Validation ce', val_ce, epoch + 1)
                writer.add_scalar('Validation mAP', mAP, epoch + 1)
                tqdm.write(f'VAL Epoch:{epoch} Train ce = {Train_ce}, '
                      f'val ce = {val_ce} , val mAP = {mAP}')
                if mAP >= best_mAP:
                    best_mAP = mAP
                    torch.save(backbone.state_dict(), os.path.join(model_save_pth, f'mAP {mAP}.pt'))

            else:
                accuracy = correct / total
                writer.add_scalar('Validation ce', val_ce, epoch + 1)
                writer.add_scalar('Validation accuracy', accuracy, epoch + 1)

                # print(f'VAL Epoch:{epoch} Train ce = {Train_ce}, '
                #       f'val ce = {val_ce} , val accuracy = {accuracy}')
                tqdm.write(f'VAL Epoch:{epoch} Train ce = {Train_ce}, '
                      f'val ce = {val_ce} , val accuracy = {accuracy}')

                torch.save(backbone.state_dict(), os.path.join(model_save_pth, f'final.pt'))
            # save_model(model_save_pth,backbone, epoch, Train_ce, accuracy)
            backbone.train()

    torch.save(backbone.state_dict(), os.path.join(model_save_pth,'final.pt'))
    # dummy_input = torch.randn([1, 1, 28, 28], requires_grad=True).cuda()
    # torch.onnx.export(backbone,  # model being run
    #                   dummy_input,  # model input (or a tuple for multiple inputs)
    #                   os.path.join(model_save_pth,'final.onnx'),  # where to save the model
    #                   export_params=True,  # store the trained parameter weights inside the model file
    #                   opset_version=10,  # the ONNX version to export the model to
    #                   do_constant_folding=True,  # whether to execute constant folding for optimization
    #                   input_names=['modelInput'],  # the model's input names
    #                   output_names=['modelOutput'],  # the model's output names
    #                   dynamic_axes={'modelInput': {0: 'batch_size'},  # variable length axes
    #                                 'modelOutput': {0: 'batch_size'}})
    writer.flush()
    writer.close()

if __name__ == '__main__':
    train()