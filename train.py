from pickle import TRUE
from re import I
import time
import csv
import os
import math
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import datetime
import torch.optim
import torch.utils.data
from torchvision import models
from torch import nn
import torch
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from tqdm import tqdm
import random
import numpy as np

from dataset import NIH
from utils import BatchIterator, Saved_items, checkpoint
from model.models import *


def ModelTrain(train_df_path, val_df_path, path_image, ModelType, nnIsTrained, 
                CriterionType, device, LR):

    # Training parameters
    batch_size = 32
    # workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers  # mean: how many subprocesses to use for data loading.
    workers = 2
    N_LABELS = 14
    start_epoch = 0
    num_epochs = 64  # number of epochs to train for (if early stopping is not triggered)

    val_df = pd.read_csv(val_df_path)
    val_df_size = len(val_df)
    print("Validation_df size",val_df_size)

    train_df = pd.read_csv(train_df_path)
    train_df_size = len(train_df)
    print("Train_df size", train_df_size)

    random_seed = 33    # random.randint(0,100)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # ============================ step 1/5 数据 ============================
    print("[Info]: Loading Data ...")

    data_transform = {
        "train": transforms.Compose([transforms.RandomHorizontalFlip(),
                            transforms.RandomRotation(10),
                            transforms.Resize(256),
                            transforms.CenterCrop(256),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(256),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 构建MyDataset实例(针对不是按文件夹分类, csv文件的图像分类(NIH Chest X-ray14),详情查看labels/train.csv)
    train_dataset = NIH(train_df, path_image=path_image, transform=data_transform["train"])
    val_dataset = NIH(val_df, path_image=path_image, transform=data_transform["val"])

    # 构建DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                               shuffle=False, num_workers=workers, pin_memory=True)
    
    # for i, data in enumerate(train_loader):
    #     images, labels, _ = data
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, 
                                             shuffle=False, num_workers=workers, pin_memory=True)
    print("[Info]: Data has been loaded ...")

    # ============================ step 2/5 模型 ============================
    print('[Info]: Loaded Model {}'.format(ModelType))
    if ModelType == 'densenet121':
        model = DenseNet121(N_LABELS, nnIsTrained).cuda()

    if ModelType == 'se_densenet121':
        model = Se_DenseNet121(N_LABELS, nnIsTrained).cuda()
    
    if ModelType == 'cbam_densenet121':
        model = CBAM_DenseNet121(N_LABELS, nnIsTrained).cuda()

    if ModelType == 'dca_cbam_densenet121':
        model = DCA_CBAM_DenseNet121(N_LABELS, nnIsTrained).cuda()

    if ModelType == 'sa_densenet121':
        model = SA_DenseNet121(N_LABELS, nnIsTrained).cuda()

    if ModelType == 'densenet161':
        model = DenseNet161(N_LABELS, nnIsTrained).cuda()

    if ModelType == 'resnet152':
        model = ResNet152(N_LABELS, nnIsTrained).cuda()

    if ModelType == 'cbam_resnet152':
        model = CBAM_ResNet152(N_LABELS, nnIsTrained).cuda()

    if ModelType == 'resnet101':
        model = ResNet101(N_LABELS, nnIsTrained).cuda()

    if ModelType == 'resnet50':
        model = ResNet50(N_LABELS, nnIsTrained).cuda()

    if ModelType == 'sa_resnet50':
        model = SA_ResNet50(N_LABELS, nnIsTrained).cuda()

    if ModelType == 'pysa_resnet50':
        model = PYSA_ResNet50(N_LABELS, nnIsTrained).cuda()
        
    if ModelType == 'pycbam_resnet50':
        model = PYCBAM_ResNet50(N_LABELS, nnIsTrained).cuda()

    if ModelType == 'py_resnet50':
        model = PY_ResNet50(N_LABELS, nnIsTrained).cuda()

    if ModelType == 'cbam_resnet50':
        model = CBAM_ResNet50(N_LABELS, nnIsTrained).cuda()

    if ModelType == 'dca_cbam_resnet50':
        model = DCA_CBAM_ResNet50(N_LABELS, nnIsTrained).cuda()

    if ModelType == 'resnet34':
        model = ResNet34(N_LABELS, nnIsTrained).cuda()
        
    if ModelType == 'resnet18':
        model = ResNet18(N_LABELS, nnIsTrained).cuda()
    
    if ModelType == 'se_resnet34':
        model = Se_ResNet34(N_LABELS).cuda()
        
    if ModelType == 'se_resnet101':
        model = Se_ResNet101(N_LABELS).cuda()

    if ModelType == 'cbam_resnet34':
        model = Cbam_ResNet34(N_LABELS).cuda()

    if ModelType == 'cbam_resnet101':
        model = Cbam_ResNet101(N_LABELS).cuda()

    if ModelType == 'Resume':
        CheckPointData = torch.load('results/checkpoint')
        model = CheckPointData['model']

    if torch.cuda.device_count() > 1:
        print('Using', torch.cuda.device_count(), 'GPUs')
        model = nn.DataParallel(model)

    print(model)
    model = model.to(device)

    # ============================ step 3/5 损失函数 ============================
    if CriterionType == 'BCELoss':
        criterion = nn.BCELoss().to(device)
    
    # ============================ step 5/5 训练 ============================
    epoch_losses_train = []
    epoch_losses_val = []

    since = time.time()

    best_loss = 999999
    best_epoch = -1

    #--------------------------Start of epoch loop
    for epoch in tqdm(range(start_epoch, num_epochs + 1)):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
    # -------------------------- Start of phase
        # timestampTime = time.strftime("%H%M%S")
        # timestampDate = time.strftime("%d%m%Y")
        # timestampSTART = timestampDate + '-' + timestampTime

        phase = 'train'
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=LR)   # 固定部分参数
        running_loss = BatchIterator(model=model, phase=phase, Data_loader=train_loader, 
                                    criterion=criterion, optimizer=optimizer, device=device)
        epoch_loss_train = running_loss / train_df_size
        epoch_losses_train.append(epoch_loss_train.item())
        print("Train_losses:", epoch_losses_train)

        phase = 'val' 
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
        running_loss = BatchIterator(model=model, phase=phase, Data_loader=val_loader, 
                                                criterion=criterion, optimizer=optimizer, device=device)
        
        epoch_loss_val = running_loss / val_df_size
        epoch_losses_val.append(epoch_loss_val.item())
        print("Validation_losses:", epoch_losses_val)

        timestampTime = time.strftime("%H%M%S")
        timestampDate = time.strftime("%d%m%Y")
        timestampEND = timestampDate + '-' + timestampTime

        # checkpoint model if has best val loss yet
        if epoch_loss_val < best_loss:
            best_loss = epoch_loss_val
            best_epoch = epoch
            checkpoint(model, best_loss, best_epoch, LR)
            print ('Epoch [' + str(epoch + 1) + '] [save] [' + timestampEND + '] loss= ' + str(epoch_loss_val))
        else:
            print ('Epoch [' + str(epoch + 1) + '] [----] [' + timestampEND + '] loss= ' + str(epoch_loss_val))

        # log training and validation loss over each epoch
        with open("results/log_train", 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            if (epoch == 1):
                logwriter.writerow(["epoch", "train_loss", "val_loss","Seed","LR"])
            logwriter.writerow([epoch, epoch_loss_train, epoch_loss_val,random_seed, LR])
    # -------------------------- End of phase

        # break if no val loss improvement in 3 epochs
        if ((epoch - best_epoch) >= 3):
            if epoch_loss_val > best_loss:
                print("decay loss from " + str(LR) + " to " + str(LR / 2) + " as not seeing improvement in val loss")
                LR = LR / 2
                print("created new optimizer with LR " + str(LR))
                if ((epoch - best_epoch) >= 10):
                    print("no improvement in 10 epochs, break")
                    break
        #old_epoch = epoch 
    #------------------------- End of epoch loop
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    Saved_items(epoch_losses_train, epoch_losses_val, time_elapsed, batch_size)
    
    checkpoint_best = torch.load('results/checkpoint')
    model = checkpoint_best['model']

    best_epoch = checkpoint_best['best_epoch']
    print(best_epoch)

    return model, best_epoch


