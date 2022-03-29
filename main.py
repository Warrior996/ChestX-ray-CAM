import time
import torch
import argparse
import pandas as pd
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from train import ModelTrain
from LearningCurve import *
from predictions import make_pred_multilabel
from nih import *

#---------------------- on q
path_image = "../../../data/ChestX-ray14/images"
train_df_path ="labels/train.csv"
test_df_path = "labels/test.csv"
val_df_path = "labels/val.csv"

# 测试程序是否能跑通
# train_df_path ="labels/train_copy.csv"
# test_df_path = "labels/test_copy.csv"
# val_df_path = "labels/val_copy.csv"


diseases = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
            'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 
            'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']


def main():

    MODE = "test"  # Select "train" or "test", "Resume", "plot", "Threshold", "plot15"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv(train_df_path)
    train_df_size = len(train_df)
    print("Train_df size", train_df_size)

    test_df = pd.read_csv(test_df_path)
    test_df_size = len(test_df)
    print("test_df size", test_df_size)

    val_df = pd.read_csv(val_df_path)
    val_df_size = len(val_df)
    print("val_df size", val_df_size)

    if MODE == "train":

       ModelType = "pycbam_resnet50"  # select 'resnet50','densenet121','resnet34', 'resnet18', 'rsnet18', 'se_resnet101', 'cbam_resnet34'
       nnIsTrained = False  # 是否使用预训练模型
       CriterionType = 'BCELoss' # select 'BCELoss'
       LR = 0.5e-3
      
       model, best_epoch = ModelTrain(train_df_path, val_df_path, path_image, ModelType,
                                      nnIsTrained, CriterionType, device, LR)
       
    #    PlotLearnignCurve()

    if MODE =="test":
        val_df = pd.read_csv(val_df_path)
        test_df = pd.read_csv(test_df_path)

        CheckPointData = torch.load('results/checkpoint')
        model = CheckPointData['model']

        make_pred_multilabel(model, test_df, val_df, path_image, device)


    if MODE == "Resume":
        ModelType = "Resume"  # select 'ResNet50','densenet','ResNet34', 'ResNet18'
        CriterionType = 'BCELoss'
        nnIsTrained = False
        LR = 0.5e-3

        model, best_epoch = ModelTrain(train_df_path, val_df_path, path_image, 
                     ModelType, nnIsTrained, CriterionType, device, LR)

        # PlotLearnignCurve()

if __name__ == "__main__":
    main()