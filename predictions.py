# from HeatmapGenerator import ModelType
from dataset import *
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from itertools import cycle
import sklearn.metrics as sklm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score


def make_pred_multilabel(model, test_df, val_df, path_image, device):
    """
    Gives predictions for test fold and calculates AUCs using previously trained model
    Args:

        model: densenet-121 from torchvision previously fine tuned to training data
        test_df : dataframe csv file
        PATH_TO_IMAGES:
    Returns:
        pred_df: dataframe containing individual predictions and ground truth for each test image
        auc_df: dataframe containing aggregate AUCs by train/test tuples
    """

    BATCH_SIZE = 32
    workers = 8
    n_classes = 14

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    dataset_test = NIH(test_df, path_image=path_image, transform=transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        normalize]))
    test_loader = DataLoader(dataset_test, BATCH_SIZE, shuffle=False, 
                             num_workers=workers, pin_memory=True)

    dataset_val = NIH(val_df, path_image=path_image, transform=transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        normalize]))
    val_loader = DataLoader(dataset_val, BATCH_SIZE, shuffle=True,
                            num_workers=workers, pin_memory=True)

    size = len(test_df)
    print("Test _df size :", size)
    size = len(val_df)
    print("val_df size :", size)



    # criterion = nn.BCELoss().to(device)
    model = model.to(device)
    # to find this thresold, first we get the precision and recall withoit this, from there we calculate f1 score, using f1score, we found this theresold which has best precsision and recall.  Then this threshold activation are used to calculate our binary output.


    PRED_LABEL = ['Atelectasis',
            'Cardiomegaly',
            'Effusion',
            'Infiltration',
            'Mass',
            'Nodule',
            'Pneumonia',
            'Pneumothorax',
            'Consolidation',
            'Edema',
            'Emphysema',
            'Fibrosis',
            'Pleural_Thickening',
            'Hernia']

    # for mode in ["Threshold"]:
    for mode in ["test"]:
        # create empty dfs
        pred_df = pd.DataFrame(columns=["path"])
        bi_pred_df = pd.DataFrame(columns=["path"])
        true_df = pd.DataFrame(columns=["path"])

        if mode == "Threshold":
            loader = val_loader
            Eval_df = pd.DataFrame(columns=["label", 'bestthr'])
            thrs = []

        if mode == "test":
            loader = test_loader
            TestEval_df = pd.DataFrame(columns=["label", 'auc', "auprc"])

            Eval = pd.read_csv("results/Threshold.csv")
            thrs = [Eval["bestthr"][Eval[Eval["label"] == "Atelectasis"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Cardiomegaly"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Effusion"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Infiltration"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Mass"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Nodule"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Pneumonia"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Pneumothorax"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Consolidation"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Edema"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Emphysema"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Fibrosis"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Pleural_Thickening"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Hernia"].index[0]]]

        for i, data in enumerate(loader):
            inputs, labels, item = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            true_labels = labels.cpu().data.numpy()

            batch_size = true_labels.shape

            model.eval()
            with torch.no_grad():
                outputs = model(inputs)
                probs = outputs.cpu().data.numpy()

            # get predictions and true values for each item in batch
            for j in range(0, batch_size[0]):
                thisrow = {}
                bi_thisrow = {}
                truerow = {}

                truerow["path"] = item[j]
                thisrow["path"] = item[j]
                if mode == "test":
                    bi_thisrow["path"] = item[j]
                    # iterate over each entry in prediction vector; each corresponds to
                    # individual label
                for k in range(len(PRED_LABEL)):
                    thisrow["prob_" + PRED_LABEL[k]] = probs[j, k]
                    truerow[PRED_LABEL[k]] = true_labels[j, k]

                    if mode == "test":
                       bi_thisrow["bi_" + PRED_LABEL[k]] = probs[j, k] >= thrs[k]

                pred_df = pred_df.append(thisrow, ignore_index=True)
                true_df = true_df.append(truerow, ignore_index=True)
                if mode == "test":
                    bi_pred_df = bi_pred_df.append(bi_thisrow, ignore_index=True)

            if (i % 200 == 0):
                print(str(i * BATCH_SIZE))


        for column in true_df:
            if column not in PRED_LABEL:
                continue
            actual = true_df[column]
            pred = pred_df["prob_" + column]
            
            thisrow = {}
            thisrow['label'] = column
            
            if mode == "test":
                bi_pred = bi_pred_df["bi_" + column]            
                thisrow['auc'] = np.nan
                thisrow['auprc'] = np.nan
            else:
                thisrow['bestthr'] = np.nan

            try:
#                 n_booatraps = 1000
#                 rng_seed = int(size / 100)
#                 bootstrapped_scores = []

#                 rng = np.random.RandomState(rng_seed)
#                 for i in range(n_booatraps):
#                     indices = rng.random_integers(0, len(actual.as_matrix().astype(int)) - 1, len(pred.as_matrix()))
#                     if len(np.unique(actual.as_matrix().astype(int)[indices])) < 2:
#                         continue

#                     score = sklm.roc_auc_score(
#                         actual.as_matrix().astype(int)[indices], pred.as_matrix()[indices])
#                     bootstrapped_scores.append(score)

#                 thisrow['auc'] = np.mean(bootstrapped_scores)
            

                if mode == "test":
                    fpr = dict()
                    tpr = dict()
                    roc_auc = dict()
                    for i in range(n_classes):
                        a = true_labels[:, i].astype(int)
                        b = probs[:, i]
                        fpr[i], tpr[i], _ = roc_curve(true_labels[:, i].astype(int), probs[:, i])
                        roc_auc[i] = auc(fpr[i], tpr[i])
                    thisrow['auc'] = roc_auc_score(actual.to_numpy().astype(int), pred.to_numpy())
                    thisrow['auprc'] = sklm.average_precision_score(actual.to_numpy().astype(int), pred.to_numpy())
                    # Compute micro-average ROC curve and ROC area
                    fpr["micro"], tpr["micro"], _ = roc_curve(true_labels[:, i], probs[:, i])
                    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                    # 绘制多分类问题的ROC曲线
                    # First aggregate all false positive rates
                    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

                    # Then interpolate all ROC curves at this points
                    mean_tpr = np.zeros_like(all_fpr)
                    for i in range(n_classes):
                        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

                    # Finally average it and compute AUC
                    mean_tpr /= n_classes

                    fpr["macro"] = all_fpr
                    tpr["macro"] = mean_tpr
                    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

                    # Plot all ROC curves
                    plt.figure()
                    lw = 2
                    plt.plot(
                        fpr["micro"],
                        tpr["micro"],
                        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
                        color="deeppink",
                        linestyle=":",
                        linewidth=4,
                    )

                    plt.plot(
                        fpr["macro"],
                        tpr["macro"],
                        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
                        color="navy",
                        linestyle=":",
                        linewidth=4,
                    )

                    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
                    for i, color in zip(range(n_classes), colors):
                        plt.plot(
                            fpr[i],
                            tpr[i],
                            color=color,
                            lw=lw,
                            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
                        )

                    plt.plot([0, 1], [0, 1], "k--", lw=lw)
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.title("Some extension of Receiver operating characteristic to multiclass")
                    plt.legend(loc="lower right")
                    plt.savefig('results/mutilroc1.png')
                else:

                    p, r, t = sklm.precision_recall_curve(actual.as_matrix().astype(int), pred.as_matrix())
                    # Choose the best threshold based on the highest F1 measure
                    f1 = np.multiply(2, np.divide(np.multiply(p, r), np.add(r, p)))
                    bestthr = t[np.where(f1 == max(f1))]

                    thrs.append(bestthr)
                    thisrow['bestthr'] = bestthr[0]


            except BaseException:
                print("can't calculate auc for " + str(column))

            if mode == "Threshold":
                Eval_df = Eval_df.append(thisrow, ignore_index=True)

            if mode == "test":
                TestEval_df = TestEval_df.append(thisrow, ignore_index=True)

        pred_df.to_csv("results/preds.csv", index=False)
        true_df.to_csv("results/True.csv", index=False)


        if mode == "Threshold":
            Eval_df.to_csv("results/Threshold.csv", index=False)

        if mode == "test":
            TestEval_df.to_csv("results/TestEval.csv", index=False)
            bi_pred_df.to_csv("results/bipred.csv", index=False)

    
    print("AUC ave:", TestEval_df['auc'].sum() / 14.0)

    print("done")

    return pred_df, Eval_df, bi_pred_df, TestEval_df  # , bi_pred_df , Eval_bi_df

if __name__=="__main__":
    val_df = pd.read_csv(val_df_path)
    test_df = pd.read_csv(test_df_path)

    CheckPointData = torch.load('results/checkpoint')
    model = CheckPointData['model']

    make_pred_multilabel(model, test_df, val_df, path_image, device)