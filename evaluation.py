import torch
import numpy as np
from torch.nn import functional as F
import sklearn.metrics as skm

@torch.no_grad()
def compute_metrics(model, dataloader, device, metrics):
    model.eval()

    actual_list = []
    predicted_list = []
    score_list = []
    loss_list = []

    for video, label in dataloader:
        video = video.to(device)
        label = label.to(device)

        logits = model(video)

        actual_list.extend(label.cpu().numpy().tolist())

        score = F.softmax(logits, dim=1)
        score_list.extend(score.cpu().numpy().tolist())

        loss = F.cross_entropy(logits, label)
        loss_list.append(loss.item())

        _, predicted = torch.max(logits, dim=1)
        predicted_list.extend(predicted.cpu().numpy().tolist())

    values = {}

    if "accuracy" in metrics:
        values["accuracy"] = skm.accuracy_score(actual_list, predicted_list) * 100

    if "confusion_matrix" in metrics:
        values["confusion_matrix"] = skm.confusion_matrix(actual_list, predicted_list)

    if "precision" in metrics:
        values["precision"] = skm.precision_score(actual_list, predicted_list, average=None)

    if "recall" in metrics:
        values["recall"] = skm.recall_score(actual_list, predicted_list, average=None)

    if "fscore" in metrics:
        values["fscore"] = skm.f1_score(actual_list, predicted_list, average=None)
    
    if "roc_auc" in metrics:
        try:
            values["roc_auc"] =  skm.roc_auc_score(actual_list, score_list, multi_class="ovr")
        except:
            values["roc_auc"] =  np.nan

    if "loss" in metrics:
        values["loss_mean"] = np.mean(loss_list)
        values["loss_std"] = np.std(loss_list)

    return values
