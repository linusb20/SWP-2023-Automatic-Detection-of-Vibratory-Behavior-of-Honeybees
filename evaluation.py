import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

import config as cfg

@torch.no_grad()
def compute_predicted_actual(model, dataloader):
    predicted_list, actual_list = [], []
    for images, image_seq_len, label in dataloader:
        images = images.to(cfg.DEVICE)
        label = label.to(cfg.DEVICE)
        logits = model((images, image_seq_len))
        _, predicted = torch.max(logits, 1)
        predicted_list.extend(predicted.cpu().numpy().tolist())
        actual_list.extend(label.cpu().numpy().tolist())
    return predicted_list, actual_list

def compute_accuracy(model, dataloader):
    predicted_list, actual_list = compute_predicted_actual(model, dataloader)
    return accuracy_score(actual_list, predicted_list) * 100

def compute_confusion_matrix(model, dataloader):
    predicted_list, actual_list = compute_predicted_actual(model, dataloader)
    return confusion_matrix(actual_list, predicted_list)

def compute_metrics(model, dataloader, metrics):
    values = {}
    predicted_list, actual_list = compute_predicted_actual(model, dataloader)

    if "accuracy" in metrics:
        values["accuracy"] = accuracy_score(actual_list, predicted_list) * 100

    if "confusion_matrix" in metrics:
        values["confusion_matrix"] = confusion_matrix(actual_list, predicted_list)

    if "precision" in metrics:
        values["precision"] = precision_score(actual_list, predicted_list, average=None)

    if "recall" in metrics:
        values["recall"] = recall_score(actual_list, predicted_list, average=None)

    if "fscore" in metrics:
        values["fscore"] = f1_score(actual_list, predicted_list, average=None)

    return values
