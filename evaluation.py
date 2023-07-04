import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

@torch.no_grad()
def compute_predicted_actual(model, dataloader, device):
    predicted_list, actual_list = [], []
    for video, label in dataloader:
        video = video.to(device)
        label = label.to(device)
        logits = model(video)
        _, predicted = torch.max(logits, 1)
        predicted_list.extend(predicted.cpu().numpy().tolist())
        actual_list.extend(label.cpu().numpy().tolist())
    return predicted_list, actual_list

def compute_accuracy(model, dataloader, device):
    predicted_list, actual_list = compute_predicted_actual(model, dataloader, device)
    return accuracy_score(actual_list, predicted_list) * 100

def compute_confusion_matrix(model, dataloader, device):
    predicted_list, actual_list = compute_predicted_actual(model, dataloader, device)
    return confusion_matrix(actual_list, predicted_list)

def compute_metrics(model, dataloader, device, metrics):
    values = {}
    predicted_list, actual_list = compute_predicted_actual(model, dataloader, device)

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
