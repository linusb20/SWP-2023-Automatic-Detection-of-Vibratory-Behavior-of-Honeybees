import torch
from sklearn.metrics import confusion_matrix

import config as cfg

def compute_accuracy(model, dataloader):
    correct, num_examples = 0, 0
    for images, image_seq_len, label in dataloader:
        images = images.to(cfg.DEVICE)
        label = label.to(cfg.DEVICE)
        logits = model((images, image_seq_len))
        _, predicted = torch.max(logits, 1)
        num_examples += logits.size(0) # batch size
        correct += (predicted == label).sum()
    return correct / num_examples * 100

def compute_confusion_matrix(model, dataloader):
    predicted_list, actual_list = [], []
    for images, image_seq_len, label in dataloader:
        images = images.to(cfg.DEVICE)
        label = label.to(cfg.DEVICE)
        logits = model((images, image_seq_len))
        _, predicted = torch.max(logits, 1)
        predicted_list.extend(predicted.to("cpu"))
        actual_list.extend(label.to("cpu"))
    return confusion_matrix(actual_list, predicted_list)
