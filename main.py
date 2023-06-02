import os
import numpy as np
import pickle
import json
import pathlib
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import config as cfg
from dataset import WDDDataset
from model import CNNEncoder, RNNDecoder
from evaluation import compute_accuracy, compute_confusion_matrix
from plotting import plot_accuracy, plot_loss, plot_confusion_matrix, playback

def load_gt_items(path):
    with open(path, "rb") as f:
        r = pickle.load(f)
        items = [(key,) + v for key, v in r.items()]
    return items

def custom_collate(data):
    image_seq_lens = torch.tensor([img.shape[0] for img, _ in data])
    images = [torch.tensor(img) for img, _ in data]
    images = pad_sequence(images, batch_first=True)
    label = torch.tensor([torch.tensor(y) for _, y in data])
    return images, image_seq_lens, label

def main():
    gt_items = load_gt_items(cfg.PATH_PICKLE) 
    def remap(p):
        head = pathlib.Path(cfg.PATH_IMAGES)
        tail = p.relative_to("/mnt/curta/storage/beesbook/wdd/")
        return head.joinpath(tail)
    gt_items = [tuple(item) + (remap(path),) for *item, path in gt_items]

    all_indices = np.arange(len(gt_items))
    mask = all_indices % 10 == 0
    test_indices = all_indices[mask]
    train_indices = all_indices[~mask]

    print(f"Found {len(test_indices)} test examples")
    print(f"Found {len(train_indices)} training examples")

    gt_train_items = [gt_items[i] for i in train_indices]
    gt_test_items = [gt_items[i] for i in test_indices]

    train_dataset = WDDDataset(gt_train_items)
    assert len(train_dataset) == len(train_indices)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS, shuffle=True, collate_fn=custom_collate)

    test_dataset = WDDDataset(gt_test_items)
    assert len(test_dataset) == len(test_indices)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS, shuffle=True, collate_fn=custom_collate) 

    model = torch.nn.Sequential(
        CNNEncoder(), 
        RNNDecoder(),
    )
    model = model.to(cfg.DEVICE)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    stats = {
        "train_acc_list": [],
        "test_acc_list": [],
        "loss_mean_list": [],
        "loss_std_list": [],
    }

    for epoch in range(cfg.NUM_EPOCHS):
        loss_list = []
        model.train()
        for batch_idx, (images, image_seq_len, label) in enumerate(train_dataloader):
            images = images.to(cfg.DEVICE)
            label = label.to(cfg.DEVICE)

            logits = model((images, image_seq_len))
            loss = torch.nn.functional.cross_entropy(logits, label)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch} Batch: {batch_idx}")
            print(f"Loss: {loss:.4f}")
            loss_list.append(loss.item())
        stats["loss_mean_list"].append(np.mean(loss_list))
        stats["loss_std_list"].append(np.std(loss_list))

        model.eval()
        with torch.no_grad():
            train_acc = compute_accuracy(model, train_dataloader)
            test_acc = compute_accuracy(model, test_dataloader)
            print(f"Epoch {epoch}")
            print(f"Training Accuracy: {train_acc:.2f}%")
            print(f"Testing Accuracy: {test_acc:.2f}%")
            stats["train_acc_list"].append(train_acc.item())
            stats["test_acc_list"].append(test_acc.item())

    with torch.no_grad():
        cm = compute_confusion_matrix(model, test_dataloader)

    os.makedirs(cfg.STATS_PATH)
    plot_accuracy(stats["train_acc_list"], stats["test_acc_list"], cfg.SAVE_PATH_ACCURACY)
    plot_loss(stats["loss_mean_list"], stats["loss_std_list"], cfg.SAVE_PATH_LOSS)
    plot_confusion_matrix(cm, test_dataset.all_labels, cfg.SAVE_PATH_CONFUSION)
    with open(cfg.SAVE_PATH_JSON, "w") as f:
        json.dump(stats, f)
    with open(cfg.SAVE_PATH_MODEL_SUMMARY, "w") as f:
        print(model, file=f)


if __name__ == "__main__":
    main()
