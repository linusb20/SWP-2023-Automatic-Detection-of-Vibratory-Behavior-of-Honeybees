import os
import argparse
import random
import numpy as np
import pickle
import json
import pathlib
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from sklearn.metrics import accuracy_score

import config
from dataset import WDDDataset, WDDSampler, custom_collate
from model import CNNEncoder, RNNDecoder
from evaluation import compute_metrics
from plotting import plot_accuracy, plot_loss, plot_confusion_matrix, playback


def train(model, dataloader, optimizer, epoch, log_interval, device):
    model.train()

    loss_list = []
    predicted_list, actual_list = [], []

    for batch_idx, (video, video_lens, label) in enumerate(dataloader):
        video = video.to(device)
        label = label.to(device)

        logits = model((video, video_lens))
        loss = F.cross_entropy(logits, label)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        _, predicted = torch.max(logits, 1)
        predicted_list.extend(predicted.cpu().numpy().tolist())
        actual_list.extend(label.cpu().numpy().tolist())
        loss_list.append(loss.item())

        if batch_idx % log_interval == 0:
            print(f"[Epoch {epoch:3}] Batch {batch_idx:3} of {len(dataloader):3}")
            print(f"Loss: {loss:.4f}")
    
    loss_mean = np.mean(loss_list)
    loss_std = np.std(loss_list)
    acc = accuracy_score(actual_list, predicted_list) * 100

    return loss_mean, loss_std, acc

def eval(model, dataloader, device):
    model.eval()

    loss_list = []
    predicted_list, actual_list = [], []

    with torch.no_grad():
        for batch_idx, (video, video_lens, label) in enumerate(dataloader):
            video = video.to(device)
            label = label.to(device)

            logits = model((video, video_lens))
            loss = F.cross_entropy(logits, label)

            _, predicted = torch.max(logits, 1)
            predicted_list.extend(predicted.cpu().numpy().tolist())
            actual_list.extend(label.cpu().numpy().tolist())
            loss_list.append(loss.item())

    loss_mean = np.mean(loss_list)
    loss_std = np.std(loss_list)
    acc = accuracy_score(actual_list, predicted_list) * 100

    return loss_mean, loss_std, acc

def train_loop(
    gt_train_items,
    gt_test_items,
    num_epochs,
    batch_size,
    num_workers,
    path_checkpoints, 
    path_stats, 
    checkpoint_pth,
    log_interval,
    save_interval,
    device,
):
    train_dataset = WDDDataset(gt_train_items)
    train_sampler = WDDSampler(class_bins=train_dataset.class_bins, batch_size=batch_size)
    train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=num_workers, collate_fn=custom_collate)

    test_dataset = WDDDataset(gt_test_items, augment=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=custom_collate)

    model = torch.nn.Sequential(
        CNNEncoder(), 
        RNNDecoder(),
    )
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    path_checkpoint = os.path.join(patch_checkpoints, checkpoint_pth) if checkpoint_pth else None

    ckpt = {}
    if path_checkpoint:
        ckpt = torch.load(path_checkpoint)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded model from {path_checkpoint}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    if path_checkpoint:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"Loaded optimizer from {path_checkpoint}")

    stats = ckpt["stats"] if "stats" in ckpt else {
        "train_acc_list": [],
        "test_acc_list": [],
        "train_loss_mean_list": [],
        "train_loss_std_list": [],
        "test_loss_mean_list": [],
        "test_loss_std_list": [],
    }

    start_epoch = ckpt["epoch"] + 1 if "epoch" in ckpt else 0

    for epoch in range(start_epoch, num_epochs):
        train_loss_mean, train_loss_std, train_acc = train(model, train_dataloader, optimizer, epoch, log_interval, device)
        test_loss_mean, test_loss_std, test_acc = eval(model, test_dataloader, device)

        stats["train_acc_list"].append(round(train_acc, 3))
        stats["test_acc_list"].append(round(test_acc, 3))
        stats["train_loss_mean_list"].append(round(train_loss_mean, 3))
        stats["train_loss_std_list"].append(round(train_loss_std, 3))
        stats["test_loss_mean_list"].append(round(test_loss_mean, 3))
        stats["test_loss_std_list"].append(round(test_loss_std, 3))

        print(f"Training Accuracy: {train_acc:.2f}%")
        print(f"Testing Accuracy: {test_acc:.2f}%")
        print(f"Training Loss Mean: {train_loss_mean:.2f}")
        print(f"Training Loss Std: {train_loss_std:.2f}")
        print(f"Testing Loss Mean: {test_loss_mean:.2f}")
        print(f"Testing Loss Std: {test_loss_std:.2f}")

        if epoch % save_interval == 0:
            save_path = os.path.join(path_checkpoints, f"epoch-{epoch:03}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "stats": stats,
            }, save_path)
            print(f"Saved model in epoch {epoch} to {save_path}")

    model.eval()
    with torch.no_grad():
        metrics = compute_metrics(model, test_dataloader, device, metrics=["confusion_matrix", "precision", "recall", "fscore"])
        stats.update({k: metrics[k].tolist() for k in ["precision", "recall", "fscore"]})
        cm = metrics["confusion_matrix"]
        other_precision, waggle_precision, ventilating_precision, activating_precision = metrics["precision"]
        other_recall, waggle_recall, ventilating_recall, activating_recall = metrics["recall"]
        other_fscore, waggle_fscore, ventilating_fscore, activating_fscore = metrics["fscore"]
        print(f"waggle_precision: {waggle_precision:.3f}, waggle_recall: {waggle_recall:.3f}, waggle_fscore: {waggle_fscore:.3f}")
        print(f"ventilating_precision: {ventilating_precision:.3f}, ventilating_recall: {ventilating_recall:.3f}, ventilating_fscore: {ventilating_fscore:.3f}")
        print(f"activating_precision: {activating_precision:.3f}, activating_recall: {activating_recall:.3f}, activating_fscore: {activating_fscore:.3f}")

    plot_accuracy(stats["train_acc_list"], stats["test_acc_list"], os.path.join(path_stats, "accuracy.jpg"))
    plot_loss(stats["train_loss_mean_list"], stats["train_loss_std_list"], os.path.join(path_stats, "train_loss.jpg"))
    plot_loss(stats["test_loss_mean_list"], stats["test_loss_std_list"], os.path.join(path_stats, "test_loss.jpg"))
    plot_confusion_matrix(cm, test_dataset.all_labels, os.path.join(path_stats, "confusion.jpg"))
    with open(os.path.join(path_stats, "stats.json"), "w") as f:
        json.dump(stats, f)
    with open(os.path.join(path_stats, "model.txt"), "w") as f:
        print(model, file=f)


def main(
    path_pickle, 
    path_images, 
    batch_size, 
    num_epochs, 
    num_workers, 
    log_interval,
    save_interval,
    path_checkpoints,
    path_stats,
    checkpoint_pth,
    device,
):
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    # Save information from pickle file
    with open(path_pickle, "rb") as f:
        r = pickle.load(f)
        gt_items = [(key,) + v for key, v in r.items()]
    def remap(p):
        head = pathlib.Path(path_images)
        tail = p.relative_to("/mnt/curta/storage/beesbook/wdd/")
        return head.joinpath(tail)
    gt_items = [tuple(item) + (remap(path),) for *item, path, in gt_items]

    os.makedirs(path_checkpoints, exist_ok=True)
    os.makedirs(path_stats, exist_ok=True)

    all_indices = np.arange(len(gt_items))
    np.random.shuffle(all_indices)
    n = int(0.85 * len(all_indices)) # 85/15 train/test split
    train_indices = all_indices[:n]
    test_indices = all_indices[n:]
     
    print(f"Found {len(test_indices)} test examples")
    print(f"Found {len(train_indices)} training examples")

    gt_train_items = [gt_items[i] for i in train_indices]
    gt_test_items = [gt_items[i] for i in test_indices]

    train_loop(
        gt_train_items,
        gt_test_items,
        num_epochs,
        batch_size,
        num_workers,
        path_checkpoints, 
        path_stats, 
        checkpoint_pth,
        log_interval,
        save_interval,
        device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--path-pickle", type=str, default=config.path_pickle)
    parser.add_argument("--path-images", type=str, default=config.path_images)
    parser.add_argument("--path-checkpoints", type=str, default=config.path_checkpoints)
    parser.add_argument("--path-stats", type=str, default=config.path_stats)
    parser.add_argument("--device", type=str, default=config.device)
    parser.add_argument("--checkpoint_pth", type=str, default="")

    parser.add_argument("--batch-size", type=int, default=config.batch_size)
    parser.add_argument("--num-epochs", type=int, default=config.num_epochs)
    parser.add_argument("--num-workers", type=int, default=config.num_workers)
    parser.add_argument("--log-interval", type=int, default=config.log_interval)
    parser.add_argument("--save-interval", type=int, default=config.save_interval)

    args = parser.parse_args()

    main(
        path_pickle = args.path_pickle, 
        path_images = args.path_images, 
        batch_size = args.batch_size, 
        num_epochs = args.num_epochs, 
        num_workers = args.num_workers, 
        log_interval = args.log_interval,
        save_interval = args.save_interval,
        path_checkpoints = args.path_checkpoints,
        path_stats = args.path_stats,
        checkpoint_pth = args.checkpoint_pth,
        device = args.device,
    )
