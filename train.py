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
from evaluation import compute_metrics
from callbacks import AugmentScheduler, StatsCallback, SaveModelCallback 
from model import C3D_RNN

def train(model, dataloader, optimizer, epoch, stats_cb, device):
    model.train()

    loss_list = []
    predicted_list, actual_list = [], []

    for batch_idx, (video, label) in enumerate(dataloader):
        video = video.to(device)
        label = label.to(device)

        logits = model(video)
        loss = F.cross_entropy(logits, label)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        _, predicted = torch.max(logits, 1)
        predicted_list.extend(predicted.cpu().numpy().tolist())
        actual_list.extend(label.cpu().numpy().tolist())
        loss_list.append(loss.item())

        stats_cb.on_batch_end(batch_idx, len(dataloader), epoch, loss.item())
    
    loss_mean = np.mean(loss_list)
    loss_std = np.std(loss_list)
    acc = accuracy_score(actual_list, predicted_list) * 100

    return loss_mean, loss_std, acc

def test(model, dataloader, device):
    metrics = compute_metrics(model, dataloader, device, metrics=["loss", "accuracy"])
    return metrics["loss_mean"], metrics["loss_std"], metrics["accuracy"]

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
    skip_training,
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
        tail = tail.parent
        return head.joinpath(tail)
    gt_items = [
        (remap(path), label if label != "trembling" else "other") 
        for _, label, _, path, in gt_items
    ]

    all_indices = np.arange(len(gt_items))
    np.random.shuffle(all_indices)
    n = int(0.85 * len(all_indices)) # 85/15 train/test split
    train_indices = all_indices[:n]
    test_indices = all_indices[n:]
    assert np.intersect1d(train_indices, test_indices).size == 0
     
    print(f"Found {len(test_indices)} test examples")
    print(f"Found {len(train_indices)} training examples")

    gt_train_items = [gt_items[i] for i in train_indices]
    gt_test_items = [gt_items[i] for i in test_indices]

    train_dataset = WDDDataset(gt_train_items)
    train_sampler = WDDSampler(class_bins=train_dataset.class_bins, batch_size=batch_size)
    train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=num_workers, collate_fn=custom_collate)

    test_dataset = WDDDataset(gt_test_items, augment=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=custom_collate)

    model = C3D_RNN()
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    stats = None
    start_epoch = 0

    if checkpoint_pth:
        assert os.path.exists(path_checkpoints) 
        path_checkpoint = os.path.join(path_checkpoints, checkpoint_pth)
        ckpt = torch.load(path_checkpoint)

        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded model from {path_checkpoint}")

        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print(f"Loaded optimizer from {path_checkpoint}")

        if "stats" in ckpt:
            stats = ckpt["stats"]

        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1

    augment_scheduler = AugmentScheduler(train_dataset)
    stats_cb = StatsCallback(log_interval, path_stats, train_dataset.all_labels, stats)
    save_model_cb = SaveModelCallback(path_checkpoints, save_interval, accuracy_threshold=70)

    if not skip_training:
        for epoch in range(start_epoch, num_epochs):
            augment_scheduler.on_epoch_begin(epoch)
            train_loss_mean, train_loss_std, train_acc = train(model, train_dataloader, optimizer, epoch, stats_cb, device)
            test_loss_mean, test_loss_std, test_acc = test(model, test_dataloader, device)

            stats_cb.on_epoch_end(
                train_loss_mean, 
                train_loss_std, 
                train_acc,
                test_loss_mean, 
                test_loss_std, 
                test_acc,
            )

            save_model_cb.on_epoch_end(model, optimizer, epoch, stats_cb.stats, test_acc)

    metrics = compute_metrics(model, test_dataloader, device, metrics=["confusion_matrix", "precision", "recall", "fscore", "roc_auc"])
    stats_cb.on_train_end(metrics, model.__str__())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--path-pickle", type=str, default=config.path_pickle)
    parser.add_argument("--path-images", type=str, default=config.path_images)
    parser.add_argument("--path-checkpoints", type=str, default=config.path_checkpoints)
    parser.add_argument("--path-stats", type=str, default=config.path_stats)
    parser.add_argument("--device", type=str, default=config.device)
    parser.add_argument("--checkpoint-pth", type=str, default="")
    parser.add_argument("--skip-training", type=bool, default=False, action=argparse.BooleanOptionalAction)

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
        skip_training = args.skip_training,
        device = args.device,
    )
