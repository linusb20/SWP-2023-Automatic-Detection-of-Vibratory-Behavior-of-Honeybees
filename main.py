import os
import numpy as np
import pickle
import json
import pathlib
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import config as cfg
from dataset import WDDDataset, WDDSampler
from model import CNNEncoder, RNNDecoder
from evaluation import compute_accuracy, compute_confusion_matrix
from plotting import plot_accuracy, plot_loss, plot_confusion_matrix, playback

def load_gt_items(path):
    '''
        pickle files contains elements in (key, value) form, like:
            KEY         VALUE
            102934      ('activating',  1.88699,  PosixPath('/mnt/curta/storage/beesbook/wdd/wdd_output_2021/cam1/2021/10/2/10/13/13/waggle.json'))

        OUTPUT
             102934    , 'activating',  1.88699,  PosixPath('/mnt/curta/storage/beesbook/wdd/wdd_output_2021/cam1/2021/10/2/10/13/13/waggle.json')
    '''
    with open(path, "rb") as f:
        r = pickle.load(f)
        items = [(key,) + v for key, v in r.items()]
    return items

def custom_collate(data):
    """ Return custom batch tensors.
    Pad videos to longest video length in a batch and save original video lengths.
    Build batch tensor for padded videos, video lengths and labels
    """
    video_lens = torch.tensor([video.shape[0] for video, _ in data])
    video = [torch.tensor(video) for video, _ in data]
    video = pad_sequence(video, batch_first=True)
    label = torch.tensor([torch.tensor(label) for _, label in data])
    return video, video_lens, label

def main():
    # LOAD PICKLE AND INITIALIZE PATHS
    gt_items = load_gt_items(cfg.PATH_PICKLE) 
    def remap(p):
        head = pathlib.Path(cfg.PATH_IMAGES) # actual system path
        tail = p.relative_to("/mnt/curta/storage/beesbook/wdd/")  # path from pickle 
        return head.joinpath(tail) # replace path
    gt_items = [tuple(item) + (remap(path),) for *item, path, in gt_items]

    all_indices = np.arange(len(gt_items))

    # SPLIT DIRECTORIES INTO TRAIN & TEST DATA (directories contain video/images + we know each label, e. g., activating)
    mask = all_indices % 10 == 0
    test_indices = all_indices[mask]
    train_indices = all_indices[~mask]
     
    print(f"Found {len(test_indices)} test examples")
    print(f"Found {len(train_indices)} training examples")

    gt_train_items = [gt_items[i] for i in train_indices]
    gt_test_items = [gt_items[i] for i in test_indices]

    # INIT. Datasets & Dataloader (Dataloader contains references to directories, such as label, path, angle, key)
    train_dataset = WDDDataset(gt_train_items, augment=True)
    assert len(train_dataset) == len(train_indices)
    train_sampler = WDDSampler(class_bins=train_dataset.class_bins, batch_size=cfg.BATCH_SIZE)
    train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=cfg.NUM_WORKERS, collate_fn=custom_collate)

    test_dataset = WDDDataset(gt_test_items, augment=False)
    assert len(test_dataset) == len(test_indices)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS, shuffle=True, collate_fn=custom_collate) 

    # DEFINE MODEL
    model = torch.nn.Sequential(
        CNNEncoder(), 
        RNNDecoder(),
    )

    # use GPU if available and if more than 2 GPU, than parallelize
    model = model.to(cfg.DEVICE)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    os.makedirs(cfg.PATH_CHECKPOINT_DIR, exist_ok=True)

    ckpt = {}
    if cfg.PATH_CHECKPOINT_RESTORE:
        ckpt = torch.load(cfg.PATH_CHECKPOINT_RESTORE)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded model from {cfg.PATH_CHECKPOINT_RESTORE}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    if cfg.PATH_CHECKPOINT_RESTORE:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"Loaded optimizer from {cfg.PATH_CHECKPOINT_RESTORE}")

    # TRAIN MODEL
    stats = ckpt["stats"] if "stats" in ckpt else {
        "train_acc_list": [],
        "test_acc_list": [],
        "loss_mean_list": [],
        "loss_std_list": [],
    }

    start_epoch = ckpt["epoch"] + 1 if "epoch" in ckpt else 0

    for epoch in range(start_epoch, cfg.NUM_EPOCHS):
        loss_list = []
        model.train()
        for batch_idx, (video, video_lens, label) in enumerate(train_dataloader):
            video = video.to(cfg.DEVICE)
            label = label.to(cfg.DEVICE)

            logits = model((video, video_lens))
            loss = torch.nn.functional.cross_entropy(logits, label)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            # print current epoch & loss
            if batch_idx % cfg.LOG_INTERVAL == 0:
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

        save_path = os.path.join(cfg.PATH_CHECKPOINT_DIR, f"epoch-{epoch:03}.pth")
        if epoch % cfg.SAVE_INTERVAL == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "stats": stats,
            }, save_path)
            print(f"Saved model in epoch {epoch} to {save_path}")

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
