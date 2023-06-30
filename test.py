import os
import numpy as np
import pickle
import json
import pathlib
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


import pandas as pd
import config as cfg
from dataset import WDDDataset
from model import CNNEncoder, RNNDecoder
from evaluation import compute_accuracy, compute_confusion_matrix
from plotting import plot_accuracy, plot_loss, plot_confusion_matrix, playback
from typing import List

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
    ''' complicated (!) pre-processing to generate `PackedSequenze` used in RNN '''
    image_seq_lens = torch.tensor([img.shape[0] for img, _ in data])
    images = [torch.tensor(img) for img, _ in data]
    images = pad_sequence(images, batch_first=True)
    label = torch.tensor([torch.tensor(y) for _, y in data])
    return images, image_seq_lens, label

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
train_dataset = WDDDataset(gt_train_items)



gt_df:pd.DataFrame = pd.DataFrame(gt_items, columns=["waggle_id", "label", "gt_angle", "path"]) # from pickle 
meta_data_paths:np.ndarray = gt_df.path.to_numpy()
labels = gt_df.label.copy()
labels[labels == "trembling"] = "other" # merge tembling and other 

all_labels:List[str] = ["other", "waggle", "ventilating", "activating"]

'''self.Y - list containing actual class labels (int) for related videos accessable under self.meta_data_paths'''
label_mapper = {s: i for i, s in enumerate(all_labels)} # dict(other: 0, waggle: 1, ...)
Y = np.array([label_mapper[l] for l in labels])

print(Y.shape)