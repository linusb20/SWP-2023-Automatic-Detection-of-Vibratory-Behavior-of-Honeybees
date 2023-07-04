import os
import copy
import random
import PIL.Image
import zipfile
import json
import pandas as pd
import numpy as np
import albumentations as A
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import config

class WDDDataset(Dataset):
    def __init__(self, gt_items, augment=True):
        self.augment = augment
        self.gt_df= pd.DataFrame(gt_items, columns=["waggle_id", "label", "gt_angle", "path"])
        self.meta_data_paths = self.gt_df.path.values
        labels = self.gt_df.label.copy()
        labels[labels == "trembling"] = "other" # merge tembling and other 
        self.all_labels = ["other", "waggle", "ventilating", "activating"]
        label_mapper = {s: i for i, s in enumerate(self.all_labels)}
        self.Y = np.array([label_mapper[l] for l in labels])
        self.class_bins = [[] for _ in range(len(self.all_labels))]
        for i, y in enumerate(self.Y):
            self.class_bins[y].append(i)
        self.augment_p = None

    def __len__(self):
        return len(self.meta_data_paths)

    def __getitem__(self, i):
        """Method called by DataLoader to get one video and its related class label for classification"""

        video = WDDDataset.load_waggle_images(self.meta_data_paths[i]) # min video length is 95
        label = self.Y[i]

        augments = ["resize", "normalize"]
        if self.augment:
            augments.extend(["shape", "quality"])

        video = self.augment_video(video, augments=augments, p=self.augment_p)
        video = [img.astype(np.float32) for img in video]
        video = np.expand_dims(video, axis=1)

        return video, label

    @staticmethod
    def load_image(filename):
        """Load one image and cast it to np.array"""
        img = PIL.Image.open(filename)
        img = np.asarray(img, dtype=np.uint8) 
        return img

    @staticmethod
    def load_waggle_images(waggle_path):
        """Load images for one video"""
        images = []
        waggle_dir = waggle_path.parent
        zip_file_path = os.path.join(waggle_dir, "images.zip")
        assert os.path.exists(zip_file_path) 
        with zipfile.ZipFile(zip_file_path, "r") as zf:
            image_fns = zf.namelist()
            for fn in image_fns:
                with zf.open(fn, "r") as f:
                    images.append(WDDDataset.load_image(f))
        return images

    @staticmethod
    def load_waggle_metadata(waggle_path):
        with open(waggle_path, "r") as f:
            metadata = json.load(f)
        waggle_angle = metadata["waggle_angle"]
        waggle_duration = metadata["waggle_duration"]
        waggle_vector = np.array([np.cos(waggle_angle), np.sin(waggle_angle)], dtype=np.float32)
        return waggle_vector, waggle_duration

    def augment_video(self, video, augments, p):
        """initializes self.augmenter by defining different augmentations"""

        """aug_resize - augmenter for resizing images (downsampling)"""
        aug_resize = None 

        """aug_quality - augmenter forfilter, e. g., noise, blur, brightness, contrast"""
        aug_quality = None 

        """aug_shape - augmenter for spatial change, e. g., shifts, tilts, rotations"""
        aug_shape = None 

        """aug_shape - augmenter for normalizing pixel range"""
        aug_normalize = None  

        p = p or 1

        transforms = []

        if "quality" in augments:
            aug_quality = A.Compose([
                # A.MultiplicativeNoise() --- produces lighter/darker image
                #
                # `arg multiplier(a,b)`
                #   value p is multiplied by a unique value v which is in the range of
                #   multiplier=(a, b).
                #   v < 1 --> p' is darker than p
                #   v > 1 --> p' is lighter than p
                # 
                # `arg elementwise=True` 
                #   True --> each pixel p uses different factor v to create p'=p*v 
                A.MultiplicativeNoise(p=0.4*p, multiplier=(0.75, 1.25), elementwise=True),

                # A.GaussNoise() --- produces noisy image
                #
                # `arg var_limit(a, b)`
                #   defines range for variance (randomly sampled), where a is min and b is max
                A.GaussNoise(p=0.5*p, var_limit=(0, 10)),

                # A.GaussianBlur() --- produces blurry image
                #
                # `arg sigma_limit(a, b)`
                #   defines range for blur, where a is min and b is max
                A.GaussianBlur(p=0.4*p, sigma_limit=(0.0, 0.75), always_apply=True),

                # A.RandomBrightnessContrast() --- affects brightness (light/dark) and contrast
                #
                # `arg brightness_limit(a, b)`
                #   defines range for brightness, where a is min and b is max
                #
                # `arg contrast_limit(a, b)`
                #   defines range for contrast_limit, where a is min and b is max
                A.RandomBrightnessContrast(p=0.7*p, brightness_limit=(-0.1, 0.5), contrast_limit=(-0.5, 0.5)),

                A.RandomGamma(p=0.6*p),
            ])
            transforms.append(aug_quality)

        if "shape" in augments:
            aug_shape = A.Compose([
                A.OneOf([
                    A.HorizontalFlip(), 
                    A.VerticalFlip()
                ], p=0.8*p),

                A.RandomRotate90(p=0.6*p),

                #  A.Affine() --- rotation, zoom, shift
                #
                # `arg translate_percent(dict)`
                #   shifts across x,y axis (min, max)
                #
                # `arg shear(a,b)`
                #   rotation on the z axis
                #
                # `arg scale(dict)`
                #   zooms on different axis
                A.Affine(
                    p=0.6*p,
                    translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                    shear=(-5, 5),
                    scale={"x": (0.9,1.1), "y": (0.9,1.1)},
                ),
            ])
            transforms.append(aug_shape)

        if "resize" in augments:
            aug_resize =  A.Compose([
                A.Resize(
                    p=1,
                    width   = config.img_w,
                    height  = config.img_h,
                )
            ])
            transforms.append(aug_resize)
        
        if "normalize" in augments:
            aug_normalize = A.Normalize(
                p=1,
                mean=config.img_mean, std=config.img_std, max_pixel_value=255
            )
            transforms.append(aug_normalize)

        dictoINIT =  {} 
        dictoCALL = {}
        for i in range(1,len(video)):
            dictoINIT[f'image{i}'] = "image"
            dictoCALL[f'image{i}'] = video[i]

        augmenter = A.Compose(transforms=transforms, additional_targets=dictoINIT)
        aug_video = augmenter(image=video[0], **dictoCALL)

        return aug_video.values()


class WDDSampler():
    """ Return an iterator over lists of batch indices in an epoch.
    For each batch generate a list `batch` of indices into the dataset
    where `len(batch) = batch_size`.
    The indices are sampled in such a way that each class ("other", "waggle", 
    "ventilating", "activating") has an equal chance of being included in a batch.
    """

    def __init__(self, class_bins, batch_size):
        self.batch_size = batch_size
        self.class_bins = class_bins
        self.len = None

    def __iter__(self):
        class_bins = copy.deepcopy(self.class_bins)
        maxl = max(len(c) for c in class_bins)
        for c in class_bins:
            c += random.choices(c, k=maxl-len(c))
        idx_list = [idx for c in class_bins for idx in c]
        random.shuffle(idx_list)

        batch_list = []
        while(len(idx_list) >= self.batch_size):
            batch = []
            for _ in range(self.batch_size):
                batch.append(idx_list.pop())
            batch_list.append(batch)
        self.len = len(batch_list)
        return iter(batch_list)

    def __len__(self):
        return self.len


def custom_collate(data):
    """ Return custom batch tensors.
    Pad videos to longest video length in a batch and save original video lengths.
    Build batch tensor for padded videos, video lengths and labels
    """
    video = [torch.tensor(video) for video, _ in data]
    video = pad_sequence(video, batch_first=True)
    video = video.permute(0, 2, 1, 3, 4)
    label = torch.tensor([torch.tensor(label) for _, label in data])
    return video, label
