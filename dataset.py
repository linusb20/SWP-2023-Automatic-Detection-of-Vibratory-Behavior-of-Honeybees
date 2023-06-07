import os
import random
import PIL.Image
import zipfile
import json
from typing import List, Tuple

import pandas as pd
import numpy as np
import skimage.transform as sk_transform
from torch.utils.data import Dataset
import albumentations as A

class WDDDataset(Dataset):
    def __init__(self, gt_items):
        '''self.gt_df - pickle-file content stored in DataFrame'''
        self.gt_df:pd.DataFrame = pd.DataFrame(gt_items, columns=["waggle_id", "label", "gt_angle", "path"]) # from pickle 

        '''self.meta_data_paths - list of paths to different waggle.json files'''
        self.meta_data_paths:pd.ArrayLike = self.gt_df.path.values

        '''self.all_labels - list of all possible class labels'''
        labels = self.gt_df.label.copy()
        labels[labels == "trembling"] = "other" # merge tembling and other 
        self.all_labels:List[str] = ["other", "waggle", "ventilating", "activating"]

        '''self.Y - list containing actual class labels (int) for related videos accessable under self.meta_data_paths'''
        label_mapper = {s: i for i, s in enumerate(self.all_labels)} # dict(other: 0, waggle: 1, ...)
        self.Y:np.ndarray[int] = np.array([label_mapper[l] for l in labels])

    def __len__(self):
        return len(self.meta_data_paths)

    def __getitem__(self, i:int) -> Tuple[List[np.ndarray], int]:
        '''method called by DataLoader'''
        video_imgs:List[np.ndarray] = WDDDataset.load_waggle_images(self.meta_data_paths[i])
        label:int = self.Y[i]

        # TODO: image augmentation (what we have at this point: 1 video from one batch, consisting of k images)
        # TODO: image shape should not change 

        transform = random.choice([
            lambda img: np.flip(img, axis=0),
            lambda img: np.flip(img, axis=1),
            lambda img: sk_transform.rotate(img, 90),
            lambda img: sk_transform.rotate(img, 180),
            lambda img: sk_transform.rotate(img, 270),
        ])
        images = [transform(img) for img in video_imgs]

        # floatify images for training
        video_imgs = [img.astype(np.float32) for img in video_imgs]

        video_imgs = np.expand_dims(video_imgs, axis=1)

        return video_imgs, label

    def init_augmenters(self) -> None:
        p = 0.2 
        
        self.augmenter_quality = A.Compose([
            A.MultiplicativeNoise(p=0.25 * p, multiplier=(0.9, 1.1), elementwise=True),
            A.GaussNoise(p=0.5 * p, var_limit=(0, 0.1)),
            A.GaussianBlur(p=0.25 * p, sigma_limit=(0.0, 0.5)),
            A.RandomBrightnessContrast(
                    p=0.5 * p, brightness_limit=(-0.5, 0.5), contrast_limit=(-0.5, 0.5)),
        ])

        self.augmenter_rescale = None 
        self.augmenter = None 


    #============== LOADING IMAGES ======================
    @staticmethod
    def load_image(filename) -> np.ndarray:
        '''loads one image and casts it to np.array'''
        img = PIL.Image.open(filename)

        # transform to uint8 (RGB 0-255)
        img:np.ndarray = np.asarray(img, dtype=np.uint8) 
      
        # mini image augmentation for every image (down sampling)
        img = img / 255 * 2 - 1  # normalize to [-1, 1]
        img = sk_transform.resize(img, (110, 110))

        return img

    @staticmethod
    def load_waggle_images(waggle_path) -> List[np.ndarray]:
        '''load images from one directory (= 1 video)'''
        images: List[np.ndarray] = []
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

