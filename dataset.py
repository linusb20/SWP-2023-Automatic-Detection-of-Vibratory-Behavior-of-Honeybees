import os
import random
import numpy as np
import pandas
import PIL.Image
import zipfile
import json
import skimage.transform as sk_transform
from torch.utils.data import Dataset

class WDDDataset(Dataset):
    def __init__(self, gt_items):
        """
        Args:
            gt_items: list of 4-tuples of `waggle_id`, `label`, `gt_angle`, `path`
        """
        self.gt_df = pandas.DataFrame(gt_items, columns=["waggle_id", "label", "gt_angle", "path"])
        self.meta_data_paths = self.gt_df.path.values
        labels = self.gt_df.label.copy()
        labels[labels == "trembling"] = "other"
        self.all_labels = ["other", "waggle", "ventilating", "activating"]
        label_mapper = {s: i for i, s in enumerate(self.all_labels)}
        self.Y = np.array([label_mapper[l] for l in labels])

    def __len__(self):
        return len(self.meta_data_paths)

    def __getitem__(self, i):
        images = WDDDataset.load_waggle_images(self.meta_data_paths[i])
        label = self.Y[i]
        transform = random.choice([
            lambda img: np.flip(img, axis=0),
            lambda img: np.flip(img, axis=1),
            lambda img: sk_transform.rotate(img, 90),
            lambda img: sk_transform.rotate(img, 180),
            lambda img: sk_transform.rotate(img, 270),
        ])
        images = [transform(img) for img in images]
        images = np.expand_dims(images, axis=1)

        return images, label

    @staticmethod
    def load_image(f):
        img = PIL.Image.open(f)
        img = np.asarray(img, dtype=np.float32)
        img = img / 255 * 2 - 1  # normalize to [-1, 1]
        img = sk_transform.resize(img, (110, 110))
        return img

    @staticmethod
    def load_waggle_images(waggle_path):
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
