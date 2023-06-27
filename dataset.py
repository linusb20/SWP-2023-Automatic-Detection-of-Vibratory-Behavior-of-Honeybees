import os
import copy
import random
import PIL.Image
import zipfile
import json
from typing import List, Tuple, Dict

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import albumentations as A

class WDDDataset(Dataset):
    def __init__(
            self, 
            gt_items,
            img_resize_factor = 0.5):

        # PICKLE FILE
        '''self.gt_df - pickle-file content stored in DataFrame'''
        self.gt_df:pd.DataFrame = pd.DataFrame(gt_items, columns=["waggle_id", "label", "gt_angle", "path"]) # from pickle 

        '''self.meta_data_paths - list of paths to different waggle.json files'''
        self.meta_data_paths:pd.ArrayLike = self.gt_df.path.values

        # CLASS LABELS
        '''self.all_labels - list of all possible class labels'''
        labels = self.gt_df.label.copy()
        labels[labels == "trembling"] = "other" # merge tembling and other 
        self.all_labels:List[str] = ["other", "waggle", "ventilating", "activating"]

        '''self.Y - list containing actual class labels (int) for related videos accessable under self.meta_data_paths'''
        label_mapper = {s: i for i, s in enumerate(self.all_labels)} # dict(other: 0, waggle: 1, ...)
        self.Y:np.ndarray[int] = np.array([label_mapper[l] for l in labels])
        self.class_bins = [[] for _ in range(len(self.all_labels))]
        for i, y in enumerate(self.Y):
            self.class_bins[y].append(i)

        # IMAGE RESOLUTION
        '''self.img_reszie_factor - factor for resizing images, e. g., 0.5 makes (400p x 300p) -> (200p x 150p)'''
        self.img_reszie_factor = img_resize_factor

        exmple_video:List[np.ndarray] = WDDDataset.load_waggle_images(self.meta_data_paths[0])
        example_image:np.ndarray = exmple_video[0]
        
        '''self.img_shape_original - original shape of image (width, height)'''
        self.img_shape_original:Tuple[int,int] = example_image.shape

    def __len__(self):
        return len(self.meta_data_paths)

    def __getitem__(self, i:int) -> Tuple[List[np.ndarray], int]:
        '''
        Method called by DataLoader to get one video and its related class label for classification.

            - loads video (= k images) from OS directory
            - performs image augmentation for all images
            - adjusts image shape: floatify and dimension        
        '''
        # load video
        video_imgs:List[np.ndarray] = WDDDataset.load_waggle_images(self.meta_data_paths[i])
        label:int = self.Y[i]

        # image augmentation (what we have at this point: 1 video from one batch, consisting of k images)
        aug_video_imgs:Dict[str, np.ndarray] = self.augment_video(video=video_imgs)

        # adjusts image shape: floatify and dimension    
        aug_video_imgs = [aug_img.astype(np.float32) for aug_img in aug_video_imgs.values()]
        aug_video_imgs = np.expand_dims(aug_video_imgs, axis=1)

        # logging shape und pixel range
        #print(f'Images had shape {self.img_shape_original}, now {np.array(aug_video_imgs[0]).shape}')
        #print(f'Images pixel range was [{np.min(video_imgs[0])},{np.max(video_imgs[0])}], now [{np.min(aug_video_imgs[0])},{np.max(aug_video_imgs[0])}]')

        return aug_video_imgs, label

    #============== LOADING IMAGES ======================
    @staticmethod
    def load_image(filename) -> np.ndarray:
        '''loads one image and casts it to np.array'''
        img = PIL.Image.open(filename)

        # transform to uint8 (RGB 0-255) for image augmentation
        img:np.ndarray = np.asarray(img, dtype=np.uint8) 

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


    #============= IMAGE AUGMENTATION
    def augment_video(self, video:List[np.ndarray] ) -> Dict[str, np.ndarray]:
        '''initializes self.augmenter by defining different augmentations'''

        '''aug_resize - augmenter for resizing images (downsampling)'''
        aug_resize = None 

        '''aug_quality - augmenter forfilter, e. g., noise, blur, brightness, contrast'''
        aug_quality = None 

        '''aug_shape - augmenter for spatial change, e. g., shifts, tilts, rotations'''
        aug_shape = None 

        '''aug_shape - augmenter for normalizing pixel range'''
        aug_normalize = None  
        
        p = 0.75

        # AUGMENTER
        aug_resize =  A.Compose([
            A.Resize(
                p=1,
                width   = int(self.img_shape_original[0] * self.img_reszie_factor), 
                height  = int(self.img_shape_original[1] * self.img_reszie_factor)
            )
        ])

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
            A.MultiplicativeNoise(p=p, multiplier=(0.75, 1.25), elementwise=True),

            # A.GaussNoise() --- produces noisy image
            #
            # `arg var_limit(a, b)`
            #   defines range for variance (randomly sampled), where a is min and b is max
            A.GaussNoise(p=p, var_limit=(0, 10)),

            # A.GaussianBlur() --- produces blurry image
            #
            # `arg sigma_limit(a, b)`
            #   defines range for blur, where a is min and b is max
            A.GaussianBlur(p=p, sigma_limit=(0.0, 0.75), always_apply=True),

            # A.RandomBrightnessContrast() --- affects brightness (light/dark) and contrast
            #
            # `arg brightness_limit(a, b)`
            #   defines range for brightness, where a is min and b is max
            #
            # `arg contrast_limit(a, b)`
            #   defines range for contrast_limit, where a is min and b is max
            A.RandomBrightnessContrast(p=p, brightness_limit=(-0.1, 0.5), contrast_limit=(-0.5, 0.5))
        ])

        aug_shape = A.Compose([
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
                p=p,
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                shear=(-5, 5),
                scale={"x": (0.9,1.1), "y": (0.9,1.1)},
            ),
        ])
        
        aug_normalize = A.Normalize(
            # normalize to [-1,1]
            p=1,
            mean=0.5, std=0.5, max_pixel_value=255
        )


        # AUGMENTER INIT
        dictoINIT =  {} 
        dictoCALL = {}
        for i in range(1,len(video)):
            dictoINIT[f'image{i}'] = "image"
            dictoCALL[f'image{i}'] = video[i]

        augmenter = A.Compose([
                aug_resize,
                aug_quality,
                aug_shape,
                aug_normalize
            ],
            additional_targets=dictoINIT
        )

        # AUGMENTER CALL
        aug_video = augmenter(image=video[0], **dictoCALL)
        return aug_video

class WDDSampler():
    def __init__(self, class_bins, batch_size):
        self.batch_size = batch_size
        self.class_bins = class_bins

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
        return iter(batch_list)
