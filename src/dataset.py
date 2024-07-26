import torch
import os
import glob
import numpy as np
import cv2
import hydra
import logging
from pathlib import Path
import cv2
import json
from config import Config, parse_config
from visualization import blend_keypoints, visualize_corner_displacements, add_axes
from utils import bounding_box_rotation, torch2cv
from augmentation import noise_augmentation, compute_image_augmentation_params, apply_affine_image, statistic_norm
from lib.utils.image import color_aug
from collections import defaultdict
import time
import warnings

def load_images(root, extensions=('png',)):
    imgs = []
    root_path = Path(root)
    # Add images and their corresponding json files to the list
    def add_json_files(path):
        for ext in extensions:
            for img_path in path.glob(f"*.{ext}"):
                json_path = img_path.with_suffix('.json')
                if img_path.exists() and json_path.exists():
                    video_id = img_path.parent.name
                    frame_id = img_path.stem
                    imgs.append((str(img_path), video_id, frame_id, str(json_path)))
    # Recursively search for images in the root directory
    def explore(path):
        if not path.is_dir():
            return
        folders = [entry for entry in path.iterdir() if entry.is_dir()]
        if folders:
            for folder in folders:
                explore(folder)
        else:
            add_json_files(path)

    explore(root_path)
    return imgs

class ObjectronDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, dataset_path:str, split:str='train'):
        super().__init__()
        self.split = split
        self.cfg:Config = parse_config(cfg)
        self.dataset_path = Path(dataset_path)

        self.tracking_mode = False
        if 'centerpose_track' in self.cfg.method.name:
            self.tracking_mode = True

        self._data_rng = np.random.RandomState(123)
        
        self.mean = np.array(self.cfg.category.mean)
        self.std = np.array(self.cfg.category.std)
        assert self.mean.shape == (3,), f"Assume mean shape is (3,) but {self.mean.shape}"
        assert self.std.shape == (3,), f"Assume std shape is (3,) but {self.std.shape}"

        self._eig_val = np.array(self.cfg.category.eig_val)
        self._eig_vec = np.array(self.cfg.category.eig_vec)
        assert self._eig_val.shape == (3,), f"Assume eig_val shape is (3,) but {self._eig_val.shape}"
        assert self._eig_vec.shape == (3, 3), f"Assume eig_vec shape is (3, 3) but {self._eig_vec.shape}"

        # if self.cfg.experimental.data_generation_mode_ratio > 0:
        #     self._setup_detector()

        print(f'Initializing objectron {self.cfg.category.name} {self.split} data.')
        self.data = self._load_data()
        print(f'Loaded {self.split} {len(self.data)} samples.')

        print(f'Grouping images to video.')
        self.videos = defaultdict()
        self._group_images_to_videos()

        
    def _load_data(self):
        return load_images(self.dataset_path, extensions=('png', 'jpg', 'jpeg'))
    
    def _group_images_to_videos(self):
        for i in self.data:
            video_name = i[1]
            if video_name not in self.videos:
                self.videos[video_name] = []
            self.videos[video_name].append(i)
    
    def _setup_detector(self):
        """ 学習データの水増しのため検出器による推定をするケース
        """
        raise NotImplementedError

    def _get_input(self, image, affine_mat):
        """ Execute following procedure.
        1. Apply affine transformation
        2. Apply PCA color augmentation if training and flag exists
        3. Statistical normalization

        Args:
            image (np.ndarray): Image
            affine_mat (np.ndarray): Affine transformation

        Returns:
            np.ndarray: image which is input of model
        """
        assert affine_mat.shape == (2, 3), f"Affine matrix required (2, 3) shape but {affine_mat.shape}"
        inp = apply_affine_image(image, affine_mat, (self.cfg.method.input_res, self.cfg.method.input_res))
        if self.split == 'train' and self.cfg.augmentation.color_aug:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        inp = statistic_norm(inp, self.mean, self.std)
        return inp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        start = time.time()
        np.random.seed(100)
        image_path, video_id, frame_id, annotation_path = self.data[idx]

        # 読み込み
        try:
            image = cv2.imread(image_path)
        except Exception as error:
            warnings.warn(f'Image cannot be read. {image_path}')
            return None
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)
        num_objs = min(len(annotation['objects']), self.cfg.method.max_objs)

        # ノイズ関連オーギュメンテーション
        if self.cfg.experimental.data_generation_mode_ratio:
            image = noise_augmentation(image)
        
        height, width = image.shape[:2]

        # 空間オーギュメンテーション(画像中心オフセット,スケール,回転)
        center_point_org = np.array([image.shape[1] / 2., image.shape[0]/2.], dtype=np.float32)
        scale_org = max(image.shape[0], image.shape[1]) * 1.0
        rot_org = 0


        center_point = annotation.get('center_point', None)
        corner_point = annotation.get('corner_point', None)
        center_displacement = annotation.get('center_displacement', None)
        corner_displacement = annotation.get('corner_displacement', None)

        supervised_info = {
            'inp': image,  # Placeholder - replace with actual image data
            'center_point': center_point,
            'corner_point': corner_point,
            'center_displacement': center_displacement,
            'corner_displacement': corner_displacement,
        }

        if 'center_uncertainty' in annotation:
            supervised_info['center_uncertainty'] = annotation['center_uncertainty']
        if 'corner_uncertainty' in annotation:
            supervised_info['corner_uncertainty'] = annotation['corner_uncertainty']

        return supervised_info