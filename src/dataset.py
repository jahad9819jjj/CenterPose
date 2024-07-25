import torch
import os
import glob
import numpy as np
import cv2
import open3d as o3d
import hydra
import logging
from pathlib import Path
import cv2
import json
from config import Config, parse_config

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
    def __init__(self, cfg, dataset_path):
        super().__init__()
        self.cfg:Config = parse_config(cfg)
        self.dataset_path = Path(dataset_path)
        self.data = self._load_data()
        
    def _load_data(self):
        return load_images(self.dataset_path, extensions=('png', 'jpg', 'jpeg'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, video_id, frame_id, annotation_path = self.data[idx]
        eigen_vals = self.cfg.category.eig_val
        
        image = cv2.imread(image_path)
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)
            
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