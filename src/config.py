from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Union

@dataclass
class ExperimentalConfig:
    objectron_root:str = None
    num_gpus:int = 1
    results_dir:str = None
    pretrained_model_path:str = None
    randomseed:int = 777

@dataclass
class TrainingParameterConfig:
    max_epoch:int = None
    batch_size:int = 2
    num_workers:int = 4
    img_normalize:bool = True
    
@dataclass
class CategorySpecificParameterConfig:
    category:str = None
    num_symmetry:int = 12
    eig_val:List[float] = None
    eig_vec:List[float] = None
    mean:List[float] = None
    std:List[float] = None

@dataclass
class AugmentationParameterConfig:
    color_aug: int = 1
    crop_aug: int = 1
    flip_aug: int = 1
    rot_aug: int = 1

@dataclass
class MethodParameterConfig:
    method: str = "centerpose_track"
    input_res: int = 512
    output_res: int = 512
    flip_idx: List[List[int]] = field(default_factory=lambda: [[1, 5], [3, 7], [2, 6], [4, 8]])
    
@dataclass
class Config:
    experimental: ExperimentalConfig = ExperimentalConfig()
    training: TrainingParameterConfig = TrainingParameterConfig()
    category: CategorySpecificParameterConfig = CategorySpecificParameterConfig()
    augmentation: AugmentationParameterConfig = AugmentationParameterConfig()
    method: MethodParameterConfig = MethodParameterConfig()
    
def parse_config(cfg_dict: Dict[str, Any]) -> Config:
    config = Config()

    # Parse Experimental Config
    if 'experimental' in cfg_dict:
        for key, value in cfg_dict['experimental'].items():
            setattr(config.experimental, key, value)

    # Parse Training Parameter Config
    if 'training' in cfg_dict:
        for key, value in cfg_dict['training'].items():
            setattr(config.training, key, value)

    # Parse Category Specific Parameter Config
    if 'category' in cfg_dict:
        for key, value in cfg_dict['category'].items():
            setattr(config.category, key, value)

    # Parse Augmentation Parameter Config
    if 'method' in cfg_dict:
        for key, value in cfg_dict['method']['augmentation'].items():
            setattr(config.augmentation, key, value)

    # Parse CenterPose Track Config
    if 'method' in cfg_dict:
        for key, value in cfg_dict['method']['centerpose_track'].items():
            setattr(config.method, key, value)

    return config