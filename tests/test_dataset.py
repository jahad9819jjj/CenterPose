# centerpose$ pytest -q tests/test_dataset.py
# 上記を見ながらテストデータセットの形状を確認するテストケースを追加

import importlib
import pathlib
import os
import sys
import numpy as np
import pytest

# Debug
from icecream import icecream 
import ipdb

ROOT = pathlib.Path(__file__).resolve().parents[1]        # centerpose/
SRC  = ROOT / "src"                                       # centerpose/src
sys.path.insert(0, str(SRC))   
opts = importlib.import_module('lib.opts')

def build_opt():
    opt = opts.opts()
    opt = opt.parser.parse_args([])
    opt.c = 'bottle500'
    opt.obj_scale = True
    opt.tracking_task = True
    opt.batch_size = 2
    opt.gpus = '3'
    opt.num_workers = 0
    opt.stereo_training = True
    opt.training_data_dir = ['/path/to/gt']
    opt.flip = 0.0
    opt.num_symmetry = 1

    opt.obj_scale_uncertainty = True
    opt.hps_uncertainty = True
    opt.tracking_label_mode = 1
    opt.render_hm_mode = 1

    opt.data_generation_mode_ratio = 0
        
    # Augmentation parameter
    opt.shift = 0.25
    opt.scale = 0.25
    # opt.shift = 0.00
    # opt.scale = 0.00
    opt.not_rand_crop = True
    opt.aug_rot = 0.0
        
    # たぶん前フレームのキーポイントを入力とするときの摂動パラメータ
    # For hm
    opt.hm_heat_random = True
    opt.hm_disturb = 0.05
    opt.lost_disturb = 0.2
    opt.fp_disturb = 0.1
    # For hm_hp
    opt.hm_hp_heat_random = True
    opt.hm_hp_disturb = 0.03
    opt.hp_lost_disturb = 0.1
    opt.hp_fp_disturb = 0.05
    opt.max_frame_dist = 3
        
    opt.reg_offset = not opt.not_reg_offset
    opt.reg_bbox = not opt.not_reg_bbox
    opt.hm_hp = not opt.not_hm_hp
    opt.reg_hp_offset = (not opt.not_reg_hp_offset) and opt.hm_hp

    # Btw frame parameter
    if opt.tracking_task:
        opt.pre_img = True
        opt.pre_hm = True
        opt.tracking = True
        opt.pre_hm_hp = True
        opt.tracking_hp = True
        opt.cur_msk = True
        opt.pre_msk = True

    opt.default_resolution = [512, 512]
    opt.flip_idx = [[1, 5], [3, 7], [2, 6], [4, 8]]
    opt.mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32).reshape(1, 1, 3)
    opt.std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32).reshape(1, 1, 3)

    return opt

def test_dataset_smoke():
    opt = build_opt()
    dataset_mdle = importlib.import_module('lib.datasets.dataset_combined')
    objpose_ds = dataset_mdle.ObjectPoseDataset(opt, split='train')
    opt = opts.opts().update_dataset_info_and_set_heads(opt, objpose_ds)
    # 10 samples only
    for i in range(10):
        sample = objpose_ds.__getitem__(i)
        assert isinstance(sample, dict)
        
        # monocular/stereo common
        batch_fields = [
            # current‑frame stuff
            'input', 'hm', 'reg_mask', 'ind', 'hps', 'hps_mask',
            'cur_msk', 'pre_msk',
            'hps_uncertainty', 'pre_img', 'pre_hm', 'pre_hm_hp',
            'tracking', 'tracking_mask', 'tracking_hp', 'tracking_hp_mask',
            'scale', 'scale_uncertainty', 'wh', 'reg', 'hm_hp',
            'hp_offset', 'hp_ind', 'hp_mask',
            # meta / misc
            'meta', 'intrinsics', 'extrinsics', 'height_org', 'width_org',
            'ct', 'pts', 'ct_3d', 'pts_3d',
        ]
        pair_only_fields = [
            'input_pair', 'hm_pair', 'reg_mask_pair', 'ind_pair',
            'hps_pair', 'hps_mask_pair', 'hps_uncertainty_pair',
            'pre_img_pair', 'pre_hm_pair', 'pre_hm_hp_pair',
            'tracking_pair', 'tracking_mask_pair',
            'tracking_hp_pair', 'tracking_hp_mask_pair',
            'scale_pair', 'scale_uncertainty_pair',
            'wh_pair', 'reg_pair', 'hm_hp_pair',
            'hp_offset_pair', 'hp_ind_pair', 'hp_mask_pair',
            'cur_msk_pair', 'pre_msk_pair',
            'ct_pair', 'pts_pair', 'ct_3d_pair', 'pts_3d_pair',
            'intrinsics_pair'
        ]
        if getattr(opt, 'stereo_training', False):
            batch_fields += pair_only_fields

        for i in batch_fields:
            print(i)
            assert i in sample

            variables = sample[i]

            # 入力画像
            if i == 'input' and isinstance(variables, np.ndarray):
                C, H, W = variables.shape
                assert C in {3}, '想定外のチャンネル数'
                assert H == opt.input_h
                assert W == opt.input_w
            
            # 前画像
            if i == 'pre_img' and isinstance(variables, np.ndarray):
                C, H, W = variables.shape
                assert C in {3}, '想定外のチャンネル数'
                assert H == opt.input_h
                assert W == opt.input_w

            # 現在マスク
            if i == 'cur_msk' and isinstance(variables, np.ndarray):
                C, H, W = variables.shape
                assert C in {2}, '想定外のチャンネル数'
                assert H == opt.input_h
                assert W == opt.input_w

            # 前マスク
            if i == 'pre_msk' and isinstance(variables, np.ndarray):
                C, H, W = variables.shape
                assert C in {2}, '想定外のチャンネル数'
                assert H == opt.input_h
                assert W == opt.input_w
            
            # 前物体3D中心のヒートマップ
            if i == 'pre_hm' and isinstance(variables, np.ndarray):
                C, H, W = variables.shape
                assert C in {1}, '想定外のチャンネル数'
                assert H == opt.input_h
                assert W == opt.input_w


            # 前物体3DBBox頂点のヒートマップ
            if i == 'pre_hm_hp' and isinstance(variables, np.ndarray):
                C, H, W = variables.shape
                assert C in {8}, '想定外のチャンネル数'
                assert H == opt.input_h
                assert W == opt.input_w
            
            # 現在物体3D中心のヒートマップ
            if i == 'hm' and isinstance(variables, np.ndarray):
                B, C, H, W = variables.shape
                # icecream.icecream.ic(B)
                # icecream.icecream.ic(opt.batch_size)

                assert B == opt.batch_size // 2 if opt.stereo_training else B == opt.batch_size, '想定外のバッチ数'
                assert C == 1, '想定外のヒートマップチャンネル数'
                assert H == opt.input_h // opt.down_ratio
                assert W == opt.input_w // opt.down_ratio

            # 現在物体3DBBox頂点のヒートマップ
            if i == 'hm_hp' and isinstance(variables, np.ndarray):
                B, C, H, W = variables.shape
                assert B == opt.batch_size // 2 if opt.stereo_training else B == opt.batch_size, '想定外のバッチ数'
                assert C == 8, '想定外のヒートマップチャンネル数'
                assert H == opt.input_h // opt.down_ratio
                assert W == opt.input_w // opt.down_ratio

            if i == 'meta' and isinstance(variables, dict):
                assert variables['c'].size == 2 # 中心シフト量
                assert variables['s'].size == 1 # スケール
                assert variables['trans_in'].shape ==  variables['trans_out'].shape == (2, 3)


        # 何らかの理由によりサンプルがないときはスキップ 
        if sample is None:
            pytest.skip('Sample is None(I/O error)')


if __name__ == "__main__":
    # build_opt()
    test_dataset_smoke()