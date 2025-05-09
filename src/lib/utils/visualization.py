import cv2
import numpy as np
import torch
import torchvision
import math
import h5py
from typing import Iterable
from pathlib import Path

def save_batch_to_hdf5(batch, filename, compression="gzip", compression_opts=9):
    """
    Save batch dictionary containing torch tensors to HDF5 file with compression
    
    Args:
        batch: Dictionary containing torch tensors
        filename: Output HDF5 filename
        compression: Compression filter to use. Options: 'gzip', 'lzf', None
        compression_opts: Compression settings (1-9 for gzip, ignored for lzf)
    """
    with h5py.File(filename, 'w') as f:
        # Iterate through all keys in batch
        for key, value in batch.items():
            # Handle torch tensors directly
            if isinstance(value, torch.Tensor):
                if value.device.type == 'cuda':
                    data = value.detach().cpu().numpy()
                else:
                    data = value.cpu().numpy()
                f.create_dataset(
                    key, 
                    data=data,
                    compression=compression,
                    compression_opts=compression_opts if compression == 'gzip' else None,
                    shuffle=True  # データタイプに応じて自動的にバイトシャッフリングを適用
                )
            
            # Handle nested dictionaries
            elif isinstance(value, dict):
                # Create a group for the nested dictionary
                group = f.create_group(key)
                # Save each item in the nested dictionary
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        if sub_value.device.type == 'cuda':
                            sub_value_np = sub_value.detach().cpu().numpy()
                        else:
                            sub_value_np = sub_value.cpu().numpy()
                        group.create_dataset(
                            sub_key, 
                            data=sub_value_np,
                            compression=compression,
                            compression_opts=compression_opts if compression == 'gzip' else None,
                            shuffle=True
                        )
                    elif isinstance(sub_value, np.ndarray):
                        group.create_dataset(
                            sub_key, 
                            data=sub_value,
                            compression=compression,
                            compression_opts=compression_opts if compression == 'gzip' else None,
                            shuffle=True
                        )
                    else:
                        try:
                            group.create_dataset(sub_key, data=sub_value)
                        except:
                            print(f"Warning: Could not save {key}/{sub_key} of type {type(sub_value)}")
                            
def process_loaded_data(data, device='cuda'):
    """
    Helper function to process loaded data and convert to appropriate type
    """
    if isinstance(data, np.ndarray):
        try:
            # Handle standard numeric types
            if data.dtype in [np.float32, np.float64, np.int64, np.int32, 
                            np.int16, np.int8, np.uint8, np.bool_]:
                data = torch.from_numpy(data)
                if device == 'cuda':
                    data = data.to(device)
                return data
            # Handle object arrays
            elif data.dtype == np.dtype('object'):
                # Try converting each element separately
                processed = []
                for item in data:
                    if isinstance(item, np.ndarray):
                        processed.append(torch.from_numpy(item))
                    else:
                        processed.append(item)
                return processed
            else:
                return data
        except Exception as e:
            print(f"Warning: Could not convert data to tensor: {str(e)}")
            return data
    return data

def load_batch_from_hdf5(filename, verbose=True, device='cuda'):
    """
    Load batch data from HDF5 file back into a dictionary with torch tensors
    
    Args:
        filename: Input HDF5 filename
        verbose: If True, print debug information
    
    Returns:
        Dictionary containing the loaded data
    """
    batch_loaded = {}
    with h5py.File(filename, 'r') as f:
        # First, print the structure if verbose is True
        if verbose:
            def print_structure(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"Dataset: {name}, Shape: {obj.shape}, Type: {obj.dtype}")
                elif isinstance(obj, h5py.Group):
                    print(f"Group: {name}")
            f.visititems(print_structure)

        for key in f.keys():
            try:
                # Handle groups (nested dictionaries)
                if isinstance(f[key], h5py.Group):
                    batch_loaded[key] = {}
                    for sub_key in f[key].keys():
                        try:
                            data = f[key][sub_key][()]
                            batch_loaded[key][sub_key] = process_loaded_data(data, device)
                            if verbose:
                                print(f"Loaded {key}/{sub_key}")
                        except Exception as e:
                            print(f"Error loading {key}/{sub_key}: {str(e)}")
                            if verbose:
                                print(f"Data type: {type(data)}")
                                if isinstance(data, np.ndarray):
                                    print(f"Array shape: {data.shape}, dtype: {data.dtype}")
                
                # Handle datasets (tensors)
                else:
                    data = f[key][()]
                    batch_loaded[key] = process_loaded_data(data, device)
                    if verbose:
                        print(f"Loaded {key}")
            
            except Exception as e:
                print(f"Error loading {key}: {str(e)}")
                continue
    
    return batch_loaded

def visualize_inst_msk(image, *masks):
    """
    Visualize multiple instance masks overlaid on an image with different colors.
    
    Args:
        image: Base image of shape (C, H, W) or (H, W, C)
        *masks: Variable number of instance masks of shape (H, W)
    
    Returns:
        Visualization image with colored masks overlaid
    """
    
    # Denormalize image if needed (assuming values are normalized)
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Create base visualization
    vis_img = image.copy()
    
    # Define colors for different instances (RGB format)
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
    ]
    
    # Overlay each instance mask with a different color
    for idx, mask in enumerate(masks):
        # Convert mask to uint8 if needed
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
            
        # Create colored mask
        color = colors[idx % len(colors)]
        colored_mask = np.dstack([
            mask * color[0],
            mask * color[1],
            mask * color[2]
        ])
        
        # Overlay mask on image
        vis_img = cv2.addWeighted(vis_img, 0.7, colored_mask, 0.3, 0)
    
    return vis_img

def visualize_heatmap(heatmap, title=None):
    heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap_color = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
    if title is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(heatmap_color, title, (5, 15), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return heatmap_color

def create_multi_channel_heatmaps(heatmap_data, prefix=''):
    """
    Create individual heatmaps for each channel in the input data using visualize_heatmap.
    
    :param heatmap_data: numpy array of shape (C, H, W) where C is the number of channels
    :param prefix: prefix for the title of each heatmap (e.g., 'GT' or 'Pred')
    :return: list of RGB images, each as numpy array of shape (H, W, 3)
    """
    heatmaps = []
    for i, channel_data in enumerate(heatmap_data):
        heatmap = visualize_heatmap(channel_data)
        heatmaps.append(heatmap)
    
    return heatmaps

def overlay_heatmaps(image, heatmaps, alpha=0.5):
    """
    image: RGB画像 (H, W, 3)
    heatmaps: リストのカラーヒートマップ [(H, W, 3)] × 8
    alpha: ブレンディングの強さ (0-1)
    """
    overlay = image.copy()
    
    for heatmap in heatmaps:
        # RGB値の平均を取って強度マップを作成
        hm_intensity = np.mean(heatmap, axis=2)
        
        # 正規化して強度を保持
        hm_norm = (hm_intensity - hm_intensity.min()) / (hm_intensity.max() - hm_intensity.min() + 1e-8)
        mask = (hm_norm > 0.1)  # 閾値以上の領域のみを表示
        
        # 各ヒートマップの最大値を保持したまま重ね合わせ
        for c in range(3):
            overlay[:,:,c] = np.where(mask, 
                np.maximum(overlay[:,:,c], heatmap[:,:,c]),  # 最大値を取る
                overlay[:,:,c])
    
    return overlay

def blend_keypoints(image, corners=None, center=None, draw_flag:str='gt', text_mode:bool=False):
    """Corner and center plot

    Args:
        image (_type_): _description_
        corners (_type_): _description_
        center (_type_): _description_
        draw_flag (str, optional): _description_. Defaults to 'gt'.

    Returns:
        _type_: _description_
    """
    colors_hp = [(0, 0, 255), (0, 165, 255), (0, 255, 255),
                 (0, 128, 0), (255, 0, 0), (130, 0, 75), (238, 130, 238),
                 (0, 0, 0), (255, 255, 0), (255, 0, 0), (255, 0, 0)]
    # LA Dataset
    # edges = [[1, 2], [1, 4], [2, 3], [3, 4],
    #          [1, 5], [2, 6], [3, 7], [4, 8],
    #          [5, 6], [5, 8], [6, 7], [7, 8]]
    # x_cross = [[4, 7], [3, 8]]
    # z_cross = [[1, 3], [2, 4]]


    # Marker Dataset
    # edges = [[1, 2], [1, 4], [2, 3], [3, 4],
    #          [1, 5], [2, 6], [3, 7], [4, 8],
    #          [5, 6], [5, 8], [6, 7], [7, 8]]
    # x_cross = [[1, 8], [4, 5]] # +x
    # y_cross = [[1, 6], [2, 5]] # +y
    # z_cross = [[1, 3], [2, 4]]   # +z

    # # Objectron Dataset
    edges = [[2, 4], [2, 6], [6, 8], [4, 8],
             [1, 2], [3, 4], [5, 6], [7, 8],
             [1, 3], [1, 5], [3, 7], [5, 7]]
    
    # z_cross = [[3, 5], [1, 7]]   
    y_cross = [[3, 8], [4, 7]] 
    x_cross = [[5, 8], [6, 7]] 
    z_cross = [[2, 8], [4, 6]]
        
    num_corners = 8
    if corners is not None:
        corners = np.array(corners, dtype=np.int32).reshape(num_corners, 2)
        # Draw corners and numbers
        for j in range(num_corners):
            if not np.isnan(corners[j]).any():
                cv2.circle(image, (corners[j, 0], corners[j, 1]), 1, colors_hp[j], -1)
                # Add number near the corner
                number_pos = (corners[j, 0] + 15, corners[j, 1] + 15)  # Offset the number slightly
                if text_mode:
                    cv2.putText(image, str(j+1), number_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, str(j+1), number_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        # Set edge color based on draw_flag
        if draw_flag == 'pred':
            edge_color = (0, 0, 255)  # bgr
        elif draw_flag == 'gt':
            edge_color = (0, 255, 0)
        else:
            edge_color = (255, 0, 0)  # default to blue if unknown flag
        # Draw edges
        for idx, e in enumerate(edges + x_cross + y_cross + z_cross):
            temp = [e[0] - 1, e[1] - 1]
            if not np.isnan(corners[j]).any():
                if any(corners[temp[0]] <= -10000) or any(corners[temp[1]] <= -10000):
                    continue
                if idx < len(edges):
                    cv2.line(image, tuple(corners[temp[0]]), tuple(corners[temp[1]]), [255, 255, 255], 1, lineType=cv2.LINE_AA)
                elif len(edges) <= idx < len(edges) + len(x_cross):
                    cv2.line(image, tuple(corners[temp[0]]), tuple(corners[temp[1]]), [0, 0, 255], 1, lineType=cv2.LINE_AA)
                elif len(edges) + len(x_cross) <= idx < len(edges) + len(x_cross) + len(y_cross):
                    cv2.line(image, tuple(corners[temp[0]]), tuple(corners[temp[1]]), [0, 255, 0], 1, lineType=cv2.LINE_AA)
                elif len(edges) + len(x_cross) + len(y_cross) <= idx < len(edges) + len(x_cross) + len(y_cross) + len(z_cross):
                    cv2.line(image, tuple(corners[temp[0]]), tuple(corners[temp[1]]), [255, 0, 0], 1, lineType=cv2.LINE_AA)


    if center is not None:
        if center.ndim != 1:
            center = center[0]
        if 'float' in center.dtype.name:
            center = center.astype(np.int32)
        # Draw center
        cv2.circle(image, (center[0], center[1]), 1, colors_hp[num_corners], -1)
        
        if text_mode:
            cv2.putText(image, 'C', (center[0] + 15, center[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, 'C', (center[0] + 15, center[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)


    return image

def save_grid(*batches,
              normalize=False,
              save_path='grid.png',
              nrow=None,
              padding=2):
    """
    複数バッチを連結してグリッド画像を保存する汎用関数。

    Parameters
    ----------
    *batches : torch.Tensor
        各バッチは shape (N, C, H, W)。
    normalize : bool | Sequence[bool], default=False
        False          -> 全バッチ正規化なし
        True           -> 全バッチ min–max 正規化 (画像単位)
        Sequence[bool] -> バッチごとに指定
    save_path : str, default='grid.png'
        出力ファイル名
    nrow : int | None, default=None
        グリッド 1 行あたりの画像数 (None なら √N 切り上げ)
    padding : int, default=2
        画像間余白ピクセル
    """
    if len(batches) == 0:
        raise ValueError("少なくとも 1 つのバッチを渡してください")

    # -------- 1. normalize の指定を整形 --------
    if isinstance(normalize, bool):
        norm_flags = [normalize] * len(batches)
    else:
        if len(normalize) != len(batches):
            raise ValueError("normalize の長さがバッチ数と一致しません")
        norm_flags = list(normalize)

    processed = []

    for batch, do_norm in zip(batches, norm_flags):
        if batch.dim() != 4:
            raise ValueError("各バッチは (N,C,H,W) である必要があります")

        # GPU Tensor でも OK ― make_grid/save_image が .cpu() へ移動
        batch = batch.clone().float()        # 破壊的代入を避ける

        # --- データが 1ch → 3ch なら複製 ---
        if batch.shape[1] == 1:
            batch = batch.repeat(1, 3, 1, 1)

        # --- 画像単位の min–max 正規化 ---
        if do_norm:
            for i in range(batch.shape[0]):
                mn, mx = batch[i].min(), batch[i].max()
                batch[i] = (batch[i] - mn) / (mx - mn + 1e-6)

        processed.append(batch)

    # -------- 2. 連結 (M,C,H,W) --------
    combo = torch.cat(processed, dim=0)

    # -------- 3. グリッド化 --------
    total = combo.shape[0]
    if nrow is None:
        nrow = math.ceil(math.sqrt(total))

    grid = torchvision.utils.make_grid(
        combo, nrow=nrow, padding=padding
    )
    torchvision.utils.save_image(grid, save_path)

def _overlay_one(img: torch.Tensor,
                 mask: torch.Tensor,
                 alpha: float = .5) -> torch.Tensor:
    """
    img  : (3,H,W)   0–1 もしくは ‑1–1 正規化どちらでも可
    mask : (2,H,W)   0/1 の二値（0–1 の float も許容）

    0ch ⇒ 認識対象を「赤」, 1ch ⇒ その他を「緑」で重ねる
    """
    # ① 画像を 0–1 にクリップ（‑1–1 の場合は平均・分散を戻すなど好みで）
    if img.min() < 0:
        img = img * 0.5 + 0.5          # ‑1–1 → 0–1 とりあえず簡易に
    img = img.clamp(0, 1)

    # ② オーバーレイ用にコピーを作る
    over = img.clone()

    # ③ マスクを bool に
    trg   = mask[0] > 0.5             # 対象        → 赤
    other = mask[1] > 0.5             # その他物体  → 緑

    over[:, trg]   = torch.tensor([1., 0., 0.], device=img.device)[:, None]
    over[:, other] = torch.tensor([0., 1., 0.], device=img.device)[:, None]

    # ④ αブレンド
    return img * (1 - alpha) + over * alpha

def save_grid_with_mask(
    imgs : Iterable[np.ndarray] | Iterable[torch.Tensor],
    masks: Iterable[np.ndarray] | Iterable[torch.Tensor],
    save_path: str | Path,
    nrow: int = 2,
    alpha: float = .5,
    padding: int = 2
) -> Path:
    """
    画像と対応する 2ch マスクをオーバーレイして 1枚のグリッド画像に保存する。

    Parameters
    ----------
    imgs, masks : 同じ長さで、順番にペアになっていること
                  いずれも (C,H,W) or (2,H,W) 形式。np / torch どちらでも可
    """
    tensors: list[torch.Tensor] = []
    for img, msk in zip(imgs, masks):
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        if isinstance(msk, np.ndarray):
            msk = torch.from_numpy(msk)
        tensors.append(_overlay_one(img.float().cpu(), msk.float().cpu(), alpha))

    grid = torchvision.utils.make_grid(torch.stack(tensors), nrow=nrow, padding=padding)
    torchvision.utils.save_image(grid, str(save_path))
    return Path(save_path)