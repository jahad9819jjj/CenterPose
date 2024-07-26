import numpy as np
import torch
from lib.utils.image import affine_transform

# Group action
def rotation_y_matrix(theta:float)->np.ndarray:
    M_R = np.array([[np.cos(theta), 0, np.sin(theta), 0],
                    [0, 1, 0, 0],
                    [-np.sin(theta), 0, np.cos(theta), 0], [0, 0, 0, 1]])
    return M_R

def bounding_box(points):
    x_coordinates, y_coordinates = zip(*points)
    return [min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates)]


def bounding_box_rotation(points, trans):
    coordinates_transformed = []
    for x, y, _ in points:
        coordinates_transformed.append(affine_transform([x, y], trans))

    return bounding_box(coordinates_transformed)

# Conversion
def torch2cv(inp:torch.tensor)->np.ndarray:
    image = (inp.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.int32)
    return image