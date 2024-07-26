import cv2
import numpy as np

def blend_keypoints(image, corners, center, draw_flag:str='gt'):
    """Corner and center plot

    Args:
        image (_type_): _description_
        corners (_type_): _description_
        center (_type_): _description_
        draw_flag (str, optional): _description_. Defaults to 'gt'.

    Returns:
        _type_: _description_
    """
    canvas = image.copy()
    num_corners = 8
    colors_hp = [(0, 0, 255), (0, 165, 255), (0, 255, 255),
                 (0, 128, 0), (255, 0, 0), (130, 0, 75), (238, 130, 238),
                 (0, 0, 0), (255, 255, 0), (255, 0, 0), (255, 0, 0)]
    edges = [[2, 4], [2, 6], [6, 8], [4, 8],
             [1, 2], [3, 4], [5, 6], [7, 8],
             [1, 3], [1, 5], [3, 7], [5, 7]]
    front_cross = [[2, 8], [4, 6]]
    top_cross = [[3, 8], [4, 7]]
    corners = np.array(corners, dtype=np.int32).reshape(num_corners, 2)
    
    # Draw corners and numbers
    for j in range(num_corners):
        cv2.circle(canvas, (corners[j, 0], corners[j, 1]), 10, colors_hp[j], -1)
        # Add number near the corner
        number_pos = (corners[j, 0] + 15, corners[j, 1] + 15)  # Offset the number slightly
        cv2.putText(canvas, str(j+1), number_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, str(j+1), number_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw center
    cv2.circle(canvas, (center[0], center[1]), 10, colors_hp[num_corners], -1)
    cv2.putText(canvas, 'C', (center[0] + 15, center[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, 'C', (center[0] + 15, center[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    # Set edge color based on draw_flag
    if draw_flag == 'pred':
        edge_color = (0, 0, 255)  # bgr
    elif draw_flag == 'gt':
        edge_color = (0, 255, 0)
    else:
        edge_color = (255, 0, 0)  # default to blue if unknown flag

    # Draw edges
    for e in edges + front_cross + top_cross:
        temp = [e[0] - 1, e[1] - 1]
        if any(corners[temp[0]] <= -10000) or any(corners[temp[1]] <= -10000):
            continue
        cv2.line(canvas, tuple(corners[temp[0]]), tuple(corners[temp[1]]), edge_color, 2, lineType=cv2.LINE_AA)

    return canvas

def visualize_corner_displacements(image, corners, displacements, draw_flag:str='gt'):
    """Corner displacement

    Args:
        image (_type_): _description_
        corners (_type_): _description_
        displacements (_type_): _description_
        draw_flag (str, optional): _description_. Defaults to 'gt'.

    Returns:
        _type_: _description_
    """
    canvas = image.copy()
    num_corners = 8
    colors_hp = [(0, 0, 255), (0, 165, 255), (0, 255, 255),
                 (0, 128, 0), (255, 0, 0), (130, 0, 75), (238, 130, 238),
                 (0, 0, 0), (255, 255, 0), (255, 0, 0), (255, 0, 0)]
    
    corners = np.array(corners, dtype=np.int32).reshape(num_corners, 2)
    displacements = np.array(displacements, dtype=np.float32).reshape(num_corners, 2)

    # Draw corners, numbers, and displacement arrows
    for j in range(num_corners):
        # Draw corner
        cv2.circle(canvas, (corners[j, 0], corners[j, 1]), 10, colors_hp[j], -1)
        
        # Add number near the corner
        number_pos = (corners[j, 0] + 15, corners[j, 1] + 15)  # Offset the number slightly
        cv2.putText(canvas, str(j+1), number_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, str(j+1), number_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw displacement arrow
        end_point = (int(corners[j, 0] + displacements[j, 0]), int(corners[j, 1] + displacements[j, 1]))
        cv2.arrowedLine(canvas, tuple(corners[j]), end_point, colors_hp[j], 5, tipLength=0.2)

    return canvas
            
def add_axes(image, M_c2o, cam_projection_matrix, vector_len_pixel, height, width):
    """ 6DoF Plot

    Args:
        image (_type_): _description_
        M_c2o (_type_): _description_
        cam_projection_matrix (_type_): _description_
        vector_len_pixel (_type_): _description_
        height (_type_): _description_
        width (_type_): _description_

    Returns:
        _type_: _description_
    """
    canvas = image.copy()
    
    # Invert M_c2o to get M_o2c
    M_o2c = np.linalg.inv(M_c2o)
    
    # Define unit vectors for each axis in object coordinate system
    dir_X = np.array([1, 0, 0, 1])
    dir_Y = np.array([0, 1, 0, 1])
    dir_Z = np.array([0, 0, 1, 1])

    # Transform the origin and axis endpoints to camera coordinate system
    origin = M_o2c @ np.array([0, 0, 0, 1])
    end_X = M_o2c @ dir_X
    end_Y = M_o2c @ dir_Y
    end_Z = M_o2c @ dir_Z

    # Project 3D points to 2D using cam_projection_matrix
    origin_2d = cam_projection_matrix @ origin
    end_X_2d = cam_projection_matrix @ end_X
    end_Y_2d = cam_projection_matrix @ end_Y
    end_Z_2d = cam_projection_matrix @ end_Z

    # Normalize homogeneous coordinates
    origin_2d = origin_2d[:2] / origin_2d[2]
    end_X_2d = end_X_2d[:2] / end_X_2d[2]
    end_Y_2d = end_Y_2d[:2] / end_Y_2d[2]
    end_Z_2d = end_Z_2d[:2] / end_Z_2d[2]

    # Map to viewport coordinates
    origin_2d = (origin_2d + 1.0) / 2.0 * np.array([width, height])
    end_X_2d = (end_X_2d + 1.0) / 2.0 * np.array([width, height])
    end_Y_2d = (end_Y_2d + 1.0) / 2.0 * np.array([width, height])
    end_Z_2d = (end_Z_2d + 1.0) / 2.0 * np.array([width, height])

    # Calculate directions in 2D
    dir_X_2d = end_X_2d - origin_2d
    dir_Y_2d = end_Y_2d - origin_2d
    dir_Z_2d = end_Z_2d - origin_2d

    # Normalize and scale directions
    dir_X_2d = dir_X_2d / np.linalg.norm(dir_X_2d) * vector_len_pixel
    dir_Y_2d = dir_Y_2d / np.linalg.norm(dir_Y_2d) * vector_len_pixel
    dir_Z_2d = dir_Z_2d / np.linalg.norm(dir_Z_2d) * vector_len_pixel

    # Calculate end points for drawing
    end_X = origin_2d + dir_X_2d
    end_Y = origin_2d + dir_Y_2d
    end_Z = origin_2d + dir_Z_2d

    # Convert to integer coordinates for drawing
    origin = tuple(map(int, origin_2d))
    end_X = tuple(map(int, end_X))
    end_Y = tuple(map(int, end_Y))
    end_Z = tuple(map(int, end_Z))

    # Draw arrows
    cv2.arrowedLine(canvas, origin, end_X, (0, 255, 0), 2, tipLength=0.2)  # Red for X-axis
    cv2.arrowedLine(canvas, origin, end_Y, (255, 0, 0), 2, tipLength=0.2)  # Green for Y-axis
    cv2.arrowedLine(canvas, origin, end_Z, (0, 0, 255), 2, tipLength=0.2)  # Blue for Z-axis

    return canvas
