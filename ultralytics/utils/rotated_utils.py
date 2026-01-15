import cv2
import numpy as np


def vertices_to_rotated_box(vertices):
    """
    Converts 4 vertex coordinates to rotated bounding box parameters
    
    Args:
        vertices: Array of shape (4, 2) containing [x1,y1, x2,y2, x3,y3, x4,y4]
    
    Returns:
        Tuple (x, y, w, h, θ) where:
        - (x, y): center coordinates
        - w, h: box width and height
        - θ: rotation angle in radians [0, π)
    """
    # Convert to integer points for OpenCV
    pts = np.array(vertices, dtype=np.float32)
    
    # Get minimum area rectangle
    rect = cv2.minAreaRect(pts)
    (x, y), (w, h), angle = rect
    
    # Convert angle to [0, π) range
    angle = np.deg2rad(angle)
    if angle < 0:
        angle += np.pi
    
    return x, y, w, h, angle


def rotated_box_to_vertices(x, y, w, h, angle):
    """
    Converts rotated box parameters back to 4 vertex coordinates
    
    Args:
        x, y: center coordinates
        w, h: box width and height
        angle: rotation angle in radians
    
    Returns:
        Array of shape (4, 2) containing vertex coordinates
    """
    # Create rotation matrix
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    
    # Get corner points relative to center
    corners = np.array([[-w/2, -h/2],
                        [ w/2, -h/2],
                        [ w/2,  h/2],
                        [-w/2,  h/2]])
    
    # Rotate and translate
    rotated_corners = np.dot(corners, R.T) + np.array([x, y])
    return rotated_corners