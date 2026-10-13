import numpy as np


def xywh2xwxy(boxes: np.ndarray) -> np.ndarray:
    """
    Convert boxes from xywh format to x1y1x2y2 format in a numpy array.
    Args:
        boxes (np.ndarray): The boxes in xywh format.
    Returns:
        np.ndarray: The boxes in x1y1x2y2 format.
    """
    if not isinstance(boxes, np.ndarray):
        boxes = np.array(boxes)
    boxes = boxes.copy()
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]
    return boxes


def resize_boxes(
    boxes: np.ndarray, new_size: tuple[int, int], original_size: tuple[int, int]
) -> np.ndarray:
    """
    Resize boxes from one size to another.
    Args:
        boxes (np.ndarray): The boxes in x1y1x2y2 format.
        new_size (tuple[int, int]): The new size of the image. Order: (width, height).
        original_size (tuple[int, int]): The original size of the image. Order: (width, height).
    Returns:
        np.ndarray: The resized boxes.
    """
    boxes = boxes.copy()
    boxes[:, 0] = boxes[:, 0] * new_size[0] / original_size[0]
    boxes[:, 1] = boxes[:, 1] * new_size[1] / original_size[1]
    boxes[:, 2] = boxes[:, 2] * new_size[0] / original_size[0]
    boxes[:, 3] = boxes[:, 3] * new_size[1] / original_size[1]
    return boxes


def normalize_boxes(boxes: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """
    Normalize the boxes coordinates.
    Args:
        boxes (np.ndarray): The boxes in x1y1x2y2 format.
        image_size (tuple[int, int]): The size of the image. Order: (width, height).
    Returns:
        np.ndarray: The normalized boxes in range [0, 1].
    """
    boxes = boxes.copy()
    boxes[:, 0] = boxes[:, 0] / image_size[0]
    boxes[:, 1] = boxes[:, 1] / image_size[1]
    boxes[:, 2] = boxes[:, 2] / image_size[0]
    boxes[:, 3] = boxes[:, 3] / image_size[1]
    return boxes
