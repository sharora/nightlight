import numpy as np
from PIL import Image


def loadOCfromPath(path):
    """Loads an occupancy grid array from an image, image must have 255 = empty
    and 0 = occupied for this method to work

    Args:
    path : a string representing the relative path to the image

    Returns:
        A normalized numpy array loaded from the given imageCoordinates
    """
    return 1 - np.array(Image.open(path))/255.0

