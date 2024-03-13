import numpy as np


class ToFloatMNIST(object):
    """Convert ndarrays in sample values to integers."""

    def __call__(self, image):
        return (image / 255).astype('f4')


class ToFloatUKBB(object):
    """Convert ndarrays in sample values to integers."""

    def __call__(self, image):
        try:
            image = image.astype('f8')
            maxv = np.max(image)
            minv = np.min(image)
            return ((image - minv) / maxv).astype('f4')
        except:
            return image
