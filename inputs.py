import os
import time

import nibabel as nib
import numpy as np
import torch
import torch.utils.data as data
from skimage.exposure import adjust_gamma, adjust_sigmoid
from skimage.filters import sobel_h
from skimage.transform import rotate, rescale


def _get_boundary(im):
    """Find the upper and lower boundary between body and background."""
    edge_sobel = sobel_h(im)
    threshold = np.max(edge_sobel) / 20
    top, bottom = np.where(edge_sobel > threshold)[0][[0, -1]]  # index arrays
    return top, bottom


def _banish_darkness(xs, ys):
    """Clip black background region from nii raw data along y-axis, to alleviate computations.
    Argument:
        A tuple consists (image_3d, label_3d)
            image_3d: int16 with shape [depth, height, width]
            label_3d: uint8 with shape [depth, height, width]
    Return:
        tuples of (images, labels, top, bottom).
        + images: [depth, reduced_height, width]
        + labels: [depth, reduced_height, width]
        + top: upper boundary
        + bottom: lower boundary
    """

    boundaries = np.array([_get_boundary(im) for im in xs])
    t, b = np.mean(boundaries, axis=0).astype(np.uint8)
    # Empirically the lower boundary is more robust.
    if (b - t) < 180:
        t = b - 180
    return xs[:, t: b, :], ys[:, t: b, :], t, b


def _augment(xs):
    """Image adjustment doesn't change image shape, but for intensity.
    Return:
        images: 4-d tensor with shape [depth, height, width, channels]
    """

    # `xs` has shape [depth, height, width] with value in [0, 1].
    gamma = np.random.uniform(low=0.9, high=1.1)
    return adjust_gamma(xs, gamma)


def _rotate_and_rescale(xs, ys):
    """Rotate images and labels and scale image and labels by a certain factor.
    Both need to swap axis from [depth, height, width] to [height, width, depth]
    required by skimage.transform library.
    """

    degree = np.int(np.random.uniform(low=-3, high=5))
    factor = np.random.uniform(low=0.85, high=0.95)
    # swap axis
    HWC_xs, HWC_ys = [np.transpose(item, [1, 2, 0]) for item in [xs, ys]]
    # rotate and rescale
    HWC_xs, HWC_ys = [rotate(item, degree, mode='symmetric', preserve_range=True) for item in [HWC_xs, HWC_ys]]
    HWC_xs, HWC_ys = [rescale(item, factor, mode='symmetric', preserve_range=True) for item in [HWC_xs, HWC_ys]]
    # swap back
    xs, ys = [np.transpose(item, [2, 0, 1]) for item in [HWC_xs, HWC_ys]]
    return xs, ys


def _translate(xs, ys):
    """Perform translate, and the displacement is skewed to 0.
        In detail, take samples from the modified power function distribution.
    """

    samples = np.random.power(5, size=4)  # samples now in range [0, 1]
    skewed_samples = np.int8((- samples + 2) * 15)  # skewed_samples in range [1, 16]
    r1, c1, r2, c2 = skewed_samples  # discard 0 for indexing `-0`
    trans_xs, trans_ys = [item[:, r1: -r2, c1: -c2] for item in [xs, ys]]
    return trans_xs, trans_ys


class DatasetFromFolder(data.Dataset):
    def __init__(self, data_path='./data/train'):
        """Assume dataset is in directory '.data/train' or './data/val'
            1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
            2. Preprocess the data (e.g. torchvision.Transform).
            3. Return a data pair (e.g. image and label).
        """
        super(DatasetFromFolder, self).__init__()
        self.data_path = data_path
        self.label_path = [os.path.join(self.data_path, p)
                           for p in os.listdir(self.data_path) if p.endswith('Label.nii')]
        self.image_path = [p.replace('_Label', '') for p in self.label_path]

    def __getitem__(self, index):
        # Set random seed for ramdom augment.
        np.random.seed(int(time.time()))

        # Load nii file.
        xs, ys = [nib.load(p[index]).get_data() for p in [self.image_path, self.label_path]]

        # Crop black region to reduce nii volumes.
        xs, ys, *_ = _banish_darkness(xs, ys)

        # Normalize, `xs` with dtype float64
        xs = (xs - np.min(xs)) / (np.max(xs) - np.min(xs))

        # Image augment.
        xs, ys = _rotate_and_rescale(xs, ys)
        xs, ys = _translate(xs, ys)
        xs = _augment(xs)

        # Regenerate the binary label, just in case.
        ys = (ys > 0.5).astype(np.uint8)

        # Add gray image channel, with shape [1, depth, height, width]
        xs, ys = [item[np.newaxis, ...] for item in [xs, ys]]
        return torch.from_numpy(xs), torch.from_numpy(ys)

    def __len__(self):
        return len(self.image_path)
