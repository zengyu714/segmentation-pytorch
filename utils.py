import os
import torch
import numpy as np

from torch.nn import init

# Training configuration
class config:
    def __init__(self):
        self.cuda = torch.cuda.is_available()
        self.batch_size = 1
        self.epochs = 100
        self.augment_size = 500
        self.training_size = 12
        self.val_size = 3
        self.learning_rate = 3e-6
        self.criterion = 'dice'
        self.seed = 714
        self.threads = 24
        self.from_scratch = False
        self.checkpoint_dir = './checkpoints/'
        self.result_dir = './results/'
        self.resume_step = -1
        self.prefix = 'May force be with you.'

def weights_init(net):
    for m in net.modules():
        if isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.ConvTranspose3d):
            init.kaiming_normal(m.weight)
            init.constant(m.bias, 0.01)

def dice_loss(y_conv, y_true):
    """Compute dice among **positive** labels to avoid unbalance.
    Argument:
        y_true: [batch_size * depth * height * width, (1)] (torch.cuda.LongTensor)
        y_conv: [batch_size * depth * height * width,  2 ] (torch.cuda.FloatTensor)
    """
    y_conv = y_conv[:, 1]
    y_true = y_true.float()
    intersection = torch.sum(y_conv * y_true, 0)

    # `dim = 0` for Tensor result
    union = torch.sum(y_conv * y_conv, 0) + torch.sum(y_true * y_true, 0)
    dice = 2.0 * intersection / union
    return 1 - torch.clamp(dice, 0.0, 1.0 - 1e-7)

import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'

def show_center_slices(im_3d, indices=None):
    """Function to display slices of 3-d image """

    if indices is None:
        indices = np.array(im_3d.shape) // 2
    assert len(indices) == 3, 'Except 3-d array, but receive %d-d array indexing.' % len(indices)

    x_th, y_th, z_th = indices
    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(im_3d[x_th, :, :])
    axes[1].imshow(im_3d[:, y_th, :])
    axes[2].imshow(im_3d[:, :, z_th])
    plt.suptitle('Center slices for spine image')
