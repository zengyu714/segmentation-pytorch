import os
import torch

from torch.nn import init

# Training configuration
class config:
    def __init__(self):
        self.cuda = torch.cuda.is_available()
        self.batch_size = 1
        self.epochs = 100
        self.augment_size = 500
        self.learning_rate = 3e-5
        self.threads = 24
        self.seed = 714
        self.from_scratch = False
        self.checkpoint_dir = './checkpoints/'
        self.resume_step = -1
        self.prefix = 'May force be with you.'

def weights_init(net):
    for m in net.modules():
        if isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.ConvTranspose3d):
            init.kaiming_normal(m.weight)
            init.constant(m.bias, 0.1)

def dice_loss(y_true, y_conv):
    """Compute dice among **positive** labels to avoid unbalance.
    Argument:
        y_true: [batch_size * 1 * depth * height * width]
        y_conv: [batch_size * depth * height * width, 2]
    """
    y_conv = y_conv[:, 1]

    intersection = torch.sum(y_conv * y_true, 0)
    # `dim = 0` for Tensor result
    union = torch.sum(y_conv * y_conv, 0) + torch.sum(y_true * y_true, 0)
    dice = 2.0 * intersection / union
    return 1 - torch.clamp(dice, 0.0, 1.0 - 1e-7)
