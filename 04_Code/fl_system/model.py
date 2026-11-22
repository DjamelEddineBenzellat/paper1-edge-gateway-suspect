"""Paper 1 models.

ECG model (PR:66): 4 conv blocks (12->32->32->64->64 channels, kernel 7,
GroupNorm with 8 groups, ReLU, max-pool 2) + global average pooling +
linear 64->5. GroupNorm (not BatchNorm) because DP-SGD needs per-example
gradients. 53,765 parameters.
"""
import torch.nn as nn

N_LEADS = 12
N_CLASSES = 5
ECG_PARAMETERS = 53_765


def _block(c_in, c_out):
    return [nn.Conv1d(c_in, c_out, kernel_size=7, padding=3), nn.GroupNorm(8, c_out), nn.ReLU(),
            nn.MaxPool1d(2)]


def ecg_cnn():
    return nn.Sequential(*_block(N_LEADS, 32), *_block(32, 32), *_block(32, 64), *_block(64, 64),
                         nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, N_CLASSES))


def n_parameters(model):
    return sum(p.numel() for p in model.parameters())
