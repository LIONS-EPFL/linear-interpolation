from torch import nn


class Generator(nn.Module):
    noise_dim = None
