# ResNet generator and discriminator
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable, grad

import numpy as np


channels = 3

class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            self.conv2
            )
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Upsample(scale_factor=2)

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, spectral_norm=False):
        super(ResBlockDiscriminator, self).__init__()

        _apply_sn = lambda x: nn.utils.spectral_norm(x) if spectral_norm else x
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)

        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(),
                _apply_sn(self.conv1),
                nn.ReLU(),
                _apply_sn(self.conv2)
                )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                _apply_sn(self.conv1),
                nn.ReLU(),
                _apply_sn(self.conv2),
                nn.AvgPool2d(2, stride=stride, padding=0)
                )
        self.bypass = nn.Sequential()
        if stride != 1:

            self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))

            self.bypass = nn.Sequential(
                _apply_sn(self.bypass_conv),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
            # if in_channels == out_channels:
            #     self.bypass = nn.AvgPool2d(2, stride=stride, padding=0)
            # else:
            #     self.bypass = nn.Sequential(
            #         SpectralNorm(nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)),
            #         nn.AvgPool2d(2, stride=stride, padding=0)
            #     )


    def forward(self, x):
        return self.model(x) + self.bypass(x)

# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, spectral_norm=False):
        super(FirstResBlockDiscriminator, self).__init__()

        _apply_sn = lambda x: nn.utils.spectral_norm(x) if spectral_norm else x
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            _apply_sn(self.conv1),
            nn.ReLU(),
            _apply_sn(self.conv2),
            nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            _apply_sn(self.bypass_conv),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class ResNetGenerator(nn.Module):
    def __init__(self, z_dim, size=128):
        super(ResNetGenerator, self).__init__()
        self.z_dim = z_dim
        self.size = size

        self.dense = nn.Linear(self.z_dim, 4 * 4 * self.size)
        self.final = nn.Conv2d(self.size, channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform(self.dense.weight.data, 1.)
        nn.init.xavier_uniform(self.final.weight.data, 1.)

        self.model = nn.Sequential(
            ResBlockGenerator(self.size, self.size, stride=2),
            ResBlockGenerator(self.size, self.size, stride=2),
            ResBlockGenerator(self.size, self.size, stride=2),
            nn.BatchNorm2d(self.size),
            nn.ReLU(),
            self.final,
            nn.Tanh())

    def forward(self, z):
        fake = self.model(self.dense(z).view(-1, self.size, 4, 4))
        
        # Important for torch-fidelity
        if not self.training:
            fake = (255 * (fake.clamp(-1, 1) * 0.5 + 0.5))
            fake = fake.to(torch.uint8)
        return fake
    
    def load(self, model):
      self.load_state_dict(model.state_dict(), strict=False)
      

class ResNetDiscriminator(nn.Module):
    def __init__(self, spectral_norm=False, size=128):
        super(ResNetDiscriminator, self).__init__()
        self.size = size
        self.spectral_norm = spectral_norm

        _apply_sn = lambda x: nn.utils.spectral_norm(x) if spectral_norm else x
        self.model = nn.Sequential(
                FirstResBlockDiscriminator(channels, self.size, stride=2, spectral_norm=spectral_norm),
                ResBlockDiscriminator(self.size, self.size, stride=2, spectral_norm=spectral_norm),
                ResBlockDiscriminator(self.size, self.size, spectral_norm=spectral_norm),
                ResBlockDiscriminator(self.size, self.size, spectral_norm=spectral_norm),
                nn.ReLU(),
                nn.AvgPool2d(8),
            )
        self.fc = nn.Linear(self.size, 1)
        nn.init.xavier_uniform(self.fc.weight.data, 1.)
        self.fc = _apply_sn(self.fc)

    def forward(self, x):
        return self.fc(self.model(x).view(-1,self.size))

    def load(self, model):
      self.load_state_dict(model.state_dict(), strict=False)

    def get_penalty(self, x_true, x_gen, mode="linear"):
        x_true = x_true.view_as(x_gen)
        if mode ==  "linear":
            alpha = torch.rand((len(x_true),)+(1,)*(x_true.dim()-1))
            if x_true.is_cuda:
                alpha = alpha.cuda(x_true.get_device())
            x_penalty = alpha*x_true + (1-alpha)*x_gen
        elif mode == "gen":
            x_penalty = x_gen.clone()
        elif mode == "data":
            x_penalty = x_true.clone()
        x_penalty.requires_grad_()
        p_penalty = self.forward(x_penalty)
        gradients = grad(p_penalty, x_penalty, grad_outputs=torch.ones_like(p_penalty).cuda(x_true.get_device()) if x_true.is_cuda else torch.ones_like(p_penalty), create_graph=True, retain_graph=True, only_inputs=True)[0]
        penalty = ((gradients.view(len(x_true), -1).norm(2, 1) - 1)**2).mean()
        return penalty
