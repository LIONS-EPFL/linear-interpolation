import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable, grad

from rapp.models.discriminator import Discriminator


_H_FILTERS = 64
channels = 3
leak = 0.1
w_g = 4


class Generator32(nn.Module):
    def __init__(self, z_dim, h_filters=_H_FILTERS):
        super(Generator32, self).__init__()
        self.z_dim = z_dim

        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 8*h_filters, 4, stride=1),
            nn.BatchNorm2d(8*h_filters),
            nn.ReLU(),
            nn.ConvTranspose2d(8*h_filters, 4*h_filters, 4, stride=2, padding=(1,1)),
            nn.BatchNorm2d(4*h_filters),
            nn.ReLU(),
            nn.ConvTranspose2d(4*h_filters, 2*h_filters, 4, stride=2, padding=(1,1)),
            nn.BatchNorm2d(2*h_filters),
            nn.ReLU(),
            nn.ConvTranspose2d(2*h_filters, h_filters, 4, stride=2, padding=(1,1)),
            nn.BatchNorm2d(h_filters),
            nn.ReLU(),
            nn.ConvTranspose2d(h_filters, channels, 3, stride=1, padding=(1,1)),
            nn.Tanh())

    def forward(self, z):
        fake = self.model(z.view(-1, self.z_dim, 1, 1))

        # Important for torch-fidelity
        if not self.training:
            fake = (255 * (fake.clamp(-1, 1) * 0.5 + 0.5))
            fake = fake.to(torch.uint8)
        return fake
    
    def load(self, model):
      self.load_state_dict(model.state_dict(), strict=False)
    

# model = Dis(power=100)
# model(input)
# model_new = Dis(Power=0)
# model_new.load(model)

class Discriminator32(nn.Module):
    def __init__(self, spectral_norm=False, h_filters=_H_FILTERS):
        super(Discriminator32, self).__init__()
        self.h_filters = h_filters
        _apply_sn = lambda x: nn.utils.spectral_norm(x, name='weight') if spectral_norm else x

        self.conv1 = _apply_sn(nn.Conv2d(channels, h_filters, 3, stride=1, padding=(1,1)))

        self.conv2 = _apply_sn(nn.Conv2d(h_filters, h_filters, 4, stride=2, padding=(1,1)))
        self.conv3 = _apply_sn(nn.Conv2d(h_filters, 2*h_filters, 3, stride=1, padding=(1,1)))
        self.conv4 = _apply_sn(nn.Conv2d(2*h_filters, 2*h_filters, 4, stride=2, padding=(1,1)))
        self.conv5 = _apply_sn(nn.Conv2d(2*h_filters, 4*h_filters, 3, stride=1, padding=(1,1)))
        self.conv6 = _apply_sn(nn.Conv2d(4*h_filters, 4*h_filters, 4, stride=2, padding=(1,1)))
        self.conv7 = _apply_sn(nn.Conv2d(4*h_filters, 8*h_filters, 3, stride=1, padding=(1,1)))

        self.fc = _apply_sn(nn.Linear(w_g * w_g * 8*h_filters, 1))

    def forward(self, x):
        m = x
        m = nn.LeakyReLU(leak)(self.conv1(m))
        m = nn.LeakyReLU(leak)(self.conv2(m))
        m = nn.LeakyReLU(leak)(self.conv3(m))
        m = nn.LeakyReLU(leak)(self.conv4(m))
        m = nn.LeakyReLU(leak)(self.conv5(m))
        m = nn.LeakyReLU(leak)(self.conv6(m))
        m = nn.LeakyReLU(leak)(self.conv7(m))

        return self.fc(m.view(-1,w_g * w_g * 8*self.h_filters))

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
    
    def load(self, model):
      self.load_state_dict(model.state_dict(), strict=False)