import torch
import torch.nn.functional as F


def define_model_loss(config):
    """returns dis/gen loss functions based on the model"""
    if config.loss == 'vangan':
        return vangan_loss_dis, vangan_loss_gen
    elif config.loss == 'hinge':
        return hinge_loss_dis, hinge_loss_gen
    elif config.loss in ['wgan', 'wgan_gp']:
        return wgan_loss_dis, wgan_loss_gen
    else:
        raise NotImplementedError('%s model is not implemented!' % config.loss)


def vangan_loss_dis(x_real, x_fake, netD, device=None):
    p_real, p_gen = netD(x_real), netD(x_fake)
    dis_loss = F.softplus(-p_real).mean() + F.softplus(p_gen).mean()
    return dis_loss, p_real, p_gen


def vangan_loss_gen(x_fake, netD, device=None):
    p_gen = netD(x_fake)
    gen_loss = F.softplus(-p_gen).mean()
    return gen_loss, p_gen


def wgan_loss_gen(x_fake, netD, device=None):
    score_gen = netD(x_fake)

    gen_loss = -score_gen.mean()
    return gen_loss, score_gen


def wgan_loss_dis(x_real, x_fake, netD, device=None):
    score_real, score_gen = netD(x_real), netD(x_fake)

    dis_loss = score_gen.mean() - score_real.mean()
    return dis_loss, score_real, score_gen


def hinge_loss_gen(x_fake, netD, device=None):
    score_gen = netD(x_fake)

    gen_loss = -score_gen.mean()
    return gen_loss, score_gen


def hinge_loss_dis(x_real, x_fake, netD, device=None):
    score_real, score_gen = netD(x_real), netD(x_fake)

    real_loss = torch.nn.ReLU()(1.0 - score_real).mean()
    fake_loss = torch.nn.ReLU()(1.0 + score_gen).mean()
    dis_loss = fake_loss + real_loss
    return dis_loss, score_real, score_gen
