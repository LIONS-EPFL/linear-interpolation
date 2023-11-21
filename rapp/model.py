import copy
import math
import numpy as np
import torch
import torchvision
from pytorch_lightning import LightningModule
import wandb
import torch_fidelity
from torch.optim.lr_scheduler import MultiStepLR

from rapp.args import Args
from rapp.optim.RAPP import RAPP, RAPPsgd
from rapp.optim.lookahead import Lookahead, LA, AdamLA, ExtraSGDLA, ExtraAdamLA, EGplusLA, EGplusAdamLA
from rapp.optim.extragradient import Extragradient, ExtraSGD, ExtraAdam
from rapp.optim.EGplus import EGplus, EGplusSGD, EGplusAdam
from rapp.losses import define_model_loss
from rapp.models.utils import tensorify, tensorify_grad, weight_clipping


NETD_IDX = 0
NETG_IDX = 1
    
    
class OSP_Model(LightningModule):
    def __init__(self, netG, netD, args: Args):
        super().__init__()
        self.automatic_optimization = False

        self.config = args
        config = args
        self.netD = netD
        self.netG = netG
        self.steps = 0
        if config.opt in ['EG', 'EA', 'EGplus', 'EAplus', 'LA-EGplus', 'LA-EAplus',
                            'LA', 'LA-Adam', 'LA-EG', 'LA-EA']:
            self.extrapolate = True
            self.netD_ea, self.netG_ea = copy.deepcopy(netD), copy.deepcopy(netG)
        else:
            self.extrapolate = False
            self.netD_ea, self.netG_ea = netD, netG
        
        self.inner_steps = config.inner_steps
        if config.opt in ['LA-EG', 'LA-EA']:
            self.inner_steps = self.inner_steps*2
        self.lam = config.lam
        if self.lam<0 and self.lam>1:
            raise RuntimeError("lam is invalid.")
        self.bz = config.nz
            
        # Load potential checkpoint before making copies
        if config.ckpt_D is not None:
            print('loading D models from %s' % config.ckpt_D)
            ckpt = torch.load(config.ckpt_D)['state_dict']
            state = {k.replace('netD.', ''):v for k,v in ckpt.items() if k.startswith('netD.')}
            self.netD.load_state_dict(state)
        
        if config.ckpt_G is not None:
            print('loading G models from %s' % config.ckpt_G)
            ckpt = torch.load(config.ckpt_G)['state_dict']
            state = {k.replace('netG.', ''):v for k,v in ckpt.items() if k.startswith('netG.')}
            self.netG.load_state_dict(state)

        self.validation_z = self._sample_noise(64)

        model_loss_dis, model_loss_gen = define_model_loss(config)
        self.model_loss_dis = model_loss_dis
        self.model_loss_gen = model_loss_gen

        # eval models that should not be saved (can't be top level)
        self.init_nets = {
            'netD': copy.deepcopy(netD),
            'netG': copy.deepcopy(netG),
        }
        
    def configure_optimizers(self):
        # setup optimizer
        config = self.config
        netD, netG = self.netD, self.netG
        netD_ea, netG_ea = self.netD_ea, self.netG_ea
        self.gen_has_extrapolated = False     
        self.local_steps = 0
        opt_D_ea = torch.optim.SGD(netD_ea.parameters(), lr=config.lrD)
        opt_G_ea = torch.optim.SGD(netG_ea.parameters(), lr=config.lrG)
        
        opt_D = self._config_optim(config.opt, netD, config.lrD)
        opt_G = self._config_optim(config.opt, netG, config.lrG)
        if self.extrapolate:
            opt_D_ea = self._config_optim(config.opt, netD_ea, config.lrD)
            opt_G_ea = self._config_optim(config.opt, netG_ea, config.lrG)
        
        sch_D = {'scheduler': MultiStepLR(opt_D, milestones=[600,1000], gamma=0.5), 'interval': 'epoch'}
        sch_D_ea = {'scheduler': MultiStepLR(opt_D_ea, milestones=[600,1000], gamma=0.5), 'interval': 'epoch'}
        sch_G = {'scheduler': MultiStepLR(opt_G, milestones=[600,1000], gamma=0.5), 'interval': 'epoch'}
        sch_G_ea = {'scheduler': MultiStepLR(opt_G_ea, milestones=[600,1000], gamma=0.5), 'interval': 'epoch'}
        return [opt_G, opt_D, opt_G_ea, opt_D_ea], [sch_G, sch_D, sch_G_ea, sch_D_ea]

    def _config_optim(self, opt, net, lr):
        if opt == 'RAPP':
            optim = RAPPsgd(net.parameters(), lr=lr)
        elif opt == 'LA':
            optim = LA(net.parameters(), lr=lr)
        elif opt == 'LA-Adam':
            optim = AdamLA(net.parameters(), lr=lr, betas=(self.config.adam1, self.config.adam2))
        elif opt == 'LA-EG':
            optim = ExtraSGDLA(net.parameters(), lr=lr)
        elif opt == 'LA-EA':
            optim = ExtraAdamLA(net.parameters(), lr=lr, betas=(self.config.adam1, self.config.adam2))
        elif opt == 'GDA':
            optim = torch.optim.SGD(net.parameters(), lr=lr)
        elif opt == 'EG':
            optim = ExtraSGD(net.parameters(), lr=lr)  
        elif opt == 'Adam':
            optim = torch.optim.Adam(net.parameters(), lr=lr, betas=(self.config.adam1, self.config.adam2))
        elif opt == 'EA':
            optim = ExtraAdam(net.parameters(), lr=lr, betas=(self.config.adam1, self.config.adam2))
        elif opt == 'EGplus':
            optim = EGplusSGD(net.parameters(), lr=lr) 
        elif opt == 'EAplus':
            optim = EGplusAdam(net.parameters(), lr=lr, betas=(self.config.adam1, self.config.adam2))
        elif opt == 'LA-EGplus':
            optim = EGplusLA(net.parameters(), lr=lr) 
        elif opt == 'LA-EAplus':
            optim = EGplusAdamLA(net.parameters(), lr=lr, betas=(self.config.adam1, self.config.adam2))
        else:
            raise ValueError('Optimizer %s not supported' % opt)       
        return optim


    def _sample_noise(self, n):
        return torch.randn(n, self.config.nz)

    def _generate_fake(self, x_real, netG):
        x_real_size = x_real.size(0)
        noise = self._sample_noise(x_real_size).type_as(x_real)
        return netG(noise)

    def _compute_model_loss_dis(self, batch, config, netD, netG):
        # netD = self.netD

        x_real = batch[0]
        x_fake = self._generate_fake(x_real, netG)
        # remember to detach as discussed here: https://github.com/PyTorchLightning/pytorch-lightning/issues/591
        errD, D_x, D_G_z1 = self.model_loss_dis(x_real, x_fake.detach(), netD)
        errD, D_x, D_G_z1 = errD.type_as(x_real), D_x.type_as(x_real), D_G_z1.type_as(x_real)

        D_x = D_x.mean().item()
        D_G_z1 = D_G_z1.mean().item()
        
        # gradient penalty
        if config.loss == 'wgan_gp':
            errD += config.grad_penalty * netD.get_penalty(x_real.detach(), x_fake.detach())

        return x_fake, errD, D_x, D_G_z1


    def _dis_step(self, opt_D, netD, netG, batch, extra=False):
        opt_D.zero_grad()
        x_fake, errD, D_x, D_G_z1 = self._compute_model_loss_dis(batch, self.config, netD, netG)
        self.manual_backward(errD)
        if extra:
            opt_D.extrapolation()
        else:
            opt_D.step()
        
        self.log('loss_D', errD, prog_bar=True)
        self.log("real_D", D_x)
        self.log("fake_D", D_G_z1)
    
    
    def _gen_step(self, opt_G, netD, netG, batch, extra=False):
        opt_G.zero_grad()
        x_fake = self._generate_fake(batch[0], netG)
        errG, D_G_z2 = self.model_loss_gen(x_fake, netD)
        self.manual_backward(errG)
        if extra:
            opt_G.extrapolation()
        else:
            opt_G.step()
        
        self.log('loss_G', errG, prog_bar=True)
        self.log("fake_G", D_G_z2.mean().item())
    
    
    def training_step(self, batch, batch_idx):
        self.steps += 1

        # requires Trainer(automatic_optimization=False)
        # reset PPM anchor
        opt_G, opt_D, opt_G_ea, opt_D_ea = self.optimizers()
        opt_G, opt_D, opt_G_ea, opt_D_ea = opt_G.optimizer, opt_D.optimizer, opt_G_ea.optimizer, opt_D_ea.optimizer
        
        if self.steps%(self.inner_steps*self.config.num_D_step)==0 and (isinstance(opt_G, RAPP) or isinstance(opt_G, Lookahead)):
            opt_G.init_anchor(force=True, lam=self.lam)
            opt_D.init_anchor(force=True, lam=self.lam)
            if (isinstance(opt_G_ea, RAPP) or isinstance(opt_G_ea, Lookahead)):
                opt_G_ea.init_anchor(force=True, lam=self.lam)
                opt_D_ea.init_anchor(force=True, lam=self.lam)
        
        if self.steps % 2500 == 0:
            self._evaluate()

        # for step in range(osp_steps):
        self._run_Ds(batch, opt_D, opt_D_ea)

        # Clipping if vanilla WGAN
        if self.config.loss == 'wgan' and self.config.clip is not None:
            weight_clipping(self.netD, self.config.clip)
            weight_clipping(self.netD_ea, self.config.clip)

        # step gen
        if self.steps % self.config.num_D_step == 0:
            self._run_Gs(batch, opt_D, opt_D_ea, opt_G, opt_G_ea)

        # logging 
        self._log_grad_norm()


    def _run_Ds(self, batch, opt_D, opt_D_ea):
        if self.extrapolate:    # Simultaneously
            if not self.gen_has_extrapolated:   # Extrapolation
                self._dis_step(opt_D_ea, self.netD_ea, self.netG, batch, extra=True)
            else:
                self._dis_step(opt_D, self.netD, self.netG_ea, batch)   
        else:       # Alternating
            self._dis_step(opt_D, self.netD, self.netG, batch)
            

    def _run_Gs(self, batch, opt_D, opt_D_ea, opt_G, opt_G_ea):
        if self.extrapolate:    # Simultaneously
            if not self.gen_has_extrapolated:
                self._gen_step(opt_G_ea, self.netD, self.netG_ea, batch, extra=True)
                opt_D.params_copy = copy.deepcopy(opt_D_ea.params_copy)
                self.gen_has_extrapolated = True
            else:
                opt_G.params_copy = copy.deepcopy(opt_G_ea.params_copy)
                self._gen_step(opt_G, self.netD_ea, self.netG, batch)
                opt_D.params_copy, opt_G.params_copy = [], []
                opt_D_ea.params_copy, opt_G_ea.params_copy = [], []
                self.netD_ea.load_state_dict(self.netD.state_dict())
                self.netG_ea.load_state_dict(self.netG.state_dict())
                self.gen_has_extrapolated = False
        else:       # Alternating
            self._gen_step(opt_G, self.netD, self.netG, batch)
                
            
    def _evaluate(self):
        self.netG.eval()
        wrapped_generator = torch_fidelity.GenerativeModelModuleWrapper(self.netG, self.config.nz, 'normal', 0)
        if self.config.dataset == 'cifar10':
            metrics = torch_fidelity.calculate_metrics(
                input1=wrapped_generator, 
                input2='cifar10-train', 
                input1_model_num_samples=self.config.num_metrics_samples,
                input2_model_num_samples=self.config.num_metrics_samples,
                cuda=True, 
                isc=True, 
                fid=True, 
                kid=False, 
                verbose=False,
                datasets_root='./data',
            )
            self.log("log_step", self.steps)
            self.log("ISC_mean", metrics['inception_score_mean'])
            self.log("ISC_std", metrics['inception_score_std'])
            self.log("FID", metrics['frechet_inception_distance'])
            # self.log("KID_mean", metrics['kernel_inception_distance_mean'])
            # self.log("KID_std", metrics['kernel_inception_distance_std'])
        self.netG.train()
        

    def training_epoch_end(self, outputs):
        # Scheduling lr  
        if self.config.dynamic_lr:
            for sch in self.lr_schedulers():
                sch.step()
        
        #print("Computing examples and IS")
        if self.current_epoch % 5 == 0:
            network_weight = next(self.netG.parameters())
            z = self.validation_z.type_as(network_weight)
            self._log_samples(z)

            # Componentwise distance to init
            self.init_nets['netD'] = self.init_nets['netD'].to(self.device)
            self.init_nets['netG'] = self.init_nets['netG'].to(self.device)
            self.log("init_dist_netD", 
                torch.norm(tensorify(self.init_nets['netD']) - tensorify(self.netD)))
            self.log("init_dist_netG", 
                torch.norm(tensorify(self.init_nets['netG']) - tensorify(self.netG)))

            self.netG.train()
            self.netG_ea.train()

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step(epoch=self.current_epoch)
    
    def _log_grad_norm(self):
        netD_grad = tuple(p.grad for p in self.netD.parameters())
        netG_grad = tuple(p.grad for p in self.netG.parameters())
        if None in netG_grad or None in netD_grad:
            return None
        netD_grad = tensorify_grad(netD_grad, ignore_none=True)
        netG_grad = tensorify_grad(netG_grad, ignore_none=True)

        netD_grad_norm = torch.norm(netD_grad)
        netG_grad_norm = torch.norm(netG_grad)
        joint_grad_norm = torch.sqrt(netD_grad_norm**2 + netG_grad_norm**2)

        self.log('stoc_grad_norm_netD', netD_grad_norm)
        self.log('stoc_grad_norm_netG', netG_grad_norm)
        self.log('stoc_grad_norm', joint_grad_norm)

    def _log_samples(self, z):
        # log sampled images
        sample_imgs = self.netG(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        #self.logger.experiment.add_image('generated_images', grid, self.current_epoch)
        self.logger.experiment.log({"generated_images": wandb.Image(grid)}, step=self.steps)
