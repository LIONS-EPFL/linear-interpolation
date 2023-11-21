import os
from typing import Literal
from datargs import argsclass


@argsclass
class Args:
    data_root: str = './data'
    workers: int = 5
    dataset: Literal['cifar10'] = 'cifar10'
    exp_dir: str = 'EXP'
    seed: int = None
    batch_size: int = 128
    log_checkpoint_every_n: int = 0
    epochs: int = 3200
    gpus: int = 1
    num_metrics_samples: int = 50000

    # Optimizer
    lrD: float = 0.1
    lrG: float = 0.02
    opt: Literal['GDA', 'Adam', 'RAPP', 'EG', 'EA', 
                 'EGplus', 'EAplus', 'LA-EGplus', 'LA-EAplus',
                 'LA', 'LA-Adam', 'LA-EG', 'LA-EA'] = 'GDA'
    opt_independent: bool = False
    inner_steps: int = 5000
    lam: float = 1
    adam1: float = 0.0
    adam2: float = 0.9

    # Model
    ckpt_D: str = None
    ckpt_G: str = None
    
    model: Literal['cnn', 'mlp', 'resnet'] = 'resnet'
    nz: int = 128
    ngf: int = 128
    ndf: int = 64

    # loss
    loss: Literal['vangan', 'wgan', 'wgan_gp', 'hinge'] = 'hinge'
    num_D_step: int = 1

    # Lipschitz constraint
    grad_penalty: float = 0.0
    clip: float = None
    spectral_norm: bool = True

    wandb_tags: str = None
    wandb_project: str = 'one-step-proximal'
    wandb_entity: str = 'epfl-lions'
    wandb_name: str = 'debug'
    wandb_id: str = None
    
    dynamic_lr: bool = False

    def setup(self):
        self.out_dir = os.path.join(self.exp_dir, self.wandb_name)
        self.checkpoint_init_path = os.path.join(self.out_dir, 'checkpoint_init.ckpt')
        if self.lrG is None:
            self.lrG = self.lrD
