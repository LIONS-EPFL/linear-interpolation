import torch
from torch.optim import Optimizer
from rapp.optim.base import GDA, Adam
from rapp.optim.extragradient import Extragradient
from rapp.optim.EGplus import EGplus

required = object()


class Lookahead(Optimizer):
    def __init__(self, params, defaults=None):
        self.t = 0
        super(Lookahead, self).__init__(params, defaults)
        self.anchor = []
        
    def update(self, p, group): 
        raise NotImplementedError

    def init_anchor(self, force=False, lam=1):
        # Check if a copy of the parameters was already made.
        is_empty = len(self.anchor) == 0
        if is_empty:
            self.anchor = []
            for group in self.param_groups:
                for p in group['params']:
                    self.anchor.append(p.data.clone())
        if force:
            i = -1
            for group in self.param_groups:
                for p in group['params']:
                    i += 1
                    self.anchor[i] = torch.mul((1-lam),self.anchor[i]) + torch.mul(lam,p.data)
                    p.data = self.anchor[i].clone()

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self.init_anchor()

        loss = None
        if closure is not None:
            loss = closure()

        i = -1
        for group in self.param_groups:
            for p in group['params']:
                i += 1
                u = self.update(p, group)
                if u is None:
                    continue
                # Update the parameters saved during the extrapolation step
                p.data.add_(u)
                
                
class LA(GDA, Lookahead):
    """Implements Lookahead. """
    def __init__(self, params, lr=1e-3):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr)
        super(LA, self).__init__(params, defaults)
                

class AdamLA(Adam, Lookahead):
    """Implements the Adam algorithm with OSP step.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
         raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
         raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
         raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                     weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamLA, self).__init__(params, defaults)


class ExtraSGDLA(GDA, Extragradient, Lookahead):
    def __init__(self, params, lr=1e-3):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr)
        super(ExtraSGDLA, self).__init__(params, defaults)
        

class ExtraAdamLA(Adam, Extragradient, Lookahead):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
         raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
         raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
         raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
         raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                     weight_decay=weight_decay, amsgrad=amsgrad)
        super(ExtraAdamLA, self).__init__(params, defaults)
        
        
class EGplusLA(GDA, EGplus, Lookahead):
    def __init__(self, params, lr=1e-3):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr)
        super(EGplusLA, self).__init__(params, defaults)
        

class EGplusAdamLA(Adam, Extragradient, Lookahead):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
         raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
         raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
         raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
         raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                     weight_decay=weight_decay, amsgrad=amsgrad)
        super(EGplusAdamLA, self).__init__(params, defaults)
