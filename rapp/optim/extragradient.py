# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# written by Hugo Berard (berard.hugo@gmail.com) while at Facebook.

import torch
from torch.optim import Optimizer
from rapp.optim.base import GDA, Adam
from typing import List, Optional

required = object()

class Extragradient(Optimizer):
    """Base class for optimizers with extrapolation step.

        Arguments:
        params (iterable): an iterable of :class:`torch.Tensor` s or
            :class:`dict` s. Specifies what Tensors should be optimized.
        defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).
    """
    def __init__(self, params, defaults=None):
        super(Extragradient, self).__init__(params, defaults)
        self.params_copy = []

    def update(self, p, group):
        raise NotImplementedError

    def extrapolation(self):
        """Performs the extrapolation step and save a copy of the current parameters for the update step.
        """
        # Check if a copy of the parameters was already made.
        is_empty = len(self.params_copy) == 0
        for group in self.param_groups:
            for p in group['params']:
                u = self.update(p, group)
                if is_empty:
                    # Save the current parameters for the update step. Several extrapolation step can be made before each update but only the parameters before the first extrapolation step are saved.
                    self.params_copy.append(p.data.clone())
                if u is None:
                    continue
                # Update the current parameters
                p.data.add_(u)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if len(self.params_copy) == 0:
            raise RuntimeError('Need to call extrapolation before calling step.')

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
                p.data = self.params_copy[i].add_(u)

        # Free the old parameters
        # self.params_copy = []
        return loss

class ExtraSGD(GDA, Extragradient):
    def __init__(self, params, lr=1e-3):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr)
        super(ExtraSGD, self).__init__(params, defaults)
        
        
class ExtraAdam(Adam, Extragradient):
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
        super(ExtraAdam, self).__init__(params, defaults)
        

class ExtraAdagrad(Extragradient):
    """Implements Adagrad algorithm.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        lr_decay (float, optional): learning rate decay (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-10)
        foreach (bool, optional): whether foreach implementation of optimizer is used (default: None)
        maximize (bool, optional): maximize the params based on the objective, instead of
            minimizing (default: False)

    .. _Adaptive Subgradient Methods for Online Learning and Stochastic
        Optimization: http://jmlr.org/papers/v12/duchi11a.html
    """

    def __init__(self, params, lr=1e-2, lr_decay=0, weight_decay=0, initial_accumulator_value=0,
        eps=1e-10, foreach: Optional[bool] = None, *, maximize: bool = False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= lr_decay:
            raise ValueError("Invalid lr_decay value: {}".format(lr_decay))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= initial_accumulator_value:
            raise ValueError("Invalid initial_accumulator_value value: {}".format(initial_accumulator_value))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(lr=lr, lr_decay=lr_decay, eps=eps, weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value, foreach=foreach, maximize=maximize,)
        super(ExtraAdagrad, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = torch.tensor(0.0)
                init_value = (
                    complex(initial_accumulator_value, initial_accumulator_value)
                    if torch.is_complex(p)
                    else initial_accumulator_value
                )
                state["sum"] = torch.full_like(
                    p, init_value, memory_format=torch.preserve_format
                )

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("foreach", None)
            group.setdefault("maximize", False)

        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(
            state_values[0]["step"]
        )
        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(float(s["step"]))
                
    def share_memory(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["sum"].share_memory_()
                
    def update(self, p, group):
        if p.grad is None:
            return None
        grad = p.grad.data
        
        state = self.state[p]
        state['step'] += 1
        clr = group['lr'] / (1 + (state['step'] - 1) * group['lr_decay'])
                
        if group['weight_decay'] != 0:
            grad = grad.add(group['weight_decay'], p.data)
            
        state["sum"].add_(torch.pow(grad,2))
        step_size = clr/(torch.sqrt(state["sum"])+group["eps"])
        print(step_size)
        self.step_size = torch.norm(torch.diag(step_size))
        return -step_size*grad

