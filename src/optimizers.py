import torch
import math
from torch.optim.optimizer import Optimizer
from collections import defaultdict


class Lion(Optimizer):
    """
    Implements Lion optimizer from the paper
    "Symbolic Discovery of Optimization Algorithms" (https://arxiv.org/abs/2302.06675)
    
    Lion optimizer has shown better convergence on vision tasks compared to AdamW
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        """
        Initialize Lion optimizer
        
        Args:
            params: iterable of parameters to optimize
            lr: learning rate
            betas: coefficients used for computing moving averages
            weight_decay: weight decay coefficient
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Perform weight decay
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                grad = p.grad
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                
                # Update moving average
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update weights
                update = exp_avg.sign() * grad.sign() 
                p.add_(update, alpha=-group['lr'])
        
        return loss


class AdamP(Optimizer):
    """
    Implements AdamP optimizer from the paper
    "AdamP: Slowing Down the Slowdown for Momentum Optimizers on Scale-invariant Weights"
    
    AdamP helps improving generalization by adjusting the weight
    updates to preserve the norm of scale-invariant weights
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, delta=0.1, wd_ratio=0.1, nesterov=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        delta=delta, wd_ratio=wd_ratio, nesterov=nesterov)
        super(AdamP, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # State initialization
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                
                # Get parameters
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                
                # AdamP weight update
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                
                # Calculate AdamP update
                step_size = group['lr'] / bias_correction1
                
                # Calculate projection component
                perturb = step_size * exp_avg / denom
                
                # Apply weight decay if needed
                if group['weight_decay'] > 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Update parameters
                p.data.add_(-perturb)
        
        return loss 


class Lookahead(Optimizer):
    """
    Lookahead Optimizer wrapper
    
    Implements the Lookahead optimization algorithm from
    "Lookahead Optimizer: k steps forward, 1 step back"
    https://arxiv.org/abs/1907.08610
    """
    def __init__(self, optimizer, k=5, alpha=0.5):
        """
        Initialize Lookahead
        
        Args:
            optimizer: The optimizer to wrap
            k: Number of lookahead steps
            alpha: Slow weights adjustment rate
        """
        self.optimizer = optimizer
        self.param_groups = self.optimizer.param_groups
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.defaults = self.optimizer.defaults
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        
        for group in self.param_groups:
            group["counter"] = 0
    
    def update(self, group):
        """
        Update slow weights
        """
        for p in group['params']:
            if p.grad is None:
                continue
                
            param_state = self.state[p]
            
            if 'slow_param' not in param_state:
                param_state['slow_param'] = torch.clone(p.data).detach()
                
            slow_param = param_state['slow_param']
            # Update slow weights
            slow_param.add_(self.alpha * (p.data - slow_param))
            # Replace current weights with slow weights
            p.data.copy_(slow_param)
    
    def step(self, closure=None):
        """
        Performs a single optimization step
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
        """
        loss = self.optimizer.step(closure)
        
        for group in self.param_groups:
            group["counter"] += 1
            # Update slow weights every k steps
            if group["counter"] % self.k == 0:
                self.update(group)
                
        return loss
    
    def state_dict(self):
        """
        Return the state of the optimizer
        """
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }
    
    def load_state_dict(self, state_dict):
        """
        Load the state of the optimizer
        """
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state
        
    def add_param_group(self, param_group):
        """
        Add a param group to the optimizer
        """
        param_group["counter"] = 0
        self.optimizer.add_param_group(param_group) 