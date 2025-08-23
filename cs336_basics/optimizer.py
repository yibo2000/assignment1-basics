from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.   

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        
        return loss
    

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8,):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        if eps < 0:
            raise ValueError(f"Invalid eps: {eps}")
        if betas[0] < 0 or betas[1] < 0:
            raise ValueError(f"Invalid betas: {betas}")
        
        defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.  
            beta1, beta2 = group["betas"] # Get the betas
            weight_decay = group["weight_decay"] # decay rate λ
            eps = group["eps"] # eps
            
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Get iteration number from the state, or initial value.
                state = self.state[p] # Get state associated with p.
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                t = state.get("t", 1)
                g = p.grad.data # Get the gradient of loss with respect to p.
                
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * g ** 2
                a_t = lr * (math.sqrt( 1 - (beta2) ** t )) / ( 1 - (beta1) ** t )

                p.data -= a_t * m / (torch.sqrt(v) + eps) # Update weight tensor in-place.
                p.data -= lr * weight_decay * p.data # Apply weight decay
                
                # update state
                state["m"] = m
                state["v"] = v
                state["t"] = t + 1 # Increment iteration number.

        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if(it < warmup_iters): 
        return it / warmup_iters * max_learning_rate # warm up
    if(it >= warmup_iters and it <= cosine_cycle_iters):
        return min_learning_rate + 0.5 * \
        (1 + math.cos( math.pi * ( it -  warmup_iters) / ( cosine_cycle_iters - warmup_iters ) )) * \
        (max_learning_rate - min_learning_rate)
    if(it > cosine_cycle_iters):
        return min_learning_rate
    return -1


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps = 1e-6) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    l2_norm = 0
    for p in parameters:
        if(p.grad == None): continue
        l2_norm += torch.sum(p.grad ** 2)
    l2_norm = l2_norm.sqrt()
    
    for p in parameters:
        if(p.grad == None): continue
        if(l2_norm < max_l2_norm): p.grad = p.grad
        else: p.grad = p.grad * (max_l2_norm / (l2_norm + eps))


if __name__ == "__main__":
    torch.manual_seed(42)
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1e2)
    for t in range(10):
        opt.zero_grad() # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean() # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward() # Run backward pass, which computes gradients.
        opt.step() # Run optimizer step.