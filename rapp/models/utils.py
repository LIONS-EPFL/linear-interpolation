import torch
from torch.nn.functional import normalize


def tensorify(model):
    weights = tuple(p.data for p in model.parameters())
    return tensorify_grad(weights, ignore_none=True)


def tensorify_grad(grad, ignore_none=False):
    if isinstance(grad, (list, tuple)):
        grad = list(grad)
        if ignore_none:
            grad = [g for g in grad if g is not None]
        for i, g in enumerate(grad):
            if g is None:
                raise Exception("One gradient was None. Remember to call backwards.")
            grad[i] = g.view(-1)
        return torch.cat(grad)
    elif isinstance(grad, torch.Tensor):
        return grad.view(-1)
    

def weight_clipping(netD, clip):
    for p in netD.parameters():
        p.data.clamp_(-clip, clip)
        
        
# def spectralnorm_poweriters(W, num_power_iters=1, eps=1e-12):
#     W_mat = W.reshape(W.shape[0], -1)
#     with torch.no_grad():
#         for _ in range(num_power_iters):
#             v = normalize(torch.matmul(W_mat.t(), u), dim=0, eps=eps)
#             u = normalize(torch.matmul(W_mat, v), dim=0, eps=eps)
#     sigma = torch.dot(u, torch.matmul(W_mat, v))
#     W = W / sigma
#     return W
