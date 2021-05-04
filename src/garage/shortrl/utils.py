import torch

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise TypeError('Boolean value expected.')


def torch_stop_grad(torch_value):
    def wrapped_fun(x):
        with torch.no_grad():
            return torch_value(torch.Tensor(x))
    return wrapped_fun