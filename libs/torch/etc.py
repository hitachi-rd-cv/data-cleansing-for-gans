import numpy as np
import torch


def optimizer_to_(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def lr_scheduler_to_(scheduler, device):
    for key, value in scheduler.__dict__.items():
        if key != 'optimizer':
            if isinstance(value, torch.Tensor):
                value.data = value.data.to(device)


def flatten_params(params):
    param_flat = torch.hstack([torch.flatten(p) for p in params])
    return param_flat


def deflatten_params_like(flatten_params, params_ref):
    shapes_params = [p.shape for p in params_ref]
    n_elems_params = [np.prod(shape).astype(int) for shape in shapes_params]

    if not len(flatten_params) == sum(n_elems_params):
        raise ValueError(f'len(block) != sum(n_elements_of_blocks_list[0]), {len(flatten_params)} != {sum(n_elems_params)}')

    flatten_param_blocks = []
    idx_start = 0
    for n_elements_of_block in n_elems_params:
        flatten_param_blocks.append(flatten_params[idx_start:idx_start + n_elements_of_block].T.contiguous())
        idx_start += n_elements_of_block

    assert len(flatten_param_blocks) == len(shapes_params)
    params = [param_flat.reshape(shape) for param_flat, shape in zip(flatten_param_blocks, shapes_params)]
    return params


def assign_params(params_replaced, params_assigned):
    assert len(params_replaced) == len(params_assigned)
    for p_old, p_new in zip(params_replaced, params_assigned):
        p_old.detach_()
        p_old.copy_(p_new)
