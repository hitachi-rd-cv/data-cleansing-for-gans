import os

from matplotlib import pyplot as plt

from constants import Metric, SetMetric, D_METRIC_LABEL
from torch import optim
import numpy as np
from libs.torch.autograd import myjvp, mygrad
from libs.finetune import set_requires_grad, backward_D, backward_G, load_models, accumulate, backward_D_scaled
from libs.evaluator import StyleGANEvaluator, LQGANEvaluator, backward_mean, backward_d_loss
import pandas as pd
from metrics.frechet_inception_distance import backward_fid
from metrics.inception_score import backward_is
from libs.models import StyledGenerator, ToyGenerator
from torchvision import utils

from torch import Tensor
from torch.optim.optimizer import _default_to_fused_or_foreach, Optimizer
import random

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from libs.dataset import MultiResolutionDataset, resize_worker
from metrics.inception import InceptionV3
import multiprocessing
from functools import partial

from PIL import Image
import lmdb
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from sklearn.ensemble import IsolationForest


def split_and_prepare_data(path, tmp_dir, ratio_valid, ratio_test, sizes=(256,), resample='lanczos', n_worker=8):
    if ratio_valid + ratio_test >= 1:
        raise ValueError("Sum of ratio_valid and ratio_test must be less than 1")
    assert len(sizes) == 1, 'Only one size is supported'

    resample_map = {"lanczos": Image.LANCZOS, "bilinear": Image.BILINEAR}
    resample = resample_map[resample]

    print(f"Make dataset of image sizes:", ", ".join(str(s) for s in sizes))

    imgset = ImageFolder(path)

    # split dataset
    n = len(imgset)
    indices = list(range(n))
    random.shuffle(indices)
    n_valid = int(n * ratio_valid)
    n_test = int(n * ratio_test)
    n_train = n - n_valid - n_test
    train_set, valid_set, test_set = random_split(imgset, [n_train, n_valid, n_test])
    train_set.imgs = [train_set.dataset.imgs[i] for i in train_set.indices]
    valid_set.imgs = [valid_set.dataset.imgs[i] for i in valid_set.indices]
    test_set.imgs = [test_set.dataset.imgs[i] for i in test_set.indices]

    print(f"Train: {n_train}, Valid: {n_valid}, Test: {n_test}")

    datasets = []
    for name, sub_dataset in zip(['train', 'valid', 'test'], [train_set, valid_set, test_set]):
        print(f"Processing {name} dataset")
        out_path = os.path.join(tmp_dir, name)
        with lmdb.open(out_path, map_size=1024 ** 4, readahead=False) as env:
            resize_fn = partial(resize_worker, sizes=sizes, resample=resample)
            sorted_imgs = sorted(sub_dataset.imgs, key=lambda x: x[0])
            files = [(i1, file) for i1, (file, label) in enumerate(sorted_imgs)]
            labels = [label1 for _, label1 in sorted_imgs]
            total = 0
            with multiprocessing.Pool(n_worker) as pool:
                for i1, imgs1 in tqdm(pool.imap_unordered(resize_fn, files)):
                    for size, img in zip(sizes, imgs1):
                        key = f"{size}-{str(i1).zfill(5)}".encode("utf-8")

                        with env.begin(write=True) as txn:
                            txn.put(key, img)

                    total += 1

                with env.begin(write=True) as txn:
                    txn.put("length".encode("utf-8"), str(total).encode("utf-8"))

        ### load dataset ###
        transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])
        dataset = MultiResolutionDataset(out_path, transform, labels, sizes[0])
        datasets.append(dataset)

    return datasets
    # make load imagset a
    if ratio_valid + ratio_test >= 1:
        raise ValueError("Sum of ratio_valid and ratio_test must be less than 1")
    assert len(sizes) == 1, 'Only one size is supported'

    resample_map = {"lanczos": Image.LANCZOS, "bilinear": Image.BILINEAR}
    resample = resample_map[resample]

    print(f"Make dataset of image sizes:", ", ".join(str(s) for s in sizes))

    imgset = ImageFolder(path)

    # split dataset
    n = len(imgset)
    n_valid = int(n * ratio_valid)
    n_test = int(n * ratio_test)
    n_train_a = n - n_valid - n_test
    train_set, valid_set, test_set = random_split(imgset, [n_train_a, n_valid, n_test])
    train_imgpairs_a = [train_set.dataset.imgs[i] for i in train_set.indices]
    valid_imgpairs = [valid_set.dataset.imgs[i] for i in valid_set.indices]
    test_imgpairs = [test_set.dataset.imgs[i] for i in test_set.indices]

    # load and split mix dataset
    imgset_mix = ImageFolder(path_mix)
    n_mix = len(imgset_mix)
    n_train_mix = int(n_mix * ratio_train_mix)
    n_remove_mix = n_mix - n_train_mix
    train_set_mix, _ = random_split(imgset_mix, [n_train_mix, n_remove_mix])
    offset_label = len(imgset.classes)
    train_imgpairs_b = [imgset_mix.imgs[i] for i in train_set_mix.indices]
    train_imgpairs_b = [(path, label + offset_label) for path, label in train_imgpairs_b]

    # concat train imgpairs
    train_imgpairs = train_imgpairs_a + train_imgpairs_b
    n_train = len(train_imgpairs)

    print(f"Train: {n_train}, Valid: {n_valid}, Test: {n_test}")

    datasets = []
    for name, imgs in zip(['train', 'valid', 'test'], [train_imgpairs, valid_imgpairs, test_imgpairs]):
        print(f"Processing {name} dataset")
        out_path = os.path.join(tmp_dir, name)
        with lmdb.open(out_path, map_size=1024 ** 4, readahead=False) as env:
            resize_fn = partial(resize_worker, sizes=sizes, resample=resample)
            sorted_imgs = sorted(imgs, key=lambda x: x[0])
            files = [(i1, file) for i1, (file, label1) in enumerate(sorted_imgs)]
            labels = [label1 for _, label1 in sorted_imgs]
            total = 0
            with multiprocessing.Pool(n_worker) as pool:
                for i1, imgs in tqdm(pool.imap_unordered(resize_fn, files)):
                    for size, img in zip(sizes, imgs):
                        key = f"{size}-{str(i1).zfill(5)}".encode("utf-8")

                        with env.begin(write=True) as txn:
                            txn.put(key, img)

                    total += 1

                with env.begin(write=True) as txn:
                    txn.put("length".encode("utf-8"), str(total).encode("utf-8"))

        ### load dataset ###
        transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])
        dataset = MultiResolutionDataset(out_path, transform, labels, sizes[0])
        datasets.append(dataset)

    return datasets

def compute_acts(dataset, batch_size=16, flip=False):
    if len(dataset) == 0:
        return torch.tensor([])
    else:
        inception = InceptionV3().cuda()
        inception.eval()

        loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=1)

        pbar = tqdm(total=len(dataset))

        acts = []
        for real_index, real_image in loader:
            real_image = real_image.cuda()
            with torch.no_grad():
                out = inception(real_image)
                out = out[0].squeeze(-1).squeeze(-1)
            acts.append(out.cpu())
            if flip:
                real_image_flip = torch.flip(real_image, dims=[3])
                with torch.no_grad():
                    out = inception(real_image_flip)
                    out = out[0].squeeze(-1).squeeze(-1)
                acts.append(out.cpu())

            pbar.update(len(real_image))
        acts = torch.concat(acts, axis=0)

        return acts

def compute_steps_per_epoch(total_samples, batch_size):
    """
    Computes the number of steps (batches) per epoch.

    Parameters:
    - total_samples: int, the total number of samples in the dataset.
    - batch_size: int, the number of samples per batch.

    Returns:
    - steps_per_epoch: int, the number of steps per epoch.
    """
    # Compute the steps per epoch with the remainder considered.
    steps_per_epoch = total_samples // batch_size
    # If there is a remainder, add an extra step to account for the last, smaller batch.
    if total_samples % batch_size != 0:
        steps_per_epoch += 1

    return steps_per_epoch

def differential_adam_step(optimizer: optim.Adam):
    param_diffs_groups = []
    for group in optimizer.param_groups:
        group['differentiable'] = True
        group['foreach'] = False

        params_with_grad = []
        grads = []
        exp_avgs = []
        exp_avg_sqs = []
        max_exp_avg_sqs = []
        state_steps = []
        beta1, beta2 = group['betas']

        optimizer._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps)

        exp_avgs = [torch.nn.Parameter(exp_avg) for exp_avg in exp_avgs]
        exp_avg_sqs = [torch.nn.Parameter(exp_avg_sq) for exp_avg_sq in exp_avg_sqs]

        foreach = group['foreach']
        capturable = group['capturable']
        differentiable = group['differentiable']
        fused = group['fused']
        scale = getattr(optimizer, "grad_scale", None)
        inf = getattr(optimizer, "found_inf", None)
        amsgrad = group['amsgrad']
        lr = group['lr']
        decay = group['weight_decay']
        eps = group['eps']
        maximize = group['maximize']
        # Respect when the user inputs False/True for foreach or fused. We only want to change
        # the default when neither have been user-specified. Note that we default to foreach
        # and pass False to use_fused. This is not a mistake--we want to give the fused impl
        # bake-in time before making it the default, even if it is typically faster.
        if fused is None and foreach is None:
            _, foreach = _default_to_fused_or_foreach(params_with_grad, differentiable, use_fused=False)
            # Do not flip on foreach for the unsupported case where lr is a Tensor and capturable=False.
            if foreach and isinstance(lr, Tensor) and not capturable:
                foreach = False
        if fused is None:
            fused = False
        if foreach is None:
            foreach = False
        # this check is slow during compilation, so we skip it
        # if it's strictly needed we can add this check back in dynamo
        if not torch._utils.is_compiling() and not all(isinstance(t, torch.Tensor) for t in state_steps):
            raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")
        if foreach and torch.jit.is_scripting():
            raise RuntimeError('torch.jit.script not supported with foreach optimizers')
        if fused and torch.jit.is_scripting():
            raise RuntimeError("torch.jit.script not supported with fused optimizers")
        assert scale is None and inf is None
        if torch.jit.is_scripting():
            # this assert is due to JIT being dumb and not realizing that the ops below
            # have overloads to handle both float and Tensor lrs, so we just assert it's
            # a float since most people using JIT are using floats
            assert isinstance(lr, float)

        param_diffs = []
        for i, param in enumerate(params_with_grad):
            grad = grads[i] if not maximize else -grads[i]
            avg = exp_avgs[i]
            sq = exp_avg_sqs[i]
            step_t = state_steps[i]

            # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
            if not torch._utils.is_compiling() and capturable:
                assert (
                        (param.is_cuda and step_t.is_cuda) or (param.is_xla and step_t.is_xla)
                ), "If capturable=True, params and state_steps must be CUDA or XLA tensors."

            # update step
            step_t += 1

            if decay != 0:
                grad = grad.add(param, alpha=decay)

            # Decay the first and second moment running average coefficient
            avg = avg.lerp(grad, 1 - beta1)
            sq = sq.mul(beta2).addcmul(grad, grad.conj(), value=1 - beta2)

            assert differentiable
            step = step_t

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            step_size = lr / bias_correction1
            step_size_neg = step_size.neg()

            bias_correction2_sqrt = bias_correction2.sqrt()

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                max_exp_avg_sq = max_exp_avg_sqs[i].clone()

                max_exp_avg_sqs[i] = torch.maximum(max_exp_avg_sq, sq)

                # Uses the max. for normalizing running avg. of gradient
                # Folds in (admittedly ugly) 1-elem step_size math here to avoid extra param-set-sized read+write
                # (can't fold it into addcdiv_ below because addcdiv_ requires value is a Number, not a Tensor)
                denom = (max_exp_avg_sqs[i].sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)
            else:
                denom = (sq.sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)

            param_diff = torch.div(avg, denom)
            param_diffs.append(param_diff)

        param_diffs_groups.append(param_diffs)

    return param_diffs_groups

def differential_sgd_step(optimizer: optim.SGD):
    param_diffs_groups = []

    for group in optimizer.param_groups:
        params_with_grad = []
        d_p_list = []
        momentum_buffer_list = []

        optimizer._init_group(group, params_with_grad, d_p_list, momentum_buffer_list)

        foreach = group['foreach']
        decay = group['weight_decay']
        momentum = group['momentum']
        lr = group['lr']
        dampening = group['dampening']
        nesterov = group['nesterov']
        maximize = group['maximize']
        if foreach is None:

            # why must we be explicit about an if statement for torch.jit.is_scripting here?
            # because JIT can't handle Optionals nor fancy conditionals when scripting
            if not torch.jit.is_scripting():
                _, foreach = _default_to_fused_or_foreach(params_with_grad, differentiable=False, use_fused=False)
            else:
                foreach = False
        if foreach and torch.jit.is_scripting():
            raise RuntimeError('torch.jit.script not supported with foreach optimizers')

        param_diffs = []
        for i, param in enumerate(params_with_grad):
            d_p = d_p_list[i] if not maximize else -d_p_list[i]

            if decay != 0:
                d_p = d_p.add(param, alpha=decay)

            if momentum != 0:
                buf = momentum_buffer_list[i]

                if buf is None:
                    buf = torch.clone(d_p).detach()
                    momentum_buffer_list[i] = buf
                else:
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                if nesterov:
                    d_p = d_p.add(buf, alpha=momentum)
                else:
                    d_p = buf

            param_diff = d_p.mul(-lr)
            param_diffs.append(param_diff)

        param_diffs_groups.append(param_diffs)

    return param_diffs_groups


def approx_influence(train_dir, name_model, mixing, rank_G, rank_D, n_epochs, lr_G, lr_D, name_optimizer, dataset_train, acts_valid, noise_dataset, from_epoch, to_epoch, from_step, n_steps, name_metric_infl, image_size,
        on_averaged_G=False, accumulate_decay=0.999, damping=0., scale=1., ckpt=None, no_gp=False, _run=None):
    assert from_epoch < to_epoch <= n_epochs, 'epochs should be less than n_epochs'

    D_target, G_target = load_models(name_model, image_size, ckpt, rank_D, rank_G, mixing)

    set_requires_grad(G_target, False, only_trainable=False)
    set_requires_grad(D_target, False, only_trainable=False)
    set_requires_grad(G_target, True)
    set_requires_grad(D_target, True)

    ### prepare evaluation metrics ###
    # Example initialization within main function or setup routine
    if name_model == 'stylegan':
        evaluator = StyleGANEvaluator(G_target, real_acts=acts_valid, noise_dataset=noise_dataset)
    elif name_model == 'lqgan':
        evaluator = LQGANEvaluator(G_target, noise_dataset)
    else:
        raise NotImplementedError(f'Model {name_model} is not supported')

    ## load LoRA parameters
    df_ckpt = pd.read_json(os.path.join(train_dir, 'df_ckpt.json'))
    idx_start = df_ckpt[(df_ckpt['epoch'] == from_epoch) & (df_ckpt['step'] == from_step)].index[0]
    idx_end = df_ckpt[(df_ckpt['epoch'] == to_epoch - 1) & (df_ckpt['step'] == n_steps - 1)].index[0] + 1

    # load the last model to compute original valid fid
    if on_averaged_G:
        G_target.load_state_dict(torch.load(df_ckpt.iloc[idx_end]['generator_ave']), strict=False)
    else:
        G_target.load_state_dict(torch.load(df_ckpt.iloc[idx_end]['generator']), strict=False)
    D_target.load_state_dict(torch.load(df_ckpt.iloc[idx_end]['discriminator']), strict=False)

    grads_metric_G = compute_grads_metric_G(D_target, G_target, evaluator, name_metric_infl, name_model)
    influences = compute_itd_influence(D_target, G_target, _run, accumulate_decay, damping, dataset_train, df_ckpt, grads_metric_G, idx_end, idx_start, lr_D, lr_G, name_optimizer, no_gp, on_averaged_G, scale)

    return influences


def approx_influence_aid(train_dir, name_model, mixing, rank_G, rank_D, batch_size, dataset_train, acts_valid, noise_dataset, to_epoch, n_steps, depth, name_metric_infl, image_size,
        on_averaged_G=False, damping=0., scale=1., ckpt=None, no_gp=False, _run=None):

    D_target, G_target = load_models(name_model, image_size, ckpt, rank_D, rank_G, mixing)

    set_requires_grad(G_target, False, only_trainable=False)
    set_requires_grad(D_target, False, only_trainable=False)
    set_requires_grad(G_target, True)
    set_requires_grad(D_target, True)

    ### prepare evaluation metrics ###
    # Example initialization within main function or setup routine
    if name_model == 'stylegan':
        evaluator = StyleGANEvaluator(G_target, real_acts=acts_valid, noise_dataset=noise_dataset)
    elif name_model == 'lqgan':
        evaluator = LQGANEvaluator(G_target, noise_dataset)
    else:
        raise NotImplementedError(f'Model {name_model} is not supported')

    ## load LoRA parameters
    df_ckpt = pd.read_json(os.path.join(train_dir, 'df_ckpt.json'))
    idx_end = df_ckpt[(df_ckpt['epoch'] == to_epoch - 1) & (df_ckpt['step'] == n_steps - 1)].index[0] + 1

    # load the last model to compute original valid fid
    if on_averaged_G:
        G_target.load_state_dict(torch.load(df_ckpt.iloc[idx_end]['generator_ave']), strict=False)
    else:
        G_target.load_state_dict(torch.load(df_ckpt.iloc[idx_end]['generator']), strict=False)
    D_target.load_state_dict(torch.load(df_ckpt.iloc[idx_end]['discriminator']), strict=False)

    grads_metric_G = compute_grads_metric_G(D_target, G_target, evaluator, name_metric_infl, name_model)
    influences = compute_aid_influence(D_target, G_target, _run, damping, dataset_train, df_ckpt, grads_metric_G, idx_end, no_gp, on_averaged_G, scale, depth, batch_size)

    return influences


def compute_grads_metric_G(D_target, G_target, evaluator, name_metric_infl, name_model):
    # compute gradient of eval metric around the final model
    if name_metric_infl == Metric.FID:
        assert name_model == 'stylegan', 'FID is only supported for StyleGAN'

        def input_processor(x, *noise):
            return x.cuda(), [x.cuda() for x in noise]

        def feature_processor(x):
            return x[0]

        backward_fid(evaluator.noise_dataset, G_target, evaluator.inception, evaluator.real_acts,
                     inputs_backward=G_target.trainable_parameters(), batch_size=16, input_processor=input_processor, feature_processor=feature_processor)

    elif name_metric_infl == Metric.IS:
        assert name_model == 'stylegan', 'FID is only supported for StyleGAN'

        def input_processor(x, *noise):
            return x.cuda(), [x.cuda() for x in noise]

        backward_is(evaluator.noise_dataset, G_target, evaluator.inception,
                    inputs_backward=G_target.trainable_parameters(), batch_size=16, input_processor=input_processor)

    elif name_metric_infl == Metric.FAKE_IM_MEAN:
        if name_model == 'stylegan':
            def input_processor(x, *noise):
                return x.cuda(), [x.cuda() for x in noise]
        else:
            input_processor = None
        backward_mean(evaluator.noise_dataset, G_target, inputs_backward=G_target.trainable_parameters(), batch_size=16, input_processor=input_processor)

    elif name_metric_infl == Metric.D_LOSS:
        if name_model == 'stylegan':
            def input_processor(x, *noise):
                return x.cuda(), [x.cuda() for x in noise]
        else:
            input_processor = None
        backward_d_loss(evaluator.noise_dataset, G_target, D_target, inputs_backward=G_target.trainable_parameters(), batch_size=16, input_processor=input_processor)

    else:
        metric = evaluator.evaluate(name_metric_infl)
        metric.backward()
    grads_metric_G = tuple([torch.zeros_like(p) if p.grad is None else p.grad.clone().detach() for p in G_target.trainable_parameters()])
    return grads_metric_G


def compute_itd_influence(D_target, G_target, _run, accumulate_decay, damping, dataset, df_ckpt, grads_metric_G, idx_end, idx_start, lr_D, lr_G, name_optimizer, no_gp, on_averaged_G, scale):
    # initialization of intermediate vectors for influence estimation
    if on_averaged_G:
        us_G = tuple([torch.zeros_like(p) for p in grads_metric_G])
    else:
        us_G = tuple([p.clone().detach() for p in grads_metric_G])
    us_D = tuple([torch.zeros_like(p) for p in D_target.trainable_parameters()])
    D_target.load_state_dict(torch.load(df_ckpt.iloc[idx_end]['discriminator']), strict=False)
    G_target.load_state_dict(torch.load(df_ckpt.iloc[idx_end]['generator']), strict=False)  # ensure reading non-averaged generator model
    # load optimizer
    if name_optimizer == 'adam':
        G_optimizer = optim.Adam(G_target.trainable_parameters(), lr=lr_G, betas=(0.0, 0.99))
        D_optimizer = optim.Adam(D_target.trainable_parameters(), lr=lr_D, betas=(0.0, 0.99))
        step_fn = differential_adam_step
    elif name_optimizer == 'sgd':
        G_optimizer = optim.SGD(G_target.trainable_parameters(), lr=lr_G)
        D_optimizer = optim.SGD(D_target.trainable_parameters(), lr=lr_D)
        step_fn = differential_sgd_step
    else:
        raise NotImplementedError(f'Optimizer {name_optimizer} is not supported')
    accumulate_scale = 1.
    dataset_size = len(dataset)
    influences = np.zeros(dataset_size)
    pbar = tqdm(range(idx_end - 1, idx_start - 1, -1), position=0, total=idx_end - idx_start)
    for idx in pbar:
        samples = torch.load(df_ckpt.iloc[idx]['samples'])
        real_index = samples['real_index']
        real_image = samples['real_image']
        samples_d_update = samples['samples_d_update']
        samples_g_update = samples['samples_g_update']

        if on_averaged_G:
            for u, g in zip(us_G, grads_metric_G):
                u.add_(g.clone().detach() * accumulate_scale * (1. - accumulate_decay))
            accumulate_scale = accumulate_scale * accumulate_decay

        # load the intermediate  model to compute the influence
        G_target.load_state_dict(torch.load(df_ckpt.iloc[idx]['generator']), strict=False)
        G_optimizer.load_state_dict(torch.load(df_ckpt.iloc[idx_end]['optimizer_g']))

        G_target.zero_grad()
        G_optimizer.zero_grad()
        G_loss_val = backward_G(G_target, D_target, samples_g_update)
        G_loss_val.backward(create_graph=True)
        param_diffs = step_fn(G_optimizer)[0]

        # grads = fill_none_grads_with_zeros(grads, G_target.trainable_parameters())
        grads_filtered = tuple([g for g in param_diffs if g is not None])
        us_G_filtered = tuple([u for u, p in zip(us_G, G_target.trainable_parameters()) if p.grad is not None])
        hvps_D, hvps_G = myjvp(grads_filtered, (D_target.trainable_parameters(), G_target.trainable_parameters()), us_G_filtered)

        with torch.no_grad():
            for u, hvp in zip(us_D, hvps_D):
                u.copy_((1. - damping) * u.clone().detach() + scale * hvp.clone().detach())
            for u, hvp in zip(us_G, hvps_G):
                u.copy_((1. - damping) * u.clone().detach() + scale * hvp.clone().detach())

        # Reset gradients to None after backward to avoid memory leak
        for param in G_target.parameters():
            param.grad = None

        D_target.load_state_dict(torch.load(df_ckpt.iloc[idx]['discriminator']), strict=False)
        D_optimizer.load_state_dict(torch.load(df_ckpt.iloc[idx]['optimizer_d']))

        loss_mask = torch.nn.Parameter(torch.ones((len(real_index), 1)).cuda())

        D_target.zero_grad()
        D_optimizer.zero_grad()
        D_loss_val = backward_D(G_target, D_target, real_image, samples_d_update, no_gp=no_gp, loss_mask=loss_mask)
        D_loss_val.backward(create_graph=True)
        param_diffs = step_fn(D_optimizer)[0]
        grads_filtered = tuple([g for g in param_diffs if g is not None])
        us_D_filtered = tuple([u for u, p in zip(us_D, D_target.trainable_parameters()) if p.grad is not None])

        hvps_D, hvps_G, influences_scaled = myjvp(grads_filtered, (D_target.trainable_parameters(), G_target.trainable_parameters(), loss_mask), us_D_filtered)

        influences[real_index.numpy()] -= scale * influences_scaled.squeeze(-1).cpu().numpy()

        with torch.no_grad():
            for u, hvp in zip(us_D, hvps_D):
                u.copy_((1. - damping) * u.clone().detach() + scale * hvp.clone().detach())
            for u, hvp in zip(us_G, hvps_G):
                u.copy_((1. - damping) * u.clone().detach() + scale * hvp.clone().detach())

        for param in D_target.parameters():
            param.grad = None

        nonzero_abs_influences = np.abs(influences[influences.nonzero()])
        mean_influence = np.mean(nonzero_abs_influences)
        max_influence = np.max(nonzero_abs_influences)
        min_influence = np.min(nonzero_abs_influences)
        pbar.set_description(f'mean={mean_influence:.2e}, max={max_influence:.2e}, min={min_influence:.2e}')

        if _run is not None:
            _run.log_scalar('mean_influence', mean_influence, idx)
            _run.log_scalar('max_influence', max_influence, idx)
            _run.log_scalar('min_influence', min_influence, idx)

        # release unused gpu memory
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary())
    return influences


def compute_aid_influence(D_target, G_target, _run, damping, dataset, df_ckpt, grads_metric_G, idx_end, no_gp, on_averaged_G, scale, depth, batch_size):
    us_G = tuple([p.clone().detach() for p in grads_metric_G])
    us_D = tuple([torch.zeros_like(p) for p in D_target.trainable_parameters()])
    D_target.load_state_dict(torch.load(df_ckpt.iloc[idx_end]['discriminator']), strict=False)
    if on_averaged_G:
        G_target.load_state_dict(torch.load(df_ckpt.iloc[idx_end]['generator_ave']), strict=False)
    else:
        G_target.load_state_dict(torch.load(df_ckpt.iloc[idx_end]['generator']), strict=False)

    dataset_size = len(dataset)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=1)
    pbar = tqdm(range(depth), position=0, total=depth)
    for idx in pbar:
        loader_iter = iter(loader)
        try:
            real_index, real_image = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            real_index, real_image = next(loader_iter)

        samples_d_update = G_target.sample_train_noise(len(real_image))
        samples_g_update = G_target.sample_train_noise(len(real_image))

        G_target.zero_grad()

        G_loss_val = backward_G(G_target, D_target, samples_g_update)
        grads_G = mygrad(G_loss_val, G_target.trainable_parameters(), create_graph=True, allow_unused=True)
        grads_G_filtered = tuple([g for g in grads_G if g is not None])
        us_G_filtered = tuple([u for u, g in zip(us_G, grads_G) if g is not None])

        hvps_D, hvps_G = myjvp(grads_G_filtered, (D_target.trainable_parameters(), G_target.trainable_parameters()), us_G_filtered)

        with torch.no_grad():
            for u, hvp in zip(us_D, hvps_D):
                u.copy_((1. - damping) * u.clone().detach() - scale * hvp.clone().detach())
            for u, hvp, g in zip(us_G, hvps_G, grads_metric_G):
                u.copy_((1. - damping) * u.clone().detach() - scale * hvp.clone().detach()) + g.clone().detach()

        # Reset gradients to None after backward to avoid memory leak
        for param in G_target.parameters():
            param.grad = None

        D_target.zero_grad()

        loss_mask = torch.nn.Parameter(torch.ones((len(real_index), 1)).cuda())
        D_loss_val = backward_D(G_target, D_target, real_image, samples_d_update, no_gp=no_gp, loss_mask=loss_mask)
        grads_D = mygrad(D_loss_val, D_target.trainable_parameters(), create_graph=True, allow_unused=True)
        grads_D_filtered = tuple([g for g in grads_D if g is not None])
        us_D_filtered = tuple([u for u, g in zip(us_D, grads_D) if g is not None])

        hvps_D, hvps_G = myjvp(grads_D_filtered, (D_target.trainable_parameters(), G_target.trainable_parameters()), us_D_filtered)

        with torch.no_grad():
            for u, hvp in zip(us_D, hvps_D):
                u.copy_((1. - damping) * u.clone().detach() - scale * hvp.clone().detach())
            for u, hvp, g in zip(us_G, hvps_G, grads_metric_G):
                u.copy_((1. - damping) * u.clone().detach() - scale * hvp.clone().detach()) + g.clone().detach()

        for param in D_target.parameters():
            param.grad = None

        norm_u_G = torch.norm(torch.cat([u.view(-1) for u in us_G], dim=0))
        norm_u_D = torch.norm(torch.cat([u.view(-1) for u in us_D], dim=0))
        pbar.set_description(f'norm_u_G={norm_u_G:.2e}, norm_u_D={norm_u_D:.2e}')

        if _run is not None:
            _run.log_scalar('norm_u_G', norm_u_G, idx)
            _run.log_scalar('norm_u_D', norm_u_D, idx)

        # release unused gpu memory
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary())

    influences = np.zeros(dataset_size)
    for real_index, real_image in loader:
        samples_d_update = G_target.sample_train_noise(len(real_image))
        G_target.zero_grad()
        D_target.zero_grad()
        loss_mask = torch.nn.Parameter(torch.ones((len(real_index), 1)).cuda())
        D_loss_val = backward_D(G_target, D_target, real_image, samples_d_update, no_gp=no_gp, loss_mask=loss_mask)
        grads_D = mygrad(D_loss_val, D_target.trainable_parameters(), allow_unused=True)
        grads_D_filtered = tuple([g for g in grads_D if g is not None])
        us_D_filtered = tuple([u for u, g in zip(us_D, grads_D) if g is not None])

        influences_scaled = myjvp(grads_D_filtered, loss_mask, us_D_filtered)

        influences[real_index.numpy()] = scale * influences_scaled.squeeze(-1).cpu().numpy()

    return influences


def run_isolation_forest(acts_train):
    clf = IsolationForest()
    clf.fit(acts_train)
    return clf.score_samples(acts_train)


def cal_true_influence(train_dir, name_model, mixing, rank_G, rank_D, n_epochs, lr_G, lr_D, name_optimizer, target_indices, from_epoch, to_epoch, from_step, n_steps, image_size,
        accumulate_decay=0.999, ckpt=None, no_gp=False):
    assert from_epoch < to_epoch <= n_epochs, 'epochs should be less than n_epochs'

    target_indices = torch.tensor(target_indices)

    D_target, G_target = load_models(name_model, image_size, ckpt, rank_D, rank_G, mixing)

    if name_optimizer == 'adam':
        G_optimizer = optim.Adam(G_target.trainable_parameters(), lr=lr_G, betas=(0.0, 0.99))
        D_optimizer = optim.Adam(D_target.trainable_parameters(), lr=lr_D, betas=(0.0, 0.99))
    elif name_optimizer == 'sgd':
        G_optimizer = optim.SGD(G_target.trainable_parameters(), lr=lr_G)
        D_optimizer = optim.SGD(D_target.trainable_parameters(), lr=lr_D)
    else:
        raise NotImplementedError(f'Optimizer {name_optimizer} is not supported')

    ## load LoRA parameters
    df_ckpt = pd.read_json(os.path.join(train_dir, 'df_ckpt.json'))
    idx_start = df_ckpt[(df_ckpt['epoch'] == from_epoch) & (df_ckpt['step'] == from_step)].index[0]
    idx_end = df_ckpt[(df_ckpt['epoch'] == to_epoch - 1) & (df_ckpt['step'] == n_steps - 1)].index[0] + 1

    # load the intermediate  model to compute the influence
    D_target.load_state_dict(torch.load(df_ckpt.iloc[idx_start]['discriminator']), strict=False)
    G_target.load_state_dict(torch.load(df_ckpt.iloc[idx_start]['generator']), strict=False)
    D_optimizer.load_state_dict(torch.load(df_ckpt.iloc[idx_start]['optimizer_d']))
    G_optimizer.load_state_dict(torch.load(df_ckpt.iloc[idx_start]['optimizer_g']))

    if name_model == 'stylegan':
        G_running_target = StyledGenerator(image_size=image_size, rank=rank_G, mixing=mixing).cuda()
    elif name_model == 'lqgan':
        G_running_target = ToyGenerator().cuda()
    else:
        raise NotImplementedError(f'Model {name_model} is not supported')

    # copy parameters including base parameters
    accumulate(G_running_target.named_parameters(), G_target.named_parameters(), decay=0)
    # load and overwrite averaged LoRA parameters
    G_running_target.load_state_dict(torch.load(df_ckpt.iloc[idx_start]['generator_ave']), strict=False)
    G_running_target.train(False)

    set_requires_grad(G_target, False, only_trainable=False)
    set_requires_grad(D_target, False, only_trainable=False)

    for idx in tqdm(range(idx_start, idx_end)):
        samples = torch.load(df_ckpt.iloc[idx]['samples'])
        real_index = samples['real_index']
        real_image = samples['real_image']
        samples_d_update = samples['samples_d_update']
        samples_g_update = samples['samples_g_update']

        D_target.zero_grad()

        set_requires_grad(G_target, False)
        set_requires_grad(D_target, True)
        ### update D ###
        isin_real_index = torch.isin(real_index, target_indices)
        if torch.any(isin_real_index):
            loss_mask = (~isin_real_index).float().unsqueeze(-1).cuda()
            D_loss_val = backward_D_scaled(G_target, D_target, real_image, samples_d_update, no_gp=no_gp, loss_mask=loss_mask)
        else:
            D_loss_val = backward_D_scaled(G_target, D_target, real_image, samples_d_update, no_gp=no_gp)
        D_loss_val.backward()

        D_optimizer.step()

        ### save results and checkpoints ###
        ### update G ###
        G_target.zero_grad()

        set_requires_grad(G_target, True)
        set_requires_grad(D_target, False)
        G_loss_val = backward_G(G_target, D_target, samples_g_update)
        G_loss_val.backward()

        G_optimizer.step()
        accumulate(G_running_target.named_trainable_parameters(), G_target.named_trainable_parameters(), decay=accumulate_decay)

    state_dicts = {
        'discriminator': {name: v for name, v in D_target.state_dict().items() if name in D_target.trainable_parameter_names()},
        'generator': {name: v for name, v in G_target.state_dict().items() if name in G_target.trainable_parameter_names()},
        'generator_ave': {name: v for name, v in G_running_target.state_dict().items() if name in G_target.trainable_parameter_names()},
        'optimizer_d': D_optimizer.state_dict(),
        'optimizer_g': G_optimizer.state_dict()
    }

    return state_dicts

def load_latest_model(train_dir, to_epoch, n_steps):
    ## load LoRA parameters
    df_ckpt = pd.read_json(os.path.join(train_dir, 'df_ckpt.json'))
    idx_end = df_ckpt[(df_ckpt['epoch'] == to_epoch - 1) & (df_ckpt['step'] == n_steps - 1)].index[0] + 1

    state_dicts = {
        'discriminator': torch.load(df_ckpt.iloc[idx_end]['discriminator']),
        'generator': torch.load(df_ckpt.iloc[idx_end]['generator']),
        'generator_ave': torch.load(df_ckpt.iloc[idx_end]['generator_ave']),
        'optimizer_d': torch.load(df_ckpt.iloc[idx_end]['optimizer_d']),
        'optimizer_g': torch.load(df_ckpt.iloc[idx_end]['optimizer_g'])
    }

    return state_dicts

def eval_latest_model(name_model, mixing, rank_G, rank_D, real_acts, noise_dataset, state_dicts, name_metric, on_averaged_G, image_size, ckpt=None):
    D_target, G_target = load_models(name_model, image_size, ckpt, rank_D, rank_G, mixing)

    ### prepare evaluation metrics ###
    # Example initialization within main function or setup routine
    if name_model == 'stylegan':
        evaluator = StyleGANEvaluator(G_target, real_acts=real_acts, noise_dataset=noise_dataset)
    elif name_model == 'lqgan':
        evaluator = LQGANEvaluator(G_target, noise_dataset)
    else:
        raise NotImplementedError(f'Model {name_model} is not supported')

    # load the last model to compute original valid fid
    if on_averaged_G:
        G_target.load_state_dict(state_dicts['generator_ave'], strict=False)
    else:
        G_target.load_state_dict(state_dicts['generator'], strict=False)
    D_target.load_state_dict(state_dicts['discriminator'], strict=False)
    with torch.no_grad():
        metric = evaluator.evaluate(name_metric).item()

    return metric

def sample_images(name_model, image_size, mixing, rank_G, rank_D, state_dict, grid_size, on_averaged_G, ckpt=None, _run=None):
    _, G_target = load_models(name_model, image_size, ckpt, rank_D, rank_G, mixing)

    ### prepare evaluation metrics ###
    # Example initialization within main function or setup routine
    if name_model == 'stylegan':
        evaluator = StyleGANEvaluator(G_target, output_dir='./sample/tmp/', _run=_run, grid_size=(grid_size))
    elif name_model == 'lqgan':
        evaluator = LQGANEvaluator(G_target, _run=_run)
    else:
        raise NotImplementedError(f'Model {name_model} is not supported')

    if on_averaged_G:
        G_target.load_state_dict(state_dict['generator_ave'], strict=False)
    else:
        G_target.load_state_dict(state_dict['generator'], strict=False)

    with torch.no_grad():
        fake_images = evaluator.gen_and_save_fake_images(f'fake_images.jpg')

    return fake_images

def sample_influential_instances(dataset, influences, grid_size, name_metric, out_dir, _run=None):
    assert len(dataset) == len(influences), 'Dataset and influences must have the same length: {len(dataset)} != {len(influences)}'

    n_samples = grid_size[0] * grid_size[1]
    if n_samples > len(dataset):
        raise ValueError(f'Number of harmful instances ({n_samples}) exceeds the dataset size ({len(dataset)})')

    sorted_indices = np.argsort(influences)
    if name_metric in SetMetric.NEGATIVE:
        harmful_indices = sorted_indices[:n_samples]
        helpful_indices = sorted_indices[-n_samples:]
    elif name_metric in SetMetric.POSITIVE:
        harmful_indices = sorted_indices[-n_samples:]
        helpful_indices = sorted_indices[:n_samples]
    else:
        raise ValueError(f'Unknown metric: {name_metric}')

    for indices, label in zip((harmful_indices, helpful_indices), ('harmful', 'helpful')):
        images = torch.cat([dataset[i][1].unsqueeze(0) for i in indices], dim=0)
        # Save fake images - adjust path as needed
        sample_path = os.path.join(out_dir, f'{label}_instances.png')
        os.makedirs(os.path.dirname(sample_path), exist_ok=True)
        utils.save_image(images, sample_path, nrow=grid_size[0], normalize=True, value_range=(-1, 1))
        if _run is not None:
            _run.add_artifact(sample_path)

    return {'harmful': harmful_indices, 'helpful': helpful_indices}

def plot_cleansing_result_wrt_from_epoch(metrics_clean, metric_ori, from_epochs, name_metric_eval, output_path, _run=None):
    print('Influence')
    for from_epoch, metric_clean in zip(from_epochs, metrics_clean):
        # print actual_diff with sign
        print(f'from_epoch: {from_epoch}, result: {metric_clean:.4f} ({metric_clean - metric_ori})')
    print('Random')

    # plot metrics and influences_sums in different y axes.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(from_epochs, metrics_clean, label='cleansed', marker='o', linestyle='')
    ax.plot(from_epochs, np.ones_like(from_epochs) * metric_ori, label='original', color='grey')
    ax.set_xlabel('From epoch')
    ax.set_ylabel(D_METRIC_LABEL[name_metric_eval])
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(output_path, 'influence.png'))
    if _run is not None:
        _run.add_artifact(os.path.join(output_path, 'influence.png'))

    return fig

def plot_cleansing_result_wrt_removal_rate(metrics_clean, metric_ori, removal_rates, name_metric_eval, output_path, _run=None):
    print('Influence')
    for removal_rate, metric_clean in zip(removal_rates, metrics_clean):
        # print actuall_diff with sign
        print(f'removal_rate: {removal_rate}, result: {metric_clean:.4f} ({metric_clean - metric_ori})')
    print('Random')

   # plot metrics and influences_sums in different y axes.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(removal_rates, metrics_clean, label='cleansed', marker='o')
    ax.plot(removal_rates, np.ones_like(removal_rates) * metric_ori, label='original', color='grey')
    ax.set_xlabel('Removal rate')
    ax.set_ylabel(D_METRIC_LABEL[name_metric_eval])
    ax.set_xscale('log')
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(output_path, 'influence.png'))
    if _run is not None:
        _run.add_artifact(os.path.join(output_path, 'influence.png'))

    return fig
