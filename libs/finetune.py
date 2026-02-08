# Modified from FreezeD (https://github.com/sangwoomo/FreezeD/tree/master/stylegan).
#
# MIT License
#
# Copyright (c) 2019 Kim Seonghyeon
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import argparse
from tqdm import tqdm

from torch import optim
from torch.nn import functional as F
from torch.autograd import grad
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from libs.dataset import MultiResolutionDataset
from libs.evaluator import StyleGANEvaluator, LQGANEvaluator
from libs.models import StyledGenerator, Discriminator, ToyGenerator, ToyDiscriminator
import torch
import pandas as pd


def accumulate(named_par1, named_par2, decay=0.999):
    par1 = dict(named_par1)
    par2 = dict(named_par2)

    with torch.no_grad():
        for k in par1.keys():
            par1[k].data.mul_(decay).add_(1. - decay, par2[k].data)


def set_requires_grad(model, value, only_trainable=True):
    if only_trainable:
        for param in model.trainable_parameters():
            param.requires_grad = value
    else:
        for param in model.parameters():
            param.requires_grad = value

def backward_D(G_target, D_target, real_image, samples, no_gp=False, loss_mask=None):
    ### update D (GAN loss) ###
    real_image = real_image.cuda()

    if real_image.shape[0] > 0:

        if not no_gp:
            real_image.requires_grad = True

        real_predict = D_target(real_image)  # before activation
        if loss_mask is not None:
            if not loss_mask.shape == real_predict.shape:
                raise ValueError(f'loss_mask.shape: {loss_mask.shape} is not equal to real_predict.shape: {real_predict.shape}')
            D_loss_real = (F.softplus(-real_predict) * loss_mask).mean()
        else:
            D_loss_real = F.softplus(-real_predict).mean()
        if no_gp:
            grad_penalty = 0.
        else:
            grad_real = grad(outputs=real_predict.sum(), inputs=real_image, create_graph=True)[0]
            grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
            grad_penalty = 10 / 2 * grad_penalty

    else:
        D_loss_real = 0.
        grad_penalty = 0.

    fake_image_tgt = G_target(**samples)
    fake_predict = D_target(fake_image_tgt)
    D_loss_fake = F.softplus(fake_predict).mean()

    D_loss = D_loss_real + D_loss_fake + grad_penalty

    return D_loss

def backward_D_scaled(G_target, D_target, real_image, samples, no_gp=False, loss_mask=None):
    ### update D (GAN loss) ###
    real_image = real_image.cuda()

    if real_image.shape[0] > 0:

        if not no_gp:
            real_image.requires_grad = True

        real_predict = D_target(real_image)  # before activation
        if loss_mask is not None:
            if not loss_mask.shape == real_predict.shape:
                raise ValueError(f'loss_mask.shape: {loss_mask.shape} is not equal to real_predict.shape: {real_predict.shape}')
            is_used = loss_mask.bool()
            D_loss_real = F.softplus(-real_predict)[is_used].mean()

        else:
            D_loss_real = F.softplus(-real_predict).mean()
        if no_gp:
            grad_penalty = 0.
        else:
            grad_real = grad(outputs=real_predict.sum(), inputs=real_image, create_graph=True)[0]
            grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
            grad_penalty = 10 / 2 * grad_penalty

    else:
        D_loss_real = 0.
        grad_penalty = 0.

    fake_image_tgt = G_target(**samples)
    fake_predict = D_target(fake_image_tgt)
    D_loss_fake = F.softplus(fake_predict).mean()

    D_loss = D_loss_real + D_loss_fake + grad_penalty

    return D_loss


def backward_G(G_target, D_target, samples):
    ### update G (GAN loss) ###

    fake_image_tgt = G_target(**samples)
    predict = D_target(fake_image_tgt)
    G_loss = F.softplus(-predict).mean()

    return G_loss

def collect_param_info(named_parameters):
    param_info = []
    accumulated_size = 0
    for name, param in named_parameters:
        size = param.data.nelement()  # Getting the parameter size
        accumulated_size += size
        param_info.append((name, size, accumulated_size))
    return param_info

# Function to collect parameter info from an optimizer
def print_param_info(param_info, optimizer_name):
    print(f"Optimizer: {optimizer_name}")
    print(f"{'Parameter ID':>12} {'Size':>10} {'Accumulated Size':>20}")
    print("-" * 80)
    for name, size, acc_size in param_info:
        print(f"{name:>40} {size:>10} {acc_size:>20}")
    print("-" * 80)
    print(f"{'Total':>40} {'':>10} {acc_size:>20}\n")


def load_models(name_model, image_size, ckpt, rank_D, rank_G, mixing):
    if name_model == 'stylegan':
        G_target = StyledGenerator(image_size=image_size, rank=rank_G, mixing=mixing).cuda()
        D_target = Discriminator(image_size=image_size, rank=rank_D).cuda()
    elif name_model == 'lqgan':
        G_target = ToyGenerator().cuda()
        D_target = ToyDiscriminator().cuda()
    else:
        raise NotImplementedError(f'Model {name_model} is not supported')

    ## load base parameters
    if ckpt:
        state_dict = torch.load(ckpt)
        missing_keys, unexpected_keys = G_target.load_state_dict(state_dict['generator'], strict=False)
        assert len(unexpected_keys) == 0, f'Error in loading generator: {unexpected_keys}'
        assert set(G_target.trainable_parameter_names()) == set(missing_keys), f'Missing keys in generator: {set(missing_keys) - set(G_target.trainable_parameter_names())}'

        missing_keys, unexpected_keys = D_target.load_state_dict(state_dict['discriminator'], strict=False)
        assert len(unexpected_keys) == 0, f'Error in loading discriminator: {unexpected_keys}'
        assert set(D_target.trainable_parameter_names()) == set(missing_keys), f'Missing keys in discriminator: {set(missing_keys) - set(D_target.trainable_parameter_names())}'

    return D_target, G_target


def finetune(
        output_dir, name_model, dataset, acts_valid, mixing, rank_G, rank_D, batch_size, n_epochs, eval_step, lr_G, lr_D, name_optimizer, names_metric, image_size,
        accumulate_decay=0.999, sample_num=5000, ckpt=None, no_gp=False, train_only=False, _run=None
):

    ### load G and D ###
    D_target, G_target = load_models(name_model, image_size, ckpt, rank_D, rank_G, mixing)

    if name_model == 'stylegan':
        G_running_target = StyledGenerator(image_size=image_size, rank=rank_G, mixing=mixing).cuda()
    elif name_model == 'lqgan':
        G_running_target = ToyGenerator().cuda()
    else:
        raise NotImplementedError(f'Model {name_model} is not supported')

    G_running_target.train(False)
    accumulate(G_running_target.named_parameters(), G_target.named_parameters(), 0.)

    if name_optimizer == 'adam':
        G_optimizer = optim.Adam(G_target.trainable_parameters(), lr=lr_G, betas=(0.0, 0.99))
        D_optimizer = optim.Adam(D_target.trainable_parameters(), lr=lr_D, betas=(0.0, 0.99))
    elif name_optimizer == 'rmsprop':
        G_optimizer = optim.RMSprop(G_target.trainable_parameters(), lr=lr_G)
        D_optimizer = optim.RMSprop(D_target.trainable_parameters(), lr=lr_D)
    elif name_optimizer == 'sgd':
        G_optimizer = optim.SGD(G_target.trainable_parameters(), lr=lr_G)
        D_optimizer = optim.SGD(D_target.trainable_parameters(), lr=lr_D)
    else:
        raise NotImplementedError(f'Optimizer {name_optimizer} is not supported')

    # Collecting parameter info from G_optimizer and D_optimizer
    g_param_info = collect_param_info([(name, param) for name, param in G_target.named_trainable_parameters()])
    d_param_info = collect_param_info([(name, param) for name, param in D_target.named_trainable_parameters()])

    # Printing parameter info for both optimizers
    print_param_info(g_param_info, "G_target")
    print_param_info(d_param_info, "D_target")

    ### create logger ###
    sample_dir = os.path.join(output_dir, 'sample')
    ckpt_dir = os.path.join(output_dir, 'checkpoint')
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = SummaryWriter(output_dir)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=1)

    ### prepare evaluation metrics ###
    # Example initialization within main function or setup routine
    if name_model == 'stylegan':
        evaluator = StyleGANEvaluator(G_running_target, acts_valid, sample_num=sample_num, output_dir=sample_dir, _run=_run)
    elif name_model == 'lqgan':
        evaluator = LQGANEvaluator(G_running_target, _run=_run)
    else:
        raise NotImplementedError(f'Model {name_model} is not supported')

    ### run experiment ###
    metrics = {}
    if eval_step > 0:
        evaluator.gen_and_save_fake_images(f'{str(0).zfill(6)}.png')
        with torch.no_grad():
            for name_metric in names_metric:
                metrics[name_metric] = evaluator.evaluate(name_metric)

        for key, val in metrics.items():
            logger.add_scalar(key, val, 0)
            evaluator.log_metrics(key, val, 0)

    set_requires_grad(G_target, False, only_trainable=False)
    set_requires_grad(D_target, False, only_trainable=False)
    df_ckpt = pd.DataFrame(columns=['epoch', 'step', 'generator', 'generator_ave', 'discriminator', 'optimizer_g', 'optimizer_d', 'samples'])
    step_total = 0
    for epoch in range(n_epochs):
        loader_iter = iter(loader)
        pbar = tqdm(enumerate(loader_iter), position=0, total=len(loader))
        for step_in_epoch, (real_index, real_image) in pbar:
            ### sample data and noise ###
            samples_d_update = G_target.sample_train_noise(len(real_image))
            samples_g_update = G_target.sample_train_noise(len(real_image))

            ### save samples ###
            if not train_only:
                d_ckpt = {
                    'epoch': epoch,
                    'step': step_in_epoch,
                    'generator': os.path.join(ckpt_dir, f'G_target_{epoch}_{step_in_epoch}.pth'),
                    'generator_ave': os.path.join(ckpt_dir, f'G_running_target_{epoch}_{step_in_epoch}.pth'),
                    'discriminator': os.path.join(ckpt_dir, f'D_target_{epoch}_{step_in_epoch}.pth'),
                    'optimizer_g': os.path.join(ckpt_dir, f'G_optimizer_{epoch}_{step_in_epoch}.pth'),
                    'optimizer_d': os.path.join(ckpt_dir, f'D_optimizer_{epoch}_{step_in_epoch}.pth'),
                    'samples': os.path.join(ckpt_dir, f'samples_{epoch}_{step_in_epoch}.pth'),
                }
                df_ckpt.loc[step_total] = d_ckpt
                torch.save({name: v for name, v in D_target.state_dict().items() if name in D_target.trainable_parameter_names()}, d_ckpt['discriminator'])
                torch.save({name: v for name, v in G_target.state_dict().items() if name in G_target.trainable_parameter_names()}, d_ckpt['generator'])
                torch.save({name: v for name, v in G_running_target.state_dict().items() if name in G_running_target.trainable_parameter_names()}, d_ckpt['generator_ave'])
                torch.save(D_optimizer.state_dict(), d_ckpt['optimizer_d'])
                torch.save(G_optimizer.state_dict(), d_ckpt['optimizer_g'])
                d_samples = {
                    'real_index': real_index,
                    'real_image': real_image,
                    'samples_d_update': samples_d_update,
                    'samples_g_update': samples_g_update,
                }
                torch.save(d_samples, d_ckpt['samples'])
                df_ckpt.to_json(os.path.join(output_dir, 'df_ckpt.json'))

            ### update D ###
            D_target.zero_grad()

            set_requires_grad(G_target, False)
            set_requires_grad(D_target, True)
            D_loss_val = backward_D(G_target, D_target, real_image, samples_d_update, no_gp=no_gp)
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

            ### save results and checkpoints ###
            accumulate(G_running_target.named_trainable_parameters(), G_target.named_trainable_parameters(), decay=accumulate_decay)

            ### save results and checkpoints ###
            if eval_step > 0:
                if (step_total + 1) % eval_step == 0:
                    evaluator.gen_and_save_fake_images(f'{str(step_total+1).zfill(6)}.png')

                    logger.add_scalar('G_loss_val', G_loss_val.item(), step_total + 1)
                    evaluator.log_metrics('G_loss_val', G_loss_val.item(), step_total + 1)
                    logger.add_scalar('D_loss_val', D_loss_val.item(), step_total + 1)
                    evaluator.log_metrics('D_loss_val', D_loss_val.item(), step_total + 1)

                    with torch.no_grad():
                        for name_metric in names_metric:
                            val = evaluator.evaluate(name_metric)
                            metrics[name_metric] = val
                            logger.add_scalar(name_metric, val, step_total + 1)
                            evaluator.log_metrics(name_metric, val, step_total + 1)

                state_msg = f'Epoch: {epoch + 1}/{n_epochs}; G: {G_loss_val.item():.3f}; D: {D_loss_val.item():.3f}'
                if metrics is not None:
                    state_msg += '; '.join([f' {key}: {val:.2f}' for (key, val) in metrics.items()])

            else:
                state_msg = f'Epoch: {epoch + 1}/{n_epochs}; G: {G_loss_val.item():.3f}; D: {D_loss_val.item():.3f}'

            pbar.set_description(state_msg)
            step_total += 1

    d_ckpt = {
        'epoch': -1,
        'step': -1,
        'generator': os.path.join(ckpt_dir, 'G_target_final.pth'),
        'generator_ave': os.path.join(ckpt_dir, 'G_running_target_final.pth'),
        'discriminator': os.path.join(ckpt_dir, 'D_target_final.pth'),
        'optimizer_g': os.path.join(ckpt_dir, 'G_optimizer_final.pth'),
        'optimizer_d': os.path.join(ckpt_dir, 'D_optimizer_final.pth'),
        'samples': {},
    }
    torch.save({name: v for name, v in D_target.state_dict().items() if name in D_target.trainable_parameter_names()}, d_ckpt['discriminator'])
    torch.save({name: v for name, v in G_target.state_dict().items() if name in G_target.trainable_parameter_names()}, d_ckpt['generator'])
    torch.save({name: v for name, v in G_running_target.state_dict().items() if name in G_running_target.trainable_parameter_names()}, d_ckpt['generator_ave'])
    torch.save(D_optimizer.state_dict(), d_ckpt['optimizer_d'])
    torch.save(G_optimizer.state_dict(), d_ckpt['optimizer_g'])
    df_ckpt.loc[step_total] = d_ckpt
    df_ckpt.to_json(os.path.join(output_dir, 'df_ckpt.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Progressive Growing of GANs (fine-tuning)')

    parser.add_argument('--dataset', type=str, required=True, help='dataset name')
    parser.add_argument('--name', type=str, default='temp', help='name of experiment')
    parser.add_argument('--name_model', type=str, default='stylegan', help='name of experiment')
    parser.add_argument('--ckpt', type=str, default='./checkpoint/stylegan-256px-new.model', help='source model')

    parser.add_argument('--n_epochs', type=int, default=150, help='number of samples used for each training n_epochss')
    parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=8, type=int, help='batch-size')
    parser.add_argument('--mixing', action='store_true', help='use mixing regularization')
    parser.add_argument('--loss', type=str, default='r1', choices=['r1'], help='class of gan loss')
    parser.add_argument('--eval_step', default=1000, type=int, help='step size for evaluation')
    parser.add_argument('--sample_num', default=5000, type=int, help='number of samples for evaluation')
    parser.add_argument('--optimizer', default='adam', type=str, help='name of the optimizer')

    parser.add_argument('--rank_G', type=int, default=16)
    parser.add_argument('--rank_D', type=int, default=16)

    args = parser.parse_args()

    finetune(args.name, args.name_model, args.dataset, args.mixing, args.rank_G, args.rank_D, args.batch_size, args.n_epochs, args.eval_step, args.lr, args.optimizer, args.sample_num, ['fid'], ckpt=args.ckpt, no_gp=False, train_only=True)
