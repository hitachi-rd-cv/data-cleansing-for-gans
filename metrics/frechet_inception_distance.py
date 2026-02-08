import torch
import scipy
from torch.autograd import Function
import numpy as np
from scipy import linalg
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy import linalg  # For numpy FID
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter as P
from torchvision.models.inception import inception_v3


def frechet_inception_distance(act1, act2):
    mu1, sigma1 = calculate_activation_statistics(act1)
    mu2, sigma2 = calculate_activation_statistics(act2)
    return torch_calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    # return calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    # mu1 = np.atleast_1d(mu1)
    # mu2 = np.atleast_1d(mu2)
    #
    # sigma1 = np.atleast_2d(sigma1)
    # sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean = sqrtm(torch.mm(sigma1, sigma2))

    # if not np.isfinite(covmean).all():
    #     msg = ('fid calculation produces singular product; '
    #            'adding %s to diagonal of cov estimates') % eps
    #     print(msg)
    #     offset = np.eye(sigma1.shape[0]) * eps
    #     covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    # if np.iscomplexobj(covmean):
    #     if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
    #         m = np.max(np.abs(covmean.imag))
    #         raise ValueError('Imaginary component {}'.format(m))
    #     covmean = covmean.real
    #
    tr_covmean = torch.trace(covmean)

    return (diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean)


# from https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py
def calculate_activation_statistics(act, batch_size=50,
                                    dims=2048, cuda=False, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    mu = torch.mean(act, dim=0)
    sigma = cov(act, rowvar=False)
    return mu, sigma


# def get_metric(self, metric_name, *args):
#     if metric_name in ('loss_d', 'loss_g'):
#         if metric_name == 'loss_d':
#             target = 'discriminator'
#         elif metric_name == 'loss_g':
#             target = 'generator'
#         return self.get_losses(target, *args)
#
#     if metric_name in ('sum_g', 'sum_d'):
#         if metric_name == 'sum_g':
#             params = self.generator.parameters()
#         elif metric_name == 'sum_d':
#             params = self.discriminator.parameters()
#         sum = 0
#         for p in params:
#             sum += p.sum()
#         return sum
#     else:
#         raise NotImplementedError

# https://github.com/pytorch/pytorch/issues/19037
def cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    """Estimates covariance matrix like numpy.cov"""
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w/w_sum)[:,None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze()

class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.
    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
    """
    @staticmethod
    def forward(ctx, input):
        m = input.detach().cpu().numpy().astype(np.float_)
        sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).to(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input

sqrtm = MatrixSquareRoot.apply

# from https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py
def calculate_frechet_distance_np(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

# from https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py
def calculate_activation_statistics_np(act):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma
#
# def frechet_inception_distance_np(act1, act2):
#     mu1, sigma1 = calculate_activation_statistics_np(act1)
#     mu2, sigma2 = calculate_activation_statistics_np(act2)
#     return calculate_frechet_distance_np(mu1, sigma1, mu2, sigma2)

def frechet_inception_distance_np(real_imgs, gen_imgs):
    m = np.mean(real_imgs, axis=0)
    m_v = np.mean(gen_imgs, axis=0)
    sigma = np.cov(real_imgs, rowvar=False)
    sigma_v = np.cov(gen_imgs, rowvar=False)
    sqcc = scipy.linalg.sqrtm(np.dot(sigma, sigma_v))
    mean = np.square(m - m_v).sum()
    trace = np.trace(sigma + sigma_v - 2 * sqcc)
    fid = mean + trace
    return fid

def test_torch_fid(act1, act2):
    fid_torch = frechet_inception_distance(act1, act2)
    fid_np = frechet_inception_distance_np(act1, act2)
    print('torch: {}, numpy: {}'.format(fid_torch, fid_np))


def backward_fid(dataset, generator, inception, real_acts, batch_size=16, inputs_backward=None, input_processor=None, output_processor=None, feature_processor=None):
    loader = DataLoader(dataset, batch_size=batch_size)

    pbar = tqdm(total=len(dataset), position=1, leave=False)
    pbar.set_description('Pre-computing activations')

    if inputs_backward is None:
        inputs_backward = tuple(generator.parameters())

    # Compute fake activations without gradient for memory efficiency
    with torch.no_grad():
        fake_acts_list = []
        for input in loader:
            if isinstance(input, (list, tuple)):
                sample_size = input[0].size(0)
                if input_processor is not None:
                    input = input_processor(*input)
                fake_image = generator(*input)
            else:
                sample_size = input.size(0)
                if input_processor is not None:
                    input = input_processor(input)
                fake_image = generator(input)

            if output_processor is not None:
                fake_image = output_processor(fake_image)

            feature = inception(fake_image)
            if feature_processor is not None:
                feature = feature_processor(feature)

            out = feature.squeeze(-1).squeeze(-1)
            fake_acts_list.append(out)
            pbar.update(sample_size)

    # Concatenate all pre-computed activations
    fake_acts = torch.cat(fake_acts_list, axis=0)

    pbar = tqdm(total=len(dataset), position=1, leave=False)
    pbar.set_description('Computing gradients')

    generator.zero_grad()
    start_index = 0
    for input in loader:
        # generator.zero_grad()
        # inception.zero_grad()

        # Recompute a batch of fake activations with gradient
        if isinstance(input, (list, tuple)):
            sample_size = input[0].size(0)
            if input_processor is not None:
                input = input_processor(*input)
            fake_image = generator(*input)
        else:
            sample_size = input.size(0)
            if input_processor is not None:
                input = input_processor(input)
            fake_image = generator(input)

        if output_processor is not None:
            fake_image = feature_processor(fake_image)

        feature = inception(fake_image)
        if feature_processor is not None:
            feature = feature_processor(feature)

        out = feature.squeeze(-1).squeeze(-1)

        # Determine the start and end indices for the current batch
        end_index = start_index + sample_size

        # Replace the corresponding part of pre-computed fake_acts
        fake_acts_batch = fake_acts.clone().detach()  # Detach and clone for safe replacement
        fake_acts_batch[start_index:end_index] = out

        # Compute FID and gradient for the current batch
        fid = frechet_inception_distance(real_acts, fake_acts_batch)

        # Accumulate gradients
        fid.backward(inputs=inputs_backward)

        # Update the start index for the next batch
        start_index = end_index

        pbar.update(sample_size)

    return fid

# Pytorch implementation of matrix sqrt, from Tsung-Yu Lin, and Subhransu Maji
# https://github.com/msubhransu/matrix-sqrt
def sqrt_newton_schulz(A, numIters, dtype=None):
    if dtype is None:
        dtype = A.type()
    batchSize = A.shape[0]
    dim = A.shape[1]
    normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
    Y = A.div(normA.view(batchSize, 1, 1).expand_as(A));
    I = torch.eye(dim, dim).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype)
    Z = torch.eye(dim, dim).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype)
    for i in range(numIters):
        T = 0.5 * (3.0 * I - Z.bmm(Y))
        Y = Y.bmm(T)
        Z = T.bmm(Z)
    sA = Y * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
    return sA

def torch_calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Pytorch implementation of the Frechet Distance.
    Taken from https://github.com/bioinf-jku/TTUR
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representive data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2
    # Run 20 itrs of newton-schulz to get the matrix sqrt of sigma1 dot sigma2
    covmean = sqrt_newton_schulz(sigma1.mm(sigma2).unsqueeze(0), 20).squeeze()
    # covmean = sqrt_newton_schulz(sigma1.mm(sigma2).unsqueeze(0), 10).squeeze()
    out = (diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2)
           - 2 * torch.trace(covmean))
    return out


if __name__ == '__main__':
    # Test the backward computation of FID
    from torch import nn
    from torch.utils.data import TensorDataset
    import torch

    input_dim = 1000
    output_dim = 1000
    feature_dim = 1000
    sample_num = 5000
    batch_size = 16
    dtype = torch.float32
    seed = 0

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    class Generator(nn.Module):
        def __init__(self, input_dim=2, output_dim=8):
            super().__init__()
            self.z_dim = input_dim

            self.model = nn.Sequential(
                    nn.Linear(input_dim, output_dim, bias=True, dtype=dtype),
                    nn.Tanh()
            )

        def forward(self, z):
            return self.model(z)

    class FeatureMapping(nn.Module):
        def __init__(self, input_dim=8, output_dim=4):
            super().__init__()
            self.model = nn.Sequential(
                    nn.Linear(input_dim, output_dim, bias=True, dtype=dtype),
                )

        def forward(self, x):
            return self.model(x)

    generator = Generator(input_dim, output_dim).cuda()
    inception = FeatureMapping(output_dim, feature_dim).cuda()

    for p in inception.parameters():
        p.requires_grad_(False)

    input_g = torch.randn(sample_num, input_dim, dtype=dtype).cuda()
    input_i = torch.randn(sample_num, output_dim, dtype=dtype).cuda()

    with torch.no_grad():
        real_acts = inception(input_i)

    # compute without minibatch
    fake_acts1 = inception(generator(input_g))
    fid1 = frechet_inception_distance(real_acts, fake_acts1)
    fid1.backward(inputs=tuple(generator.parameters()))
    grads1 = [p.grad.detach().clone() for p in generator.parameters()]

    # compute with minibatch
    dataset = TensorDataset(input_g)
    fid2 = backward_fid(dataset, generator, inception, real_acts, batch_size=batch_size)
    grads2 = [p.grad for p in generator.parameters()]

    # compare gradients
    for g1, g2 in zip(grads1, grads2):
        if torch.allclose(g1, g2, atol=1e-6):
            print('Gradients match!')
        else:
            print('Gradients do not match!')
        print(g1)
        print(g2)

    if torch.allclose(fid1, fid2, atol=1e-6):
        print('FID match!')
    else:
        print('FID does not match!')

    print(fid1)
    print(fid2)


