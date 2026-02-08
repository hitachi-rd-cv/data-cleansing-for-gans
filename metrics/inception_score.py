
"""The core implementation of Inception Score and FID."""

from typing import List, Union, Tuple, Any

import numpy as np
import torch
from numpy import ndarray, dtype, floating, float_
from numpy._typing import _64Bit
from scipy import linalg
from torch import FloatTensor, Tensor
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from .inception import InceptionV3


# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_inception_feature(
    images: Union[List[torch.FloatTensor], DataLoader],
    dims: List[int],
    model: InceptionV3 = None,
    batch_size: int = 50,
    use_torch: bool = False,
    verbose: bool = False,
    device: torch.device = torch.device('cuda:0'),
):
    """Calculate Inception Score and FID.

    For each image, only a forward propagation is required to calculating
    features for FID and Inception Score.

    Args:
        images: List of tensor or torch.utils.data.Dataloader. The return image
                must be float tensor of range [0, 1].
        dims: List of int, see InceptionV3.BLOCK_INDEX_BY_DIM for
              available dimension.
        batch_size: int, The batch size for calculating activations. If
                    `images` is torch.utils.data.Dataloader, this argument is
                    ignored.
        use_torch: When True, use torch to calculate FID. Otherwise, use numpy.
        verbose: Set verbose to False for disabling progress bar. Otherwise,
                 the progress bar is showing when calculating activations.
        device: the torch device which is used to calculate inception feature
    Returns:
        inception_score: float tuple, (mean, std)
        fid: float
    """
    assert all(dim in InceptionV3.BLOCK_INDEX_BY_DIM for dim in dims)

    is_dataloader = isinstance(images, DataLoader)
    if is_dataloader:
        num_images = min(len(images.dataset), images.batch_size * len(images))
        batch_size = images.batch_size
    else:
        num_images = len(images)

    block_idxs = [InceptionV3.BLOCK_INDEX_BY_DIM[dim] for dim in dims]
    if model is None:
        model = InceptionV3(block_idxs).to(device)
    model.eval()

    if use_torch:
        features = [torch.empty((num_images, dim)).to(device) for dim in dims]
    else:
        features = [np.empty((num_images, dim)) for dim in dims]

    pbar = tqdm(
        total=num_images, dynamic_ncols=True, leave=False,
        disable=not verbose, desc="get_inception_feature")
    looper = iter(images)
    start = 0
    while start < num_images:
        # get a batch of images from iterator
        if is_dataloader:
            batch_images = next(looper)
        else:
            batch_images = images[start: start + batch_size]
        end = start + len(batch_images)

        # calculate inception feature
        batch_images = batch_images.to(device)
        outputs = model(batch_images)
        for feature, output, dim in zip(features, outputs, dims):
            if use_torch:
                feature[start: end] = output.view(-1, dim)
            else:
                feature[start: end] = output.view(-1, dim).cpu().numpy()
        start = end
        pbar.update(len(batch_images))
    pbar.close()
    return features


def calculate_inception_score(
    probs: Union[torch.FloatTensor, np.ndarray],
    splits: int = 10,
    use_torch: bool = False,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    # Inception Score
    scores = []
    for i in range(splits):
        part = probs[
            (i * probs.shape[0] // splits):
            ((i + 1) * probs.shape[0] // splits), :]
        if use_torch:
            kl = part * (
                torch.log(part) -
                torch.log(torch.unsqueeze(torch.mean(part, 0), 0)))
            kl = torch.mean(torch.sum(kl, 1))
            scores.append(torch.exp(kl))
        else:
            kl = part * (
                np.log(part) -
                np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
    if use_torch:
        scores = torch.stack(scores)
        inception_score = torch.mean(scores)
        std = torch.std(scores)
    else:
        inception_score, std = (np.mean(scores), np.std(scores))
    del probs, scores
    return inception_score, std

def get_inception_score(
    images: Union[torch.FloatTensor, DataLoader],
    dims: List[int] = [1008],
    splits: int = 10,
    use_torch: bool = False,
    **kwargs,
) -> Tuple[FloatTensor, FloatTensor]:
    """Calculate Inception Score.

    Args:
        images: List of tensor or torch.utils.data.Dataloader. The return image
                must be float tensor of range [0, 1].
        splits: The number of bins of Inception Score.
        use_torch: When True, use torch to calculate FID. Otherwise, use numpy.
        **kwargs: The arguments passed to
                  `pytorch_gan_metrics.core.get_inception_feature`.

    Returns:
        Inception Score
    """
    out = get_inception_feature(images, dims=dims, use_torch=use_torch, **kwargs)
    probs = out[dims.index(1008)]
    inception_score, std = calculate_inception_score(probs, splits, use_torch)
    return (inception_score, std)


def backward_is(dataset, generator, inception, batch_size=16, inputs_backward=None, input_processor=None, output_processor=None):
    loader = DataLoader(dataset, batch_size=batch_size)

    pbar = tqdm(total=len(dataset), position=1, leave=False)
    pbar.set_description('Pre-computing activations')

    if inputs_backward is None:
        inputs_backward = tuple(generator.parameters())

    dims = [InceptionV3.DIM_BY_BLOCK_INDEX[idx] for idx in inception.output_blocks]
    idx_probs = dims.index(1008)
    # Compute fake activations without gradient for memory efficiency
    with torch.no_grad():
        probs = []
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

            out = get_inception_feature(fake_image, model=inception, dims=dims, use_torch=True, batch_size=sample_size)
            probs_batch = out[idx_probs]
            probs.append(probs_batch)
            pbar.update(sample_size)

    # Concatenate all pre-computed activations
    probs = torch.cat(probs, axis=0)

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

        out = get_inception_feature(fake_image, model=inception, dims=dims, use_torch=True, batch_size=sample_size)
        probs_batch = out[idx_probs]

        # Determine the start and end indices for the current batch
        end_index = start_index + sample_size

        # Replace the corresponding part of pre-computed fake_acts
        probs_tmp = probs.clone().detach()  # Detach and clone for safe replacement
        probs_tmp[start_index:end_index] = probs_batch

        # Compute FID and gradient for the current batch
        inception_score, _ = calculate_inception_score(probs_tmp, use_torch=True)

        # Accumulate gradients
        inception_score.backward(inputs=inputs_backward)

        # Update the start index for the next batch
        start_index = end_index

        pbar.update(sample_size)

