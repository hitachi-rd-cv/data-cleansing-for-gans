import os
import pickle

import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import utils
from tqdm import tqdm
from torch.nn import functional as F

from metrics.frechet_inception_distance import frechet_inception_distance
from metrics.inception import InceptionV3
from metrics.inception_score import get_inception_score
from constants import Metric
from metrics.improved_precision_recall import compute_pairwise_distances, distances2radii, compute_metric, Manifold
from metrics.prdc import compute_prdc

class StyleGANEvaluator:
    def __init__(self, generator, real_acts=None, noise_dataset=None, grid_size=(10, 5), sample_num=5000, output_dir='./sample/', _run=None):
        """
        Initializes the Metric class.

        Args:
        - noise_dim (tuple): The dimensions for the noise vector.
        - grid_size (tuple): The grid size for generating fixed noise.
        - inception_model (nn.Module, optional): An Inception model for feature extraction.
        """
        self.generator = generator
        self.grid_size = grid_size
        self.output_dir = output_dir
        self._run = _run

        if real_acts is None:
            self.real_acts = None
        else:
            self.real_acts = real_acts.cuda()

        self.fixed_noise = torch.randn(*grid_size, generator.code_dim).cuda()
        self.sample_num = sample_num
        self.inception = InceptionV3([3, 4]).cuda()
        self.inception.eval()
        self.noise_dataset = noise_dataset

    def compute_activations(self, batch_size=16):
        if self.noise_dataset:
            dataset = self.noise_dataset
        else:
            noise_input = torch.randn(self.sample_num, self.generator.code_dim)
            noises = []
            for i in range(self.generator.step + 1):
                size = 4 * 2 ** i
                noises.append(torch.randn(self.sample_num, 1, size, size).cuda())

            dataset = TensorDataset(noise_input, *noises)

        loader = DataLoader(dataset, batch_size=batch_size)

        pbar = tqdm(total=len(dataset), position=1, leave=False)
        pbar.set_description('Get acts of fake images')

        acts = []
        for gen_in, *noise in loader:
            gen_in = gen_in.cuda()  # list -> tensor
            fake_image = self.generator(gen_in, noise)
            out = self.inception(fake_image)
            out = out[0].squeeze(-1).squeeze(-1)
            acts.append(out)
            pbar.update(len(gen_in))

        acts = torch.cat(acts, axis=0)  # N x d
        return acts

    def gen_fake_images(self, batch_size=16):
        if self.noise_dataset:
            dataset = self.noise_dataset
        else:
            noise_input = torch.randn(self.sample_num, self.generator.code_dim)
            noises = []
            for i in range(self.generator.step + 1):
                size = 4 * 2 ** i
                noises.append(torch.randn(self.sample_num, 1, size, size).cuda())

            dataset = TensorDataset(noise_input, *noises)

        loader = DataLoader(dataset, batch_size=batch_size)

        pbar = tqdm(total=len(dataset), position=1, leave=False)
        pbar.set_description('Get fake images and acts')

        fake_images = []
        for gen_in, *noise in loader:
            gen_in = gen_in.cuda()  # list -> tensor
            fake_image = self.generator(gen_in, noise)
            fake_images.append(fake_image)
            pbar.update(len(gen_in))

        fake_images = torch.cat(fake_images, axis=0)  # N x d
        return fake_images

    def gen_and_save_fake_images(self, filename):
        # Generate fake images
        fake_images = []
        with torch.no_grad():
            for noise in self.fixed_noise:
                fake_images.append(self.generator(noise).cpu())
        fake_images = torch.cat(fake_images, dim=0)
        # Save fake images - adjust path as needed
        sample_path = os.path.join(self.output_dir, filename)
        os.makedirs(os.path.dirname(sample_path), exist_ok=True)
        utils.save_image(fake_images, sample_path, nrow=self.grid_size[0], normalize=True, value_range=(-1, 1))
        if self._run is not None:
            self._run.add_artifact(sample_path)

        return fake_images

    def evaluate(self, name_metric, batch_size=16):
        if name_metric == Metric.FID:
            fake_acts = self.compute_activations(batch_size)
            fid = frechet_inception_distance(self.real_acts, fake_acts)
            return fid

        elif name_metric == Metric.IS:
            fake_images = self.gen_fake_images(batch_size)
            dims = [InceptionV3.DIM_BY_BLOCK_INDEX[idx] for idx in self.inception.output_blocks]
            score, _ = get_inception_score(fake_images, model=self.inception, dims=dims, batch_size=batch_size, use_torch=True)
            return score

        elif name_metric == Metric.FAKE_IM_MEAN:
            loader = DataLoader(self.noise_dataset, batch_size=batch_size)

            out = 0.
            for gen_in, *noise in loader:
                gen_in = gen_in.cuda()  # list -> tensor
                fake_image = self.generator(gen_in, noise)
                out += torch.sum(torch.mean(fake_image, dim=(1,2,3)))

            out /= len(self.noise_dataset)
            return out

        elif name_metric == Metric.D_LOSS:
            raise NotImplementedError
            # assert self.noise_dataset is not None
            # assert self.discriminator is not None
            #
            # loader_noise = DataLoader(self.noise_dataset, batch_size=batch_size)
            # loss_fake = 0.
            # for gen_in, *noise in loader_noise:
            #     gen_in = gen_in.cuda()  # list -> tensor
            #     fake_image_tgt = self.generator(gen_in, noise)
            #     fake_predict = self.discriminator(fake_image_tgt)
            #     loss_fake += F.softplus(fake_predict).sum()
            #
            # loss_fake /= len(self.noise_dataset)
            # return loss_fake

        elif name_metric in (Metric.DENSITY, Metric.COVERAGE, Metric.PRECISION, Metric.RECALL):
            fake_acts = self.compute_activations(batch_size).cpu().numpy()
            real_acts = self.real_acts.cpu().numpy()

            prdc = compute_prdc(real_features=real_acts,
                                    fake_features=fake_acts,
                                    nearest_k=5)

            if name_metric == Metric.DENSITY:
                return prdc['density']
            elif name_metric == Metric.COVERAGE:
                return prdc['coverage']
            elif name_metric == Metric.PRECISION:
                return prdc['precision']
            elif name_metric == Metric.RECALL:
                return prdc['recall']
            else:
                raise ValueError(f'Invalid metric name: {name_metric}')
        else:
            raise ValueError(f'Invalid metric name: {name_metric}')

    def log_metrics(self, name, val, iteration=None):
        if self._run is not None:
            self._run.log_scalar(name, val, iteration)

class LQGANEvaluator:
    def __init__(self, generator, noise_dataset=None, _run=None):
        """
        Initializes the Metric class.

        Args:
        - noise_dim (tuple): The dimensions for the noise vector.
        - grid_size (tuple): The grid size for generating fixed noise.
        - inception_model (nn.Module, optional): An Inception model for feature extraction.
        """
        self.generator = generator
        self.noise_dataset = noise_dataset
        self._run = _run

    def gen_and_save_fake_images(self, iteration):
        pass

    def evaluate(self, name_metric, batch_size=16):
        assert self.noise_dataset is not None, 'Noise dataset is required for evaluation'

        loader = DataLoader(self.noise_dataset, batch_size=batch_size)

        out = 0.
        for gen_in in loader:
            gen_in = gen_in[0].cuda()  # list -> tensor
            out += torch.sum(torch.mean(self.generator(gen_in), dim=1)) / len(self.noise_dataset)

        return out


    def log_metrics(self, name, val, iteration=None):
        if self._run is not None:
            self._run.log_scalar(name, val, iteration)


def backward_mean(dataset, generator, batch_size=16, inputs_backward=None, input_processor=None, output_processor=None):
    loader = DataLoader(dataset, batch_size=batch_size)

    pbar = tqdm(total=len(dataset), position=1, leave=False)
    pbar.set_description('Computing gradients')

    generator.zero_grad()
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

        dims_to_reduce = list(range(1, fake_image.dim()))

        out = torch.sum(torch.mean(fake_image, dim=dims_to_reduce)) / len(dataset)

        out.backward(inputs=inputs_backward)

        pbar.update(sample_size)


def backward_d_loss(dataset, generator, discriminator, batch_size=16, inputs_backward=None, input_processor=None, output_processor=None):
    loader = DataLoader(dataset, batch_size=batch_size)

    pbar = tqdm(total=len(dataset), position=1, leave=False)
    pbar.set_description('Computing gradients')

    generator.zero_grad()
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

        fake_predict = discriminator(fake_image)
        loss = F.softplus(fake_predict).sum() / len(dataset)

        loss.backward(inputs=inputs_backward)

        pbar.update(sample_size)
