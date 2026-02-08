from libs import waluigi
from libs.finetune import finetune as train

from methods import *
import os
import numpy as np
from libs.plot import gen_error_fig
from libs.models import StyledGenerator, ToyGenerator
import matplotlib.pyplot as plt
from constants import D_DATASET_PATH, SetMetric
import sklearn

class MakeDatasets(waluigi.TaskBase):
    name_dataset: str = waluigi.Parameter()
    ratio_valid: float = waluigi.FloatParameter()
    ratio_test: float = waluigi.FloatParameter()
    image_size: int = waluigi.IntParameter()

    _ver: int = waluigi.IntParameter(1)

    def run(self):
        output_dir = self.make_and_get_temporary_directory()
        path = D_DATASET_PATH[self.name_dataset]

        dataset_train, dataset_valid, dataset_test = split_and_prepare_data(
                path=path,
                tmp_dir=output_dir,
                ratio_valid=self.ratio_valid,
                ratio_test=self.ratio_test,
                sizes=(self.image_size, ),
        )
        self.dump(dict(train=dataset_train, valid=dataset_valid, test=dataset_test))

class PrecomputeActivations(waluigi.TaskBase):
    '''
    This task precomputes the activations of the classifier.
    The output is the precomputed activations.
    '''
    task_datasets: waluigi.TaskOnKart = waluigi.TaskInstanceParameter()

    _ver: int = waluigi.IntParameter(1)

    def requires(self):
        return self.task_datasets

    def run(self):
        d_datasets = self.load()
        acts_train = compute_acts(d_datasets['train'])
        acts_valid = compute_acts(d_datasets['valid'])
        acts_test = compute_acts(d_datasets['test'])
        self.dump(dict(train=acts_train, valid=acts_valid, test=acts_test))


class TrainWithCheckpoints(waluigi.TaskBase):
    '''
    This task performs ASGD training.
    The output is the trained parameters and stored information (e.g., intermediate parameters, latent variables, and mini-batch indices)
    '''
    task_datasets: waluigi.TaskOnKart = waluigi.TaskInstanceParameter()
    task_acts: waluigi.TaskOnKart = waluigi.TaskInstanceParameter(significant=False)

    n_epochs: int = waluigi.IntParameter()
    lr_G: float = waluigi.FloatParameter()
    lr_D: float = waluigi.FloatParameter()
    mixing: bool = waluigi.BoolParameter()
    batch_size: int = waluigi.IntParameter()
    eval_step: int = waluigi.IntParameter(significant=False)
    rank_D: int = waluigi.IntParameter()
    rank_G: int = waluigi.IntParameter()
    name_model: str = waluigi.Parameter()
    name_optimizer: str = waluigi.Parameter()
    names_metric: list = waluigi.ListParameter()
    ckpt: str = waluigi.Parameter()
    no_gp: bool = waluigi.BoolParameter()
    image_size: int = waluigi.IntParameter()
    accumulate_decay: float = waluigi.FloatParameter()

    _ver: int = waluigi.IntParameter(16)

    def requires(self):
        return self.task_datasets, self.task_acts

    def run(self):
        d_datasets, d_acts  = self.load()
        output_dir = self.make_and_get_temporary_directory()

        self.run_in_sacred_experiment(
                train,
                output_dir=output_dir,
                name_model=self.name_model,
                dataset=d_datasets['train'],
                rank_G=self.rank_G,
                rank_D=self.rank_D,
                batch_size=self.batch_size,
                mixing=self.mixing,
                n_epochs=self.n_epochs,
                eval_step=self.eval_step,
                lr_G=self.lr_G,
                lr_D=self.lr_D,
                name_optimizer=self.name_optimizer,
                names_metric=self.names_metric,
                no_gp=self.no_gp,
                ckpt=self.ckpt,
                train_only=False,
                acts_valid=d_acts['valid'],
                image_size=self.image_size,
                accumulate_decay=self.accumulate_decay,
        )
        self.dump(output_dir)


class GetEpochSize(waluigi.TaskBase):
    task_datasets: waluigi.TaskOnKart = waluigi.TaskInstanceParameter()
    batch_size: int = waluigi.IntParameter()

    def requires(self):
        return self.task_datasets

    def run(self):
        d_dataset = self.load()
        dataset_size = len(d_dataset['train'])
        n_steps = compute_steps_per_epoch(dataset_size, self.batch_size)
        self.dump(n_steps)

class GenNoiseDataset(waluigi.TaskBase):
    noise_size: int = waluigi.IntParameter()
    name_model: str = waluigi.Parameter()
    _ver: int = waluigi.IntParameter(2)

    def run(self):
        self.dump(self.main(self.noise_size, self.name_model))

    @staticmethod
    def main(n_samples, name_model):
        if name_model == 'stylegan':
            return StyledGenerator().sample_valid_noise_dataset(n_samples)
        elif name_model == 'lqgan':
            return ToyGenerator().sample_valid_noise_dataset(n_samples)


class InfluenceEstimationITD(waluigi.TaskBase):
    task_train: waluigi.TaskOnKart = waluigi.TaskInstanceParameter()
    task_noise: waluigi.TaskOnKart = waluigi.TaskInstanceParameter()
    task_n_steps: waluigi.TaskOnKart = waluigi.TaskInstanceParameter()
    task_datasets: waluigi.TaskOnKart = waluigi.TaskInstanceParameter()
    task_acts: waluigi.TaskOnKart = waluigi.TaskInstanceParameter()

    n_epochs: int = waluigi.IntParameter()
    lr_G: float = waluigi.FloatParameter()
    lr_D: float = waluigi.FloatParameter()
    mixing: bool = waluigi.BoolParameter()
    rank_D: int = waluigi.IntParameter()
    rank_G: int = waluigi.IntParameter()
    name_model: str = waluigi.Parameter()
    ckpt: str = waluigi.Parameter()
    image_size: int = waluigi.IntParameter()
    accumulate_decay: float = waluigi.FloatParameter()
    name_optimizer: str = waluigi.Parameter()

    from_epoch: int = waluigi.IntParameter()
    to_epoch: int = waluigi.IntParameter()
    from_step: int = waluigi.IntParameter()
    name_metric_infl: str = waluigi.Parameter()
    on_averaged_G: bool = waluigi.BoolParameter()

    damping: float = waluigi.FloatParameter(0.)
    scale: float = waluigi.FloatParameter(1.)
    use_gp_when_infl: bool = waluigi.BoolParameter()
    dataset_split_infl: str = waluigi.Parameter()

    _ver: int = waluigi.IntParameter(20)

    def requires(self):
        return self.task_train, self.task_noise, self.task_n_steps, self.task_datasets, self.task_acts

    def run(self):
        train_dir, noise_dataset, n_steps, d_datasets, d_acts = self.load()
        approx_diffs = self.run_in_sacred_experiment(approx_influence,
        # approx_diffs = approx_influence(
                train_dir=train_dir,
                name_model=self.name_model,
                dataset_train=d_datasets['train'],
                acts_valid=d_acts[self.dataset_split_infl],
                rank_G=self.rank_G,
                rank_D=self.rank_D,
                mixing=self.mixing,
                n_epochs=self.n_epochs,
                lr_G=self.lr_G,
                lr_D=self.lr_D,
                no_gp=not self.use_gp_when_infl,
                ckpt=self.ckpt,
                noise_dataset=noise_dataset,
                from_epoch=self.from_epoch,
                to_epoch=self.to_epoch,
                from_step=self.from_step,
                n_steps=n_steps,
                name_metric_infl=self.name_metric_infl,
                damping=self.damping,
                scale=self.scale,
                image_size=self.image_size,
                on_averaged_G=self.on_averaged_G,
                accumulate_decay=self.accumulate_decay,
                name_optimizer=self.name_optimizer,
        )
        self.dump(approx_diffs)



class InfluenceEstimationAID(waluigi.TaskBase):
    task_train: waluigi.TaskOnKart = waluigi.TaskInstanceParameter()
    task_noise: waluigi.TaskOnKart = waluigi.TaskInstanceParameter()
    task_n_steps: waluigi.TaskOnKart = waluigi.TaskInstanceParameter()
    task_datasets: waluigi.TaskOnKart = waluigi.TaskInstanceParameter()
    task_acts: waluigi.TaskOnKart = waluigi.TaskInstanceParameter()

    mixing: bool = waluigi.BoolParameter()
    rank_D: int = waluigi.IntParameter()
    rank_G: int = waluigi.IntParameter()
    name_model: str = waluigi.Parameter()
    ckpt: str = waluigi.Parameter()
    image_size: int = waluigi.IntParameter()
    batch_size: int = waluigi.IntParameter()

    to_epoch: int = waluigi.IntParameter()
    name_metric_infl: str = waluigi.Parameter()
    on_averaged_G: bool = waluigi.BoolParameter()

    damping: float = waluigi.FloatParameter(0.)
    scale: float = waluigi.FloatParameter(1.)
    use_gp_when_infl: bool = waluigi.BoolParameter()
    dataset_split_infl: str = waluigi.Parameter()
    depth: int = waluigi.IntParameter()

    _ver: int = waluigi.IntParameter(20)

    def requires(self):
        return self.task_train, self.task_noise, self.task_n_steps, self.task_datasets, self.task_acts

    def run(self):
        train_dir, noise_dataset, n_steps, d_datasets, d_acts = self.load()
        approx_diffs = self.run_in_sacred_experiment(approx_influence_aid,
                                                     # approx_diffs = approx_influence_aid(
                                                     train_dir=train_dir,
                                                     name_model=self.name_model,
                                                     dataset_train=d_datasets['train'],
                                                     acts_valid=d_acts[self.dataset_split_infl],
                                                     rank_G=self.rank_G,
                                                     rank_D=self.rank_D,
                                                     mixing=self.mixing,
                                                     no_gp=not self.use_gp_when_infl,
                                                     ckpt=self.ckpt,
                                                     noise_dataset=noise_dataset,
                                                     to_epoch=self.to_epoch,
                                                     n_steps=n_steps,
                                                     name_metric_infl=self.name_metric_infl,
                                                     damping=self.damping,
                                                     scale=self.scale,
                                                     image_size=self.image_size,
                                                     on_averaged_G=self.on_averaged_G,
                                                     batch_size=self.batch_size,
                                                     depth=self.depth,
                                                     )
        self.dump(approx_diffs)


class IsolationForest(waluigi.TaskBase):
    task_acts: waluigi.TaskOnKart = waluigi.TaskInstanceParameter()
    _ver: int = waluigi.IntParameter(20)

    def requires(self):
        return self.task_acts

    def run(self):
        d_acts = self.load()
        approx_diffs = run_isolation_forest(
                acts_train=d_acts['train'],
        )
        self.dump(approx_diffs)


class GetHarmfulIndices(waluigi.TaskBase):
    task_influence: waluigi.TaskOnKart = waluigi.TaskInstanceParameter()
    name_metric_infl: str = waluigi.Parameter()
    ratio: float = waluigi.FloatParameter()

    _ver: int = waluigi.IntParameter(0)

    def requires(self):
        return self.task_influence

    def run(self):
        approx_diffs = self.load()
        target_indices = self.get_harmful_indices(approx_diffs, self.ratio, self.name_metric_infl)
        self.dump(target_indices)

    def get_harmful_indices(self, approx_diffs, ratio, metric):
        n_targets = int(len(approx_diffs) * ratio)
        if n_targets == 0:
            return np.array([])
        else:
            sorted_indices = np.argsort(approx_diffs)
            if metric in SetMetric.NEGATIVE:
                harmful_indices = sorted_indices[:n_targets]
            elif metric in SetMetric.POSITIVE:
                harmful_indices = sorted_indices[-n_targets:]
            else:
                raise ValueError(f'Unknown metric: {metric}')
            return harmful_indices

class GenRandomInfluenceValues(waluigi.TaskBase):
    task_datasets: waluigi.TaskOnKart = waluigi.TaskInstanceParameter()

    _ver: int = waluigi.IntParameter(1)

    def requires(self):
        return self.task_datasets

    def run(self):
        d_datasets = self.load()
        self.dump(np.random.randn(len(d_datasets['train'])))

class CounterfactualTrain(waluigi.TaskBase):
    task_train: waluigi.TaskOnKart = waluigi.TaskInstanceParameter()
    task_n_steps: waluigi.TaskOnKart = waluigi.TaskInstanceParameter()
    task_indices: waluigi.TaskOnKart = waluigi.TaskInstanceParameter()

    n_epochs: int = waluigi.IntParameter()
    lr_G: float = waluigi.FloatParameter()
    lr_D: float = waluigi.FloatParameter()
    mixing: bool = waluigi.BoolParameter()
    rank_D: int = waluigi.IntParameter()
    rank_G: int = waluigi.IntParameter()
    name_model: str = waluigi.Parameter()
    name_optimizer: str = waluigi.Parameter()
    ckpt: str = waluigi.Parameter()
    no_gp: bool = waluigi.BoolParameter()
    accumulate_decay: float = waluigi.FloatParameter()

    from_epoch: int = waluigi.IntParameter()
    to_epoch: int = waluigi.IntParameter()
    from_step: int = waluigi.IntParameter(0)
    image_size: int = waluigi.IntParameter()

    _ver: int = waluigi.IntParameter(13)

    def requires(self):
        return self.task_train, self.task_n_steps, self.task_indices

    def run(self):
        train_dir, n_steps, target_indices = self.load()

        state_dicts = cal_true_influence(
                train_dir=train_dir,
                name_model=self.name_model,
                rank_G=self.rank_G,
                rank_D=self.rank_D,
                mixing=self.mixing,
                n_epochs=self.n_epochs,
                lr_G=self.lr_G,
                lr_D=self.lr_D,
                name_optimizer=self.name_optimizer,
                no_gp=self.no_gp,
                ckpt=self.ckpt,
                target_indices=target_indices,
                from_epoch=self.from_epoch,
                to_epoch=self.to_epoch,
                from_step=self.from_step,
                n_steps=n_steps,
                image_size=self.image_size,
                accumulate_decay=self.accumulate_decay,
        )

        self.dump(state_dicts)



class LoadLatestModel(waluigi.TaskBase):
    task_train: waluigi.TaskOnKart = waluigi.TaskInstanceParameter()
    task_n_steps: waluigi.TaskOnKart = waluigi.TaskInstanceParameter()

    to_epoch: int = waluigi.IntParameter()

    _ver: int = waluigi.IntParameter(8)

    def requires(self):
        return self.task_train, self.task_n_steps

    def run(self):
        train_dir, n_steps  = self.load()

        state_dicts = load_latest_model(
                train_dir=train_dir,
                to_epoch=self.to_epoch,
                n_steps=n_steps,
        )

        self.dump(state_dicts)


class EvalModel(waluigi.TaskBase):
    task_state_dict: waluigi.TaskOnKart = waluigi.TaskInstanceParameter()
    task_noise: waluigi.TaskOnKart = waluigi.TaskInstanceParameter()
    task_acts: waluigi.TaskOnKart = waluigi.TaskInstanceParameter()

    dataset_split_eval: str = waluigi.Parameter()

    name_model: str = waluigi.Parameter()
    mixing: bool = waluigi.BoolParameter()
    rank_G: int = waluigi.IntParameter()
    rank_D: int = waluigi.IntParameter()
    name_metric_eval: str = waluigi.Parameter()
    ckpt: str = waluigi.Parameter()
    on_averaged_G: bool = waluigi.BoolParameter()
    image_size: int = waluigi.IntParameter()

    _ver: int = waluigi.IntParameter(1)

    def requires(self):
        return self.task_state_dict, self.task_noise, self.task_acts

    def run(self):
        state_dicts, noise_dataset, d_acts = self.load()
        metric = eval_latest_model(
                name_model=self.name_model,
                mixing=self.mixing,
                rank_G=self.rank_G,
                rank_D=self.rank_D,
                real_acts=d_acts[self.dataset_split_eval],
                noise_dataset=noise_dataset,
                state_dicts=state_dicts,
                name_metric=self.name_metric_eval,
                ckpt=self.ckpt,
                on_averaged_G=self.on_averaged_G,
                image_size=self.image_size,
        )
        self.dump(metric)


class GetItem(waluigi.TaskBase):
    tasks: list = waluigi.ListTaskInstanceParameter()
    task_index: waluigi.TaskOnKart = waluigi.TaskInstanceParameter()

    def requires(self):
        return self.tasks, self.task_index

    def run(self):
        inputs, idx = self.load()
        self.dump(inputs[idx])

class GetArgBestMetric(waluigi.TaskBase):
    tasks: list = waluigi.ListTaskInstanceParameter()
    name_metric_eval: str = waluigi.Parameter()

    def requires(self):
        return self.tasks

    def run(self):
        metrics = self.load()
        if self.name_metric_eval in SetMetric.POSITIVE:
            idx = np.argmax(metrics)
        elif self.name_metric_eval in SetMetric.NEGATIVE:
            idx = np.argmin(metrics)
        else:
            raise ValueError(f'Unknown metric: {self.name_metric_infl}')
        self.dump(idx)


class PlotCleansing(waluigi.TaskBase):
    tasks_clean: list = waluigi.ListTaskInstanceParameter()
    task_ori: waluigi.TaskOnKart = waluigi.TaskInstanceParameter()
    removal_rates: list = waluigi.ListParameter()

    name_metric_eval: str = waluigi.Parameter()

    _ver: int = waluigi.IntParameter(1)

    def requires(self):
        return self.tasks_clean, self.task_ori

    def run(self):
        metrics_clean, metric_ori = self.load()
        fig = self.run_in_sacred_experiment(
                plot_cleansing_result_wrt_removal_rate,
                metric_ori=metric_ori,
                metrics_clean=np.array(metrics_clean),
                removal_rates=self.removal_rates,
                output_path=self.local_temporary_directory,
                name_metric_eval=self.name_metric_eval,
        )
        self.dump(fig)


class PlotBestCleansings(waluigi.TaskBase):
    tasks_clean: list = waluigi.ListTaskInstanceParameter()
    task_ori: waluigi.TaskOnKart = waluigi.TaskInstanceParameter()
    from_epochs: list = waluigi.ListParameter()

    name_metric_eval: str = waluigi.Parameter()

    _ver: int = waluigi.IntParameter(1)

    def requires(self):
        return self.tasks_clean, self.task_ori

    def run(self):
        metrics_clean, metric_ori = self.load()
        fig = self.run_in_sacred_experiment(plot_cleansing_result_wrt_from_epoch,
        # fig = plot_cleansing_result_wrt_from_epoch(
                metric_ori=metric_ori,
                metrics_clean=np.array(metrics_clean),
                from_epochs=self.from_epochs,
                output_path=self.local_temporary_directory,
                name_metric_eval=self.name_metric_eval,
        )
        self.dump(fig)

class SampleInfluentialInstances(waluigi.TaskBase):
    task_infl: waluigi.TaskOnKart = waluigi.TaskInstanceParameter()
    task_datasets: waluigi.TaskOnKart = waluigi.TaskInstanceParameter()

    grid_size: tuple = waluigi.TupleParameter((10, 5))

    name_metric_infl: str = waluigi.Parameter()

    _ver: int = waluigi.IntParameter(2)

    def requires(self):
        return self.task_infl, self.task_datasets

    def run(self):
        approx_diffs, d_datasets = self.load()
        out = self.run_in_sacred_experiment(sample_influential_instances,
                                            # sample_harmful_instances(
                                            dataset=d_datasets['train'],
                                            influences=approx_diffs,
                                            grid_size=self.grid_size,
                                            name_metric=self.name_metric_infl,
                                            out_dir=self.local_temporary_directory,
                                            )
        self.dump(out)



class GenerateImages(waluigi.TaskBase):
    task_model: waluigi.TaskOnKart = waluigi.TaskInstanceParameter()

    mixing: bool = waluigi.BoolParameter()
    rank_G: int = waluigi.IntParameter()
    rank_D: int = waluigi.IntParameter()
    name_model: str = waluigi.Parameter()
    ckpt: str = waluigi.Parameter()
    image_size: int = waluigi.IntParameter()
    on_averaged_G: bool = waluigi.BoolParameter()
    grid_size: tuple = waluigi.TupleParameter((100, 5))

    _ver: int = waluigi.IntParameter(3)

    def requires(self):
        return self.task_model

    def run(self):
        state_dict = self.load()
        fake_images = self.run_in_sacred_experiment(sample_images,
        # out = sample_images(
                name_model=self.name_model,
                state_dict=state_dict,
                image_size=self.image_size,
                mixing=self.mixing,
                rank_G=self.rank_G,
                rank_D=self.rank_D,
                grid_size=self.grid_size,
                ckpt=self.ckpt,
                on_averaged_G=self.on_averaged_G,
        )
        self.dump(fake_images)


class PipelineCleansing(waluigi.TaskBase):
        # dataset
    name_dataset: str = waluigi.Parameter('afhq_v2_cat')
    ratio_valid: float = waluigi.FloatParameter(0.2)
    ratio_test: float = waluigi.FloatParameter(0.2)

    # train
    n_epochs: int = waluigi.IntParameter(50)
    lr_G: float = waluigi.FloatParameter(0.002)
    lr_D: float = waluigi.FloatParameter(0.002)
    mixing: bool = waluigi.BoolParameter()
    batch_size: int = waluigi.IntParameter(8)
    eval_step: int = waluigi.IntParameter(0, significant=False)
    rank_D: int = waluigi.IntParameter(32)
    rank_G: int = waluigi.IntParameter(32)
    name_model: str = waluigi.Parameter('stylegan')
    names_metric: list = waluigi.ListParameter([Metric.FID], significant=False)
    name_optimizer: str = waluigi.Parameter('adam')
    ckpt: str = waluigi.Parameter('./checkpoint/stylegan-256px-new.model')
    no_gp: bool = waluigi.BoolParameter()
    image_size: int = waluigi.IntParameter(256)
    accumulate_decay: float = waluigi.FloatParameter(0.999)

    # influence
    noise_size_train: int = waluigi.IntParameter(5000)
    noise_size_valid: int = waluigi.IntParameter(5000)
    noise_size_test: int = waluigi.IntParameter(5000)
    n_targets: int = waluigi.IntParameter(0)
    from_epoch: int = waluigi.IntParameter(0)
    to_epoch: int = waluigi.IntParameter(50)
    from_step: int = waluigi.IntParameter(0)
    name_metric_infl: str = waluigi.Parameter(Metric.FID)
    name_metric_valid: str = waluigi.Parameter(Metric.FID)
    name_metric_eval: str = waluigi.Parameter(Metric.FID)
    on_averaged_G: bool = waluigi.BoolParameter()
    damping: float = waluigi.FloatParameter(0.)
    scale: float = waluigi.FloatParameter(1.)
    use_gp_when_infl: bool = waluigi.BoolParameter()
    method_influence: str = waluigi.Parameter('itd')
    depth: int = waluigi.IntParameter(1000)

    # valid
    jaccard_size: float = waluigi.FloatParameter(0.1)
    names_score: list = waluigi.ListParameter(['r2', 'jaccardindex', 'kendalltau'])

    # step growth
    from_epoch_growth: int = waluigi.IntParameter(9)
    from_step_growth: int = waluigi.IntParameter(0)
    to_epoch_growth: int = waluigi.IntParameter(10)
    
    dataset_split_infl: str = waluigi.Parameter('valid')
    dataset_split_eval: str = waluigi.Parameter('test')

    run_only: str = waluigi.Parameter('all')

    removal_rates: list = waluigi.ListParameter([0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,0.7,0.9])
    from_epochs: list = waluigi.ListParameter([49,0])
    scales: list = waluigi.ListParameter([1.])

    def requires(self):
        assert len(self.from_epochs) == len(self.scales), f'{len(self.from_epochs)} != {len(self.scales)}'

        task_datasets = self.clone(MakeDatasets)
        task_acts = self.clone(PrecomputeActivations, task_datasets=task_datasets)
        task_train = self.clone(TrainWithCheckpoints, task_datasets=task_datasets, task_acts=task_acts)
        d_task_noise = {
            'valid': self.clone(GenNoiseDataset, noise_size=self.noise_size_valid),
            'test': self.clone(GenNoiseDataset, noise_size=self.noise_size_test, fix_random_seed_value=self.fix_random_seed_value + 1),
            'train': self.clone(GenNoiseDataset, noise_size=self.noise_size_train, fix_random_seed_value=self.fix_random_seed_value + 2),
        }
        task_n_steps = self.clone(GetEpochSize, task_datasets=task_datasets)
        
        # eval original model
        task_ori_model = self.clone(LoadLatestModel, task_train=task_train, task_n_steps=task_n_steps)
        task_eval_ori = self.clone(EvalModel, task_state_dict=task_ori_model, task_noise=d_task_noise[self.dataset_split_eval], task_acts=task_acts)
        task_gen_ori = self.clone(GenerateImages, task_model=task_ori_model)

        # cleansing
        tasks_eval_clean = []
        tasks_gen_clean = []
        tasks_infl_instance = []
        tasks_plot_cleansing = []
        tasks_best_model_clean = []
        taskss_eval_valid = []
        taskss_eval_test = []
        for from_epoch, scale in zip(self.from_epochs, self.scales):
            if self.name_metric_infl in (Metric.D_LOSS, Metric.FID, Metric.FAKE_IM_MEAN, Metric.D_LOSS):
                if self.method_influence == 'itd':
                    task_approx = self.clone(InfluenceEstimationITD, task_train=task_train, task_noise=d_task_noise[self.dataset_split_infl], from_epoch=from_epoch, task_n_steps=task_n_steps, task_datasets=task_datasets, task_acts=task_acts, scale=scale)
                elif self.method_influence == 'aid':
                    task_approx = self.clone(InfluenceEstimationAID, task_train=task_train, task_noise=d_task_noise[self.dataset_split_infl], task_n_steps=task_n_steps, task_datasets=task_datasets, task_acts=task_acts, scale=scale)
                else:
                    raise ValueError(f'Unknown method: {self.method_influence}')
            elif self.name_metric_infl == Metric.RANDOM:
                task_approx = self.clone(GenRandomInfluenceValues, task_datasets=task_datasets)
            elif self.name_metric_infl == Metric.IF:
                task_approx = self.clone(IsolationForest, task_acts=task_acts)
            else:
                raise ValueError(f'Unknown metric: {self.name_metric_infl}')

            task_harmful = self.clone(SampleInfluentialInstances, task_infl=task_approx, task_datasets=task_datasets)
            tasks_infl_instance.append(task_harmful)

            # Cleansing
            tasks_eval_valid = []
            tasks_eval_test = []
            tasks_clean_model = []
            for removal_rate in self.removal_rates:
                task_indices = self.clone(GetHarmfulIndices, task_influence=task_approx, ratio=removal_rate)
                task_clean_model = self.clone(CounterfactualTrain, task_train=task_train, task_indices=task_indices, task_n_steps=task_n_steps, from_epoch=from_epoch)
                task_eval_valid = self.clone(EvalModel, task_state_dict=task_clean_model, task_noise=d_task_noise[self.dataset_split_infl], dataset_split_eval=self.dataset_split_infl, task_acts=task_acts, name_metric_eval=self.name_metric_valid)
                task_eval_test = self.clone(EvalModel, task_state_dict=task_clean_model, task_noise=d_task_noise[self.dataset_split_eval], dataset_split_eval=self.dataset_split_eval, task_acts=task_acts, name_metric_eval=self.name_metric_eval)
                tasks_eval_valid.append(task_eval_valid)
                tasks_eval_test.append(task_eval_test)
                tasks_clean_model.append(task_clean_model)

            taskss_eval_valid.append(tasks_eval_valid)
            taskss_eval_test.append(tasks_eval_test)

            task_best_idx_clean = self.clone(GetArgBestMetric, tasks=tasks_eval_valid, name_metric_eval=self.name_metric_valid)
            task_best_model_clean = self.clone(GetItem, tasks=tasks_clean_model, task_index=task_best_idx_clean)
            tasks_best_model_clean.append(task_best_model_clean)

            task_eval_test_best = self.clone(EvalModel, task_state_dict=task_best_model_clean, task_noise=d_task_noise[self.dataset_split_eval], task_acts=task_acts)
            task_gen_clean = self.clone(GenerateImages, task_model=task_best_model_clean)
            tasks_eval_clean.append(task_eval_test_best)
            tasks_gen_clean.append(task_gen_clean)

            task_plot_cleansing = self.clone(PlotCleansing, tasks_clean=tasks_eval_valid, task_ori=task_eval_ori, from_epochs=[from_epoch], name_metric_eval=self.name_metric_eval)
            tasks_plot_cleansing.append(task_plot_cleansing)

        task_plot = self.clone(PlotBestCleansings, tasks_clean=tasks_eval_clean, task_ori=task_eval_ori)

        return task_plot, tasks_plot_cleansing, tasks_infl_instance, task_eval_ori, tasks_eval_clean, task_gen_ori, tasks_gen_clean, taskss_eval_valid, taskss_eval_test

    def run(self):
        self.dump(self.load())

