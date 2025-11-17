import functools
import inspect
import json
import math
import os
import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from easydict import EasyDict
from easytorch.utils import master_only
from easytorch.config import get_ckpt_save_dir
from tqdm import tqdm

from ..metrics import (masked_mae, masked_mape, masked_mse, masked_rmse,
                       masked_wape)
from ..utils import get_dataset_name
from .base_epoch_runner import BaseEpochRunner
from .custom_eval import eval_csv_unnorm, eval_csv_norm


class BaseTimeSeriesForecastingRunner(BaseEpochRunner):
    """
    Runner for multivariate time series forecasting tasks.

    Features:
        - Supports evaluation at pre-defined horizons (optional) and overall performance assessment.
        - Metrics: MAE, RMSE, MAPE, WAPE, and MSE. Customizable. The best model is selected based on the smallest MAE on the validation set.
        - Supports `setup_graph` for models that operate similarly to TensorFlow.
        - Default loss function is MAE (masked_mae), but it can be customized.
        - Supports curriculum learning.
        - Users only need to implement the `forward` function.

    Customization:
        - Model:
            - Args:
                - history_data (torch.Tensor): Historical data with shape [B, L, N, C], 
                  where B is the batch size, L is the sequence length, N is the number of nodes, 
                  and C is the number of features.
                - future_data (torch.Tensor or None): Future data with shape [B, L, N, C]. 
                  Can be None if there is no future data available.
                - batch_seen (int): The number of batches seen so far.
                - epoch (int): The current epoch number.
                - train (bool): Indicates whether the model is in training mode.
            - Return:
                - Dict or torch.Tensor:
                    - If returning a Dict, it must contain the 'prediction' key. Other keys are optional and will be passed to the loss and metric functions.
                    - If returning a torch.Tensor, it should represent the model's predictions, with shape [B, L, N, C].

        - Loss & Metrics (optional):
            - Args:
                - prediction (torch.Tensor): Model's predictions, with shape [B, L, N, C].
                - target (torch.Tensor): Ground truth data, with shape [B, L, N, C].
                - null_val (float): The value representing missing data in the dataset.
                - Other args (optional): Additional arguments will be matched with keys in the model's return dictionary, if applicable.
            - Return:
                - torch.Tensor: The computed loss or metric value.

        - Dataset (optional):
            - Return: The returned data will be passed to the `forward` function as the `data` argument.
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        
        # region declare all functions before define them
        # self.forward = self.define_forward()
        # self.metric_forward = self.define_metric_forward()
        # endregion

        # define some private variables
        self.null_val = cfg.get('NULL_VAL', 0.0)

        # initialize scaler
        self.scaler = self.build_scaler(cfg)

        # define loss function
        self.loss = cfg['TRAIN']['LOSS']

        # metrics
        self.metrics = {
            'MAE': masked_mae,
            'MSE': masked_mse,
            'RMSE': masked_rmse,
            'MAPE': masked_mape,
            'WAPE': masked_wape
        }

        # model forward arguments
        self.forward_features = [
            'history_data', 'future_data', 'history_data_norm', 'future_data_norm'
        ]
        # specific model forward args
        if cfg.get('MODEL', {}).get('FORWARD_FEATURES') is not None:
            self.forward_features = cfg['MODEL']['FORWARD_FEATURES']

        # metrics configuration
        self.target_metrics = cfg.get('METRICS', {}).get('TARGET', 'MAE')
        self.metrics_best = cfg.get('METRICS', {}).get('BEST', 'min')
        
        # 智能判断指标的最佳方向，如果配置中没有明确指定
        if 'METRICS' in cfg and 'BEST' not in cfg['METRICS']:
            # 对于常见的损失指标，默认为min（越小越好）
            common_min_metrics = ['mae', 'mse', 'rmse', 'loss']
            if self.target_metrics.lower() in common_min_metrics:
                self.metrics_best = 'min'
                self.logger.info(f'Automatically set metrics_best to "min" for {self.target_metrics}')
            else:
                self.logger.warning(f'Unknown metric {self.target_metrics}, using default "min". Consider setting CFG.METRICS.BEST explicitly.')
        
        assert self.target_metrics in self.metrics or self.target_metrics == 'loss', f'Target metric {self.target_metrics} not found in metrics.'
        assert self.metrics_best in ['min', 'max'], f'Invalid best metric {self.metrics_best}.'

        # specific normalization args for data loader
        # if null, will not normalize data
        if cfg.get('TRAIN', {}).get('DATA', {}).get('NORM_EACH_CHANNEL', None) is not None:
            norm_each_channel = cfg['TRAIN']['DATA']['NORM_EACH_CHANNEL']
        else:
            norm_each_channel = False

        self.norm_each_channel = norm_each_channel

        # curriculum learning
        self.cl_param = cfg.get('TRAIN', {}).get('CL', None)
        if self.cl_param is not None:
            self.warm_up_epochs = self.cl_param.get('WARM_EPOCHS', 0)
            self.cl_epochs = self.cl_param.get('CL_EPOCHS')
            self.prediction_length = self.cl_param.get('PREDICTION_LENGTH')

        self.need_setup_graph = cfg.get('MODEL', {}).get('SETUP_GRAPH', False)

        # Evaluation settings
        self.if_evaluate_on_gpu = cfg.get('EVAL', EasyDict()).get('USE_GPU', True)
        self.evaluation_horizons = [_ - 1 for _ in cfg.get('EVAL', EasyDict()).get('HORIZONS', [])]
        assert len(self.evaluation_horizons) == 0 or min(self.evaluation_horizons) >= 0, 'The horizon should start counting from 1.'

        # target columns for evaluation
        if hasattr(cfg, 'target_columns'):
            self.target_columns = cfg.target_columns
        else:
            self.target_columns = None

    def build_scaler(self, cfg: Dict):
        """Build scaler.

        Args:
            cfg (Dict): Configuration.

        Returns:
            Scaler instance or None if no scaler is declared.
        """

        if 'SCALER' in cfg:
            return cfg['SCALER']['TYPE'](**cfg['SCALER']['PARAM'])
        return None

    def setup_graph(self, cfg: Dict, train: bool):
        """Setup all parameters and the computation graph.

        Some models (e.g., DCRNN, GTS) require creating parameters during the first forward pass, similar to TensorFlow.

        Args:
            cfg (Dict): Configuration.
            train (bool): Whether the setup is for training or inference.
        """

        dataloader = self.build_test_data_loader(cfg=cfg) if not train else self.build_train_data_loader(cfg=cfg)
        data = next(iter(dataloader))  # get the first batch
        self.forward(data=data, epoch=1, iter_num=0, train=train)

    def count_parameters(self):
        """Count the number of parameters in the model."""

        num_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f'Number of parameters: {num_parameters}')

    def init_training(self, cfg: Dict):
        """Initialize training components, including loss, meters, etc.

        Args:
            cfg (Dict): Configuration.
        """

        if self.need_setup_graph:
            self.setup_graph(cfg=cfg, train=True)
            self.need_setup_graph = False

        super().init_training(cfg)
        self.count_parameters()

        self.register_epoch_meter('train/loss', 'train', '{:.4f}')
        for key in self.metrics:
            self.register_epoch_meter(f'train/{key}', 'train', '{:.4f}')

    def init_validation(self, cfg: Dict):
        """Initialize validation components, including meters.

        Args:
            cfg (Dict): Configuration.
        """

        super().init_validation(cfg)
        self.register_epoch_meter('val/loss', 'val', '{:.4f}')
        for key in self.metrics:
            self.register_epoch_meter(f'val/{key}', 'val', '{:.4f}')

    def init_test(self, cfg: Dict):
        """Initialize test components, including meters.

        Args:
            cfg (Dict): Configuration.
        """

        if self.need_setup_graph:
            self.setup_graph(cfg=cfg, train=False)
            self.need_setup_graph = False

        super().init_test(cfg)
        self.register_epoch_meter('test/loss', 'test', '{:.4f}')
        for key in self.metrics:
            self.register_epoch_meter(f'test/{key}', 'test', '{:.4f}')

    def build_train_dataset(self, cfg: Dict):
        """Build the training dataset.

        Args:
            cfg (Dict): Configuration.

        Returns:
            Dataset: The constructed training dataset.
        """

        if 'DATASET' not in cfg:
            # TODO: support building different datasets for training, validation, and test.
            if 'logger' in inspect.signature(cfg['TRAIN']['DATA']['DATASET']['TYPE'].__init__).parameters:
                cfg['TRAIN']['DATA']['DATASET']['PARAM']['logger'] = self.logger
            if 'mode' in inspect.signature(cfg['TRAIN']['DATA']['DATASET']['TYPE'].__init__).parameters:
                cfg['TRAIN']['DATA']['DATASET']['PARAM']['mode'] = 'train'
            dataset = cfg['TRAIN']['DATA']['DATASET']['TYPE'](**cfg['TRAIN']['DATA']['DATASET']['PARAM'])
            self.logger.info(f'Train dataset length: {len(dataset)}')
            batch_size = cfg['TRAIN']['DATA']['BATCH_SIZE']
            self.iter_per_epoch = math.ceil(len(dataset) / batch_size)
        else:
            dataset = cfg['DATASET']['TYPE'](mode='train', logger=self.logger, **cfg['DATASET']['PARAM'])
            self.logger.info(f'Train dataset length: {len(dataset)}')
            batch_size = cfg['TRAIN']['DATA']['BATCH_SIZE']
            self.iter_per_epoch = math.ceil(len(dataset) / batch_size)

        return dataset

    def build_val_dataset(self, cfg: Dict):
        """Build the validation dataset.

        Args:
            cfg (Dict): Configuration.

        Returns:
            Dataset: The constructed validation dataset.
        """

        if 'DATASET' not in cfg:
            # TODO: support building different datasets for training, validation, and test.
            if 'logger' in inspect.signature(cfg['VAL']['DATA']['DATASET']['TYPE'].__init__).parameters:
                cfg['VAL']['DATA']['DATASET']['PARAM']['logger'] = self.logger
            if 'mode' in inspect.signature(cfg['VAL']['DATA']['DATASET']['TYPE'].__init__).parameters:
                cfg['VAL']['DATA']['DATASET']['PARAM']['mode'] = 'valid'
            dataset = cfg['VAL']['DATA']['DATASET']['TYPE'](**cfg['VAL']['DATA']['DATASET']['PARAM'])
            self.logger.info(f'Validation dataset length: {len(dataset)}')
        else:
            dataset = cfg['DATASET']['TYPE'](mode='valid', logger=self.logger, **cfg['DATASET']['PARAM'])
            self.logger.info(f'Validation dataset length: {len(dataset)}')

        return dataset

    def build_test_dataset(self, cfg: Dict):
        """Build the test dataset.

        Args:
            cfg (Dict): Configuration.

        Returns:
            Dataset: The constructed test dataset.
        """

        if 'DATASET' not in cfg:
            # TODO: support building different datasets for training, validation, and test.
            if 'logger' in inspect.signature(cfg['TEST']['DATA']['DATASET']['TYPE'].__init__).parameters:
                cfg['TEST']['DATA']['DATASET']['PARAM']['logger'] = self.logger
            if 'mode' in inspect.signature(cfg['TEST']['DATA']['DATASET']['TYPE'].__init__).parameters:
                cfg['TEST']['DATA']['DATASET']['PARAM']['mode'] = 'test'
            dataset = cfg['TEST']['DATA']['DATASET']['TYPE'](**cfg['TEST']['DATA']['DATASET']['PARAM'])
            self.logger.info(f'Test dataset length: {len(dataset)}')
        else:
            dataset = cfg['DATASET']['TYPE'](mode='test', logger=self.logger, **cfg['DATASET']['PARAM'])
            self.logger.info(f'Test dataset length: {len(dataset)}')

        return dataset

    def curriculum_learning(self, epoch: int = None) -> int:
        """Calculate task level for curriculum learning.

        Args:
            epoch (int, optional): Current epoch if in training process; None otherwise. Defaults to None.

        Returns:
            int: Task level for the current epoch.
        """

        if epoch is None:
            return self.prediction_length
        epoch -= 1
        # generate curriculum length
        if epoch < self.warm_up_epochs:
            # still in warm-up phase
            cl_length = self.prediction_length
        else:
            progress = ((epoch - self.warm_up_epochs) // self.cl_epochs + 1) * self.cl_step_size
            cl_length = min(progress, self.prediction_length)
        return cl_length

    def forward(self, data: tuple, epoch: int = None, iter_num: int = None, train: bool = True, **kwargs) -> Dict:
        """
        Performs the forward pass for training, validation, and testing. 
        Note: The outputs are not re-scaled.

        Args:
            data (Dict): A dictionary containing 'target' (future data) and 'inputs' (history data) (normalized by self.scaler).
            epoch (int, optional): Current epoch number. Defaults to None.
            iter_num (int, optional): Current iteration number. Defaults to None.
            train (bool, optional): Indicates whether the forward pass is for training. Defaults to True.

        Returns:
            Dict: A dictionary containing the keys:
                  - 'inputs': Selected input features.
                  - 'prediction': Model predictions.
                  - 'target': Selected target features.

        Raises:
            AssertionError: If the shape of the model output does not match [B, L, N].
        """

        raise NotImplementedError()

    def metric_forward(self, metric_func, args: Dict) -> torch.Tensor:
        """Compute metrics using the given metric function.

        Args:
            metric_func (function or functools.partial): Metric function.
            args (Dict): Arguments for metrics computation.

        Returns:
            torch.Tensor: Computed metric value.
        """

        covariate_names = inspect.signature(metric_func).parameters.keys()
        args = {k: v for k, v in args.items() if k in covariate_names}

        if isinstance(metric_func, functools.partial):
            if 'null_val' not in metric_func.keywords and 'null_val' in covariate_names: # null_val is required but not provided
                args['null_val'] = self.null_val
            metric_item = metric_func(**args)
        elif callable(metric_func):
            if 'null_val' in covariate_names: # null_val is required
                args['null_val'] = self.null_val
            metric_item = metric_func(**args)
        else:
            raise TypeError(f'Unknown metric type: {type(metric_func)}')
        return metric_item

    def train_iters(self, epoch: int, iter_index: int, data: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        """Training iteration process.

        Args:
            epoch (int): Current epoch.
            iter_index (int): Current iteration index.
            data (Union[torch.Tensor, Tuple]): Data provided by DataLoader.

        Returns:
            torch.Tensor: Loss value.
        """

        iter_num = (epoch - 1) * self.iter_per_epoch + iter_index
        forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)

        if self.cl_param:
            cl_length = self.curriculum_learning(epoch=epoch)
            forward_return['prediction'] = forward_return['prediction'][:, :cl_length, :, :]
            forward_return['target'] = forward_return['target'][:, :cl_length, :, :]
        loss = self.metric_forward(self.loss, forward_return)
        self.update_epoch_meter('train/loss', loss.item())

        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            self.update_epoch_meter(f'train/{metric_name}', metric_item.item())
        return loss

    def val_iters(self, iter_index: int, data: Union[torch.Tensor, Tuple]):
        """Validation iteration process.

        Args:
            iter_index (int): Current iteration index.
            data (Union[torch.Tensor, Tuple]): Data provided by DataLoader.
        """

        forward_return = self.forward(data=data, epoch=None, iter_num=iter_index, train=False)
        loss = self.metric_forward(self.loss, forward_return)
        self.update_epoch_meter('val/loss', loss.item())

        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            self.update_epoch_meter(f'val/{metric_name}', metric_item.item())

    def compute_evaluation_metrics(self, returns_all: Dict):
        """Compute metrics for evaluating model performance during the test process.

        Args:
            returns_all (Dict): Must contain keys: inputs, prediction, target.
        """

        metrics_results = {}
        for i in self.evaluation_horizons:
            pred = returns_all['prediction'][:, i, :, :]
            real = returns_all['target'][:, i, :, :]

            metrics_results[f'horizon_{i + 1}'] = {}
            metric_repr = ''
            for metric_name, metric_func in self.metrics.items():
                if metric_name.lower() == 'mase':
                    continue # MASE needs to be calculated after all horizons
                metric_item = self.metric_forward(metric_func, {'prediction': pred, 'target': real})
                metric_repr += f', Test {metric_name}: {metric_item.item():.4f}'
                metrics_results[f'horizon_{i + 1}'][metric_name] = metric_item.item()
            self.logger.info(f'Evaluate best model on test data for horizon {i + 1}{metric_repr}')

        metrics_results['overall'] = {}
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, returns_all)
            self.update_epoch_meter(f'test/{metric_name}', metric_item.item())
            metrics_results['overall'][metric_name] = metric_item.item()

        return metrics_results

    @torch.no_grad()
    @master_only
    def test(self, train_epoch: Optional[int] = None, save_metrics: bool = False, save_results: bool = False) -> Dict:
        """Test process.
        
        Args:
            train_epoch (Optional[int]): Current epoch if in training process.
            save_metrics (bool): Save the test metrics. Defaults to False.
            save_results (bool): Save the test results. Defaults to False.
        """

        prediction, target, inputs = [], [], []
        prediction_unnorm, target_unnorm = [], []

        for data in tqdm(self.test_data_loader):
            forward_return = self.forward(data, epoch=None, iter_num=None, train=False)

            loss = self.metric_forward(self.loss, forward_return)
            self.update_epoch_meter('test/loss', loss.item())

            if not self.if_evaluate_on_gpu:
                forward_return['prediction'] = forward_return['prediction'].detach().cpu()
                forward_return['target'] = forward_return['target'].detach().cpu()
                forward_return['inputs'] = forward_return['inputs'].detach().cpu()
                forward_return['prediction_unnorm'] = forward_return['prediction_unnorm'].detach().cpu()
                forward_return['target_unnorm'] = forward_return['target_unnorm'].detach().cpu()

            prediction.append(forward_return['prediction'])
            target.append(forward_return['target'])
            inputs.append(forward_return['inputs'])
            prediction_unnorm.append(forward_return['prediction_unnorm'])
            target_unnorm.append(forward_return['target_unnorm'])

        prediction = torch.cat(prediction, dim=0)
        target = torch.cat(target, dim=0)
        inputs = torch.cat(inputs, dim=0)
        prediction_unnorm = torch.cat(prediction_unnorm, dim=0)
        target_unnorm = torch.cat(target_unnorm, dim=0)

        returns_all = {'prediction': prediction, 'target': target, 'inputs': inputs, 
                       'prediction_unnorm': prediction_unnorm, 'target_unnorm': target_unnorm}
        metrics_results = self.compute_evaluation_metrics(returns_all)

        # save
        if save_results:
            # save returns_all to self.ckpt_save_dir/test_results.npz
            test_results = {k: v.cpu().numpy() for k, v in returns_all.items()}
            np.savez(os.path.join(self.ckpt_save_dir, 'test_results.npz'), **test_results)

        if save_metrics:
            # save metrics_results to self.ckpt_save_dir/test_metrics.json
            with open(os.path.join(self.ckpt_save_dir, 'test_metrics.json'), 'w') as f:
                json.dump(metrics_results, f, indent=4)

        csv_path = os.path.join(self.ckpt_save_dir,'test_metrics.csv')
        eval_csv_unnorm(returns_all, self.target_columns, csv_path)
        # eval_csv_norm(returns_all, self.target_columns, csv_path)
        
        return returns_all

    @torch.no_grad()
    @master_only
    def evaluate_train(self, save_metrics: bool = False, save_results: bool = False) -> Dict:
        """Evaluate process on training dataset.
        
        Args:
            save_metrics (bool): Save the train metrics. Defaults to False.
            save_results (bool): Save the train results. Defaults to False.
        """

        self.logger.info('Start evaluation on training dataset.')
        
        prediction, target, inputs = [], [], []
        prediction_unnorm, target_unnorm = [], []

        # Initialize training dataset evaluation meters
        for key in self.metrics:
            self.register_epoch_meter(f'train_eval/{key}', 'train_eval', '{:.4f}')
        self.register_epoch_meter('train_eval/loss', 'train_eval', '{:.4f}')

        for data in tqdm(self.train_data_loader):
            forward_return = self.forward(data, epoch=None, iter_num=None, train=False)

            loss = self.metric_forward(self.loss, forward_return)
            self.update_epoch_meter('train_eval/loss', loss.item())

            if not self.if_evaluate_on_gpu:
                forward_return['prediction'] = forward_return['prediction'].detach().cpu()
                forward_return['target'] = forward_return['target'].detach().cpu()
                forward_return['inputs'] = forward_return['inputs'].detach().cpu()
                forward_return['prediction_unnorm'] = forward_return['prediction_unnorm'].detach().cpu()
                forward_return['target_unnorm'] = forward_return['target_unnorm'].detach().cpu()

            prediction.append(forward_return['prediction'])
            target.append(forward_return['target'])
            inputs.append(forward_return['inputs'])
            prediction_unnorm.append(forward_return['prediction_unnorm'])
            target_unnorm.append(forward_return['target_unnorm'])

        prediction = torch.cat(prediction, dim=0)
        target = torch.cat(target, dim=0)
        inputs = torch.cat(inputs, dim=0)
        prediction_unnorm = torch.cat(prediction_unnorm, dim=0)
        target_unnorm = torch.cat(target_unnorm, dim=0)

        returns_all = {'prediction': prediction, 'target': target, 'inputs': inputs, 
                       'prediction_unnorm': prediction_unnorm, 'target_unnorm': target_unnorm}
        
        # Compute evaluation metrics for training set
        metrics_results = {}
        for i in self.evaluation_horizons:
            pred = returns_all['prediction'][:, i, :, :]
            real = returns_all['target'][:, i, :, :]

            metrics_results[f'horizon_{i + 1}'] = {}
            metric_repr = ''
            for metric_name, metric_func in self.metrics.items():
                if metric_name.lower() == 'mase':
                    continue # MASE needs to be calculated after all horizons
                metric_item = self.metric_forward(metric_func, {'prediction': pred, 'target': real})
                metric_repr += f', Train {metric_name}: {metric_item.item():.4f}'
                metrics_results[f'horizon_{i + 1}'][metric_name] = metric_item.item()
            self.logger.info(f'Evaluate model on training data for horizon {i + 1}{metric_repr}')

        metrics_results['overall'] = {}
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, returns_all)
            self.update_epoch_meter(f'train_eval/{metric_name}', metric_item.item())
            metrics_results['overall'][metric_name] = metric_item.item()

        # Print training evaluation meters
        self.print_epoch_meters('train_eval')

        # save
        if save_results:
            # save returns_all to self.ckpt_save_dir/train_results.npz
            train_results = {k: v.cpu().numpy() for k, v in returns_all.items()}
            np.savez(os.path.join(self.ckpt_save_dir, 'train_results.npz'), **train_results)
            self.logger.info(f'Training results saved to {os.path.join(self.ckpt_save_dir, "train_results.npz")}.')

        if save_metrics:
            # save metrics_results to self.ckpt_save_dir/train_metrics.json
            with open(os.path.join(self.ckpt_save_dir, 'train_metrics.json'), 'w') as f:
                json.dump(metrics_results, f, indent=4)
            self.logger.info(f'Training metrics saved to {os.path.join(self.ckpt_save_dir, "train_metrics.json")}.')

        csv_path = os.path.join(self.ckpt_save_dir,'train_metrics.csv')
        eval_csv_unnorm(returns_all, self.target_columns, csv_path)
        # eval_csv_norm(returns_all, self.target_columns, csv_path)
        self.logger.info(f'Training evaluation CSV saved to {csv_path[:-4]}_unnorm.csv and {csv_path[:-4]}_result_unnorm.csv.')
        
        return returns_all

    @torch.no_grad()
    @master_only
    def evaluate_val(self, save_metrics: bool = False, save_results: bool = False) -> Dict:
        """Evaluate process on validation dataset and save predictions and ground-truth.

        Args:
            save_metrics (bool): Save the validation metrics. Defaults to False.
            save_results (bool): Save the validation results (pred/gt). Defaults to False.
        """

        self.logger.info('Start evaluation on validation dataset.')

        prediction, target, inputs = [], [], []
        prediction_unnorm, target_unnorm = [], []

        # Initialize validation dataset evaluation meters
        for key in self.metrics:
            self.register_epoch_meter(f'val_eval/{key}', 'val_eval', '{:.4f}')
        self.register_epoch_meter('val_eval/loss', 'val_eval', '{:.4f}')

        for data in tqdm(self.val_data_loader):
            forward_return = self.forward(data, epoch=None, iter_num=None, train=False)

            loss = self.metric_forward(self.loss, forward_return)
            self.update_epoch_meter('val_eval/loss', loss.item())

            if not self.if_evaluate_on_gpu:
                forward_return['prediction'] = forward_return['prediction'].detach().cpu()
                forward_return['target'] = forward_return['target'].detach().cpu()
                forward_return['inputs'] = forward_return['inputs'].detach().cpu()
                forward_return['prediction_unnorm'] = forward_return['prediction_unnorm'].detach().cpu()
                forward_return['target_unnorm'] = forward_return['target_unnorm'].detach().cpu()

            prediction.append(forward_return['prediction'])
            target.append(forward_return['target'])
            inputs.append(forward_return['inputs'])
            prediction_unnorm.append(forward_return['prediction_unnorm'])
            target_unnorm.append(forward_return['target_unnorm'])

        prediction = torch.cat(prediction, dim=0)
        target = torch.cat(target, dim=0)
        inputs = torch.cat(inputs, dim=0)
        prediction_unnorm = torch.cat(prediction_unnorm, dim=0)
        target_unnorm = torch.cat(target_unnorm, dim=0)

        returns_all = {'prediction': prediction, 'target': target, 'inputs': inputs,
                       'prediction_unnorm': prediction_unnorm, 'target_unnorm': target_unnorm}

        # Compute evaluation metrics for validation set
        metrics_results = {}
        for i in self.evaluation_horizons:
            pred = returns_all['prediction'][:, i, :, :]
            real = returns_all['target'][:, i, :, :]

            metrics_results[f'horizon_{i + 1}'] = {}
            metric_repr = ''
            for metric_name, metric_func in self.metrics.items():
                if metric_name.lower() == 'mase':
                    continue
                metric_item = self.metric_forward(metric_func, {'prediction': pred, 'target': real})
                metric_repr += f', Val {metric_name}: {metric_item.item():.4f}'
                metrics_results[f'horizon_{i + 1}'][metric_name] = metric_item.item()
            self.logger.info(f'Evaluate model on validation data for horizon {i + 1}{metric_repr}')

        metrics_results['overall'] = {}
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, returns_all)
            self.update_epoch_meter(f'val_eval/{metric_name}', metric_item.item())
            metrics_results['overall'][metric_name] = metric_item.item()

        # Print validation evaluation meters
        self.print_epoch_meters('val_eval')

        # save
        if save_results:
            # save returns_all to self.ckpt_save_dir/val_results.npz
            val_results = {k: v.cpu().numpy() for k, v in returns_all.items()}
            np.savez(os.path.join(self.ckpt_save_dir, 'val_results.npz'), **val_results)
            self.logger.info(f'Validation results saved to {os.path.join(self.ckpt_save_dir, "val_results.npz")}.)')

        if save_metrics:
            # save metrics_results to self.ckpt_save_dir/val_metrics.json
            with open(os.path.join(self.ckpt_save_dir, 'val_metrics.json'), 'w') as f:
                json.dump(metrics_results, f, indent=4)
            self.logger.info(f'Validation metrics saved to {os.path.join(self.ckpt_save_dir, "val_metrics.json")}.)')

        csv_path = os.path.join(self.ckpt_save_dir, 'val_metrics.csv')
        eval_csv_unnorm(returns_all, self.target_columns, csv_path)
        # eval_csv_norm(returns_all, self.target_columns, csv_path)
        self.logger.info(f'Validation evaluation CSV saved to {csv_path[:-4]}_unnorm.csv and {csv_path[:-4]}_result_unnorm.csv.')

        return returns_all

    @master_only
    def on_validating_end(self, train_epoch: Optional[int]):
        """Callback at the end of the validation process.

        Args:
            train_epoch (Optional[int]): Current epoch if in training process.
        """
        greater_best = not self.metrics_best == 'min'
        if train_epoch is not None:
            self.save_best_model(train_epoch, 'val/' + self.target_metrics, greater_best=greater_best)
