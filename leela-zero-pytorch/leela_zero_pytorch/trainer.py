import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import pytorch_lightning as pl

from typing import Tuple, List
from torch.utils.data import DataLoader

from flambe.learn import Trainer
from flambe.dataset import Dataset


logger = logging.getLogger(__name__)


class GoTrainer(Trainer):

    def __init__(self,
                 dataset,
                 train_sampler,
                 val_sampler,
                 model,
                 loss_fn,
                 metric_fn,
                 optimizer,
                 **kwargs):
        super().__init__(
            dataset,
            train_sampler,
            val_sampler,
            model,
            loss_fn,
            metric_fn,
            optimizer,
            **kwargs)
        self.parallel_model = None
        if torch.cuda.device_count() > 1:
            logger.info(f'{torch.cuda.device_count()} GPUs, using DataParallel')
            self.parallel_model = nn.DataParallel(self.model)

    def get_model(self):
        return self.parallel_model if self.parallel_model is not None else self.model

    def _compute_batch(self, batch: Tuple[torch.Tensor, ...],
                       metrics: List[Tuple] = []) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = self._batch_to_device(batch)
        pred, target = self.get_model()(*batch)
        for metric, state in metrics:
            metric.aggregate(state, pred, target)
        loss = self.loss_fn(pred, target)
        return pred, target, loss


class LightningTask:

    def __init__(self,
                 model: pl.LightningModule,
                 trainer: pl.Trainer):
        self.model = model
        self.trainer = trainer

    def run(self) -> bool:
        self.trainer.fit(self.model)
        return False


class GoLightningModule(pl.LightningModule):

    def __init__(self,
                 dataset: Dataset,
                 model: torch.nn.Module,
                 alpha: float):
        super().__init__()
        self.dataset = dataset
        self.model = model
        self.alpha = alpha

    def _calculate_loss(self, pred: Tuple[torch.Tensor, torch.Tensor],
                        target: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        pred_move, pred_val = pred
        pred_move = pred_move.squeeze()
        pred_val = pred_val.squeeze()
        target_move, target_val = target
        cross_entropy_loss = F.cross_entropy(pred_move, target_move)
        mse_loss = F.mse_loss(pred_val, target_val)
        return self.alpha * mse_loss + cross_entropy_loss

    def forward(self, planes, target_pol, target_val):
        return self.model(planes, target_pol, target_val)

    def training_step(self, batch, batch_idx):
        pred, target = self.forward(*batch)
        loss = self._calculate_loss(pred, target)
        return {
            'loss': loss,
            'log': {'training_loss': loss},
        }

    def validation_step(self, batch, batch_idx):
        pred, target = self.forward(*batch)
        return {'val_loss': self._calculate_loss(pred, target)}

    def validation_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'log': {'validation_loss': val_loss_mean}}

    def test_step(self, batch, batch_idx):
        pred, target = self.forward(*batch)
        return {'test_loss': self._calculate_loss(pred, target)}

    def test_end(self, outputs):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        return {'log': {'test_loss': test_loss_mean}}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(
            self.dataset.train,
            batch_size=16,
            shuffle=True)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(
            self.dataset.val,
            batch_size=16,
            shuffle=True)

    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(
            self.dataset.test,
            batch_size=16,
            shuffle=True)
