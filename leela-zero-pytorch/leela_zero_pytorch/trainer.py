import torch
import torch.nn as nn
import logging

from typing import Tuple, List

from flambe.learn import Trainer


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
