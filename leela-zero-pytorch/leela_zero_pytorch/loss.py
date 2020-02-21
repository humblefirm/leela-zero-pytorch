import torch
import torch.nn.functional as F

from typing import Tuple

from flambe.metric import Metric


class MSECrossEntropyLoss(Metric):

    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def compute(self, pred: Tuple[torch.Tensor, torch.Tensor],
                target: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        pred_move, pred_val = pred
        pred_move = pred_move.squeeze()
        pred_val = pred_val.squeeze()
        target_move, target_val = target
        cross_entropy_loss = F.cross_entropy(pred_move, target_move)
        mse_loss = F.mse_loss(pred_val, target_val)
        return self.alpha * mse_loss + cross_entropy_loss

    def aggregate(self, state: dict, *args, **kwargs) -> dict:
        score = self.compute(*args, **kwargs)
        score_np = score.cpu().detach().numpy() \
            if isinstance(score, torch.Tensor) \
            else score
        try:
            num_samples = args[0][0].size(0)
        except (ValueError, AttributeError):
            raise ValueError(f'Cannot get size from {type(args[0][0])}')
        if not state:
            state['accumulated_score'] = 0.
            state['sample_count'] = 0
        state['accumulated_score'] = \
            (state['sample_count'] * state['accumulated_score'] +
             num_samples * score_np.item()) / \
            (state['sample_count'] + num_samples)
        state['sample_count'] = state['sample_count'] + num_samples
        return state
