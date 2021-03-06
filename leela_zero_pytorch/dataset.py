import logging
import glob
import os
import torch
import numpy as np
import gzip

from typing import Tuple, List
from itertools import cycle
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)

# (planes, move probs, game outcome)
DataPoint = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def stone_plane(plane: np.ndarray) -> torch.Tensor:
    bits = np.unpackbits(plane)[7:]
    return torch.tensor(bits).float().view(19, 19)


def move_plane(turn: int) -> List[torch.Tensor]:
    # 0 = black, 1 = white
    # 17) All 1 if black is to move, 0 otherwise
    # 18) All 1 if white is to move, 0 otherwise
    ones = torch.ones(19, 19)
    zeros = torch.zeros(19, 19)
    if turn == 0:
        # black's turn to move
        return [ones, zeros]
    return [zeros, ones]


def hex_to_ndarray(hex: str) -> np.ndarray:
    return np.array(bytearray.fromhex('0' + hex))


def get_data_from_file(fname: str) -> Tuple[List[np.ndarray], List[int], List[np.ndarray], List[int]]:
    stone_planes: List[np.ndarray] = []
    move_planes: List[int] = []
    move_probs: List[np.ndarray] = []
    outcomes: List[int] = []
    with gzip.open(fname, 'rt') as f:
        for i in cycle(range(19)):
            try:
                line = next(f).strip()
            except StopIteration:
                break
            if i < 16:
                stone_planes.append(hex_to_ndarray(line))
            elif i == 16:
                move_planes.append(int(line))
            elif i == 17:
                move_probs.append(np.array([int(p) for p in line.split()], dtype=np.uint8))
            else:
                # i == 18
                outcomes.append(int(line))
    return stone_planes, move_planes, move_probs, outcomes


class Dataset:

    def __init__(self, filenames: List[str]):
        stone_planes: List[np.ndarray] = []
        move_planes: List[int] = []
        move_probs: List[np.ndarray] = []
        outcomes: List[int] = []
        self.raw_datapoints: List[List[str]] = []
        with ProcessPoolExecutor() as executor:
            for data in executor.map(get_data_from_file, filenames):
                f_stone_planes, f_move_planes, f_move_probs, f_outcomes = data
                stone_planes.extend(f_stone_planes)
                move_planes.extend(f_move_planes)
                move_probs.extend(f_move_probs)
                outcomes.extend(f_outcomes)
        self.stone_planes = np.stack(stone_planes) if len(stone_planes) > 0 else np.empty((0, 19, 19))
        self.move_planes = np.array(move_planes)
        self.move_probs = np.stack(move_probs) if len(move_probs) > 0 else np.empty((0, 19*19+1))
        self.outcomes = np.array(outcomes)

    def __getitem__(self, idx: int) -> DataPoint:
        input_planes: List[torch.Tensor] = []
        for plane in self.stone_planes[idx * 16: (idx + 1) * 16]:
            input_planes.append(stone_plane(plane))
        input_planes.extend(move_plane(self.move_planes[idx]))
        return (
            torch.stack(input_planes),
            torch.tensor(self.move_probs[idx].argmax()),
            torch.tensor(self.outcomes[idx]).float(),
        )

    def __len__(self):
        return len(self.outcomes)

    @classmethod
    def from_data_dir(cls, path: str):
        return cls(glob.glob(os.path.join(path, '*.gz')))
