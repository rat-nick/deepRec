from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class Params:
    w: torch.Tensor = None
    v: torch.Tensor = None
    h: torch.Tensor = None


@dataclass
class Metrics:
    trainRMSE: list = field(default_factory=list)
    trainMAE: list = field(default_factory=list)
    validRMSE: list = field(default_factory=list)
    validMAE: list = field(default_factory=list)

    @property
    def bestRMSE(self) -> dict:
        return {"epoch": np.argmin(self.validRMSE), "value": np.min(self.validRMSE)}

    @property
    def bestMAE(self) -> dict:
        return {"epoch": np.argmin(self.validMAE), "value": np.min(self.validMAE)}


@dataclass
class TrainingParams:
    # decay: lambda x: x
    prev_wd: torch.Tensor = None
    prev_vd: torch.Tensor = None
    prev_hd: torch.Tensor = None
    epoch: int = 0
    current_patience: int = 0


@dataclass
class HyperParams:
    batch_size: int
    lr: float
    l1: float = 0
    l2: float = 0
    momentum: float = 0
    max_epochs: int = 100
    patience: int = 10
    early_stopping: bool = False
