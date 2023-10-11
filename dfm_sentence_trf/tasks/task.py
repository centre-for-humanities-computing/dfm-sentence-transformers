from abc import abstractproperty
from typing import Callable, Protocol

from torch.utils.data import DataLoader


class Task(Protocol):
    dataset: str

    @abstractproperty
    def dataloader(self) -> DataLoader:
        pass

    @abstractproperty
    def loss(self) -> Callable:
        pass
