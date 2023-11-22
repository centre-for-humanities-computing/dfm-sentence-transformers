from abc import abstractmethod, abstractproperty
from itertools import chain, groupby
from typing import Callable, Iterable, List, Protocol, Tuple, Union

from datasets import Dataset, DatasetDict
from sentence_transformers import InputExample, SentenceTransformer
from torch.utils.data import DataLoader


class Task(Protocol):
    dataset: Union[Dataset, DatasetDict]

    @abstractproperty
    def examples(self) -> List[InputExample]:  # type: ignore
        pass

    @abstractproperty
    def loss(self) -> Callable:  # type: ignore
        pass

    @abstractmethod
    def evaluate(self, model: SentenceTransformer) -> float:
        pass


def join_examples(tasks: Iterable[Task]) -> List[InputExample]:
    examples = (task.examples for task in tasks)
    return list(chain.from_iterable(examples))


def to_objectives(
    tasks: List[Task], model: SentenceTransformer, batch_size: int
) -> List[Tuple]:
    """Finalizes all objectives, joins all tasks that
    have the same loss function, this way the datasets from
    different objectives can be mixed in a batch."""
    objectives = []
    for _, group in groupby(tasks, key=str):
        group = list(group)
        examples = join_examples(group)
        loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
        loss = group[0].loss(model)
        objectives.append((loader, loss))
    return objectives
