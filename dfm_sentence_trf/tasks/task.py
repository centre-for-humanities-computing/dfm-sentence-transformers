from abc import abstractproperty
from itertools import chain, groupby
from typing import Callable, Iterable, Protocol, Union

from datasets import Dataset, DatasetDict
from sentence_transformers import InputExample, SentenceTransformer
from torch.utils.data import DataLoader


class Task(Protocol):
    dataset: Union[Dataset, DatasetDict]

    @abstractproperty
    def examples(self) -> list[InputExample]:  # type: ignore
        pass

    @abstractproperty
    def loss(self) -> Callable:  # type: ignore
        pass


def join_examples(tasks: Iterable[Task]) -> list[InputExample]:
    examples = (task.examples for task in tasks)
    return list(chain.from_iterable(examples))


def to_objectives(
    tasks: list[Task], model: SentenceTransformer, batch_size: int
) -> list[tuple]:
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
