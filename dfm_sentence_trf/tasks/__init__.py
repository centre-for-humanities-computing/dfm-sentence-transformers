from typing import Union

import catalogue
from confection import registry
from datasets import Dataset, DatasetDict

from dfm_sentence_trf.tasks.cosine_similarity import CosineSimilarity
from dfm_sentence_trf.tasks.multiplenegativeranking import (
    MultipleNegativesRanking,
)
from dfm_sentence_trf.tasks.softmax import Softmax
from dfm_sentence_trf.tasks.task import Task, to_objectives

registry.tasks = catalogue.create("confection", "tasks", entry_points=False)

__all__ = [
    "MultipleNegativesRanking",
    "Task",
    "to_objectives",
    "CosineSimilarity",
    "Softmax",
]


@registry.tasks.register("multiple_negatives_ranking")
def make_multiple_negatives_ranking(
    dataset: Union[Dataset, DatasetDict],
    sentence1: str,
    sentence2: str,
    scale: float = 20.0,
) -> MultipleNegativesRanking:
    return MultipleNegativesRanking(dataset, sentence1, sentence2, scale)


@registry.tasks.register("cosine_similarity")
def make_cosine_similarity(
    dataset: Union[Dataset, DatasetDict],
    sentence1: str,
    sentence2: str,
    similarity: str,
) -> CosineSimilarity:
    return CosineSimilarity(dataset, sentence1, sentence2, similarity)


@registry.tasks.register("softmax")
def make_softmax(
    dataset: Union[Dataset, DatasetDict],
    sentence1: str,
    sentence2: str,
    label: str,
) -> Softmax:
    return Softmax(dataset, sentence1, sentence2, label)
