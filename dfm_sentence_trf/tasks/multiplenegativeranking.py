from typing import Union

from datasets import Dataset, DatasetDict
from sentence_transformers import InputExample, losses

from dfm_sentence_trf.tasks.task import Task


class MultipleNegativesRanking(Task):
    def __init__(
        self,
        dataset: Union[Dataset, DatasetDict],
        sentence1: str,
        sentence2: str,
        scale: float = 20.0,
    ):
        self.dataset = dataset
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        self.scale = scale

    @property
    def examples(self) -> list[InputExample]:
        examples = []
        if isinstance(self.dataset, Dataset):
            ds = self.dataset
        else:
            ds = self.dataset["train"]
        for entry in ds:
            example = InputExample(
                texts=[entry[self.sentence1], entry[self.sentence2]], label=1
            )
            examples.append(example)
        return examples

    @property
    def loss(self):
        return lambda model: losses.MultipleNegativesRankingLoss(
            model, scale=self.scale
        )

    def __str__(self):
        return f"MultipleNegativesRanking(scale={self.scale})"
