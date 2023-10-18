from typing import Union

from datasets import Dataset, DatasetDict
from sentence_transformers import InputExample, losses

from dfm_sentence_trf.tasks.task import Task


class CosineSimilarity(Task):
    def __init__(
        self,
        dataset: Union[Dataset, DatasetDict],
        sentence1: str,
        sentence2: str,
        similarity: str,
    ):
        self.dataset = dataset
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        self.similarity = similarity

    @property
    def examples(self) -> list[InputExample]:
        examples = []
        if isinstance(self.dataset, Dataset):
            ds = self.dataset
        else:
            ds = self.dataset["train"]
        for entry in ds:
            example = InputExample(
                texts=[entry[self.sentence1], entry[self.sentence2]],
                label=entry[self.similarity],
            )
            examples.append(example)
        return examples

    @property
    def loss(self):
        return lambda model: losses.CosineSimilarityLoss(model)

    def __str__(self):
        return "CosineSimilarity()"
