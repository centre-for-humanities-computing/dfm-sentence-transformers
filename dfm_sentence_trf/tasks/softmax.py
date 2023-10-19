from typing import Union

from datasets import Dataset, DatasetDict
from sentence_transformers import InputExample, SentenceTransformer, losses

from dfm_sentence_trf.tasks.task import Task


class Softmax(Task):
    def __init__(
        self,
        dataset: Union[Dataset, DatasetDict],
        sentence1: str,
        sentence2: str,
        label: str,
    ):
        self.dataset = dataset
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        self.label = label

    @property
    def n_labels(self) -> int:
        if isinstance(self.dataset, Dataset):
            ds = self.dataset
        else:
            ds = self.dataset["train"]
        labels = ds[self.label]
        return len(set(labels))

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
                label=entry[self.label],
            )
            examples.append(example)
        return examples

    @property
    def loss(self):
        def _loss(model: SentenceTransformer):
            return losses.SoftmaxLoss(
                model=model,
                sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                num_labels=self.n_labels,
            )

        return _loss

    def __str__(self):
        # We have no reasonable way of telling if the labels mean the same,
        # so all tasks should get a different representation.
        return f"Softmax(scale={id(self)})"
