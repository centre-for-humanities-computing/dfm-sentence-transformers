import warnings
from typing import List, Union

from datasets import Dataset, DatasetDict
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import LabelAccuracyEvaluator
from torch.utils.data import DataLoader

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
        if isinstance(self.dataset, Dataset):
            self.train_ds = self.dataset
            self.test_ds = None
        else:
            self.train_ds = self.dataset["train"]
            self.test_ds = self.dataset["test"]

    @property
    def n_labels(self) -> int:
        labels = self.train_ds[self.label]
        return len(set(labels))

    @property
    def examples(self) -> List[InputExample]:
        examples = []
        for entry in self.train_ds:
            example = InputExample(
                texts=[entry[self.sentence1], entry[self.sentence2]],
                label=entry[self.label],
            )
            examples.append(example)
        return examples

    @property
    def test_examples(self) -> List[InputExample]:
        if self.test_ds is None:
            return []
        shuffled = self.test_ds.shuffle()[:1000]
        examples = []
        for entry in shuffled:
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

    def evaluate(self, model: SentenceTransformer) -> float:
        if self.test_ds is None:
            warnings.warn("No test data in task, returning 0 on evaluation.")
            return 0
        model_with_loss = self.loss(model)
        test_loader = DataLoader(
            self.test_examples, shuffle=True, batch_size=16
        )
        evaluator = LabelAccuracyEvaluator(test_loader)
        return evaluator(model_with_loss)

    def __str__(self):
        # We have no reasonable way of telling if the labels mean the same,
        # so all tasks should get a different representation.
        return f"Softmax(scale={id(self)})"
