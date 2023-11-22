import warnings
from typing import List, Union

from datasets import Dataset, DatasetDict
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import TranslationEvaluator

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
        if isinstance(self.dataset, Dataset):
            self.train_ds = self.dataset
            self.test_ds = None
        else:
            self.train_ds = self.dataset["train"]
            self.test_ds = self.dataset["test"]

    @property
    def examples(self) -> List[InputExample]:
        examples = []
        for entry in self.train_ds:
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

    def evaluate(self, model: SentenceTransformer) -> float:
        if self.test_ds is None:
            warnings.warn("No test data in task, returning 0 on evaluation.")
            return 0
        shuffled = self.test_ds.shuffle()[:1000]
        evaluator = TranslationEvaluator(
            source_sentences=shuffled[self.sentence1],
            target_sentences=shuffled[self.sentence2],
        )
        return evaluator(model)

    def __str__(self):
        return f"MultipleNegativesRanking(scale={self.scale})"
