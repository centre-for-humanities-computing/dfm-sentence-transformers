import warnings
from typing import List, Union

from datasets import Dataset, DatasetDict
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

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
                texts=[entry[self.sentence1], entry[self.sentence2]],
                label=entry[self.similarity],
            )
            examples.append(example)
        return examples

    @property
    def loss(self):
        return lambda model: losses.CosineSimilarityLoss(model)

    def evaluate(self, model: SentenceTransformer) -> float:
        if self.test_ds is None:
            warnings.warn("No test data in task, returning 0 on evaluation.")
            return 0
        shuffled = self.test_ds.shuffle()[:1000]
        evaluator = EmbeddingSimilarityEvaluator(
            sentences1=shuffled[self.sentence1],
            sentences2=shuffled[self.sentence2],
            scores=shuffled[self.similarity],
        )
        return evaluator(model)

    def __str__(self):
        return "CosineSimilarity()"
