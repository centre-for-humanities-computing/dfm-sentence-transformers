from datasets import Dataset, load_dataset
from sentence_transformers import InputExample, losses
from torch.utils.data import DataLoader

from dfm_sentence_trf.tasks.task import Task
from dfm_sentence_trf.tasks.utils import coinflip, collect_negative_ids


class ContrastiveParallel(Task):
    def __init__(
        self,
        dataset: str,
        sentence1: str,
        sentence2: str,
        batch_size: int = 128,
        negative_samples: int = 5,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        self.batch_size = batch_size
        self.negative_samples = negative_samples
        self.shuffle = shuffle

    @property
    def dataloader(self) -> DataLoader:
        ds: Dataset = load_dataset(self.dataset, split="train")  # type: ignore
        examples: list[InputExample] = []
        for positive_id, positive_entry in enumerate(ds):
            positive_example = InputExample(
                texts=[
                    positive_entry[self.sentence1],
                    positive_entry[self.sentence2],
                ],
                label=1,
            )
            examples.append(positive_example)
            for negative_id in collect_negative_ids(
                len(ds), self.negative_samples, positive_id
            ):
                negative_entry = ds[negative_id]
                if coinflip():
                    texts = [
                        positive_entry[self.sentence1],
                        negative_entry[self.sentence2],
                    ]
                else:
                    texts = [
                        negative_entry[self.sentence1],
                        positive_entry[self.sentence2],
                    ]
                negative_example = InputExample(texts=texts, label=0)
                examples.append(negative_example)
        return DataLoader(
            examples, batch_size=self.batch_size, shuffle=self.shuffle
        )

    @property
    def loss(self):
        return losses.ContrastiveLoss
