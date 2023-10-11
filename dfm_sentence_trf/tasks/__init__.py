import catalogue
from confection import registry

from dfm_sentence_trf.tasks.contrastive_parallel import ContrastiveParallel
from dfm_sentence_trf.tasks.task import Task

registry.tasks = catalogue.create("confection", "tasks", entry_points=False)

__all__ = ["ContrastiveParallel", "Task"]


@registry.tasks.register("contrastive-parallel")
def make_contrastive_parallel(
    dataset: str,
    sentence1: str,
    sentence2: str,
    batch_size: int = 128,
    negative_samples: int = 5,
    shuffle: bool = True,
):
    return ContrastiveParallel(
        dataset=dataset,
        sentence1=sentence1,
        sentence2=sentence2,
        batch_size=batch_size,
        negative_samples=negative_samples,
        shuffle=shuffle,
    )
