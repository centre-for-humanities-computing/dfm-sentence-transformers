from typing import List

from sentence_transformers.evaluation import SentenceEvaluator

from dfm_sentence_trf.tasks.task import Task


class TaskListEvaluator(SentenceEvaluator):
    """Evaluator that evaluates the model on all test sets
    from all tasks and returns the sum of their scores."""

    def __init__(self, tasks: List[Task]):
        self.tasks = tasks

    def __call__(
        self, model, output_path=None, epoch: int = -1, steps: int = -1
    ) -> float:
        scores = []
        for task in self.tasks:
            score = task.evaluate(model)
            scores.append(score)
        return sum(scores)
