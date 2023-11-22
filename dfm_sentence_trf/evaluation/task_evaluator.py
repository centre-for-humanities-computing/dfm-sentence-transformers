import logging
from typing import Dict

from sentence_transformers.evaluation import SentenceEvaluator

from dfm_sentence_trf.tasks.task import Task

logger = logging.getLogger(__name__)


class TaskListEvaluator(SentenceEvaluator):
    """Evaluator that evaluates the model on all test sets
    from all tasks and returns the sum of their scores."""

    def __init__(self, tasks: Dict[str, Task], log_to_wandb: bool = False):
        self.tasks = tasks
        self.log_to_wandb = log_to_wandb

    def __call__(
        self, model, output_path=None, epoch: int = -1, steps: int = -1
    ) -> float:
        scores = dict()
        for task_name, task in self.tasks.items():
            score = task.evaluate(model)
            scores[task_name] = score
        sum_score = sum(scores.values())
        scores["Summed Score"] = sum_score
        if self.log_to_wandb:
            import wandb

            wandb.log(scores)
        else:
            logger.info("Evaluation results: " + str(scores))
        return sum_score
