import json
from typing import Iterable, List, Optional, Tuple

import torch
from sentence_transformers import (__MODEL_HUB_ORGANIZATION__,
                                   SentenceTransformer, __version__)
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.model_card_templates import ModelCardTemplate
from sentence_transformers.util import batch_to_device, fullname
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange


def fit_sentence_transformer(
    sentence_transformer,
    train_objectives: List[Tuple[DataLoader, nn.Module]],
    evaluator: Optional[SentenceEvaluator] = None,
    epochs: int = 1,
    steps_per_epoch=None,
    scheduler: str = "WarmupLinear",
    warmup_steps: int = 10000,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1,
) -> Iterable[SentenceTransformer]:
    """Custom training loop so that we can actually save stuff
    to disk and hub after each epoch.
    The function yields the current state of the model after each epoch,
    this allows top-level users to do anything with it.
    """
    optimizer_class = torch.optim.AdamW
    info_loss_functions = []
    for dataloader, loss in train_objectives:
        info_loss_functions.extend(
            ModelCardTemplate.get_train_objective_info(dataloader, loss)
        )
    info_loss_functions = "\n\n".join([text for text in info_loss_functions])

    info_fit_parameters = json.dumps(
        {
            "evaluator": fullname(evaluator),
            "epochs": epochs,
            "steps_per_epoch": steps_per_epoch,
            "scheduler": scheduler,
            "warmup_steps": warmup_steps,
            "optimizer_class": str(optimizer_class),
            "optimizer_params": {"lr": learning_rate},
            "weight_decay": weight_decay,
            "max_grad_norm": max_grad_norm,
        },
        indent=4,
        sort_keys=True,
    )
    sentence_transformer._model_card_text = None
    sentence_transformer._model_card_vars[
        "{TRAINING_SECTION}"
    ] = ModelCardTemplate.__TRAINING_SECTION__.replace(
        "{LOSS_FUNCTIONS}", info_loss_functions
    ).replace(
        "{FIT_PARAMETERS}", info_fit_parameters
    )
    sentence_transformer.to(sentence_transformer.device)
    dataloaders = [dataloader for dataloader, _ in train_objectives]
    # Use smart batching
    for dataloader in dataloaders:
        dataloader.collate_fn = sentence_transformer.smart_batching_collate
    loss_models = [loss for _, loss in train_objectives]
    for loss_model in loss_models:
        loss_model.to(sentence_transformer.device)
    sentence_transformer.best_score = -9999999
    if steps_per_epoch is None or steps_per_epoch == 0:
        steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])
    num_train_steps = int(steps_per_epoch * epochs)
    # Prepare optimizers
    optimizers = []
    schedulers = []
    for loss_model in loss_models:
        param_optimizer = list(loss_model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = optimizer_class(
            optimizer_grouped_parameters, lr=learning_rate
        )
        scheduler_obj = sentence_transformer._get_scheduler(
            optimizer,
            scheduler=scheduler,
            warmup_steps=warmup_steps,
            t_total=num_train_steps,
        )
        optimizers.append(optimizer)
        schedulers.append(scheduler_obj)
    global_step = 0
    data_iterators = [iter(dataloader) for dataloader in dataloaders]
    num_train_objectives = len(train_objectives)
    skip_scheduler = False
    for epoch in trange(epochs, desc="Epoch"):
        training_steps = 0
        for loss_model in loss_models:
            loss_model.zero_grad()
            loss_model.train()
        for _ in trange(
            steps_per_epoch,
            desc="Iteration",
            smoothing=0.05,
        ):
            for train_idx in range(num_train_objectives):
                loss_model = loss_models[train_idx]
                optimizer = optimizers[train_idx]
                scheduler = schedulers[train_idx]
                data_iterator = data_iterators[train_idx]
                try:
                    data = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(dataloaders[train_idx])
                    data_iterators[train_idx] = data_iterator
                    data = next(data_iterator)
                features, labels = data
                labels = labels.to(sentence_transformer.device)
                features = [
                    batch_to_device(batch, sentence_transformer.device)
                    for batch in features
                ]
                loss_value = loss_model(features, labels)
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(
                    loss_model.parameters(), max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad()
                if not skip_scheduler:
                    scheduler.step()
            training_steps += 1
            global_step += 1
        sentence_transformer._eval_during_training(
            evaluator, None, False, epoch, -1, None
        )
        yield sentence_transformer
