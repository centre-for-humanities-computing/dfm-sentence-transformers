import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import catalogue
from confection import Config, registry
from datasets import Dataset, DatasetDict, load_dataset
from radicli import Arg, Radicli
from sentence_transformers import SentenceTransformer, models
from tqdm import tqdm

from dfm_sentence_trf.config import (default_angle_config,
                                     default_cleaning_config, default_config)
from dfm_sentence_trf.evaluation.task_evaluator import TaskListEvaluator
from dfm_sentence_trf.filtering.cleaning import generate_cleaned_pairs
from dfm_sentence_trf.hub import save_to_hub
from dfm_sentence_trf.tasks import to_objectives
from dfm_sentence_trf.training.train import fit_sentence_transformer

logger = logging.getLogger(__name__)
cli = Radicli()

registry.loaders = catalogue.create(
    "confection", "loaders", entry_points=False
)


@registry.loaders.register("load_dataset")
def loaders_load_dataset(
    path: str,
    name: Optional[str] = None,
    additional_arguments: Optional[Dict[str, Any]] = None,
) -> Union[Dataset, DatasetDict]:
    if additional_arguments is None:
        additional_arguments = dict()
    return load_dataset(path, name=name, **additional_arguments)


@cli.command(
    "finetune",
    config_path=Arg(help="Config file containing information about training."),
    output_folder=Arg(
        "--output-folder", "-o", help="Folder to save the finalized model."
    ),
    cache_folder=Arg(
        "--cache-folder",
        "-c",
        help="Folder to cache models into while training.",
    ),
)
def finetune(
    config_path: str,
    output_folder: str = "./model",
    cache_folder: str = "./model_cache",
):
    raw_config = Config().from_disk(config_path)
    raw_config = default_config.merge(raw_config)
    wandb_project = raw_config["training"].get("wandb_project")
    if wandb_project is not None:
        import wandb

        wandb.init(project=wandb_project, config=dict(raw_config))

    cfg = registry.resolve(raw_config)
    sent_trf_kwargs = dict()
    sent_trf_kwargs["device"] = cfg["model"]["device"]
    sent_trf_kwargs["cache_folder"] = cache_folder

    logger.info("Initialize SentenceTransformer model")
    embedding = models.Transformer(
        cfg["model"]["base_model"],
        max_seq_length=cfg["model"]["max_seq_length"],
    )
    pooling = models.Pooling(
        word_embedding_dimension=embedding.get_word_embedding_dimension(),
    )
    model = SentenceTransformer(
        modules=[embedding, pooling], **sent_trf_kwargs
    )

    epochs = cfg["training"]["epochs"]
    warmup_steps = cfg["training"]["warmup_steps"]
    batch_size = cfg["training"]["batch_size"]
    steps_per_epoch = cfg["training"]["steps_per_epoch"]
    checkpoint_repo = cfg["training"]["checkpoint_repo"]

    tasks = list(cfg["tasks"].values())
    evaluator = TaskListEvaluator(
        dict(cfg["tasks"]), log_to_wandb=(wandb_project is not None)
    )
    logger.info("Convert tasks to objectives")
    objectives = to_objectives(tasks, model, batch_size)
    logger.info("Starting Model Training")
    # Getting an iterable of model checkpoints after each epoch
    model_checkpoints = fit_sentence_transformer(
        model,
        objectives,
        epochs=epochs,
        warmup_steps=warmup_steps,
        evaluator=evaluator,
        steps_per_epoch=steps_per_epoch,
    )
    checkpoint_path = Path(cache_folder)
    checkpoint_path.mkdir(exist_ok=True)
    for i_epoch, checkpoint in enumerate(model_checkpoints):
        # We will save the model and push it to hub
        # if the user specifies a checkpoint repo.
        checkpoint.save(str(checkpoint_path))
        if checkpoint_repo is not None:
            try:
                save_to_hub(
                    checkpoint,
                    checkpoint_repo,
                    commit_message=f"Saved model checkpoint after epoch {i_epoch}",
                    exist_ok=True,
                    replace_model_card=True,
                )
            except ValueError as error_message:
                warnings.warn(
                    "Couldn't push model checkpoint."
                    "Did you forget to log into huggingface hub?\n"
                    f"{error_message}"
                )

    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    model.save(output_folder)
    raw_config.to_disk(output_path.joinpath("config.cfg"))
    if wandb_project is not None:
        import wandb

        wandb.finish()


@cli.command(
    "push_to_hub",
    config_path=Arg(help="Config file containing information about training."),
    model_path=Arg(
        "--model_path",
        help="Path to the trained model to be pushed to the Hub.",
    ),
    commit_message=Arg(
        "--commit_message",
        "-m",
        help="Message to commit while pushing.",
    ),
    exist_ok=Arg(
        "--exist_ok",
        help="Indicates whether the model should be"
        "allowed to be pushed in an existent repo.",
    ),
    replace_model_card=Arg(
        "--replace_model_card",
        help="Indicates whether the README should be"
        "replaced with a newly generated model card.",
    ),
)
def push_to_hub(
    config_path: str,
    model_path: str,
    commit_message: str = "Add new SentenceTransformer model.",
    exist_ok: bool = False,
    replace_model_card: bool = False,
):
    raw_config = Config().from_disk(config_path)
    cfg = registry.resolve(raw_config)
    repo_name = cfg["model"]["name"]
    model = SentenceTransformer(model_path)
    save_to_hub(
        model,
        repo_name,
        commit_message=commit_message,
        exist_ok=exist_ok,
        replace_model_card=replace_model_card,
    )


@cli.command(
    "angle_finetune",
    config_path=Arg(
        help="Angle finetuning config file containing information about training."
    ),
    output_folder=Arg(
        "--output-folder", "-o", help="Folder to save the finalized model."
    ),
    cache_folder=Arg(
        "--cache-folder",
        "-c",
        help="Folder to cache models into while training.",
    ),
)
def angle_finetune(
    config_path: str,
    output_folder: str = "./model",
    cache_folder: str = "./model_cache",
):
    from dfm_sentence_trf.training.angle import (angle_to_sentence_transformer,
                                                 finetune_with_angle)

    raw_config = Config().from_disk(config_path)
    raw_config = default_angle_config.merge(raw_config)
    cfg = registry.resolve(raw_config)
    sent_trf_kwargs = dict()
    sent_trf_kwargs["device"] = cfg["model"]["device"]
    sent_trf_kwargs["cache_folder"] = cache_folder

    logger.info("Initializing model")
    base_model = cfg["model"]["base_model"]
    max_seq_length = cfg["model"]["max_seq_length"]
    epochs = cfg["training"]["epochs"]
    warmup_steps = cfg["training"]["warmup_steps"]
    batch_size = cfg["training"]["batch_size"]
    device = cfg["model"]["device"]
    dataset = cfg["angle"]["dataset"]
    checkpoint_path = Path(cache_folder)
    checkpoint_path.mkdir(exist_ok=True)

    logger.info("Starting model training")
    angle = finetune_with_angle(
        base_model_name=base_model,
        dataset=dataset,
        sentence1=cfg["angle"]["sentence1"],
        sentence2=cfg["angle"]["sentence2"],
        label=cfg["angle"]["label"],
        max_seq_length=max_seq_length,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        checkpoint_directory=str(checkpoint_path),
        warmup_steps=warmup_steps,
    )

    logger.info("Turning model into SentenceTransformer.")
    model = angle_to_sentence_transformer(base_model=base_model, angle=angle)
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    model.save(output_folder)
    raw_config.to_disk(output_path.joinpath("config.cfg"))


@cli.command(
    "clean_dataset",
    config_path=Arg(help="Path to dataset config."),
)
def clean_dataset(
    config_path: str,
):
    raw_config = Config().from_disk(config_path)
    raw_config = default_cleaning_config.merge(raw_config)
    cfg = registry.resolve(raw_config)
    logger.info("Initialize SentenceTransformer model")
    trf = SentenceTransformer(
        cfg["cleaning"]["model"]["name"],
        device=cfg["cleaning"]["model"]["device"],
    )
    specificity = cfg["cleaning"]["specificity"]
    batch_size = cfg["cleaning"]["batch_size"]
    dataset_name = cfg["cleaning"]["name"]
    name = dataset_name.split("/")[-1]
    out_filename = f"{name}.jsonl"
    with open(out_filename, "w") as out_file:
        for dataset_name, data in cfg["data"].items():
            logger.info(f"Cleaning: {dataset_name}")
            pairs = generate_cleaned_pairs(
                trf,
                dataset_name,
                **data,
                batch_size=batch_size,
                specificity=specificity,
            )
            for entry in pairs:
                out_file.write(json.dumps(entry) + "\n")
    logger.info("Done.")


@cli.command("push_dataset", config_path=Arg(help="Path to dataset config"))
def push_dataset(config_path: str):
    """Shuffes and pushes dataset to hub with a train and test split."""
    raw_config = Config().from_disk(config_path)
    cfg = default_cleaning_config.merge(raw_config)
    # We do not resolve, since we don't want to load all the datasets again.
    name = cfg["cleaning"]["name"].split("/")[-1]
    out_filename = f"{name}.jsonl"
    ds = Dataset.from_json(out_filename)
    logger.info("Shuffling dataset.")
    ds = ds.shuffle(seed=42)
    logger.info("Splitting dataset.")
    ds = ds.train_test_split(test_size=0.2)
    logger.info("Pushing dataset to hub.")
    ds.push_to_hub(cfg["cleaning"]["name"])
