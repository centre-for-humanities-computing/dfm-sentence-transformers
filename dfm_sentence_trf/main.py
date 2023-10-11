from pathlib import Path
from typing import Optional

from confection import Config, registry
from radicli import Arg, Radicli
from sentence_transformers import SentenceTransformer, models

from dfm_sentence_trf.tasks import Task

cli = Radicli()


def train_model(
    model: SentenceTransformer,
    tasks: dict[str, Task],
    epochs: int,
    warmup_steps: int,
):
    for epoch in range(epochs):
        for task_name, task in tasks.items():
            model.fit(
                train_objectives=[(task.dataloader, task.loss(model))],
                warmup_steps=warmup_steps,
            )


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
    output_folder: str,
    cache_folder: Optional[str] = None,
):
    raw_config = Config().from_disk(config_path)
    cfg = registry.resolve(raw_config)
    sent_trf_kwargs = {}
    sent_trf_kwargs["device"] = cfg["model"].get("device", "cpu")
    if cache_folder is not None:
        sent_trf_kwargs["cache_folder"] = cache_folder
    pooling_kwargs = {}
    pooling_kwargs["pooling_mode"] = cfg["model"].get("pooling_mode", None)
    pooling_kwargs["pooling_mode_cls_token"] = cfg["model"].get(
        "pooling_mode_cls_token", False
    )
    pooling_kwargs["pooling_mode_max_tokens"] = cfg["model"].get(
        "pooling_mode_max_tokens", False
    )
    pooling_kwargs["pooling_mode_mean_tokens"] = cfg["model"].get(
        "pooling_mode_mean_tokens", True
    )
    pooling_kwargs["pooling_mode_mean_sqrt_len_tokens"] = cfg["model"].get(
        "pooling_mode_mean_sqrt_len_tokens", False
    )
    training_kwargs = {}
    training_kwargs["epochs"] = cfg["training"].get("epochs", 5)
    training_kwargs["warmup_steps"] = cfg["training"].get("warmup_steps", 100)

    embedding = models.Transformer(cfg["model"]["base_model"])
    pooling = models.Pooling(
        word_embedding_dimension=embedding.get_word_embedding_dimension(),
        **pooling_kwargs,
    )
    model = SentenceTransformer(
        modules=[embedding, pooling], **sent_trf_kwargs
    )
    train_model(model, cfg["tasks"], **training_kwargs)
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    model.save(output_folder)
    raw_config.to_disk(output_path.joinpath("config.cfg"))


@cli.command(
    "push_to_hub",
    config_path=Arg(help="Config file containing information about training."),
    model_path=Arg(
        "--model_path",
        help="Path to the trained model to be pushed to the Hub.",
    ),
    organization=Arg(
        "--organization",
        help="Organization in which you want to push your model.",
    ),
    commit_message=Arg(
        "--commit_message",
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
    organization: Optional[str] = None,
    commit_message: str = "Add new SentenceTransformer model.",
    exist_ok: bool = False,
    replace_model_card: bool = False,
):
    raw_config = Config().from_disk(config_path)
    cfg = registry.resolve(raw_config)
    datasets = [task.dataset for task in cfg["tasks"]]
    repo_name = cfg["model"]["name"]
    model = SentenceTransformer(model_path)
    model.save_to_hub(
        repo_name=repo_name,
        organization=organization,
        commit_message=commit_message,
        exist_ok=exist_ok,
        replace_model_card=replace_model_card,
        train_datasets=datasets,
    )
