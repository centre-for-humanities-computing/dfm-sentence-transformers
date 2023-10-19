from pathlib import Path
from typing import Optional, Union

import catalogue
from confection import Config, registry
from datasets import Dataset, DatasetDict, load_dataset
from radicli import Arg, Radicli
from sentence_transformers import SentenceTransformer, models

from dfm_sentence_trf.config import default_config
from dfm_sentence_trf.hub import save_to_hub
from dfm_sentence_trf.tasks import to_objectives

cli = Radicli()

registry.loaders = catalogue.create(
    "confection", "loaders", entry_points=False
)


@registry.loaders.register("load_dataset")
def _load_dataset(
    path: str, name: Optional[str] = None
) -> Union[Dataset, DatasetDict]:
    return load_dataset(path, name=name)  # type: ignore


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
    raw_config = default_config.merge(raw_config)
    cfg = registry.resolve(raw_config)
    sent_trf_kwargs = dict()
    sent_trf_kwargs["device"] = cfg["model"]["device"]
    if cache_folder is not None:
        sent_trf_kwargs["cache_folder"] = cache_folder

    embedding = models.Transformer(cfg["model"]["base_model"])
    pooling = models.Pooling(
        word_embedding_dimension=embedding.get_word_embedding_dimension(),
    )
    model = SentenceTransformer(
        modules=[embedding, pooling], **sent_trf_kwargs
    )

    epochs = cfg["training"]["epochs"]
    warmup_steps = cfg["training"]["warmup_steps"]
    batch_size = cfg["training"]["batch_size"]
    tasks = list(cfg["tasks"].values())
    objectives = to_objectives(tasks, model, batch_size)
    model.fit(objectives, epochs=epochs, warmup_steps=warmup_steps)
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
