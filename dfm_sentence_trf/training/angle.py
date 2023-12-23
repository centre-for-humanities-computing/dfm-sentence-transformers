from angle_emb import AnglE, AngleDataTokenizer
from datasets import Dataset, DatasetDict
from sentence_transformers import SentenceTransformer, models


def finetune_with_angle(
    base_model_name: str,
    dataset: DatasetDict,
    sentence1: str,
    sentence2: str,
    label: str,
    checkpoint_directory: str = "./_model_checkpoints/",
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    epochs: int = 5,
    device: str = "cpu",
    warmup_steps: int = 100,
    pooling_mode: str = "mean",
    max_seq_length: int = 512,
) -> AnglE:
    if pooling_mode == "mean":
        pooling_mode = "avg"
    ds = dataset.rename_columns(
        {sentence1: "text1", sentence2: "text2", label: "label"}
    )
    ds = ds.select_columns(["text1", "text2", "label"])
    train_ds = ds["train"]
    try:
        test_ds = ds["test"]
    except KeyError:
        test_ds = None
    angle = AnglE.from_pretrained(
        base_model_name,
        pooling_strategy=pooling_mode,
        max_length=max_seq_length,
    )
    angle.backbone.to(device)
    train_ds = train_ds.shuffle().map(
        AngleDataTokenizer(angle.tokenizer, angle.max_length), num_proc=8
    )
    if test_ds is not None:
        test_ds = test_ds.map(
            AngleDataTokenizer(angle.tokenizer, angle.max_length), num_proc=8
        )
    angle.fit(
        train_ds=train_ds,
        valid_ds=test_ds,
        output_dir=checkpoint_directory,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        save_steps=1000,
        eval_steps=5000,
        warmup_steps=warmup_steps,
        gradient_accumulation_steps=1,
        loss_kwargs={
            "w1": 1.0,
            "w2": 1.0,
            "w3": 1.0,
            "cosine_tau": 20,
            "ibn_tau": 20,
            "angle_tau": 1.0,
        },
        fp16=True,
        logging_steps=400,
    )
    return angle


def angle_to_sentence_transformer(
    base_model: str, angle: AnglE, pooling_mode: str = "mean"
) -> SentenceTransformer:
    model = models.Transformer(base_model)
    model.max_seq_length = angle.max_length
    model.auto_model = angle.backbone
    model.tokenizer = angle.tokenizer
    pooling = models.Pooling(
        model.get_word_embedding_dimension(), pooling_mode=pooling_mode
    )
    trf = SentenceTransformer(modules=[model, pooling])
    return trf
