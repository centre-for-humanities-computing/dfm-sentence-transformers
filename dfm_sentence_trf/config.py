from confection import Config

default_config = Config(
    dict(
        training=dict(
            epochs=5,
            warmup_steps=100,
            batch_size=30,
            max_seq_length=128,
            steps_per_epoch=None,
            checkpoint_repo=None,
            wandb_project=None,
        ),
        model=dict(
            device="cpu",
            max_seq_length=128,
        ),
    ),
)

default_angle_config = Config(
    dict(
        training=dict(
            epochs=5,
            warmup_steps=100,
            batch_size=30,
        ),
        model=dict(
            device="cpu",
            max_seq_length=128,
        ),
    ),
)

default_cleaning_config = Config(
    dict(
        cleaning=dict(batch_size=1000, specificity=1.2),
        model=dict(
            device="cpu",
        ),
    ),
)
