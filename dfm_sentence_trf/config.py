from confection import Config

default_config = Config(
    dict(
        training=dict(
            epochs=5,
            warmup_steps=100,
            batch_size=30,
            max_seq_length=128,
            steps_per_epoch=None,
        ),
        model=dict(device="cpu"),
    ),
)
