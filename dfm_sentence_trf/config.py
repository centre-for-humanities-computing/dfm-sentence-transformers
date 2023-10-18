from confection import Config

default_config = Config(
    dict(
        training=dict(epochs=5, warmump_steps=100, batch_size=30),
        model=dict(device="cpu"),
    ),
)
