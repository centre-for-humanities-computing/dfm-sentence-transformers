from typing import Dict, List, Optional

import jax.numpy as jnp
import numpy as np

from dfm_sentence_trf.weak_supervision.double_logistic_model import (
    default_hyperparameters,
    fit_model,
    get_mean_prior,
    loglikelihood,
)


def binary_entropy(observations, params):
    n_obs = observations.shape[0]
    p = jnp.exp(loglikelihood(observations, y=jnp.ones(n_obs), **params))
    return -p * jnp.log2(p) - (1 - p) * jnp.log2(1 - p)


def propose(entropy, temperature=1.0):
    probs = jnp.exp(entropy / temperature)
    probs = probs / jnp.sum(probs)
    proposal = np.random.multinomial(1, probs)
    return proposal


class ThresholdLearner:
    def __init__(
        self,
        sentence_transformer_name: str,
        sentences1: List[str],
        sentences2: List[str],
        hyperparameters: Optional[Dict[str, float]] = None,
    ):
        self.hyperparameters = hyperparameters or default_hyperparameters
        self.params = get_mean_prior(hyperparameters)
        self.x = []
        self.y = []

    def update_params(self):
        self.params = fit_model(
            self.x, self.y, self.hyperparameters, self.params
        )

    def display_proposal(self):
        pass
