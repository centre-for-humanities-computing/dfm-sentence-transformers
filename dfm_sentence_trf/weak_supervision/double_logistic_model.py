import jax
import jax.numpy as jnp
import optax
from jax.scipy import stats
from scipy.stats import gamma

default_hyperparameters = dict(
    mu_mu=0.7,
    mu_scale=0.1,
    width_shape=0.2,
    width_scale=1.0,
    shape_shape=8,
    shape_scale=1.0,
)


def double_logistic(x, mu, alpha, beta):
    """Soft decision boundaries on both sides."""
    return jnp.exp(-jnp.float_power(jnp.abs(x - mu) / alpha, beta))


def loglikelihood(x, y, mu, alpha, beta):
    """Returns log likelihood of each observation."""
    prob = double_logistic(x, mu, alpha, beta)
    return jnp.abs((1 - y) - prob)


def logprior(
    mu,
    alpha,
    beta,
    mu_mu,
    mu_scale,
    width_shape,
    width_scale,
    shape_shape,
    shape_scale,
):
    """Returns log prior of the parameters."""
    mu_prior = stats.norm.logpdf(mu, loc=mu_mu, scale=mu_scale)
    width_prior = stats.gamma.logpdf(alpha, width_shape, scale=width_scale)
    shape_prior = stats.gamma.logpdf(beta, shape_shape, scale=shape_scale)
    return mu_prior + width_prior + shape_prior


def logposterior(
    x,
    y,
    params,
    hyperparams,
):
    """Logdensity of the model given all hyperparameters, parameters and data."""
    return jnp.sum(loglikelihood(x, y, **params)) + logprior(
        **params, **hyperparams
    )


def get_mean_prior(hyperparameters):
    initial_params = dict(
        mu=hyperparameters["mu_mu"],
        alpha=gamma.mean(
            hyperparameters["width_shape"],
            scale=hyperparameters["width_scale"],
        ),
        beta=gamma.mean(
            hyperparameters["shape_shape"],
            scale=hyperparameters["shape_scale"],
        ),
    )
    return initial_params


def fit_model(
    x, y, hyperparameters, initial_params, learning_rate=0.01, n_steps=200
):
    # We set initial parameters to the mean of the prior
    params = initial_params
    loss = lambda params: -logposterior(x, y, params, hyperparameters)
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state):
        loss_value, grads = jax.value_and_grad(loss)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    for i in range(n_steps):
        params, opt_state, loss_value = step(params, opt_state)
    return initial_params, params
