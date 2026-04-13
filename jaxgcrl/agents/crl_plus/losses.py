import jax
import jax.numpy as jnp

from jaxgcrl.agents.crl.losses import energy_fn, update_actor_and_alpha

__all__ = ["update_actor_and_alpha", "update_critic"]


def update_critic(config, networks, transitions, training_state, key):
    def critic_loss(critic_params, transitions, key):
        sa_encoder_params, g_encoder_params = (
            critic_params["sa_encoder"],
            critic_params["g_encoder"],
        )

        state = transitions.observation[:, : config["state_size"]]
        action = transitions.action

        sa_repr = networks["sa_encoder"].apply(
            sa_encoder_params, jnp.concatenate([state, action], axis=-1)
        )
        g_repr = networks["g_encoder"].apply(
            g_encoder_params, transitions.observation[:, config["state_size"] :]
        )

        logits = energy_fn(config["energy_fn"], sa_repr[:, None, :], g_repr[None, :, :])

        # Per-row fwd_infonce so we can apply per-row hard-positive weights.
        diag = jnp.diag(logits)
        row_loss = -(diag - jax.nn.logsumexp(logits, axis=1))

        # Hard-positive reweighting: rows whose positive pair is far apart in
        # embedding space (i.e. large -diag) get higher weight. Normalized so
        # mean(weights) == 1, keeping the effective loss scale stable.
        hard_score = jax.lax.stop_gradient(-diag)
        beta = config["hard_positive_beta"]
        batch = row_loss.shape[0]
        softmax_weights = jax.nn.softmax(beta * hard_score)
        weights = softmax_weights * batch
        critic_loss = jnp.mean(weights * row_loss)

        # logsumexp regularisation (kept uniform across rows).
        logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
        critic_loss += config["logsumexp_penalty_coeff"] * jnp.mean(logsumexp**2)

        I = jnp.eye(logits.shape[0])
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        weight_ess = 1.0 / jnp.sum(softmax_weights**2)
        weight_max = jnp.max(weights)

        return critic_loss, (
            logsumexp,
            I,
            correct,
            logits_pos,
            logits_neg,
            weight_ess,
            weight_max,
        )

    (loss, (logsumexp, I, correct, logits_pos, logits_neg, weight_ess, weight_max)), grad = (
        jax.value_and_grad(critic_loss, has_aux=True)(
            training_state.critic_state.params, transitions, key
        )
    )
    new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
    training_state = training_state.replace(critic_state=new_critic_state)

    metrics = {
        "categorical_accuracy": jnp.mean(correct),
        "logits_pos": logits_pos,
        "logits_neg": logits_neg,
        "logsumexp": logsumexp.mean(),
        "critic_loss": loss,
        "hard_pos_weight_ess": weight_ess,
        "hard_pos_weight_max": weight_max,
    }

    return training_state, metrics
