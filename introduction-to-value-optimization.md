#
Value-function estimation

Consider the problem of determining a value function with function approximation. In the tabular case, we could use back-ups to compute our value function exactly. For function approximation, we must choose a gradient of the state or state-action pair which moves us in a direction towards optimality.

However, how can we compute the loss of our current value function, **if we don’t know the optimal value function**?

The solution is actually quite simple. Rather than performing a backup—where we assign the value of a state to a new estimate—we instead perform a *gradient step* in the direction of a backup. We perform a step in the direction of the gradient—rather than backing up—because we are using an approximation, and have to balance prediction errors from several states at once.

## Value function gradients

If we knew the optimal value function, we could simply move the weights in a direction towards the local optimum of the error estimation.

$$$\theta_{t+1} \gets \theta_t + \alpha \left \lbrac v_\pi(S_t) - \hat{v}(S_t | \theta_t) \right \rbrac \nabla \hat{v} (S_t | \theta_t)$$$

Note that this is the gradient of the mean squared value error:

$$
\sum_{s \in \mathcal{S}} d(s) \left[ v_\pi(s) - \hat{v}(s | \theta)\right]^2
$$

However, since $$v_\pi$$ is not known, we must substitute in some **estimate** $$\mathbb{E} \lbrac U_t \rbrac \approx v_\pi $$ which is preferably unbiased. These estimates can be found in the backup equations of tabular, dynamic programming RL methods like TD($$\lambda$$) or Monte Carlo. Some examples of possible $$U_t$$ values are listed:

| Loss function | $$U_t$$ value | Parameter update step expression | Biased? |
| --- | --- | --- | --- |
| Monte Carlo return | $$U_t \gets G_t$$ | $$\theta_{t+1} \gets \theta_t + \alpha \left[ G_t - \hat{v}(S_t | \theta_t \right] \nabla \hat{v} (S_t | \theta_t)$$ | No |
| TD(0) return | $$U_t \gets R_t + \gamma \hat{v} (S' | \theta_t)$$| $$\theta_{t+1} \gets \theta_t + \alpha \left[ R_t + \gamma \hat{v} (S' | \theta_t) - \hat{v}(S_t | \theta_t \right] \nabla \hat{v} (S_t | \theta_t)$$ | **Yes** |

Note that the TD(0) return is *biased*, since the target is defined in terms of the parameters $$\theta$$. Methods that use such a target are known as *semi-gradient methods*, because they only take into account the effect of the gradient on the estimate, but not the target.

The Monte Carlo update step is an unbiased estimate of $$v_\pi$$, by the definition of $$v_\pi$$. Although the unbiased nature of the MC estimate is appealing, the update step must be performed offline—the return $$G_t$$ is only known at the end of an episode. In contrast, the biased TD(0) estimate only relies on TD errors and hence can learn in non-episodic environments, and can learn online.