# Action-value function estimation

Similar to how TD(0) and Monte Carlo value estimation could be applied to the case of function approximation, we can apply $$n$$-step SARSA and Q-learning to the gradient case.

Our goal is to learn a parametric approximation $$\hat{q}(s, a | \theta) \approx q_*(s, a)$$ for on-policy control. Instead of performing a gradient step moving $$S_t$$ towards some target $$U_t$$, we now turn to performing a gradient stop on $$S_t, A_t$$ towards a target $$U_t$$. Hence, the gradient update step is of the form:

$$
\theta_{t+1} \gets \theta_t + \alpha \left[ U_t - \hat{q}(S_t, A_t | \theta_t) \right] \nabla \hat{q}(S_t, A_t | \theta_t)
$$

## Examples

In the case of (one-step) SARSA, this update is:

$$
\theta_{t+1} \gets \theta_t + \alpha \left[ R_{t+1} + \gamma \hat{q}(S_{t+1}, A_{t+1} | \theta_t) - \hat{q}(S_t, A_t | \theta_t) \right] \nabla \hat{q}(S_t, A_t | \theta_t)
$$

This algorithm is known as *episodic semi-gradient one-step SARSA*, and it would have the same convergence properties as TD(0) if the policy were constant.

And of course, there is also an analogue for $$n$$-step SARSA as well:

$$
G_t^{(n)} \gets R_{t+1} + \gamma R_{t+2} + ... + \gamma^{n-1}R_{t+n} + \gamma^n \hat{q}(S_{t+n}, A_{t+n}, \theta_{t+n-1})
$$

$$
\theta_{t+n} \gets \theta_{t+n-1} + \alpha \left[G_t^{(n)}  - \hat{q}(S_t, A_t | \theta_{t+n-1}) \right] \nabla \hat{q}(S_t, A_t | \theta_{t+n-1})
$$

As well as semi-gradient Q-learning:

$$
\theta_{t+1} \gets \theta_{t} + \alpha \left[ R_{t+1} + \gamma \max_{a'}  \hat{q}(S_{t+1}, a' | \theta_t ) \right] \nabla \hat{q}(S_t, A_t | \theta_t)
$$

Any of these methods may be used with any gradient-based optimizer, such as RMSProp, Adam, or vanilla mini batch stochastic gradient descent.

#### Application to control problems

To apply these gradient action-value methods to control problems, simply combine them with a suitable exploration policy. For example, a commonly used exploration policy is an $$\epsilon$$-greedy policy with $$\epsilon$$ annealed linearly from a high value to a low value.

## Remaining issues

We should still be wary about these function approximation methods for the reasons outlined in the “Challenges of RL section”: training will be *unstable*. This is due to a non stationary distribution of experiences and temporal correlations as outlined in the chapter “Challenges of Deep RL”.

Fortunately, there are many solutions to the training instability problem for control. The first successful application of learning from high-dimensional state spaces is the *deep Q-network*, which learned policies from raw pixel values that beat human experts at 49 Atari games.