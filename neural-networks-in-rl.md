#
Why *deep* reinforcement learning?

In reinforcement learning, our goal is to learn an optimal policy. Deep learning provides a way to use action and state spaces that are extremely large or continuous.

## Tabular methods

If one aims to find an optimal policy for a small, discrete state space, then we can use *tabular methods*: those that associate a value with each state (or state-action pair). These values can be learned exactly for every state—and to find the optimal policy, they *must* be learned for every state:

* To find an optimal policy from a **value function**, choose the value which maximizes the value $$\pi(s) = \arg \max_a \sum_{s', r} p(s', r | s, a) \left[ r + \gamma v(s') \right]$$. Note that this requires the environment’s transition model.
* To find an optimal policy from an **action-value function**, select each action greedily: $$\pi(s) = \arg \max_a q(s, a)$$.

These methods have some nice convergence properties. For example, tabular backup-based methods are guaranteed to have an improved value function at every iteration, until they converge to the optimal value function.

All of these methods rely on some form of *back-up*—an estimation of a value for a given state to another “backed-up” estimate for that state.

| Method | Backup |
| ------ | ------ |
| Monte Carlo simulation | $$S_t \gets G_t$$ |
| TD(0) | $$S_t \gets R_{t+1} + \gamma \hat{v} (S_{t+1}, \theta_t)$$ |
| $$n$$-step backup | $$S_t \gets G_t^{(n)}$$ |
| Dynamic programming policy evaluation backup | $$s \gets \mathbb{E}_\pi \left[ R_{t+1} + \gamma \hat{v} (S_{t+1}, \theta_t) \right]$$ |

Despite these properties, tabular solution methods can only be used on a small subset of problems we encounter in RL. Namely, it greatly restricts the class of problems one can tackle, to problems with a small, discrete state space.

For example, consider the case of learning a value function from pixel values. Even for a small image (32 x 32 pixels, 256 grayscale values per pixel), the size of the state space is $$256^{(32 \times 32)}$$. There are unthinkably many more states in this problem than there are *atoms in the universe*.

Fortunately, we need not compute the exact value function to have behavior that is good enough for real-world applications. After all, humans do not have to act perfectly in order to behave intelligently.

## Function approximation

To solve the problem of large state spaces, we make a few assumptions:

* **Parameterization of the value function**: the value function need not be represented in a tabular way, and instead can be *approximated* in a finite number of parameters—with the number of parameters being much smaller than the number of states.
* **The smoothness prior**: in order to generalize to new experiences, we estimate that the value of nearby states will be similar. For example, consider an agent which assigns high value to states containing pictures of apples. If the image is *almost exactly the same* as another image which has a high value, then we assume the value will also be high.

Hence, our goal now is to find $$\hat{v}(s | \theta) \approx v_\pi Z(s)$$ (note that the value function estimate is now parameterized by a finite vector $$\theta$$).

### Parameterization

Function approximation is flexible enough to encode a wide variety of function families. For example, the following can all be used in this framework:

* The space of **linear functions**—that is, functions which perform a sum of weighted inputs
* The space of **decision trees**
* The space of **artificial neural networks**—functions that can be represented by a series of linear combinations

We will focus on the latter case, because neural networks excel at approximating complex, nonlinear functions, such as those found in the RL problem. But in principle, you can use any differentiable, parameterized policy representation with the methods outlined in this book.

The workflow for deep RL problems typically looks like this:

1. Define a neural network architecture, which is believed to contain a good approximation of the value function in the space of its representable functions
2. Repeat the following steps:
3. Act according to the parameterized policy to receive experience
4. Adjust the weights of the network according to an error signal which is expected to improve expected behavior

We begin this book by discussing on-policy value function approximation.