# Deep Q Networks

###### Link to the paper: [cs.toronto.edu](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) (2014)

#### Basic idea

Rather than finding a policy directly, we can approximate the $$Q^*$$-function. Then, finding the optimal policy would trivially be $$\pi^*(s) = \arg \max_a Q^*(s, a)$$.

First, we can write the Bellman optimality equation for $$Q^\pi$$:

$$
Q^\pi(s,a) = r + \gamma Q^\pi(s', \pi(s))
$$

However, this Bellman optimality equation only holds when $$\pi$$ has converged. Otherwise, there will be some difference between $$Q(s, a)$$ and $$\gamma Q(s', \pi(s))$$. This is known as the _temporal difference error_ $$\delta$$:

$$
\delta = \hat{Q}(s, a) - \left( r + \gamma \max_a \hat{Q}(s', a) \right)
$$