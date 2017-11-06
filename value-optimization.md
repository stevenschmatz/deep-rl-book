# Challenges of Deep RL

In the supervised learning setting, the goal is *pattern recognition*: given many input-output pairs, use the gradient of the prediction error to guide learning. The error signal is used to guide the parameter search in a direction which minimizes training error.

The most important assumption of supervised learning is that the training and test examples are **independent and identically distributed (i.i.d.)**—all the examples come from the same distribution. Hence, information from the training set can be applied to new examples, and generalization is possible.

## 1.  RL violates the i.i.d. assumption.

The reason is that the sample distribution is non-stationary. There are many reasons for this, among the following:

1.  In on-policy learning, the policy used to generate the experiences ($$\pi$$) is also the policy which is being improved. Hence, when the value of $$\pi$$ changes, the distribution of training examples changes.
2. Also, we often want to learn policy-specific value functions—estimating $$q_\pi$$ or $$v_\pi$$ for example. However, as in generalized policy iteration, we need to estimate $$q\pi$$ while $$\pi$$ changes.

Hence, our function approximation methods must support online learning.

## 2. RL generates temporally correlated samples.

#TODO

