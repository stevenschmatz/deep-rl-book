# Deep Reinforcement Learning

Reinforcement learning (RL) is the study of learning intelligent _behavior_. Learning how to act is arguably a much more difficult problem than vanilla supervised learning—in addition to perception, many other challenges exist:

* Exploration vs. exploitation
* Delayed credit assignment
* Temporal correlations

Deep learning brings powerful function approximation and hierarchical understanding to the state spaces encountered in reinforcement learning problems. In the past five years, huge progress has been made in bringing the best of both worlds together. **It is, arguably, the closest humanity has got to achieving general AI.**

The purpose of this book is to provide a guide to new researchers and practitioners interested in using deep architectures in reinforcement learning problems.

_This book is a very early work in progress. If you have any suggestions, ideas, or concerns, please [let me know](https://twitter.com/stevenschmatz)!_

I am keeping this free and publicly available to help accelerate the pace of RL research and implementation. **If this book helped you, please consider donating to my [Patreon page](https://www.patreon.com/stevenschmatz)!**

#### Prerequisites

This book assumes knowledge of deep learning and basic reinforcement learning. There are many ways to learn these two topics, but I suggest you to read the following resources first:

* [_Reinforcement Learning: An Introduction_, Sutton & Barto, 2nd edition](http://ufal.mff.cuni.cz/~straka/courses/npfl114/2016/sutton-bookdraft2016sep.pdf). This book is the most famous textbook in RL.
* [_Deep Learning_, Goodfellow, Bengio & Courville](http://www.deeplearningbook.org/). This book is the most famous textbook in deep learning.

If you enjoyed either of these books, please support the authors by purchasing them.

#### Book Structure

The structure of this book is split up into two main sections:

* **Unique challenges** of deep reinforcement learning
* **Value optimization**: learning value and action-value functions, which are then used to predict a reward;
* **Policy optimization**: directly optimizing the policy, using the gradient of expected discounted reward.
* **Special topics**, which include hierarchical RL, multi-agent RL, imitation learning, multi-task learning, inverse RL, etc.

[![](/assets/algorithms.png)](https://github.com/NervanaSystems/coach)