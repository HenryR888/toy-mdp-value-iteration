# 5-Step Corridor Treasure Hunt MDP using Dynamic Programming Algorithm with Value Iteration Approach 

This repository implements a toy Markov Decision Process (MDP) designed to demonstrate **Value Iteration** using the **Dynamic Programming** framework in a fully specified environment.

This was a toy project to implement the algorithm from first principles, as described in the MARL book.
The code touches on:
- Bellman optimality
- Fixed-point convergence
- Policy extraction from optimal value functions

---

## Problem Overview

The environment is a one-dimensional corridor with five discrete states:

TRAP | 1 | 2 | 3 | TREASURE


### Objective

- Reach **State 4 (Treasure)** → +1 reward (terminal)
- Avoid **State 0 (Trap)** → -1 reward (terminal)
- All non-terminal transitions incur a **living cost** of -0.04

---

## MDP Formulation

### State Space
$$
S = \{0, 1, 2, 3, 4\}
$$

### Action Space
$$
A = \{0, 1\}
$$
- 0 → Move Left  
- 1 → Move Right  

### Transition Model
The transition function is fully specified and known:
- The intended action succeeds with probability 0.8
- With probability 0.2, the agent slips and moves in the opposite direction

### Reward Function
$$
R(s, a, s') =
\begin{cases}
+1 & \text{if } s' = 4 \\
-1 & \text{if } s' = 0 \\
-0.04 & \text{otherwise}
\end{cases}
$$

### Discount Factor
$$
\gamma = 0.95
$$

---

## Algorithm

This project implements **Value Iteration**, applying the Bellman Optimality Operator:

$$
V_{k+1}(s) = \max_{a \in A} \sum_{s' \in S} \tau(s' \mid s, a)\left[R(s, a, s') + \gamma V_k(s')\right]
$$

The iteration continues until convergence:

$$
\max_{s \in S} \left|V_{k+1}(s) - V_k(s)\right| < \varepsilon
$$

Once converged, the **optimal policy** is extracted greedily:

$$
\pi^{\*}(s) = \arg\max_a Q^{\*}(s, a),
$$

where,

$$
\Q^{\*}(s,a) = \sum_{s' \in S} \tau(s' \mid s, a)\left[R(s, a, s') + \gamma V_k(s')\right]
$$


known as the action-value function.

---

## Author

Henry Rochester

