"""
This is a Toy Markov Decision Process (MDP) Example called the '5-step Corridor Treasure Hunt'. This is
implemented using the Dynamic Processing Algorithm with the Value Iteration Approach. 

The goal is to get to the end of the corridor (state number 4) and avoid state number 0, which is a trap that will immobilise the agent.

The action set is comprised of left and right.

The Transition process is the piecewise function: 80% to go where intended, and 20% chance of slipping and going the opposite direction.

The Reward function is made up of +1 for reaching the treasure; -1 for becoming immobilised; and -0.04 for going to a non-terminal state - which we shall call the 'living cost'.

The discount factor is 0.95.

We define our MDP more formally as follows: 

State-Space = {0,1,2,3,4}

A = {0,1} where 0 is for left and 1 is for right

tau = {0.8, if go where intended; 0.2, slip and go the other direction

R = {-0.04, for s in {1,2,3}; 1, for s = 4; -1 for s = 0

gamma = 0.95

"""

import numpy as np

# --------
# MDP Setup
# --------

S = 5 # we have 5 states
A = 2 # 2 actions: left and right

LEFT, RIGHT = 0, 1

terminal_states = {0,4}

gamma=0.95
p_intended = 0.8
p_slip = 0.2

living_cost = -0.04


def is_terminal(s: int) -> bool:
    """
    check whether we are in terminal state:
    """
    return s in terminal_states

def next_state(s: int, action: int) -> int:
    """
    Terminal state is absorbing, 
    otherwise next state is -1 of input (for LEFT) or +1 of input (for RIGHT)
    """

    if is_terminal(s):
        return s
    return s-1 if action == LEFT else s+1



# --------
# Build out Transition Distribution and Reward Table
# --------

Tau = np.zeros((S,A,S))
R = np.zeros((S,A,S))

def create_TauR():


    for s in range(S):
        for a in range(A):

            if is_terminal(s):
                Tau[s,a,s] = 1.0
                R[s,a,s]= 0.0
                continue
            
            INTENDED = LEFT if a == LEFT else RIGHT
            SLIP = RIGHT if INTENDED == LEFT else LEFT

            sp_intended = next_state(s, INTENDED)
            sp_slip = next_state(s, SLIP)

            Tau[s,a,sp_intended] += p_intended
            Tau[s,a,sp_slip] += p_slip

            for sp in [sp_intended,sp_slip]:
                if sp == 4:
                    R[s,a,sp] = 1.0
                elif sp == 0:
                    R[s,a,sp] = -1.0
                else:
                    R[s,a,sp] = living_cost



# --------
# One-Step Bellman Back Q(s,a) computation
# --------

def compute_Q(V: np.ndarray) -> np.ndarray:
    """
    Q[s,a] = sum_sp.Tau(s'|s,a).[R(s,a,s') + gamma.V(s')]
    """

    Q = np.zeros((S,A))

    for s in range(S):
        for a in range(A):
            Q[s,a] = np.sum(Tau[s,a,:] * (R[s,a,:] + gamma * V))

    return Q



# --------
# Value Iteration
# --------

def value_iteration(tol: float = 1e-8, max_iters = 10_000):

    V = np.zeros(S)

    for _ in range(max_iters):

        Q = compute_Q(V)
        V_new = V.copy()
        for s in range(S):
            if is_terminal(s):
               V_new[s] = 0.0
            else:
               V_new[s] = np.max(Q[s])
        
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
    
        V = V_new

    # Extract the Greedy Policy 
    Q_final = compute_Q(V)
    pi = np.zeros(S, dtype=int)
    for s in range(S):
        if is_terminal(s):
            pi[s] = 0
        else:
            pi[s] = np.argmax(Q_final[s])
    return V, pi

            
# --------
# Display the Policy with the Values
# --------

def action_name(a: int) -> str:
    return "L" if a == LEFT else "R"

create_TauR()
V, pi = value_iteration()
print("VALUE ITERATION SOLUTION:")
print("----------")
for s in range(S):
    if is_terminal(s):
        print(f"s={s}: V={V[s]: .6f} terminal")
    else:
        print(f"s={s}: V={V[s]: .6f} pi = {action_name(pi[s])}")
print("----------")
print('The Policy is:')
print(pi)



  


