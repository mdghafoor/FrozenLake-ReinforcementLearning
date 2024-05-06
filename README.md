# FrozenLake-ReinforcementLearning

Deep Q Reinforcement learning project using OpenAI's Gymnasium and inspiration from course guidelines by Andrew Ng. 
https://www.coursera.org/specializations/machine-learning-introduction
https://gymnasium.farama.org

## Results:
100% Accuracy for "4x4" map non-slippery mode.

Performance Video:

https://github.com/mdghafoor/FrozenLake-ReinforcementLearning/assets/158994486/25ae75aa-90f2-4a5d-8113-bda06d325a42

Performance Plots:

![FrozenLakeDeepQLearningStats](https://github.com/mdghafoor/FrozenLake-ReinforcementLearning/assets/158994486/1a305e5e-cd0f-4a7c-ac48-9242d65a3777)

## Description
```Text
This script executes the following Deep Q learning with Experience Replay algorithm:
1. Initialize memory buffer D with capacity N
2. Initialize Q network with random weights w
3. Initialize target Q^ network with random weights w- = w
4. For episode i=1 to M do:
5.   Receive initial observation state S_1
6.     While not solved, or not fallen through environment do:
7.         Observe state S_t and choose action A_t using ε greedy policy.
8.         Take action A_t and recieve reward, and nexet state S_t+1
9.         Store experience replay (S_t,A_t,R_t,S_t+1) in memory buffer.
10.        Every C steps, perform update:
11.            Sample random mini batch of experience replays from memory buffer D (S_j, A_j, R_j, S_j+1)
12.            Set y_j to R_j if episodes terminate at j+1 else, set y_j to R_J + γ max_a' Q^(s_j+1,a')
13.            Perform a gradient decent step on (y_j-Q(s_j,a_j;w))^2 with respect to Q Network weights w
14.            Update the weights of target Q^ Network using soft update method
15.        Every X steps, perform hard update:
16.            Update weights of target Q^ Network with Q Network weights using hard upate method
17.    end
18. end
```

## Important notes/explanations for algorithm: 
```Text
1. Estimate Action Value function interatively using Bellman equation:
    Q_(i+1) (s,a) = R + γ max_a' Q^_i (s',a')
    This iterative method converges Q*(s,a) as i->infinity where Q*(s,a) is the optimal action-value function.
    Using a neural network an estimate of Q(s,a) can be obtained where Q(s,a) approximately equals Q*(s,a) by adjustings
    the weights of Q(s,a) at each iteration to minimize the MSE in the bellman equation. 
2. In order to obtain the MSE to update Q(s,a), first the target values are required. These are obtained using the following formula:
    y = R + γ max_a' Q(s',a';w) where w are the weights of the Q Network
3. We adjust the weights w at each iteration by minimizing the following error:
    R + γ max_a' Q(s',a';w) - Q(s,a;w)
4. Since y_targets change at each iteration, in order to reduce oscillations and instability, a seperate neural network,
    target Q-Network is created to generate y_targets. Therefore the formula is adjusted to:
    R + γ max_a' Q^(s',a';w-) - Q(s,a;w) where w- is the weights of the target Q-Network and w are the weights of the Q-Network
5. When updating the weights of the target Q-network using the weights of the Q-Network, a soft update approach is used every C steps.
    This is controlled by hyperparameter TAU where:
    w- <- TAU*w + (1-TAU)*w-.
    This helps ensure y_targets update slowly to improve stability.
    However, due to scarce rewards, a hard update is also performed every X steps where
    w- <- w
```
