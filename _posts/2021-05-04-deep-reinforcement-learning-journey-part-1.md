---
layout: post
title:  "Two Mortals’ Journey into a Deep Reinforcement Learning Project"
author: "Daniel Szemerey & Mark Aron Szulyovszky"
comments: false
tags: ops
---


![Cover Image](/assets/posts/drl-multi-1/1 - Title Illustration.png)

You can find the entire [codebase here](https://github.com/applied-exploration/multi-drl)

## **Why Deep Reinforcement learning?**

Deep Reinforcement Learning is about learning from mistakes - although all machine learning algorithms depend on making sense of the magnitude of error they’ve made. With RL, you can learn how to solve tasks that have an iterative, sequential decision making process (called [Markov Decision Process](https://en.wikipedia.org/wiki/Markov_decision_process)), like driving a car, playing chess or beating the human champion of… Breakout!

If you dig beneath the surface, you’ll quickly find out that although DRL t is still [relatively unstable](https://www.alexirpan.com/2018/02/14/rl-hard.html), it was built from just a few assumptions, and is a universal approach - you still need to retrain the agent for each environment, but you don’t need to change its underlying structure. It also handles multi-agent, stochastic and partially observable environments well, and is a model-free approach, so it works without us coding up any specific knowledge upfront.

But ultimately, look at the videos of DRL demonstrating super-human skills in a number of environments, like [Starcraft](https://deepmind.com/blog/article/AlphaStar-Grandmaster-level-in-StarCraft-II-using-multi-agent-reinforcement-learning) (Deepmind’s AlphaStar) or control problems like [this robot hand by openAI](https://openai.com/blog/solving-rubiks-cube/), and you’ll understand why!

Driven by curiosity (and Daniel by the need to apply the new skills he acquired while completing Udacity’s DRL nanodegree), we were trying to find a nail to mash with our Deep RL hammer! We thought it's’ unlikely that we’ll be able to come up with a completely new DRL agent structure that beats the well-established algorithms on the first go, so the decision was made: we’ll focus on the environment!


## **Ideation**

We were talking about the potential direction of our first project, and a bunch of questions came up:



1. Should it be about a novel environment, let's say, a bridge builder or a board game, like monopoly? (Or [Hive](http://tdk.bme.hu/VIK/DownloadPaper/Mely-megerositeses-tanulas-alapu-Hive-jatekos)? [Exploding Kittens](https://web.stanford.edu/class/aa228/reports/2018/final135.pdf)? [Uno](https://web.stanford.edu/class/aa228/reports/2020/final79.pdf)? [Risk](http://kth.diva-portal.org/smash/get/diva2:1514096/FULLTEXT01.pdf)? [Sim City](https://arxiv.org/abs/2002.03896)?  It looks like someone has been there already…)
2. Or about the interaction between agents, eg. a multi-agent environment, where to reach the goal, the agents need to balance between collaboration and competition?
3. Or an existing, difficult environment, which we can tackle with a new kind of agent?

In the end what got us excited is the question if local agents together can converge to state level solutions. What do we mean by this? Take a car that is driving from point A to a goal B (on a grid). With a single agent the solution is trivial, you just need to take the shortest route. Things become interesting once we add more agents, and a collision criteria, partial visibility… But first, we needed to get answers to these questions:


1. Will a DQN Agent and a REINFORCE Agent coverage on fixed or randomized starting and end positions?
2. Will a DQN Agent and a REINFORCE Agent coverage on a stochastic  environment?
3. Will a DQN Agent and a REINFORCE Agent coverage on a stochastic, multi-agent, fully observable environment?
4. Will a larger neural network perform better in all cases?


## Part I: Creating an Environment

Conforming to the OpenAI gym interface is actually quite straightforward - you write a step, a reset and a render method. You have plenty of good examples in the[ gym repo](https://github.com/openai/gym). Having a multi-agent environment is not that tricky either: instead of passing in a single action to step, it takes an array of actions, one for each agent.

Despite its triviality, creating an environment from scratch has been an immensely useful exercise - so much that we were wondering why this isn't the first step in any course! It gives you a good perspective on the subtleties involved in bringing an RL project into life, that no amount of research paper browsing could for us.

![The basic grid setup](/assets/posts/drl-multi-1/2 - The Grid Setup - Our Base Grid.png)


The agents see this: `([[0.625, 0.625, 0.875, 0.0]], [-1], False)`

There are four types of actions: South, West, North, East.

Rewards: Each time step -1, except destination which is +20. If two agents step on the same tile, they both get a penalty of -20.

_The observation space_ \
Do you want the agent to see the whole grid, and make the environment fully observable, and let it see everyone else's position? Or do you want to limit the agent's view? Or, simply return the starting and goal coordinates, and drop the matrix representation entirely? 

All of these decisions will have some implications: some of the agent architectures may not be able to find the patterns you want them to, with a certain setup. If you return the whole grid, your agent first will need to figure out a position of a player in a matrix, then compare it to the goal's position, and conclude which direction to take. If your observation space is only a set of coordinates, the solution is a simple add/subtract operation. But figuring out the rules of simple arithmetic may not be as straightforward within an MDP...

_Reward function_ \
Do you give clues frequently enough? Should you add a penalty if the agent goes outside of the grid? Or if they collide? Could scaling the rewards lead to different agent behaviour? Do you want to give some intermediary reward, when the agent is getting closer to the goal?

_Stochasticity of the environment_ \
What happens when you randomize the starting positions? Or the goals? Or both? \
We didn't initially notice the fact that the basic OpenAI gym environments (like Taxi) have a fixed start/end position, and the same agent (without tweaking any of the hyperparameters) may not be able to solve the environment if you randomize them.

_Variations of the environment, and generalization_ \
Does an agent generalize well to variations of the environments? What happens when you make the grid larger? Smaller? If you add a wall? If you randomize the player’s starting position?

![We incrementally increase the difficulty of the task](/assets/posts/drl-multi-1/3 - The Grid Setup - More Advanced Parameters.png)




## Part II: Creating Agents

Deep Reinforcement Learning is not the only game in town, there are other approaches to solve RL problems:



*   Constraint propagation is often used for simple problems like sudoku or tic-tac-toe. For more difficult problems it is used in combination with other methods, eg.: to significantly decrease the state space that an algorithm has to traverse for an optimal solution. In our case, we could eliminate hazardous cells up front, but the number of possibilities is still too large to efficiently solve the environment.
*   Searching the state space with A* or Monte Carlo Methods. An agent is guaranteed to converge on a solution, given enough computational resources, if the starting and ending positions are fixed. 
*   Evolving strategies with Genetic Algorithms, which are a subset could be successfully used to create intelligent behaviour, like [Karl Sims’ demonstration](https://www.youtube.com/watch?v=bBt0imn77Zg) from two decades ago. GA’s are a promising alternative and we might explore solving this environment later on.

But let’s get back to what we’re here for!


### Choosing the Agent

When choosing the right agents it is worth considering the structure of the problem, the key properties, including:



*   Is the environment fully observable or are observations limited to certain aspects of the environment?
*   Are actions surely, deterministically executed or are there probabilistic transitions between states?
*   Is our agent alone or are there multiple agents? Do agents compete, collaborate or are they neutral to each other?
*   Are environment states continuous or discrete? 
*   Are actions that the agent can take continuous or discrete?

Our grid world consists of discrete states and agents can take four different discrete actions. Although the environment seems simple compared to a more elaborate environment (like openAI’s Atari games), the project aims to infuse progressively more difficult properties: We will be adding probabilistic transition functions, multiple agents, bigger grids, and random initialization of start and goal locations. 

Let’s see what our expectations are with the following approaches.


### Baseline Agents: Value-based: DQN, Policy-based: REINFORCE

The success of one of the most famous algorithms, Deep Q-Learning (DQN) kickstarted the current wave of interest in Reinforcement Learning. It substitutes the original Q-Table from the classical Q-Learning algorithm with a Deep Learning Architecture, thereby allowing agents to tackle environments with continuous states. The Deep Neural Network is tuned to approximate the value of selecting and action in a given state. We then select actions according to a heuristic.

![The difference between Q-Learning and Deep Q-Learning](/assets/posts/drl-multi-1/4 - Q-Learning vs Deep Q-Learning.png)


REINFORCE is a more direct approach in finding a right policy. Instead of tuning a DNN to approximate state-action-values the agent outputs either the probabilities over actions, deterministically the best action in a given state or, in environments with continuous actions, the magnitude of each action that you can take. REINFORCE is great with both continuous states and continuous action spaces.

As we are most confident with DQN and REINFORCE out of the “modern” drl algorithms, we decided to run the core experiments on them.

Resources to dig deeper: 

*   See our implementation of [DQN](https://github.com/applied-exploration/multi-drl/tree/main/agents/agent_deepqn) and [REINFORCE](https://github.com/applied-exploration/multi-drl/tree/main/agents/agent_reinforce) on github. The code has been built on top of Udacity’s Deep Reinforcement Learning [repository](https://github.com/udacity/deep-reinforcement-learning).
*   If you want to look at another implementation, Shangtong Zhang DQN and REINFORCE, along with other agents can be found on [github](https://github.com/ShangtongZhang/DeepRL/tree/master/deep_rl/agent).
*   The original [DQN paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) can be found here.


### More Advanced Agent, DDPG

Actor-Critic methods are more advanced algorithms, that combine the value-based approach with a policy-based one. The idea is that by combining the two we get lower variance and lower bias, thereby speeding  up the training process and achieving better results overall.

Deep Deterministic Policy Gradients are interesting Actor-Critic methods that can be thought of as an extension to DQNs. We use an Actor, a Deep Neural Network to determine the next step when calculating the value of a state-action with the Temporal Difference method,  instead of using greedy or epsilon-greedy policy. This means DDPG can work on continuous actions.

We are interested if DDPG outperforms DQN and REINFORCE.

Resources to dig deeper: 



*   See our implementation of DDPG on github. The code has been built on top of Udacity’s Deep Reinforcement Learning [repository](https://github.com/udacity/deep-reinforcement-learning).
*   The original [DDPG paper c](https://arxiv.org/abs/1509.02971)an be found here.


## Part III. Running Experiments

We quickly found ourselves in the endless loop of tweaking a hyperparameter and re-running training, waiting for n minutes, to see the results. This meant a slow iteration cycle: setting up the experiment, running it and finally analysing it.

What we instead wanted: define a list of experiments and run them in one batch during the evening, thereby  not just drastically saving time for us, but also lowering our energy cost.

Implementing a declarative way to define an experiment (an environment - agents pair) was straightforward to implement - if we had done it sooner, it would have saved us some time. Not sure what the best hidden network size is? Running a grid search of the possible hyperparameters is now a possibility!

![Declarative vs Imperative Experiment Process](/assets/posts/drl-multi-1/5 - Experiment Process.png)


This works with any environment that conforms to the OpenAI gym interface, with an environment .reset(), environment.step() method, but it doesn’t necessarily work with agents that require full access to the MDP. The abstract training loop is only there to request an action from an agent, then passes that to the environment, and repeat, until the episode ends.

Code to the abstract training loop can [be found here](https://github.com/applied-exploration/multi-drl/blob/main/experiments/training.py).

For now, this has ruled out PPO, as its efficacy is dependent on being able to create multiple trajectories in parallel, recording the results and learning on them. 


## Part IV. Results

We incrementally made the problem more and more difficult. This way we defined more than a 100 experiments which we clustered according to questions in 4 different batches:

**Question 1:**  Will a DQN Agent and a REINFORCE Agent coverage dependent on whether goals and players are randomized? 

Hypothesis: Both DQN agents and REINFORCE agents are able to generalize. We assume this because the complexity of the problem is low and there is precedent of agents solving similar environments.

![Experiment 1A - Random Initialization DQN](/assets/posts/drl-multi-1/6 - Experiment 1A - Random Initialization.png)

![Experiment 1A - Random Initialization REINFORCE](/assets/posts/drl-multi-1/7 - Experiment 1B - Random Initialization.png)


**Question 2:** Will both Agents still perform well if the environment becomes stochastic?

**Hypothesis:** Both agents are able to solve the environment.

![Adding probabilistic transitions](/assets/posts/drl-multi-1/8 - Experiment 2 - Stochasticity.png)



**Question 3:** Will agents solve the environment from the previous batch if we add multiple agents?

**Hypothesis:** Agents will need more training time to learn the correct behaviour.

![Testing with multiple agents](/assets/posts/drl-multi-1/9 - Experiment 3 - Multiple Agents.png)


**Question 4:** Will agents perform equally well on a larger environment?

**Hypothesis:** Agents will need more training time to learn the correct behaviour.

![Testing on bigger grids](/assets/posts/drl-multi-1/10 - Experiment 4 - Bigger Grid size.png)


## Part V. Future steps

Our current environment is barely sophisticated enough to be able to test the basics, but now we’ll have the foundations to explore more sophisticated questions:



*   Do agents learn to successfully avoid each other? 
*   Will agents sacrifice individual points for the greater good?
*   Do different Deep Reinforcement Learning agents come up with fundamentally different strategies?
*   Would these agents work under partially observable inputs?
*   Could we introduce additional difficulties, such as holes were agents fall off (terminate with a substantial negative reward), like in [OpenAI’s FrozenLake environment](https://gym.openai.com/envs/FrozenLake-v0/)?

We’re looking forward to getting some answers, and to make  a fully fledged, stochastic, multi-agent frozen lake environment… keep tuned!


## Part VI. Our thoughts so far on Reinforcement Learning

The argument goes like this: there is a gap between real-life application of Deep Reinforcement Learning and research. That even if an agent is model-free (no human knowledge was "implanted" in it), interaction with a highly controlled environment is not the same as interactions with the real world. Even if a deep RL agent can beat a [human champion in DOTA 2](https://openai.com/projects/five/), it may need years of playtime to adjust when the developer adds a new unit type, while a human can effectively change their strategy in a fraction of that time.

The slightly more subtle version of this argument is that the more synthetic, easy-to-simulate the environment is, the more possibilities you have to overfit.  We have better and better tools to create synthetic environments - as we get further away from the constraints of the real world, we get more and more optimization possibilities, but we quickly lose the agent’s ability to adapt - change the environment a little (for example, randomize the starting position of the agent) and watch your pre-trained agent trying to do absurd things. Imagine trying to face the non-stationary reality with the same approach!

Reinforcement Learning may be the key to AGI (because of the inherent possibility of meta-learning), but when it comes to training agents on toy environments, it does feel like benchmarking in exactly the same context can mislead us thinking that the agent architectures we tried are fit for purpose. We’ll see in the next part of the project!