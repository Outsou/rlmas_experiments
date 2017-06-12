import asyncio

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
from creamas.core.simulation import Simulation
from creamas.grid import GridEnvironment
from scipy.stats import norm

from agents.spiro.learning_agent import LearningAgent
from rl.q_learner import QLearner

if __name__ == "__main__":
    env = GridEnvironment.create(('localhost', 5555))
    env.gs = (2, 2)

    mu1 = 1.5
    mu2 = 2
    sigma1 = 0.5
    sigma2 = 0.5

    for i in range(2):
        LearningAgent(environment=env, learner=QLearner(1, 2), difficulty_preference=mu1, standard_deviation=sigma1)
        LearningAgent(environment=env, learner=QLearner(1, 2), difficulty_preference=mu2, standard_deviation=sigma2)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(env.set_agent_neighbors())

    agents = env.get_agents(address=False)


    for agent in agents:
        agent.map_actions_to_neighbors()

    sim = Simulation(env=env)

    for _ in range(100):
        sim.step()

    preferences = {}
    for agent in agents:
        preferences[agent.addr] = agent.difficulty_preference

    print()

    for agent in agents:
        print(agent.addr)

        if preferences[agent.actions[0]] == agent.difficulty_preference:
            print("Should prefer first")
        else:
            print("Should prefer second")

        #print(agent.actions)
        print(agent.learner.q_table)
        print()

    sim.end()

    cutoff_prob = 0.001
    left1 = norm.ppf(cutoff_prob, loc=mu1, scale=sigma1)
    left2 = norm.ppf(cutoff_prob, loc=mu2, scale=sigma2)
    left = min(left1, left2)
    right1 = norm.ppf(1-cutoff_prob, loc=mu1, scale=sigma1)
    right2 = norm.ppf(1-cutoff_prob, loc=mu2, scale=sigma2)
    right = max(right1, right2)
    x = np.linspace(left, right, 100)

    plt.plot(x, mlab.normpdf(x, mu1, sigma1))
    plt.plot(x, mlab.normpdf(x, mu2, sigma2))
    plt.show()

