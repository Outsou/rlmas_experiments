import asyncio

from creamas.core.simulation import Simulation
from creamas.grid import GridEnvironment

from agents.learning_agent import LearningAgent
from rl.q_learner import QLearner

if __name__ == "__main__":
    env = GridEnvironment.create(('localhost', 5555))
    env.gs = (2, 2)

    for i in range(2):
        LearningAgent(environment=env, learner=QLearner(1, 2), difficulty_preference=1)
        LearningAgent(environment=env, learner=QLearner(1, 2), difficulty_preference=2)


    loop = asyncio.get_event_loop()
    loop.run_until_complete(env.set_agent_neighbors())

    agents = env.get_agents(address=False)


    for agent in agents:
        agent.map_actions_to_neighbors()

    sim = Simulation(env=env)

    for _ in range(1000):
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


