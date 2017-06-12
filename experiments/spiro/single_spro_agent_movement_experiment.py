import numpy as np
from creamas.core.environment import Environment
from creamas.core.simulation import Simulation

from agents.spiro.moving_spr_agent import QMovingSprAgent, BasicMovingSprAgent

if __name__ == "__main__":
    log_folder = 'movement_logs'

    env = Environment.create(('localhost', 5555))

    step_size = 10
    search_width = 10
    start_location = np.array([0, 0])
    mem_size = 100

    q_agent = QMovingSprAgent(environment=env,
                              desired_novelty=-1,
                              step_size=step_size,
                              search_width=search_width,
                              start_location = start_location,
                              initial_values=10,
                              log_folder=log_folder,
                              memsize=mem_size,
                              discount_factor=0.85,
                              learning_factor=0.8)
    basic_agent = BasicMovingSprAgent(environment=env, desired_novelty=-1, search_width=search_width, start_location=start_location, memsize=mem_size)
    random_agent = BasicMovingSprAgent(environment=env, desired_novelty=-1, search_width=search_width, start_location=start_location, jump='random', memsize=mem_size)

    sim = Simulation(env=env)
    sim.steps(100)
    sim.end()

    print("Total rewards")
    print("Q-learned movement: " + str(q_agent.total_reward))
    print("Ad hoc movement: " + str(basic_agent.total_reward))
    print("Random movement: " + str(random_agent.total_reward))
