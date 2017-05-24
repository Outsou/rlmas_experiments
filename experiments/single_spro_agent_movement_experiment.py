from creamas.core.environment import Environment
from creamas.core.simulation import Simulation
from agents.moving_spr_agent import MovingSprAgent


if __name__ == "__main__":
    log_folder = 'movement_logs'

    env = Environment.create(('localhost', 5555))
    agent = MovingSprAgent(environment=env, desired_novelty=-1, step_size=10, search_width=10, log_folder=log_folder)



    sim = Simulation(env=env)
    sim.steps(200)
    sim.end()