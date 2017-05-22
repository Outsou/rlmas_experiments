from environments.spr_environment import SprEnvironment
from creamas.core.simulation import Simulation
from agents.spr_agent import SprAgent
import asyncio
import aiomas
from utilities.serializers import get_spiro_ser


if __name__ == "__main__":
    log_folder = 'logs'

    env = SprEnvironment.create(('localhost', 5557), codec=aiomas.MsgPack, extra_serializers=[get_spiro_ser])

    # art = SprAgent(env, ((120, 120))).create(50, 100)
    # scipy.misc.imsave('test.jpg', art)
    # art2 = SprAgent(env, ((120, 120))).create(50, -100)
    # scipy.misc.imsave('test2.jpg', art2)

    rand = True

    SprAgent(env, ((25, 25), (85, 85)), rand=rand, log_folder=log_folder, desired_novelty=10)
    SprAgent(env, ((25, -25), (85, -85)), rand=rand, log_folder=log_folder, desired_novelty=10)
    SprAgent(env, ((-25, 25), (-85, 85)), rand=rand, log_folder=log_folder, desired_novelty=10)

    env.set_agent_acquaintances()

    loop = asyncio.get_event_loop()

    sim = Simulation(env=env, log_folder=log_folder)

    for step in range(100):
        sim.step()

    total_reward = 0
    for agent in env.get_agents(address=False):
        total_reward += agent.total_reward

    print('Total reward: ' + str(total_reward))

    sim.end()

