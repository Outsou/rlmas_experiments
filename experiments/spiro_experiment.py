from environments.spr_environment import SprEnvironment
from creamas.core.simulation import Simulation
from creamas.mp import MultiEnvManager
from creamas.mp import EnvManager
from creamas.core.environment import Environment
from agents.spr_agent import SprAgent
import logging
import asyncio
import aiomas
from utilities.serializers import get_spiro_ser


if __name__ == "__main__":
    log_folder = 'logs'

    addr = ('localhost', 5558)
    addrs = [('localhost', 5560),
             ('localhost', 5561),
             ('localhost', 5562)
             ]

    #env = SprEnvironment.create(('localhost', 5557), codec=aiomas.MsgPack, extra_serializers=[get_spiro_ser])

    env = SprEnvironment(addr, env_cls=Environment,
                                mgr_cls=MultiEnvManager,
                                slave_env_cls=Environment,
                                slave_mgr_cls=EnvManager,
                                slave_addrs=addrs, log_folder=log_folder,
                                log_level=logging.INFO,
                                extra_ser=[get_spiro_ser])

    loop = asyncio.get_event_loop()
    ret = loop.run_until_complete(env._set_host_managers())
    #ret = loop.run_until_complete(env._wait_slaves(30))
    ret = loop.run_until_complete(env.is_ready())

    print(ret)



    # art = SprAgent(env, ((120, 120))).create(50, 100)
    # scipy.misc.imsave('test.jpg', art)
    # art2 = SprAgent(env, ((120, 120))).create(50, -100)
    # scipy.misc.imsave('test2.jpg', art2)

    # rand = False
    #
    # SprAgent(env, ((25, 25), (85, 85)), rand=rand, log_folder=log_folder, desired_novelty=0.00000000000000001)
    # SprAgent(env, ((25, -25), (85, -85)), rand=rand, log_folder=log_folder, desired_novelty=10)
    # SprAgent(env, ((-25, 25), (-85, 85)), rand=rand, log_folder=log_folder, desired_novelty=10)
    #
    # env.set_agent_acquaintances()

    rand = False

    agent1 = aiomas.run(until=env.spawn('agents.spr_agent:SprAgent',
                                     states=((25, 25), (85, 85)),
                                     rand=rand,
                                     desired_novelty=-1,
                                     log_folder=log_folder))[0]

    print(aiomas.run(until=env.spawn('agents.spr_agent:SprAgent',
                                     states=((25, -25), (85, -85)),
                                     rand=rand,
                                     desired_novelty=-1,
                                     log_folder=log_folder)))
    print(aiomas.run(until=env.spawn('agents.spr_agent:SprAgent',
                                     states=((-25, 25), (-85, 85)),
                                     rand=rand,
                                     desired_novelty=-1,
                                     log_folder=log_folder)))

    env.set_agent_acquaintances()

    sim = Simulation(env=env, log_folder=log_folder)
    sim.async_steps(1000)

    #
    # for step in range(10):
    #     sim.step()
    #
    # total_reward = 0
    # for agent in env.get_agents(address=False):
    #     total_reward += agent.total_reward
    #
    # print('Total reward: ' + str(total_reward))
    #
    sim.end()

