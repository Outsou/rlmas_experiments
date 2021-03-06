import asyncio
import logging

import aiomas
from creamas.core.environment import Environment
from creamas.core.simulation import Simulation
from creamas.mp import EnvManager
from creamas.mp import MultiEnvManager

from environments.spiro.spr_environment import SprEnvironment
from utilities.serializers import get_spiro_ser

if __name__ == "__main__":
    log_folder = 'logs'

    addr = ('localhost', 5555)
    addrs = [('localhost', 5560),
             ('localhost', 5561),
             ('localhost', 5562)
             ]

    env = SprEnvironment(addr, env_cls=Environment,
                                mgr_cls=MultiEnvManager,
                                slave_env_cls=Environment,
                                slave_mgr_cls=EnvManager,
                                slave_addrs=addrs, log_folder=log_folder,
                                log_level=logging.INFO,
                                extra_ser=[get_spiro_ser])

    loop = asyncio.get_event_loop()
    ret = loop.run_until_complete(env.set_host_managers())
    ret = loop.run_until_complete(env.wait_slaves(30))
    ret = loop.run_until_complete(env.is_ready())

    # art = SprAgent(env, ((120, 120))).create(50, 100)
    # scipy.misc.imsave('test.jpg', art)
    # art2 = SprAgent(env, ((120, 120))).create(50, -100)
    # scipy.misc.imsave('test2.jpg', art2)

    rand = False

    print(aiomas.run(until=env.spawn('spiro.spr_agent:SprAgent',
                                     states=((25, 25), (85, 85)),
                                     rand=rand,
                                     desired_novelty=0.001,
                                     log_folder=log_folder)))

    print(aiomas.run(until=env.spawn('spiro.spr_agent:SprAgent',
                                     states=((25, -25), (85, -85)),
                                     rand=rand,
                                     desired_novelty=10,
                                     log_folder=log_folder)))
    print(aiomas.run(until=env.spawn('spiro.spr_agent:SprAgent',
                                     states=((-25, 25), (-85, 85)),
                                     rand=rand,
                                     desired_novelty=10,
                                     log_folder=log_folder)))

    env.set_agent_acquaintances()

    sim = Simulation(env=env, log_folder=log_folder, callback=env.log_situation)
    sim.async_steps(100)
    total_reward = env.get_total_reward()
    sim.end()

    print('Total reward: ' + str(total_reward))
