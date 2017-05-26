from environments.spr_environment import SprEnvironment
from creamas.core.simulation import Simulation
from creamas.mp import MultiEnvManager
from creamas.mp import EnvManager
from creamas.core.environment import Environment
import logging
import asyncio
import aiomas
from utilities.serializers import get_spriro_ser_mp


if __name__ == "__main__":
    log_folder = 'critic_logs'

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
                                extra_ser=[get_spriro_ser_mp])

    loop = asyncio.get_event_loop()
    ret = loop.run_until_complete(env._set_host_managers())
    #ret = loop.run_until_complete(env._wait_slaves(30))
    ret = loop.run_until_complete(env.is_ready())

    rand = False

    print(aiomas.run(until=env.spawn('agents.critic_test_agent:CriticTestAgent',
                                     desired_novelty=-1,
                                     log_folder=log_folder,
                                     memsize=10)))

    print(aiomas.run(until=env.spawn('agents.critic_test_agent:CriticTestAgent',
                                     desired_novelty=-1,
                                     log_folder=log_folder,
                                     memsize=10)))
    print(aiomas.run(until=env.spawn('agents.critic_test_agent:CriticTestAgent',
                                     desired_novelty=-1,
                                     log_folder=log_folder,
                                     memsize=1000)))

    env.set_agent_acquaintances()

    sim = Simulation(env=env, log_folder=log_folder)
    sim.async_steps(3)
    #total_reward = env.get_total_reward()
    sim.end()

    #print('Total reward: ' + str(total_reward))