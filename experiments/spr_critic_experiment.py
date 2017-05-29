from environments.spr_environment import SprEnvironment
from creamas.core.simulation import Simulation
from creamas.examples.spiro.spiro_agent_mp import SpiroMultiEnvManager
from creamas.examples.spiro.spiro_agent_mp import SpiroEnvManager
from creamas.core.environment import Environment
import logging
import asyncio
import aiomas
from utilities.serializers import get_spriro_ser_mp

import operator


if __name__ == "__main__":
    log_folder = 'critic_logs'

    addr = ('localhost', 5555)
    addrs = [('localhost', 5560),
             ('localhost', 5561),
             ('localhost', 5562),
             ('localhost', 5563)
             ]

    env = SprEnvironment(addr, env_cls=Environment,
                         mgr_cls=SpiroMultiEnvManager,
                         slave_env_cls=Environment,
                         slave_mgr_cls=SpiroEnvManager,
                         slave_addrs=addrs, log_folder=log_folder,
                         log_level=logging.INFO,
                         extra_ser=[get_spriro_ser_mp])

    loop = asyncio.get_event_loop()
    ret = loop.run_until_complete(env._set_host_managers())
    #ret = loop.run_until_complete(env._wait_slaves(30))
    ret = loop.run_until_complete(env.is_ready())

    rand = False

    critic_threshold = 0.06

    for _ in range(4):
        print(aiomas.run(until=env.spawn('agents.critic_test_agent:CriticTestAgent',
                                         desired_novelty=-1,
                                         log_folder=log_folder,
                                         memsize=5,
                                         critic_threshold=critic_threshold)))

    for _ in range(1):
        print(aiomas.run(until=env.spawn('agents.critic_test_agent:CriticTestAgent',
                                         desired_novelty=-1,
                                         log_folder=log_folder,
                                         memsize=30,
                                         critic_threshold=critic_threshold)))

    env.set_agent_acquaintances()

    sim = Simulation(env=env, log_folder=log_folder, callback=env.vote_and_save_info)
    sim.async_steps(5)
    acquaintance_counts = env.get_acquaintance_counts()
    acquaintance_values = env.get_acquaintance_values()
    total_comparisons = env.get_comparison_count()
    sim.end()

    for acquaintance, counts in acquaintance_counts.items():
        most_chosen = max(counts, key=operator.itemgetter(1))
        print('Agent {} chose {} ({} times)'.format(acquaintance, most_chosen[0], most_chosen[1]))
        print(counts)

    print()

    acquaintance_avgs = {}

    for acquaintance, values in acquaintance_values.items():
        acquaintance_avgs[acquaintance[:22]] = 0
        print(acquaintance)
        print(values)

    print()

    for acquaintance, values in acquaintance_values.items():
        for agent, value in values.items():
            acquaintance_avgs[agent] += value

    for key, value in acquaintance_avgs.items():
        acquaintance_avgs[key] = value/(len(acquaintance_avgs) - 1)
        print('Agent {} avg: {}'.format(key, acquaintance_avgs[key]))

    print()

    print('Comparisons: ' + str(total_comparisons))
