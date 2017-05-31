from environments.spr_environment import SprEnvironment
from creamas.core.simulation import Simulation
from creamas.examples.spiro.spiro_agent_mp import SpiroMultiEnvManager
from creamas.examples.spiro.spiro_agent_mp import SpiroEnvManager
from creamas.core.environment import Environment
import logging
import asyncio
import aiomas
from utilities.serializers import get_spiro_ser_own
from utilities.math import gini
from utilities.result_analyzer import analyze

import operator
import numpy as np
import pickle
import os


if __name__ == "__main__":
    log_folder = 'critic_logs'

    addr = ('localhost', 5555)
    addrs = [('localhost', 5560),
             ('localhost', 5561),
             ('localhost', 5562),
             ('localhost', 5563)
             ]

    stats = {'comps': [], 'novelty': [], 'gini': []}

    # Run simulation x times and record stats
    for _ in range(5):

        env = SprEnvironment(addr, env_cls=Environment,
                             mgr_cls=SpiroMultiEnvManager,
                             slave_env_cls=Environment,
                             slave_mgr_cls=SpiroEnvManager,
                             slave_addrs=addrs, log_folder=log_folder,
                             log_level=logging.INFO,
                             extra_ser=[get_spiro_ser_own])

        loop = asyncio.get_event_loop()
        ret = loop.run_until_complete(env.set_host_managers())
        ret = loop.run_until_complete(env.wait_slaves(30))
        ret = loop.run_until_complete(env.is_ready())

        critic_threshold = 0.08
        veto_threshold = 0.08
        ask_passing = True

        normal_mem = 20
        critic_mem = 20

        num_of_artifacts = 400

        for _ in range(4):
            print(aiomas.run(until=env.spawn('agents.critic_test_agent:CriticTestAgent',
                                             desired_novelty=-1,
                                             log_folder=log_folder,
                                             memsize=normal_mem,
                                             critic_threshold=critic_threshold,
                                             veto_threshold=veto_threshold,
                                             ask_passing=ask_passing)))

        for _ in range(1):
            print(aiomas.run(until=env.spawn('agents.critic_test_agent:CriticTestAgent',
                                             desired_novelty=-1,
                                             log_folder=log_folder,
                                             memsize=critic_mem,
                                             critic_threshold=critic_threshold,
                                             veto_threshold=veto_threshold,
                                             ask_passing=ask_passing)))

        env.set_agent_acquaintances()

        sim = Simulation(env=env, log_folder=log_folder, callback=env.vote_and_save_info)

        while len(env.artifacts) <= num_of_artifacts:
             sim.async_step()

        #sim.async_steps(500)

        env._consistent = False
        acquaintance_counts = env.get_acquaintance_counts()
        acquaintance_values = env.get_acquaintance_values()
        total_comparisons = env.get_comparison_count()
        mean, _, _ = env._calc_distances()
        num_of_accepted_artifacts = len(env.artifacts)
        artifacts = env.artifacts
        sim.end()

        # Print who chose who and how many times
        for acquaintance, counts in acquaintance_counts.items():
            most_chosen = max(counts, key=operator.itemgetter(1))
            print('Agent {} chose {} ({} times)'.format(acquaintance, most_chosen[0], most_chosen[1]))
            print(counts)

        print()

        acquaintance_avgs = {}

        # Print how the agents value other's opinions
        for acquaintance, values in acquaintance_values.items():
            acquaintance_avgs[acquaintance[:22]] = 0
            print(acquaintance)
            print(values)

        print()

        # Calculate and print for each agent how the other agents value them on average
        for acquaintance, values in acquaintance_values.items():
            for agent, value in values.items():
                acquaintance_avgs[agent] += value

        for key, value in acquaintance_avgs.items():
            acquaintance_avgs[key] = value/(len(acquaintance_avgs) - 1)
            print('Agent {} avg: {}'.format(key, acquaintance_avgs[key]))

        print()

        print('Comparisons: ' + str(total_comparisons))

        print()

        print('Number of accepted artifacts: ' + str(num_of_accepted_artifacts))

        # Calculate and print how many times agents got their artifacts accepted
        creator_counts = {}

        for artifact in artifacts:
            if artifact.creator not in creator_counts:
                creator_counts[artifact.creator] = 1
            else:
                creator_counts[artifact.creator] += 1

        for creator, count in creator_counts.items():
            print('Agent {} created {} accepted artifacts'.format(creator, count))

        # Calculate gini coefficient of accepted artifacts
        gini_coef = gini(np.array(list(creator_counts.values())).astype(float))
        print('\nGini coefficient for amount of accepted artifacts: ' + str(gini_coef))
        print()

        stats['comps'].append(total_comparisons)
        stats['novelty'].append(mean)
        stats['gini'].append(gini_coef)

    # Create result directory if needed
    directory = 'results'
    if not os.path.exists(directory):
        os.makedirs(directory)

    file = "{}/stats_mem{}-{}_artifacts{}.p".format(directory, normal_mem, critic_mem, num_of_artifacts)
    pickle.dump(stats, open(file, "wb"))

    analyze(file)