import asyncio
import logging
import operator
import os
import pickle

import aiomas
import numpy as np
import time

from creamas.core.environment import Environment
from creamas.core.simulation import Simulation
from creamas.examples.spiro.spiro_agent_mp import SpiroEnvManager
from creamas.examples.spiro.spiro_agent_mp import SpiroMultiEnvManager
from creamas.util import run

from environments.spiro.spr_environment import SprEnvironment
from utilities.math import gini
from utilities.result_analyzer import analyze
from utilities.serializers import get_spiro_ser_own

if __name__ == "__main__":

    # PARAMETERS

    critic_threshold = 0.10
    veto_threshold = 0.10
    ask_passing = True
    random_choosing = False

    normal_mem = 16
    critic_mem = 144
    search_width_normal = 16
    search_width_critic = 16

    num_of_normal_agents = 14
    num_of_critics = 2

    critic_type = 'agents.spiro.critic_test_agent:CriticTestAgent'
    #critic_type = 'agents.spiro.critic_only_agent:CriticOnlyAgent'

    num_of_artifacts = 200
    num_of_simulations = 10
    num_of_steps = 10

    use_steps = False # Stop when enough steps or when enough artifacts

    # Other stuff

    log_folder = 'critic_logs'

    addr = ('localhost', 5555)
    addrs = [('localhost', 5560),
             ('localhost', 5561),
             ('localhost', 5562),
             ('localhost', 5563)
             ]

    stats = {'comps': [], 'novelty': [], 'gini': [], 'bestie_find_speed': [], 'time': [], 'steps': []}

    env_kwargs = {'extra_serializers': [get_spiro_ser_own], 'codec': aiomas.MsgPack}
    slave_kwargs = [{'extra_serializers': [get_spiro_ser_own], 'codec': aiomas.MsgPack} for _ in range(len(addrs))]
    logger = None

    # Run simulation x times and record stats
    for _ in range(num_of_simulations):

        menv = SprEnvironment(addr,
                              env_cls=Environment,
                              mgr_cls=SpiroMultiEnvManager,
                              logger=logger,
                              **env_kwargs)

        loop = asyncio.get_event_loop()

        ret = run(menv.spawn_slaves(slave_addrs=addrs,
                                    slave_env_cls=Environment,
                                    slave_mgr_cls=SpiroEnvManager,
                                    slave_kwargs=slave_kwargs))

        ret = run(menv.wait_slaves(30))
        ret = run(menv.set_host_managers())
        ret = run(menv.is_ready())

        for _ in range(num_of_normal_agents):
            print(aiomas.run(until=menv.spawn('agents.spiro.critic_test_agent:CriticTestAgent',
                                              desired_novelty=-1,
                                              log_folder=log_folder,
                                              memsize=normal_mem,
                                              critic_threshold=critic_threshold,
                                              veto_threshold=veto_threshold,
                                              ask_passing=ask_passing,
                                              rand=random_choosing,
                                              search_width=search_width_normal)))

        for _ in range(num_of_critics):
            print(aiomas.run(until=menv.spawn(critic_type,
                                              desired_novelty=-1,
                                              log_folder=log_folder,
                                              memsize=critic_mem,
                                              critic_threshold=critic_threshold,
                                              veto_threshold=veto_threshold,
                                              ask_passing=ask_passing,
                                              rand=random_choosing,
                                              search_width=search_width_critic)))

        menv.set_agent_acquaintances()

        sim = Simulation(env=menv, log_folder=log_folder, callback=menv.vote_and_save_info)

        start_time = time.time()

        if use_steps:
            sim.async_steps(num_of_steps)
        else:
            while len(menv.artifacts) < num_of_artifacts:
                 sim.async_step()

        stats['time'].append(time.time()-start_time)
        stats['steps'].append(sim.age)

        menv._consistent = False
        acquaintance_counts = menv.get_acquaintance_counts()
        acquaintance_values = menv.get_acquaintance_values()
        total_comparisons = menv.get_comparison_count()
        mean, _, _ = menv._calc_distances()
        num_of_accepted_artifacts = len(menv.artifacts)
        artifacts = menv.artifacts
        last_changes = menv.get_last_best_acquaintance_changes()
        overcame_own_threshold_counts = menv.get_overcame_own_threshold_counts()
        steps = menv.age
        criticism_stats = menv.get_criticism_stats()

        sim.end()

        # Print who chose who and how many times
        for acquaintance, counts in acquaintance_counts.items():
            most_chosen = max(counts, key=operator.itemgetter(1))
            print('Agent {} chose {} ({} times)'.format(acquaintance, most_chosen[0], most_chosen[1]))
            print(counts)

        print()

        acquaintance_avgs = {}

        # Print how the spiro value other's opinions
        for acquaintance, values in acquaintance_values.items():
            acquaintance_avgs[acquaintance[:22]] = 0
            print(acquaintance)
            print(values)

        print()

        # Print when spiro had learned their best friend
        for name, last_change in last_changes.items():
            print('{} learned best friend at iteration: {}'.format(name, last_change))

        print()

        # Calculate and print for each agent how the other spiro value them on average
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

        # Calculate and print how many times spiro got their artifacts accepted
        creator_counts = {}

        for artifact in artifacts:
            if artifact.creator not in creator_counts:
                creator_counts[artifact.creator] = 1
            else:
                creator_counts[artifact.creator] += 1

        for creator, count in creator_counts.items():
            print('Agent {} created {} accepted artifacts'.format(creator, count))


        print()

        # Print overcoming own thershold counts
        for name, count in overcame_own_threshold_counts.items():
            print('{} overcame itself {}/{} times'.format(name, count, steps))

        print()

        # Print criticism stats
        for name, criticism_stats in criticism_stats.items():
            print('{} rejected {}/{} times'.format(name, criticism_stats[0], criticism_stats[1]))

        print()

        # Calculate gini coefficient of accepted artifacts
        gini_coef = gini(np.array(list(creator_counts.values())).astype(float))
        print('\nGini coefficient for amount of accepted artifacts: ' + str(gini_coef))
        print()

        stats['comps'].append(total_comparisons)
        stats['novelty'].append(mean)
        stats['gini'].append(gini_coef)
        stats['bestie_find_speed'].append(last_changes)

    #Create result directory if needed
    directory = 'results'
    if not os.path.exists(directory):
        os.makedirs(directory)

    file = "{}/stats_mem{}-{}_artifacts{}_passing{}_rand{}.p".format(directory, normal_mem, critic_mem, num_of_artifacts, ask_passing, random_choosing)
    pickle.dump(stats, open(file, "wb"))

    analyze(file)

    # print()
    #
    # for data in stats['bestie_find_speed']:
    #     for name, last_change in data.items():
    #         print('{} learned best friend at iteration: {}'.format(name, last_change))
    #     print()