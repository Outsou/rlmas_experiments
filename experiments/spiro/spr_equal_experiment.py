import asyncio
import logging
import operator
import os
import pickle

import aiomas
import numpy as np
from creamas.core.environment import Environment
from creamas.core.simulation import Simulation
from creamas.examples.spiro.spiro_agent_mp import SpiroEnvManager
from creamas.examples.spiro.spiro_agent_mp import SpiroMultiEnvManager

from environments.spiro.spr_environment_equal import SprEnvironmentEqual
from utilities.math import gini
from utilities.result_analyzer import analyze
from utilities.serializers import get_spiro_ser_own

if __name__ == "__main__":

    # ALL PARAMETERS

    critic_threshold = 0.08
    veto_threshold = 0.08
    ask_passing = True
    random_choosing = False
    memory_states = (5, 80)
    initial_state = 0
    invent_n = 80

    num_of_agents = 5
    num_of_artifacts = 400
    num_of_simulations = 5
    num_of_steps = 10

    use_steps = False # Stop when enough steps or when enough artifacts

    # Other stuff

    log_folder = 'equal_logs'

    addr = ('localhost', 5555)
    addrs = [('localhost', 5560),
             ('localhost', 5561),
             ('localhost', 5562),
             ('localhost', 5563)
             ]

    stats = {'comps': [], 'novelty': [], 'gini': [], 'bestie_find_speed': []}

    # Run simulation x times and record stats
    for _ in range(num_of_simulations):

        env = SprEnvironmentEqual(addr, env_cls=Environment,
                             mgr_cls=SpiroMultiEnvManager,
                             slave_env_cls=Environment,
                             slave_mgr_cls=SpiroEnvManager,
                             slave_addrs=addrs, log_folder=log_folder,
                             log_level=logging.INFO,
                             extra_ser=[get_spiro_ser_own])

        loop = asyncio.get_event_loop()
        ret = loop.run_until_complete(env.set_host_managers())
        ret = loop.run_until_complete(env.wait_slaves(30, check_ready=True))
        ret = loop.run_until_complete(env.is_ready())

        for _ in range(num_of_agents):
            print(aiomas.run(until=env.spawn('spiro.critic_equal_agent:CriticEqualAgent',
                                             desired_novelty=-1,
                                             log_folder=log_folder,
                                             critic_threshold=critic_threshold,
                                             veto_threshold=veto_threshold,
                                             ask_passing=ask_passing,
                                             rand=random_choosing,
                                             memory_states=memory_states,
                                             initial_state=initial_state,
                                             invent_n=invent_n)))

        env.set_agent_acquaintances()

        sim = Simulation(env=env, log_folder=log_folder, callback=env.vote_and_save_info)

        if use_steps:
            sim.async_steps(num_of_steps)
        else:
            while len(env.artifacts) < num_of_artifacts:
                 sim.async_step()

        env._consistent = False
        acquaintance_counts = env.get_acquaintance_counts()
        acquaintance_values = env.get_acquaintance_values()
        total_comparisons = env.get_comparison_count()
        mean, _, _ = env._calc_distances()
        num_of_accepted_artifacts = len(env.artifacts)
        artifacts = env.artifacts
        last_changes = env.get_last_best_acquaintance_changes()
        overcame_own_threshold_counts = env.get_overcame_own_threshold_counts()
        steps = env.age
        criticism_stats = env.get_criticism_stats()
        memory_state_times = env.get_memory_state_times()
        agents = env.get_agents(addr=True)

        sim.end()

        chosen_counts = {}

        for agent in agents:
            chosen_counts[agent] = 0

        # Print who chose who and how many times
        for acquaintance, counts in acquaintance_counts.items():
            most_chosen = max(counts, key=operator.itemgetter(1))

            chosen_counts[most_chosen[0]] += 1

            print('Agent {} chose {} ({} times)'.format(acquaintance, most_chosen[0], most_chosen[1]))
            print(counts)

        print()

        # Print how many times the spiro were chosen
        for agent, count in chosen_counts.items():
            print('{} was chosen {} times'.format(agent, count))

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
        for name, c_stats in criticism_stats.items():
            print('{} rejected {}/{} times'.format(name, c_stats[0], c_stats[1]))

        print()

        # Print total rewards
        for agent in agents:
            reward = 0
            if agent in criticism_stats:
                reward += criticism_stats[agent][0]
            if agent in creator_counts:
                reward += creator_counts[agent]
            print('{} total reward: {}'.format(agent, reward))

        print()

        # Print memory state times
        for name, thing in memory_state_times.items():
            print(name)
            print(thing)

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
    directory = 'results_equal'
    if not os.path.exists(directory):
        os.makedirs(directory)

    file = "{}/stats_artifacts{}.p".format(directory, num_of_artifacts)
    pickle.dump(stats, open(file, "wb"))

    analyze(file)

    # print()
    #
    # for data in stats['bestie_find_speed']:
    #     for name, last_change in data.items():
    #         print('{} learned best friend at iteration: {}'.format(name, last_change))
    #     print()