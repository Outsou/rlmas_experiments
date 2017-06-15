import asyncio
import operator
import os
import pickle

import aiomas
import numpy as np
from creamas.core.environment import Environment
from creamas.core.simulation import Simulation
from creamas.examples.spiro.spiro_agent_mp import SpiroEnvManager
from creamas.examples.spiro.spiro_agent_mp import SpiroMultiEnvManager
from creamas.util import run

from environments.spiro.spr_environment_equal import SprEnvironmentEqual
from utilities.math import gini
from utilities.result_analyzer import analyze
from utilities.serializers import get_spiro_ser_own


def print_and_save_stuff():
    text = ''

    chosen_counts = {}

    for agent in agents:
        chosen_counts[agent] = 0

    # Print who chose who and how many times
    for acquaintance, counts in acquaintance_counts.items():
        most_chosen = max(counts, key=operator.itemgetter(1))

        chosen_counts[most_chosen[0]] += 1

        text += 'Agent {} chose {} ({} times)\n'.format(acquaintance, most_chosen[0], most_chosen[1])
        text += str(counts) + '\n'

    text += '\n'

    # Print how many times the spiro were chosen
    for agent, count in chosen_counts.items():
        text += '{} was chosen {} times\n'.format(agent, count)

    text += '\n'

    acquaintance_avgs = {}

    # Print how the spiro value other's opinions
    for acquaintance, values in acquaintance_values.items():
        acquaintance_avgs[acquaintance[:22]] = 0
        text += str(acquaintance) + '\n'
        text += str(values) + '\n'

    text += '\n'

    # Print when spiro had learned their best friend
    for name, last_change in last_changes.items():
        text += '{} learned best friend at iteration: {}\n'.format(name, last_change)

    text += '\n'

    # Calculate and print for each agent how the other spiro value them on average
    for acquaintance, values in acquaintance_values.items():
        for agent, value in values.items():
            acquaintance_avgs[agent] += value

    for key, value in acquaintance_avgs.items():
        acquaintance_avgs[key] = value/(len(acquaintance_avgs) - 1)
        text += 'Agent {} avg: {}\n'.format(key, acquaintance_avgs[key])

    text += '\n'
    text += 'Comparisons: ' + str(total_comparisons) + '\n'
    text += 'Mean novelty: ' + str(mean) + '\n'
    text += '\n'

    text += 'Number of accepted artifacts: ' + str(num_of_accepted_artifacts) + '\n'

    # Calculate and print how many times spiro got their artifacts accepted
    creator_counts = {}

    for artifact in artifacts:
        if artifact.creator not in creator_counts:
            creator_counts[artifact.creator] = 1
        else:
            creator_counts[artifact.creator] += 1

    for creator, count in creator_counts.items():
        text += 'Agent {} created {} accepted artifacts\n'.format(creator, count)

    text += '\n'

    # Print overcoming own thershold counts
    for name, count in overcame_own_threshold_counts.items():
        text += '{} overcame itself {}/{} times\n'.format(name, count, steps)

    text += '\n'

    # Print criticism stats
    for name, c_stats in criticism_stats.items():
        text += '{} rejected {}/{} times\n'.format(name, c_stats[0], c_stats[1])

    text += '\n'

    # Print total rewards
    total_reward = 0
    for agent in agents:
        reward = 0
        if agent in criticism_stats:
            reward += criticism_stats[agent][0]
        if agent in creator_counts:
            reward += creator_counts[agent]
        total_reward += reward
        text += '{} total reward: {}\n'.format(agent, reward)

    text+= 'total total reward: ' + str(total_reward)
    text += '\n\n'

    # Print memory state times
    for name, thing in memory_state_times.items():
        total_memory_state_times[name] = [sum(x) for x in zip(total_memory_state_times[name], thing)]
        text += name + '\n'
        text += '{}, total: {}\n'.format(thing, tuple(total_memory_state_times[name]))

    text += '\n'

    # Calculate gini coefficient of accepted artifacts
    gini_coef = gini(np.array(list(creator_counts.values())).astype(float))
    text += '\nGini coefficient for amount of accepted artifacts: ' + str(gini_coef)
    text += '\n'

    print(text)

    with open(save_file, "a") as file:
        file.write('\n\n***** {} *****\n\n'.format(round))
        file.write(text + '\n\n\n\n')

    stats['comps'].append(total_comparisons)
    stats['novelty'].append(mean)
    stats['gini'].append(gini_coef)
    stats['bestie_find_speed'].append(last_changes)


if __name__ == "__main__":
    directory = 'results_q'
    save_file = "{}/print.txt".format(directory)

    # remove old save file
    if os.path.exists(save_file):
        os.remove(save_file)

    #Create result directory if needed
    if not os.path.exists(directory):
        os.makedirs(directory)

    # ALL PARAMETERS

    critic_threshold = 0.08
    veto_threshold = 0.08
    ask_passing = True
    random_choosing = False
    memory_states = (10, 60, 120)
    initial_state = 1
    invent_n = 120

    discount_factor = 0.99
    learning_factor = 0.9

    num_of_agents = 5
    num_of_artifacts = 10
    num_of_simulations = 5
    num_of_steps = 10

    use_steps = False # Stop when enough steps or when enough artifacts

    # Other stuff

    log_folder = 'q_logs'
    logger = None

    addr = ('localhost', 5555)
    addrs = [('localhost', 5560),
             ('localhost', 5561),
             ('localhost', 5562),
             ('localhost', 5563)
             ]

    env_kwargs = {'extra_serializers': [get_spiro_ser_own], 'codec': aiomas.MsgPack}
    slave_kwargs = [{'extra_serializers': [get_spiro_ser_own], 'codec': aiomas.MsgPack} for _ in range(len(addrs))]

    stats = {'comps': [], 'novelty': [], 'gini': [], 'bestie_find_speed': []}

    # Run simulation x times and record stats
    for _ in range(num_of_simulations):

        env = SprEnvironmentEqual(addr,
                                  env_cls=Environment,
                                  mgr_cls=SpiroMultiEnvManager,
                                  logger=logger,
                                  **env_kwargs)

        loop = asyncio.get_event_loop()

        ret = run(env.spawn_slaves(slave_addrs=addrs,
                                    slave_env_cls=Environment,
                                    slave_mgr_cls=SpiroEnvManager,
                                    slave_kwargs=slave_kwargs))

        ret = loop.run_until_complete(env.set_host_managers())
        ret = loop.run_until_complete(env.wait_slaves(30, check_ready=True))
        ret = loop.run_until_complete(env.is_ready())

        for _ in range(num_of_agents):
            print(aiomas.run(until=env.spawn('agents.spiro.critic_q_agent:CriticQAgent',
                                             desired_novelty=-1,
                                             log_folder=log_folder,
                                             critic_threshold=critic_threshold,
                                             veto_threshold=veto_threshold,
                                             ask_passing=ask_passing,
                                             rand=random_choosing,
                                             memory_states=memory_states,
                                             initial_state=initial_state,
                                             invent_n=invent_n,
                                             discount_factor=discount_factor,
                                             learning_factor=learning_factor)))

        env.set_agent_acquaintances()

        sim = Simulation(env=env, log_folder=log_folder, callback=env.vote_and_save_info)

        agents = env.get_agents(addr=True)
        total_memory_state_times = {}
        round = 0

        for agent in agents:
            total_memory_state_times[agent] = [0] * len(memory_states)

        if use_steps:
            sim.async_steps(num_of_steps)
        else:
            while True:
                while len(env.artifacts) < num_of_artifacts:
                    sim.async_step()

                # gather info
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
                round += 1

                print_and_save_stuff()

                # reset stuff
                env._artifacts = []
                env.reset_agents()

        sim.end()



    file = "{}/stats_artifacts{}.p".format(directory, num_of_artifacts)
    pickle.dump(stats, open(file, "wb"))

    analyze(file)

    # print()
    #
    # for data in stats['bestie_find_speed']:
    #     for name, last_change in data.items():
    #         print('{} learned best friend at iteration: {}'.format(name, last_change))
    #     print()

