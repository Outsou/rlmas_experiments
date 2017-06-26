from creamas.core import Environment
from creamas.mp import MultiEnvManager, EnvManager
from creamas.util import run
from utilities.serializers import get_maze_ser, get_func_ser
import creamas.nx as cnx
from environments.maze.environments import GatekeeperMazeMultiEnvironment
from mazes.growing_tree import choose_first, choose_random, choose_last
from utilities.result_analyzer import analyze
from utilities.graphs import graph_from_connections
from experiment_simulation import ExperimentSimulation

import asyncio
import aiomas
import networkx as nx
import operator
import os
import shutil
import logging
import time
import pickle
import matplotlib.pyplot as plt


def print_stuff():
    text = ''
    chosen_by_agent_counts = {}
    chosen_counts = {}

    for agent in agents:
        if agent not in gatekeeper_agents:
            continue
        chosen_by_agent_counts[agent] = 0
        chosen_counts[agent] = 0

    # Print who chose who and how many times
    for connection, counts in sorted(connection_counts.items()):
        if 'gk' in connection:
            continue

        most_chosen = max(counts.items(), key=operator.itemgetter(1))
        chosen_by_agent_counts[most_chosen[0]] += 1

        for agent, count in counts.items():
            chosen_counts[agent] += count

        text += 'Agent {} chose {} ({} times)\n'.format(connection, most_chosen[0], most_chosen[1])
        text += str(counts) + '\n'

    text += '\n'

    # Print how many times agent was chosen
    for agent, count in sorted(chosen_by_agent_counts.items()):
        text += '{} was chosen by {} agents, {} times\n'.format(agent, count, chosen_counts[agent])

    text += '\n'

    comps = sum(comparison_counts.values())
    artifacts = sum(artifacts_created.values())
    text += 'Comparisons: {}\n'.format(comps)
    text += 'Artifacts created: {}\n\n'.format(artifacts)

    # Print chosen function counts
    for agent, counts in sorted(choose_func_counts.items()):
        text += '{}:\n'.format(agent)
        for func, count in sorted(counts.items(), key=lambda x: x[0].__name__):
            text += '{}: {}\n'.format(func.__name__, count)

    text += '\n'

    # Print overcame self criticism counts
    for agent, counts in sorted(passed_self_criticism_counts.items()):
        if 'gk' not in agent:
            text += '{} passed self criticism {} and was published {} times\n'.format(agent, counts, published_counts[agent])

    stats['comps'].append(comps)
    stats['artifacts'].append(artifacts)

    print(text)


if __name__ == "__main__":
    # PARAMS

    num_of_low_gatekeepers = 1
    num_of_high_gatekeepers = 1
    num_of_low_creators = 5
    num_of_high_creators = 5

    maze_shape = (32, 32)

    gatekeeper_memsize = 2000
    normal_memsize = 2000
    normal_search_width = 10

    low_hedonic = 20
    high_hedonic = 60

    ask_criticism = True
    ask_random = False

    critic_threshold = 0.9
    veto_threshold = 0.9
    hedonic_std = 10
    choose_funcs = [choose_first, choose_last]

    num_of_steps = 200
    num_of_simulations = 1
    fully_connected = False

    # OTHER STUFF

    log_folder = 'gatekeeper_logs'

    creator_maze_folder = 'creator_mazes'

    if os.path.exists(creator_maze_folder):
        shutil.rmtree(creator_maze_folder)
    os.makedirs(creator_maze_folder)

    addr = ('localhost', 5550)
    addrs = [('localhost', 5560),
             ('localhost', 5561),
             # ('localhost', 5562),
             # ('localhost', 5563),
             # ('localhost', 5564),
             # ('localhost', 5565),
             # ('localhost', 5566),
             # ('localhost', 5567),
             ]

    env_kwargs = {'extra_serializers': [get_maze_ser, get_func_ser], 'codec': aiomas.MsgPack}
    slave_kwargs = [{'extra_serializers': [get_maze_ser, get_func_ser], 'codec': aiomas.MsgPack} for _ in range(len(addrs))]

    logger = None
    stats = {'comps': [], 'time': [], 'steps': [], 'artifacts': []}

    sim_count = 0

    for _ in range(num_of_simulations):
        sim_count += 1

        # Create the environments

        menv = GatekeeperMazeMultiEnvironment(addr,
                                              env_cls=Environment,
                                              mgr_cls=MultiEnvManager,
                                              logger=logger,
                                              **env_kwargs)

        loop = asyncio.get_event_loop()

        ret = run(menv.spawn_slaves(slave_addrs=addrs,
                                    slave_env_cls=Environment,
                                    slave_mgr_cls=EnvManager,
                                    slave_kwargs=slave_kwargs))

        ret = run(menv.wait_slaves(30))
        ret = run(menv.set_host_managers())
        ret = run(menv.is_ready())

        # Create the agents

        gatekeeper_agents = []

        def spawn_gatekeeper(desired_novelty):
            ret = aiomas.run(until=menv.spawn('agents.maze.gatekeeper_agent:GatekeeperAgent',
                                              log_folder=log_folder,
                                              memsize=gatekeeper_memsize,
                                              critic_threshold=critic_threshold,
                                              veto_threshold=veto_threshold,
                                              log_level=logging.DEBUG,
                                              maze_shape=maze_shape,
                                              ask_criticism=ask_criticism,
                                              ask_random=ask_random,
                                              desired_novelty=desired_novelty,
                                              hedonic_std=hedonic_std))
            menv.gatekeepers.append(ret[0])
            gatekeeper_agents.append(run(ret[0].get_addr()))
            print(ret)

        def spawn_creator(desired_novelty):
            ret = aiomas.run(until=menv.spawn('agents.maze.creator_agent:CreatorAgent',
                                              log_folder=log_folder,
                                              memsize=normal_memsize,
                                              critic_threshold=critic_threshold,
                                              veto_threshold=veto_threshold,
                                              log_level=logging.DEBUG,
                                              maze_shape=maze_shape,
                                              search_width=normal_search_width,
                                              ask_criticism=ask_criticism,
                                              ask_random=ask_random,
                                              choose_funcs=choose_funcs,
                                              desired_novelty=desired_novelty,
                                              hedonic_std=hedonic_std))
            print(ret)

        print('Low gatekeepers:')
        for _ in range(num_of_low_gatekeepers):
            spawn_gatekeeper(low_hedonic)

        print('High gatekeepers:')
        for _ in range(num_of_low_gatekeepers):
            spawn_gatekeeper(high_hedonic)

        print('Low creators:')
        for _ in range(num_of_low_creators):
            spawn_creator(low_hedonic)

        print('High creators:')
        for _ in range(num_of_high_creators):
            spawn_creator(high_hedonic)

        agents = sorted(menv.get_agents(addr=True))

        # Create connection graph

        if fully_connected:
            G = nx.complete_graph(num_of_normal_agents + num_of_gatekeeper_agents)
        else:
            G = nx.DiGraph()
            G.add_nodes_from(list(range(len(agents))))

            # Create edges to connect all agents to critics
            gatekeeper_idx = [idx for idx in range(len(agents)) if agents[idx] in gatekeeper_agents]

            edges = []
            for i in G.nodes():
                for j in gatekeeper_idx:
                    if i != j:
                        edges.append((i, j))

            G.add_edges_from(edges)

        cnx.connections_from_graph(menv, G)
        # nx.draw(graph_from_connections(menv, True))
        # plt.show()

        sim = ExperimentSimulation(menv, sim_count, log_folder=log_folder, callback=menv.publish_artifacts)

        start_time = time.time()

        sim.async_steps(num_of_steps)

        stats['time'].append(time.time()-start_time)
        stats['steps'].append(sim.age)

        connection_counts = menv.get_connection_counts()
        comparison_counts = menv.get_comparison_counts()
        artifacts_created = menv.get_artifacts_created()
        choose_func_counts = menv.get_choose_func_counts()
        passed_self_criticism_counts = menv.get_passed_self_criticism_counts()
        published_counts = menv.get_published_counts()

        #menv.save_creator_artifacts(creator_maze_folder)

        sim.end()

        print_stuff()

    directory = 'gatekeeper_results'
    if not os.path.exists(directory):
        os.makedirs(directory)

    file = "{}/stats_mem{}-{}_passing{}_rand{}.p".format(directory, normal_memsize, gatekeeper_memsize, ask_criticism, ask_random)
    pickle.dump(stats, open(file, "wb"))

    analyze(file)
