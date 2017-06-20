from creamas.core import Simulation, Environment
from creamas.mp import MultiEnvManager
from creamas.util import run
from utilities.serializers import get_maze_ser, get_func_ser
import creamas.nx as cnx
from environments.maze.environments import MazeMultiEnvironment
from creamas.vote import VoteEnvironment
from creamas.vote import VoteManager
from creamas.logging import ObjectLogger
from mazes.growing_tree import choose_first, choose_random, choose_last
from utilities.result_analyzer import analyze
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
        chosen_by_agent_counts[agent] = 0
        chosen_counts[agent] = 0

    # Print who chose who and how many times
    for connection, counts in sorted(connection_counts.items()):
        most_chosen = max(counts.items(), key=operator.itemgetter(1))
        chosen_by_agent_counts[most_chosen[0]] += 1

        for agent, count in counts.items():
            chosen_counts[agent] += count

        text += 'Agent {} chose {} ({} times)\n'.format(connection, most_chosen[0], most_chosen[1])
        text += str(counts)  + '\n'

    text += '\n'

    # Print how many times agent was chosen
    for agent, count in sorted(chosen_by_agent_counts.items()):
        text += '{} was chosen by {} agents, {} times\n'.format(agent, count, chosen_counts[agent])

    text += '\n'

    comps = sum(comparison_counts.values())
    artifacts = sum(artifacts_created.values())
    text += 'Comparisons: {}\n'.format(comps)
    text += 'Artifacts created: {}\n'.format(artifacts)
    text += 'Domain novelty: {}\n'.format(mean)

    stats['comps'].append(comps)
    stats['artifacts'].append(artifacts)

    print(text)


if __name__ == "__main__":
    # PARAMS

    num_of_critic_agents = 10
    num_of_normal_agents = 90

    maze_shape = (32, 32)

    critic_memsize = 144
    normal_memsize = 16
    critic_search_width = 2
    normal_search_width = 16

    ask_criticism = True
    ask_random = False

    critic_threshold = 0.015
    veto_threshold = 0.015
    cell_choosing_func = choose_random

    num_of_artifacts = 200
    num_of_simulations = 5
    #num_of_steps = 5

    # OTHER STUFF

    log_folder = 'experiment1_logs'
    domain_save_folder = 'experiment1_mazes'

    if os.path.exists(domain_save_folder):
        shutil.rmtree(domain_save_folder)
    os.makedirs(domain_save_folder)

    addr = ('localhost', 5551)
    addrs = [('localhost', 5560),
             ('localhost', 5561),
             ('localhost', 5562),
             ('localhost', 5563),
             ('localhost', 5564),
             ('localhost', 5565),
             ('localhost', 5566),
             ('localhost', 5567),
             ]

    env_kwargs = {'extra_serializers': [get_maze_ser, get_func_ser], 'codec': aiomas.MsgPack}
    slave_kwargs = [{'extra_serializers': [get_maze_ser, get_func_ser], 'codec': aiomas.MsgPack} for _ in range(len(addrs))]

    logger = None
    stats = {'comps': [], 'novelty': [], 'time': [], 'steps': [], 'artifacts': []}

    sim_count = 0

    for _ in range(num_of_simulations):
        sim_count += 1

        menv = MazeMultiEnvironment(addr,
                                    env_cls=Environment,
                                    mgr_cls=MultiEnvManager,
                                    logger=logger,
                                    **env_kwargs)

        loop = asyncio.get_event_loop()

        ret = run(menv.spawn_slaves(slave_addrs=addrs,
                                    slave_env_cls=VoteEnvironment,
                                    slave_mgr_cls=VoteManager,
                                    slave_kwargs=slave_kwargs))

        ret = run(menv.wait_slaves(30))
        ret = run(menv.set_host_managers())
        ret = run(menv.is_ready())

        critic_agents = []

        print('Critics:')
        for _ in range(num_of_critic_agents):
            ret = aiomas.run(until=menv.spawn('agents.maze.maze_agent:MazeAgent',
                                              log_folder=log_folder,
                                              memsize=critic_memsize,
                                              critic_threshold=critic_threshold,
                                              veto_threshold=veto_threshold,
                                              log_level=logging.DEBUG,
                                              choose_func=cell_choosing_func,
                                              maze_shape=maze_shape,
                                              search_width=critic_search_width,
                                              ask_criticism=ask_criticism,
                                              ask_random=ask_random))
            print(ret)
            critic_agents.append(run(ret[0].get_name(), loop))

        print('Normies:')
        for _ in range(num_of_normal_agents):
            ret = aiomas.run(until=menv.spawn('agents.maze.maze_agent:MazeAgent',
                                              log_folder=log_folder,
                                              memsize=normal_memsize,
                                              critic_threshold=critic_threshold,
                                              veto_threshold=veto_threshold,
                                              log_level=logging.DEBUG,
                                              choose_func=cell_choosing_func,
                                              maze_shape=maze_shape,
                                              search_width=normal_search_width,
                                              ask_criticism=ask_criticism,
                                              ask_random=ask_random))
            print(ret)

        agents = sorted(menv.get_agents(addr=True))

        # Create connection graph
        #G = nx.complete_graph(num_of_normal_agents + num_of_critic_agents)
        G = nx.DiGraph()
        G.add_nodes_from(list(range(len(agents))))

        # Create edges to connect all agents to critics
        critic_idx = [idx for idx in range(len(agents)) if agents[idx] in critic_agents]

        edges = []
        for i in G.nodes():
            for j in critic_idx:
                if i != j:
                    edges.append((i, j))

        G.add_edges_from(edges)
        cnx.connections_from_graph(menv, G)

        sim = ExperimentSimulation(menv, sim_count, log_folder=log_folder, callback=menv.vote_and_save_info)

        start_time = time.time()

        while len(menv.artifacts) < num_of_artifacts:
            sim.async_step()
        #sim.async_steps(num_of_steps)

        stats['time'].append(time.time()-start_time)
        stats['steps'].append(sim.age)

        connection_counts = menv.get_connection_counts()
        comparison_counts = menv.get_comparison_counts()
        artifacts_created = menv.get_artifacts_created()
        mean, _, _ = menv._calc_distances()

        stats['novelty'].append(mean)

        #menv.save_domain_artifacts(domain_save_folder)

        sim.end()

        print_stuff()

    directory = 'exp1_results'
    if not os.path.exists(directory):
        os.makedirs(directory)

    file = "{}/stats_mem{}-{}_artifacts{}_passing{}_rand{}.p".format(directory, normal_memsize, critic_memsize, num_of_artifacts, ask_criticism, ask_random)
    pickle.dump(stats, open(file, "wb"))

    analyze(file)
