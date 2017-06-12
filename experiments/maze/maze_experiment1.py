from creamas.core import Simulation, Environment
from creamas.mp import MultiEnvironment, MultiEnvManager, EnvManager
from creamas.util import run
from utilities.serializers import get_maze_ser
from creamas.nx import connections_from_graph

import asyncio
import aiomas
import networkx as nx


if __name__ == "__main__":
    # PARAMS

    num_of_agents = 5

    num_of_steps = 10

    # OTHER STUFF

    log_folder = 'experiment1_logs'

    addr = ('localhost', 5550)
    addrs = [('localhost', 5560),
             ('localhost', 5561),
             ('localhost', 5562),
             ('localhost', 5563)
             ]

    log_folder = None
    env_kwargs = {'extra_serializers': [get_maze_ser], 'codec': aiomas.MsgPack}

    menv = MultiEnvironment(addr,
                            env_cls=Environment,
                            mgr_cls=MultiEnvManager,
                            logger=None,
                            **env_kwargs)

    loop = asyncio.get_event_loop()
    slave_kwargs = [{'extra_serializers': [get_maze_ser], 'codec': aiomas.MsgPack} for _ in range(len(addrs))]

    ret = run(menv.spawn_slaves(slave_addrs=addrs,
                                slave_env_cls=Environment,
                                slave_mgr_cls=EnvManager,
                                slave_kwargs=slave_kwargs))

    ret = run(menv.wait_slaves(30))
    ret = run(menv.set_host_managers())
    ret = run(menv.is_ready())

    print(ret)
    for _ in range(num_of_agents):
        ret = aiomas.run(until=menv.spawn('agents.maze.maze_agent:MazeAgent',
                                          log_folder=log_folder))
        print(ret)

    G = nx.complete_graph(num_of_agents)
    connections_from_graph(menv, G)

    sim = Simulation(menv, log_folder=log_folder)
    sim.async_steps(num_of_steps)
    sim.end()
