from creamas.core import Environment, Simulation
from creamas.mp import MultiEnvManager, EnvManager, MultiEnvironment
from creamas.util import run
from utilities.serializers import get_primitive_ser, get_terminal_ser, get_primitive_set_ser, get_func_ser

import asyncio
import aiomas


if __name__ == "__main__":
    # OTHER STUFF

    log_folder = 'dsgdgs_logs'

    addr = ('localhost', 5550)
    addrs = [('localhost', 5560),
             ('localhost', 5561),
             ('localhost', 5562),
             ('localhost', 5563),
             ('localhost', 5564),
             ('localhost', 5565),
             ('localhost', 5566),
             ('localhost', 5567),
             ]

    env_kwargs = {'extra_serializers': [get_primitive_ser, get_terminal_ser, get_primitive_set_ser, get_func_ser], 'codec': aiomas.MsgPack}
    slave_kwargs = [{'extra_serializers': [get_primitive_ser, get_terminal_ser, get_primitive_set_ser, get_func_ser], 'codec': aiomas.MsgPack} for _ in range(len(addrs))]

    logger = None

    # Create the environments
    menv = MultiEnvironment(addr,
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

    from deap import base
    from deap import tools
    from deap import creator
    from deap import gp
    import operator
    import numpy as np

    pset = gp.PrimitiveSet("MAIN", arity=2)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    # pset.addPrimitive(max, 2)
    # pset.addPrimitive(min, 2)
    # pset.addPrimitive(divide, 2)
    pset.addPrimitive(np.sin, 1)
    pset.addPrimitive(np.cos, 1)
    pset.addPrimitive(np.tan, 1)
    #pset.addPrimitive(np.arcsin, 1)
    #pset.addPrimitive(np.arccos, 1)
    #pset.addPrimitive(np.arctan, 1)
    # pset.addPrimitive(exp, 1)
    #pset.addPrimitive(np.sqrt, 1)
    # pset.addPrimitive(log, 1)
    # pset.addEphemeralConstant('rand', lambda: np.random.randint(1, 4))

    pset.renameArguments(ARG0="x")
    pset.renameArguments(ARG1="y")

    for _ in range(1):
        ret = aiomas.run(until=menv.spawn('deappi.pop_agent:PopAgent',
                                          log_folder=log_folder,
                                          pset=pset,
                                          mate_func=gp.cxOnePoint))

        print(ret)

    for _ in range(1):
        ret = aiomas.run(until=menv.spawn('deappi.pop_agent:PopAgent',
                                          log_folder=log_folder,
                                          pset=pset,
                                          mate_func=gp.cxOnePointLeafBiased))

        print(ret)

    sim = Simulation(menv, log_folder=log_folder)
    sim.async_steps(2)
    sim.end()
