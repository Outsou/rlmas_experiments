from creamas.core import Environment, Simulation
from creamas.mp import MultiEnvManager, EnvManager
from environments.generic import StatEnvironment
from creamas.util import run
from creamas.rules.rule import RuleLeaf
from creamas.mappers import LinearMapper
from utilities.serializers import get_primitive_ser, get_terminal_ser, \
    get_primitive_set_typed_ser, get_func_ser, get_toolbox_ser, get_type_ser, get_rule_leaf_ser
from utilities.math import box_count
from artifacts.genetic_image_artifact import GeneticImageArtifact
import creamas.features as ft

from deap import base
from deap import tools
from deap import gp
import operator
import numpy as np

import asyncio
import aiomas
import os
import shutil


def combine(num1, num2, num3):
    return [num1, num2, num3]

def log(a):
    if a <= 0:
        a = 0.000001
    return np.log(a)

def exp(a):
    if a > 100:
        a = 100
    elif a < -100:
        a = -100
    return np.exp(a)

def create_pset():
    pset = gp.PrimitiveSetTyped("main", [float, float], list)
    pset.addPrimitive(combine, [float, float, float], list)
    pset.addPrimitive(operator.mul, [float, float], float)
    pset.addPrimitive(operator.add, [float, float], float)
    pset.addPrimitive(operator.sub, [float, float], float)
    pset.addPrimitive(np.sin, [float], float)
    pset.addPrimitive(np.cos, [float], float)
    pset.addPrimitive(np.tan, [float], float)
    pset.addPrimitive(exp, [float], float)
    pset.addPrimitive(log, [float], float)

    pset.renameArguments(ARG0="x")
    pset.renameArguments(ARG1="y")
    return pset

def create_toolbox():
    toolbox = base.Toolbox()
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutShrink)
    toolbox.register("select", tools.selDoubleTournament, fitness_size=3, parsimony_size=1.4, fitness_first=True)
    return toolbox

def create_environment():
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

    env_kwargs = {'extra_serializers': [get_type_ser, get_primitive_ser, get_terminal_ser,
                                        get_primitive_set_typed_ser, get_func_ser, get_toolbox_ser,
                                        get_rule_leaf_ser], 'codec': aiomas.MsgPack}
    slave_kwargs = [{'extra_serializers': [get_type_ser, get_primitive_ser, get_terminal_ser,
                                           get_primitive_set_typed_ser, get_func_ser, get_toolbox_ser,
                                           get_rule_leaf_ser], 'codec': aiomas.MsgPack} for _ in range(len(addrs))]

    menv = StatEnvironment(addr,
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

    return menv


if __name__ == "__main__":

    # Parameters
    critic_threshold = 0.00001
    veto_threshold = 0.00001

    pset = create_pset()
    shape = (32, 32)
    create_kwargs = {'pset': pset,
                     'toolbox': create_toolbox(),
                     'pop_size': 10,
                     'shape': shape}

    # Make the rules

    rules = []
    box_count_max = box_count(np.ones(shape))
    rules.append((RuleLeaf(ft.ImageComplexityFeature(), LinearMapper(0, box_count_max, '01')), 1.))

    memsize = 1000

    # Environment and simulation

    log_folder = 'exp_test_logs'
    save_folder = 'exp_test_artifacts'
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    os.makedirs(save_folder)

    logger = None
    menv = create_environment()

    for _ in range(1):
        ret = aiomas.run(until=menv.spawn('agents.generic.feature_agent:FeatureAgent',
                                          log_folder=log_folder,
                                          artifact_cls=GeneticImageArtifact,
                                          create_kwargs=create_kwargs,
                                          rules=rules,
                                          memsize=memsize,
                                          critic_threshold=critic_threshold,
                                          veto_threshold=veto_threshold))

        print(ret)

    sim = Simulation(menv, log_folder=log_folder)
    sim.async_steps(100)
    menv.save_artifacts(save_folder)
    sim.end()

