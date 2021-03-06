from creamas.core import Environment, Simulation
from creamas.mp import MultiEnvManager, EnvManager
from environments.generic import StatEnvironment
from creamas.util import run
from creamas.image import fractal_dimension
from creamas.mappers import LinearMapper
import creamas.features as ft
from utilities.serializers import *
from utilities.bitwise import float_or, float_xor, float_and

from deap import base
from deap import tools
from deap import gp
import operator
import numpy as np

import asyncio
import aiomas


def combine(num1, num2, num3):
    return [float(num1), float(num2), float(num3)]


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

def divide(a, b):
    if b == 0:
        b = 0.000001
    return np.divide(a, b)

def sign(a):
    if a < 0:
        return -1
    elif a > 0:
        return 1
    else:
        return 0

def mdist(a, b):
    return abs(a-b)

def safe_pow(a, b):
    if a == 0 and b < 0:
        return 0
    return pow(a, b)

def abs_sqrt(a):
    return np.sqrt(abs(a))

def create_pset():
    pset = gp.PrimitiveSetTyped("main", [float, float], list)
    pset.addPrimitive(combine, [float, float, float], list)
    pset.addPrimitive(operator.mul, [float, float], float)
    pset.addPrimitive(divide, [float, float], float)
    pset.addPrimitive(operator.add, [float, float], float)
    pset.addPrimitive(operator.sub, [float, float], float)
    pset.addPrimitive(np.sin, [float], float)
    pset.addPrimitive(np.cos, [float], float)
    #pset.addPrimitive(np.tan, [float], float)
    pset.addPrimitive(min, [float, float], float)
    pset.addPrimitive(max, [float, float], float)
    pset.addPrimitive(np.abs, [float], float)
    #pset.addPrimitive(exp, [float], float)
    #pset.addPrimitive(log, [float], float)
    #pset.addPrimitive(safe_pow, [float, float], float)
    pset.addPrimitive(abs_sqrt, [float], float)
    #pset.addEphemeralConstant('rand', lambda: np.random.random() * 2 - 1, float)
    pset.addPrimitive(sign, [float], float)
    pset.addPrimitive(mdist, [float, float], float)
    pset.addPrimitive(float_or, [float, float], float)
    pset.addPrimitive(float_xor, [float, float], float)
    pset.addPrimitive(float_and, [float, float], float)

    pset.renameArguments(ARG0="x")
    pset.renameArguments(ARG1="y")
    return pset


def mutate(individual, pset, expr):
    rand = np.random.rand()
    if rand <= 0.25:
        return gp.mutShrink(individual),
    elif rand <= 0.5:
        return gp.mutInsert(individual, pset)
    elif rand <= 0.75:
        return gp.mutNodeReplacement(individual, pset)
    return gp.mutUniform(individual, expr, pset)


def create_toolbox(pset):
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", mutate, expr=toolbox.expr)
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
                                        get_rule_leaf_ser, get_genetic_image_artifact_ser,
                                        get_ndarray_ser, get_dummy_ser], 'codec': aiomas.MsgPack}
    slave_kwargs = [{'extra_serializers': [get_type_ser, get_primitive_ser, get_terminal_ser,
                                           get_primitive_set_typed_ser, get_func_ser, get_toolbox_ser,
                                           get_rule_leaf_ser, get_genetic_image_artifact_ser,
                                           get_ndarray_ser, get_dummy_ser], 'codec': aiomas.MsgPack} for _ in range(len(addrs))]

    menv = StatEnvironment(addr,
                            env_cls=Environment,
                            mgr_cls=MultiEnvManager,
                            logger=None,
                            **env_kwargs)

    ret = run(menv.spawn_slaves(slave_addrs=addrs,
                                slave_env_cls=Environment,
                                slave_mgr_cls=EnvManager,
                                slave_kwargs=slave_kwargs))

    ret = run(menv.wait_slaves(30))
    ret = run(menv.set_host_managers())
    ret = run(menv.is_ready())

    return menv


def get_rules(img_shape):
    rules = {}
    red_rule = RuleLeaf(ft.ImageRednessFeature(), LinearMapper(0, 1, '01'))
    rules['red'] = red_rule
    green_rule = RuleLeaf(ft.ImageGreennessFeature(), LinearMapper(0, 1, '01'))
    rules['green'] = green_rule
    blue_rule = RuleLeaf(ft.ImageBluenessFeature(), LinearMapper(0, 1, '01'))
    rules['blue'] = blue_rule
    complexity_rule = RuleLeaf(ft.ImageComplexityFeature(), LinearMapper(0, fractal_dimension(np.ones(img_shape)), '01'))
    rules['complexity'] = complexity_rule
    intensity_rule = RuleLeaf(ft.ImageIntensityFeature(), LinearMapper(0, 1, '01'))
    rules['intensity'] = intensity_rule
    return rules
