from creamas.examples.spiro.spiro_agent import SpiroArtifact
from creamas.examples.spiro.spiro_agent_mp import SpiroArtifact as SpiroArtifactMP
from creamas.rules import RuleLeaf
from artifacts.spr_artifact import SpiroArtifact as SpiroArtifactOwn
from artifacts.maze_artifact import MazeArtifact
from artifacts.genetic_image_artifact import GeneticImageArtifact
from deap.gp import Primitive, Terminal, PrimitiveSet, PrimitiveSetTyped
from deap.base import Toolbox

import pickle
from types import FunctionType


def get_spiro_ser():
    return SpiroArtifact, pickle.dumps, pickle.loads


def get_spriro_ser_mp():
    return SpiroArtifactMP, pickle.dumps, pickle.loads


def get_spiro_ser_own():
    return SpiroArtifactOwn, pickle.dumps, pickle.loads


def get_maze_ser():
    return MazeArtifact, pickle.dumps, pickle.loads


def get_func_ser():
    return FunctionType, pickle.dumps, pickle.loads


def get_primitive_ser():
    return Primitive, pickle.dumps, pickle.loads


def get_terminal_ser():
    return Terminal, pickle.dumps, pickle.loads


def get_primitive_set_ser():
    return PrimitiveSet, pickle.dumps, pickle.loads

def get_primitive_set_typed_ser():
    return PrimitiveSetTyped, pickle.dumps, pickle.loads

def get_toolbox_ser():
    return Toolbox, pickle.dumps, pickle.loads

def get_type_ser():
    return type, pickle.dumps, pickle.loads

def get_rule_leaf_ser():
    return RuleLeaf, pickle.dumps, pickle.loads

def get_genetic_image_artifact_ser():
    return GeneticImageArtifact, pickle.dumps, pickle.loads
