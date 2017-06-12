import pickle

from creamas.examples.spiro.spiro_agent import SpiroArtifact
from creamas.examples.spiro.spiro_agent_mp import SpiroArtifact as SpiroArtifactMP

from artifacts.spr_artifact import SpiroArtifact as SpiroArtifactOwn
from artifacts.maze_artifact import MazeArtifact

def get_spiro_ser():
    return SpiroArtifact, pickle.dumps, pickle.loads

def get_spriro_ser_mp():
    return SpiroArtifactMP, pickle.dumps, pickle.loads

def get_spiro_ser_own():
    return SpiroArtifactOwn, pickle.dumps, pickle.loads

def get_maze_ser():
    return MazeArtifact, pickle.dumps, pickle.loads