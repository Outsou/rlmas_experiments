import pickle

from creamas.examples.spiro.spiro_agent import SpiroArtifact
from creamas.examples.spiro.spiro_agent_mp import SpiroArtifact as SpiroArtifactMP

def get_spiro_ser():
    return SpiroArtifact, pickle.dumps, pickle.loads

def get_spriro_ser_mp():
    return SpiroArtifactMP, pickle.dumps, pickle.loads