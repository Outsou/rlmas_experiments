import pickle

from creamas.examples.spiro.spiro_agent import SpiroArtifact

def get_spiro_ser():
    return SpiroArtifact, pickle.dumps, pickle.loads