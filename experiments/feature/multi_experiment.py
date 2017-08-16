from experiments.feature.util import *
from artifacts.dummy_artifact import DummyFeature
from creamas.rules.rule import RuleLeaf
from creamas.mappers import GaussianMapper
import creamas.nx as cnx
from artifacts.dummy_artifact import DummyArtifact

import aiomas
import networkx as nx
import numpy as np


if __name__ == "__main__":
    # Parameters
    num_of_agents = 60
    num_of_features = 5
    std = 0.2
    search_width = 10

    create_kwargs = {'length': num_of_features}

    # Environment and simulation

    log_folder = 'gd_test_logs'
    menv = create_environment()

    active = True

    for _ in range(num_of_agents):
        rules = []

        for i in range(num_of_features):
            rules.append(RuleLeaf(DummyFeature(i), GaussianMapper(np.random.rand(), std)))
        # rules.append(RuleLeaf(DummyFeature(0), GaussianMapper(0.4, std)))
        # rules.append(RuleLeaf(DummyFeature(1), GaussianMapper(0.8, std)))
        # rules.append(RuleLeaf(DummyFeature(2), GaussianMapper(0.1, std)))

        # rule_weights = [0.1, 0.3, 0.6]
        rule_weights = []
        for _ in range(len(rules)):
            rule_weights.append(np.random.random())

        ret = aiomas.run(until=menv.spawn('agents.generic.multi_agent:MultiAgent',
                                          log_folder=log_folder,
                                          artifact_cls=DummyArtifact,
                                          create_kwargs=create_kwargs,
                                          rules=rules,
                                          rule_weights=rule_weights,
                                          std=std,
                                          active=active,
                                          search_width=search_width))
        print(ret)
        active = False

    # Connect everyone to the main agent
    G = nx.Graph()
    G.add_nodes_from(list(range(num_of_agents)))
    edges = [(0, x) for x in range(1, num_of_agents)]
    G.add_edges_from(edges)

    cnx.connections_from_graph(menv, G)

    sim = Simulation(menv, log_folder=log_folder)
    sim.async_steps(100)
    sim.end()
