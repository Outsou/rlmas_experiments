from artifacts.dummy_artifact import DummyArtifact, DummyFeature
from experiments.feature.util import *
import creamas.nx as cnx

import aiomas
import networkx as nx
import numpy as np


if __name__ == "__main__":
    # Parameters
    num_of_agents = 1
    critic_threshold = 0.5
    num_of_features = 3
    memsize = 0

    create_kwargs = {'length': num_of_features}

    # Make the rules
    rules = []

    for i in range(num_of_features):
        rule = RuleLeaf(DummyFeature(i), LinearMapper(0, 1, '01'))
        rules.append(rule)

    # Environment and simulation

    log_folder = 'dummy_logs'
    menv = create_environment()

    for _ in range(num_of_agents):
        rule_weights = []
        for _ in range(len(rules)):
            rule_weights.append(np.random.random())
        ret = aiomas.run(until=menv.spawn('agents.generic.feature_agent:FeatureAgent',
                                          log_folder=log_folder,
                                          artifact_cls=DummyArtifact,
                                          create_kwargs=create_kwargs,
                                          rules=rules,
                                          rule_weights=rule_weights,
                                          critic_threshold=critic_threshold,
                                          memsize=memsize))

        print(ret)

    # Connections
    G = nx.complete_graph(num_of_agents)
    cnx.connections_from_graph(menv, G)

    sim = Simulation(menv, log_folder=log_folder)
    sim.async_steps(100)
    sim.end()

