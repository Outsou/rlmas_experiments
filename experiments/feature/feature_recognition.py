from creamas.mappers import LinearMapper
from utilities.math import box_count
from creamas.image import fractal_dimension
import creamas.features as ft
from experiments.feature.util import *
import creamas.nx as cnx

import os
import shutil
import numpy as np
import networkx as nx


if __name__ == "__main__":

    # Parameters

    critic_threshold = 0.00001
    veto_threshold = 0.00001
    memsize = 1000

    pset = create_pset()
    shape = (32, 32)
    create_kwargs = {'pset': pset,
                     'toolbox': create_toolbox(),
                     'pop_size': 10,
                     'shape': shape}

    # Rules

    rules = []

    red_rule = RuleLeaf(ft.ImageRednessFeature(), LinearMapper(0, 1, '01'))
    rules.append((red_rule, 1))
    green_rule = RuleLeaf(ft.ImageGreennessFeature(), LinearMapper(0, 1, '01'))
    rules.append((green_rule, 1))
    blue_rule = RuleLeaf(ft.ImageBluenessFeature(), LinearMapper(0, 1, '01'))
    rules.append((blue_rule, 1))
    complexity_rule = RuleLeaf(ft.ImageComplexityFeature(), LinearMapper(0, fractal_dimension(np.ones(shape)), '01'))
    rules.append((complexity_rule, 1))
    intensity_rule = RuleLeaf(ft.ImageIntensityFeature(), LinearMapper(0, 1, '01'))
    rules.append((intensity_rule, 1))

    # Create environment and agents

    menv = create_environment()

    log_folder = 'feature_recognition_logs'
    save_folder = 'exp_test_artifacts'
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    os.makedirs(save_folder)

    ret = aiomas.run(until=menv.spawn('agents.generic.creator_agent:CreatorAgent',
                                      log_folder=log_folder,
                                      artifact_cls=GeneticImageArtifact,
                                      create_kwargs=create_kwargs,
                                      rules=rules,
                                      memsize=memsize,
                                      critic_threshold=critic_threshold,
                                      veto_threshold=veto_threshold))

    creator = ret[1]
    print(ret)

    for rule in rules:
        ret = aiomas.run(until=menv.spawn('agents.generic.critic_agent:CriticAgent',
                                          log_folder=log_folder,
                                          artifact_cls=GeneticImageArtifact,
                                          create_kwargs=create_kwargs,
                                          rules=[rule],
                                          memsize=memsize,
                                          critic_threshold=critic_threshold,
                                          veto_threshold=veto_threshold))

        print(ret)

    # Create connections

    agents = sorted(menv.get_agents(addr=True))
    creator_idx = agents.index(creator)

    G = nx.DiGraph()
    G.add_nodes_from(list(range(len(agents))))

    edges = []
    for i in range(len(agents)):
        if i != creator_idx:
            edges.append((creator_idx, i))
    G.add_edges_from(edges)

    cnx.connections_from_graph(menv, G)

    # Run simulation

    sim = Simulation(menv, log_folder=log_folder)
    sim.async_steps(1000)
    sim.end()

