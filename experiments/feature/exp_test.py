from creamas.mappers import LinearMapper
from utilities.math import box_count
from artifacts.genetic_image_artifact import GeneticImageArtifact
import creamas.features as ft
from experiments.feature.util import *

import numpy as np
import aiomas
import os
import shutil


if __name__ == "__main__":
    # Parameters
    critic_threshold = 0.001
    veto_threshold = 0.001

    novelty_weight = 0.85

    pset = create_pset()

    shape = (128, 128)
    create_kwargs = {'pset': pset,
                     'toolbox': create_toolbox(),
                     'pop_size': 10,
                     'shape': shape}

    # Make the rules

    rule_dict = get_rules(shape)

    rules = []
    rule_weights = []
    # box_count_max = box_count(np.ones(shape))
    # rules.append((RuleLeaf(ft.ImageComplexityFeature(), LinearMapper(0, box_count_max, '01')), 1.))
    rules.append(rule_dict['red'])
    rule_weights.append(1)

    memsize = 100000

    # Environment and simulation

    log_folder = 'exp_test_logs'
    save_folder = 'exp_test_artifacts'
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    os.makedirs(save_folder)

    menv = create_environment()

    for _ in range(1):
        ret = aiomas.run(until=menv.spawn('agents.generic.feature_agent:FeatureAgent',
                                          log_folder=log_folder,
                                          artifact_cls=GeneticImageArtifact,
                                          create_kwargs=create_kwargs,
                                          rules=rules,
                                          rule_weights=rule_weights,
                                          memsize=memsize,
                                          critic_threshold=critic_threshold,
                                          veto_threshold=veto_threshold,
                                          novelty_weight=novelty_weight))

        print(ret)

    sim = Simulation(menv, log_folder=log_folder)
    sim.async_steps(100)
    menv.save_artifacts(save_folder)
    sim.end()

