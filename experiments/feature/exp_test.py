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

