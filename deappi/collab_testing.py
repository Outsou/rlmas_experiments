from experiments.feature.util import *
from creamas.util import run

import aiomas
import os
import shutil
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # # Parameters
    # critic_threshold = 0.001
    # veto_threshold = 0.001
    #
    # novelty_weight = 0.85
    #
    shape = (32, 32)


    # Make the rules

    rule_dict = get_rules(shape)

    # rule_weights = []
    # box_count_max = box_count(np.ones(shape))
    # rules.append((RuleLeaf(ft.ImageComplexityFeature(), LinearMapper(0, box_count_max, '01')), 1.))

    # rule_weights.append(1)
    #
    # memsize = 100000

    # Environment and simulation

    log_folder = 'collab_test_logs'
    # save_folder = 'exp_test_artifacts'
    # if os.path.exists(save_folder):
    #     shutil.rmtree(save_folder)
    # os.makedirs(save_folder)

    menv = create_environment()
    agents = []

    pset = create_pset()

    create_kwargs = {'pset': pset,
                     'toolbox': create_toolbox(pset),
                     'pop_size': 10,
                     'shape': shape}

    rules = [rule_dict['green'], rule_dict['red']]
    rule_weights = [0.3, 0.7]

    agent1 = aiomas.run(until=menv.spawn('deappi.collab_agent:CollabAgent',
                                         log_folder=log_folder,
                                         artifact_cls=GeneticImageArtifact,
                                         create_kwargs=create_kwargs,
                                         rules=rules,
                                         rule_weights=rule_weights))

    print(agent1)

    rules = [rule_dict['green'], rule_dict['blue']]
    rule_weights = [0.3, 0.7]

    agent2 = aiomas.run(until=menv.spawn('deappi.collab_agent:CollabAgent',
                                         log_folder=log_folder,
                                         artifact_cls=GeneticImageArtifact,
                                         create_kwargs=create_kwargs,
                                         rules=rules,
                                         rule_weights=rule_weights))

    print(agent2)


    run(agent1[0].init_connection(run(agent2[0].get_name())))
    run(agent2[0].init_connection(run(agent1[0].get_name())))

    sim = Simulation(menv, log_folder=log_folder)
    sim.async_steps(12)
    # artifact = run(agent1[0].get_artifact())
    # eval1 = run(agent1[0].get_eval())
    # eval2 = run(agent2[0].get_eval())

    best1 = run(agent1[0].get_best_received())
    best2 = run(agent2[0].get_best_received())

    sim.end()

    # print('Eval1: ' + str(eval1))
    # print('Eval2: ' + str(eval2))

    # plt.imshow(artifact.obj)
    # plt.show()

    print(best1.evals)
    print(best2.evals)

    plt.imshow(best1.obj)
    plt.show()
    plt.imshow(best2.obj)
    plt.show()
