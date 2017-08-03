from experiments.feature.util import *
import creamas.nx as cnx

import aiomas
import networkx as nx
import numpy as np


if __name__ == "__main__":
    # Parameters
    num_of_agents = 5
    critic_threshold = 0.5
    impressionability = 1

    pset = create_pset()

    shape = (32, 32)
    create_kwargs = {'pset': pset,
                     'toolbox': create_toolbox(),
                     'pop_size': 10,
                     'shape': shape}

    # Make the rules

    rule_dict = get_rules(shape)

    rules = []
    rules.append(rule_dict['red'])
    rules.append(rule_dict['green'])
    rules.append(rule_dict['blue'])

    # Environment and simulation

    log_folder = 'gd_test_logs'
    menv = create_environment()

    for _ in range(num_of_agents):
        rule_weights = []
        for _ in range(len(rules)):
            rule_weights.append(np.random.random())
        ret = aiomas.run(until=menv.spawn('agents.generic.gd_agent:GDAgent',
                                          log_folder=log_folder,
                                          artifact_cls=GeneticImageArtifact,
                                          create_kwargs=create_kwargs,
                                          rules=rules,
                                          rule_weights=rule_weights,
                                          critic_threshold=critic_threshold,
                                          impressionability=impressionability))

        print(ret)

    # Connections
    G = nx.complete_graph(num_of_agents)
    cnx.connections_from_graph(menv, G)

    sim = Simulation(menv, log_folder=log_folder)
    sim.async_steps(100)
    recommendations = menv.get_recommendations()
    total_rewards = menv.get_total_rewards()
    sim.end()

    avg_passed = 0
    num_of_passed = 0
    valid_recommendations = 0
    for recommendation in recommendations.values():
        if len(recommendation['passed']) > 0:
            avg_passed += np.sum(recommendation['passed']) / len(recommendation['passed'])
            num_of_passed += len(recommendation['passed'])
            valid_recommendations += 1
    if valid_recommendations > 0:
        avg_passed /= valid_recommendations

    print('Average value of passed recommendations: ' + str(avg_passed))
    print('Number of passed recommendations: ' + str(num_of_passed))

    total_reward = 0
    for reward in total_rewards.values():
        total_reward += reward

    print('Total reward: ' + str(total_reward))
