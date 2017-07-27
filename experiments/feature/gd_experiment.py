from experiments.feature.util import *
import creamas.nx as cnx

import aiomas
import networkx as nx


if __name__ == "__main__":
    # Parameters
    num_of_agents = 12

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
                                          rule_weights=rule_weights))

        print(ret)

    # Connections
    G = nx.complete_graph(num_of_agents)
    cnx.connections_from_graph(menv, G)

    sim = Simulation(menv, log_folder=log_folder)
    sim.async_steps(500)
    sim.end()

