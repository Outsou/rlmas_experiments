import networkx as nx
from networkx import Graph, DiGraph
from creamas.util import run


def graph_from_connections(env, directed=False):
    '''Create NetworkX graph from agent connections in a given environment.

    :param env:
        Environment where the agents live. The environment must be derived from
        :class:`~creamas.core.environment.Environment`,
        :class:`~creamas.mp.MultiEnvironment` or
        :class:`~creamas.ds.DistributedEnvironment`.

    :param bool directed:
        If ``True``, creates an instance of :class:`~networkx.digraph.DiGraph`,
        otherwise creates an instance of :class:`~networkx.graph.Graph`.

    :returns: The created NetworkX graph.
    :rtype:
        :class:`~networkx.digraph.DiGraph` or :class:`~networkx.graph.Graph`

    .. note::

        If the created graph is undirected and two connected agents have
        different attitudes towards each other, then the value of
        ``"attitude"`` key in the resulting graph for the edge is chosen
        randomly from the two values.
    '''

    G = DiGraph() if directed else Graph()
    conn_list = env.get_connections(data=True)
    labels = {}
    for agent, conns in conn_list:
        agent_proxy = run(env.connect(agent))
        desired_novelty = run(agent_proxy.get_desired_novelty())
        print(desired_novelty)
        labels[agent] = desired_novelty

        G.add_node(agent)
        ebunch = []
        for nb, data in conns.items():
            ebunch.append((agent, nb, data))
        if len(ebunch) > 0:
            G.add_edges_from(ebunch)
    pos = nx.spring_layout(G)
    nx.draw_networkx_labels(G, pos, labels, font_size=16)
    return G