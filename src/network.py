import random
from dataclasses import dataclass
from os import listdir
import networkx as nx
from networkx import Graph, is_weighted, all_pairs_dijkstra_path_length, convert_node_labels_to_integers


@dataclass
class Network:
    graph: Graph
    topology: str

    def delay_matrix(self):
        return dict(
            all_pairs_dijkstra_path_length(
                self.graph,
                weight="weight"
            )
        )

    def is_network_weighted(self):
        return is_weighted(self.graph)

    def n_switches(self):
        return len(self.graph.nodes)


def random_as(n_nodes):
    graph = nx.random_internet_as_graph(n_nodes)

    for (u, v) in graph.edges():
        graph.edges[u, v]['weight'] = random.randint(10, 100)

    return Network(
        graph=graph,
        topology="random_as"
    )


def rnp(latency_files=None):
    graph = Graph()

    graph.add_edges_from([
        ('RO', 'MT'),
        ('RO', 'AC'),
        ('MT', 'MS'),
        ('MT', 'GO'),
        ('AC', 'DF'),
        ('GO', 'TO'),
        ('GO', 'DF'),
        ('MS', 'PR'),
        ('DF', 'TO'),
        ('DF', 'SP'),
        ('DF', 'RJ'),
        ('DF', 'MG'),
        ('DF', 'MA'),
        ('DF', 'CE'),
        ('DF', 'AM'),
        ('TO', 'PA'),
        ('MG', 'SP'),
        ('MG', 'RJ'),
        ('PR', 'RS'),
        ('PR', 'SP'),
        ('PR', 'SC'),
        ('RS', 'SC'),
        ('RS', 'SP'),
        ('SC', 'SP'),
        ('SP', 'RJ'),
        ('SP', 'CE'),
        ('RJ', 'ES'),
        ('ES', 'BA'),
        ('BA', 'CE'),
        ('BA', 'PB'),
        ('BA', 'SE'),
        ('SE', 'AL'),
        ('AL', 'PE'),
        ('PE', 'PI'),
        ('PE', 'PB'),
        ('PI', 'MA'),
        ('PB', 'RN'),
        ('MA', 'PA'),
        ('RN', 'CE'),
        ('RR', 'AM'),
        ('RR', 'CE'),
        ('AP', 'AM'),
        ('AP', 'PA'),
    ])

    if latency_files:
        graph = _load_rnp_weight(latency_files, graph)

    return Network(
        graph=convert_node_labels_to_integers(graph),
        topology="rnp"
    )


def _load_rnp_weight(path, graph):
    for file in listdir(path):
        node_name = file.split('_')[0].split('-')[1]

        with open(path + file) as reader:
            for line in reader.readlines():
                edge_node = line.split('=')[0].split('-')[1]
                weight = line.split('=')[1].split(':')[2]

                try:
                    graph[node_name][edge_node]['weight'] = round(float(weight), 2)
                except:
                    continue

    return graph
