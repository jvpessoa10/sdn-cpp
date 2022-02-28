import os

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from networkx import all_pairs_dijkstra_path_length, convert_node_labels_to_integers
from pandas import DataFrame
from pymoo.visualization.scatter import Scatter

from src.cpp import PsoMogaCpp
import pandas as pd
import plotly.express as px
from src.network import rnp, random_as

N_CONTROLLERS = 3
N_SWITCHES = 27
POP_SIZE = 200
N_GEN = 200
CROSSOVER_WEIGHT = 0.8


def print_population_evolution(iterations, x_label, y_label):
    data = pd.DataFrame()
    for i, ite in enumerate(iterations):
        population = pd.DataFrame(data=ite.pop.get("F"), columns=[x_label, y_label])
        population["Generation"] = i

        data = data.append(population)

    fig = px.scatter(data, x=x_label, y=y_label, animation_frame="Generation")
    fig.show()


def save_dataframe(data, path):
    data.to_csv(path)


def save_scatter(data, path, x_label, y_label):
    data = pd.DataFrame(data=data, columns=[x_label, y_label])
    fig = px.scatter(data, x=x_label, y=y_label)
    fig.write_image(path)


def save_graph(graph, path):
    nx.draw(graph, with_labels=True)
    plt.savefig(path)

def main():
    root_path = os.path.dirname(__file__)

    network = rnp(
        latency_files=os.path.join(
            root_path,
            'topologies\\rnp\\20190830\\'
        )
    )

    # network = random_as(N_SWITCHES)

    network_enumerated = convert_node_labels_to_integers(network)

    network_delay_matrix = dict(
        all_pairs_dijkstra_path_length(
            network_enumerated,
            weight="weight"
        )
    )

    cpp = PsoMogaCpp(N_CONTROLLERS, N_SWITCHES, POP_SIZE, N_GEN, CROSSOVER_WEIGHT, network_delay_matrix)
    cpp.execute()

    output_dir = os.path.join(root_path, "output")
    os.makedirs(output_dir, exist_ok=True)

    pop = cpp.res.pop

    save_scatter(
        pop.get("F"),
        os.path.join(output_dir, "results_scatter.png"),
        x_label="SC_Avg_Delay",
        y_label="CC_Avg_Delay"
    )
    save_dataframe(DataFrame(pop.get("X")), os.path.join(output_dir, "population.csv"))
    save_dataframe(DataFrame(pop.get("F")), os.path.join(output_dir, "results.csv"))
    save_graph(network_enumerated, os.path.join(output_dir, "network.png"))

    n_evals = np.array([e.evaluator.n_eval for e in cpp.res.history])
    opt = np.array([e.opt[0].F for e in cpp.res.history])

    plt.title("Convergence")
    plt.plot(n_evals, opt, "--")
    plt.show()

    # print_population_evolution(cpp.res.history, x_label="CC_Avg_Delay", y_label="C_Load_Imbalance")


if __name__ == "__main__":
    main()
