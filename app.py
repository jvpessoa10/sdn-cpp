import os
import numpy as np
from networkx import all_pairs_dijkstra_path_length, convert_node_labels_to_integers

from src.data.config import Config
from src.cpp import PsoMogaCpp
from src.network import rnp
from src.persistence import _save_scatter, _save_line_graph

N_CONTROLLERS = 7
N_SWITCHES = 27
POP_SIZE = 200
N_GEN = 200
CROSSOVER_WEIGHT = 0.8


def main():
    _execute(Config(
        n_controllers=15,
        pop_size=200,
        n_generations=200,
        network=rnp(),
        use_generic_operators=True
    ))

    _execute(Config(
        n_controllers=10,
        pop_size=200,
        n_generations=200,
        network=rnp(),
        use_generic_operators=True
    ))

    _execute(Config(
        n_controllers=5,
        pop_size=200,
        n_generations=200,
        network=rnp(),
        use_generic_operators=True
    ))



def _execute(config: Config):
    cpp = PsoMogaCpp(config)
    cpp.execute()

    _save_result(cpp.res, config)


def _save_result(result, config: Config):
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    default_out_path = os.path.join(absolute_path, "output")
    output_dir = os.path.join(default_out_path, config.name())
    os.makedirs(output_dir, exist_ok=True)

    pop = result.pop
    _save_population_result(pop, os.path.join(output_dir, "results.png"))
    _save_optimum_per_iteration(result.history, os.path.join(output_dir, "optimum_per_iteration.png"))
    _save_results_avg_per_iteration(result.history, os.path.join(output_dir, "avg_per_iteration.png"))
    _save_config(config, os.path.join(output_dir, "config.txt"))


def _save_population_result(population, output_dir):
    _save_scatter(
        population.get("F"),
        output_dir,
        x_label="SC_Avg_Delay",
        y_label="CC_Avg_Delay",
        title="Final population performance"
    )


def _save_optimum_per_iteration(history, output_dir):
    n_evals = np.array([e.evaluator.n_eval for e in history])
    opt = np.array([e.opt.size for e in history])

    _save_line_graph(
        n_evals,
        opt,
        path=output_dir,
        x_label="Iterations",
        y_label="Individuals",
        title="Nº Optimum Individuals / Iteration"
    )


def _save_results_avg_per_iteration(history, output_dir):
    opt_avg = np.array([np.average(e.opt.get("F"), axis=0) for e in history])
    n_evals = np.array([e.evaluator.n_eval for e in history])

    _save_line_graph(
        n_evals,
        opt_avg[:, 0],
        path=output_dir,
        x_label="Iterations",
        y_label="Avg SC Delay"
    )


def _save_config(config: Config, output_dir):
    with open(output_dir, 'w') as writer:
        writer.write(
            "Number controllers: " + str(config.n_controllers) + "\n" +
            "Number switches: " + str(config.network.n_switches()) + "\n" +
            "Pop. Size: " + str(config.pop_size) + "\n" +
            "N Generations: " + str(config.n_generations) + "\n" +
            "Has generic operators: " + str(config.use_generic_operators) + "\n" +
            "Topology: " + config.network.topology + "\n" +
            "Is Weighted: " + str(config.network.is_network_weighted()) + "\n"
        )


if __name__ == "__main__":
    main()
