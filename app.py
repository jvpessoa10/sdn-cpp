import os
import numpy as np

from src.config import Config
from src.cpp import PsoMogaCpp
from src.network import rnp
from src.persistence import _save_scatter, _save_line_graph


def main():
    n_controllers_var = [5, 10, 15, 20]
    pop_size_var = [200, 400, 800]
    n_generations_var = [200, 400, 800]

    possibilities = np.array(np.meshgrid(n_controllers_var, pop_size_var, n_generations_var)).T.reshape(-1, 3)

    for n_controllers, pop_size, n_generations in possibilities:
        _execute(Config(
            n_controllers=n_controllers,
            pop_size=pop_size,
            n_generations=n_generations,
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
    _save_population_results(pop, output_dir)
    _save_optimum_per_iteration(result.history, output_dir)
    _save_results_avg_per_iteration(result.history, output_dir)
    _save_config(config, os.path.join(output_dir, "config.txt"))


def _save_population_results(population, output_dir):
    _save_scatter(
        population.get("F")[:, [0, 1]],
        os.path.join(output_dir, "SC_Avg_X_CC_Avg_Delay.png"),
        x_label="SC_Avg_Delay",
        y_label="CC_Avg_Delay",
        title="Final population performance"
    )

    _save_scatter(
        population.get("F")[:, [0, 2]],
        os.path.join(output_dir, "SC_Avg_X_Load_Imbalance.png"),
        x_label="SC_Avg_Delay",
        y_label="Load_Imbalance",
        title="Final population performance"
    )

    _save_scatter(
        population.get("F")[:, [1, 2]],
        os.path.join(output_dir, "CC_Avg_X_Load_Imbalance.png"),
        x_label="CC_Avg_Delay",
        y_label="Load_Imbalance",
        title="Final population performance"
    )


def _save_optimum_per_iteration(history, output_dir):
    n_evals = np.array([e.evaluator.n_eval for e in history])
    opt = np.array([e.opt.size for e in history])

    _save_line_graph(
        n_evals,
        opt,
        path=os.path.join(output_dir, "optimum_per_iteration.png"),
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
        path=os.path.join(output_dir, "avg_sc_delay_per_iteration.png"),
        x_label="Iterations",
        y_label="Avg SC Delay"
    )

    _save_line_graph(
        n_evals,
        opt_avg[:, 1],
        path=os.path.join(output_dir, "avg_cc_delay_per_iteration.png"),
        x_label="Iterations",
        y_label="Avg CC Delay"
    )

    _save_line_graph(
        n_evals,
        opt_avg[:, 2],
        path=os.path.join(output_dir, "load_imbalance_per_iteration.png"),
        x_label="Iterations",
        y_label="Load Imbalance"
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
