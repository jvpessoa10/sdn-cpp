import numpy.random
import numpy as np
import networkx as nx
from networkx import all_pairs_shortest_path_length
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.core.sampling import Sampling
import logging


class CPPMutation(Mutation):
    def __init__(self):
        super().__init__()

    def _do(self, problem, x, **kwargs):
        return x


class CPPCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 2)

    def _do(self, problem, x, **kwargs):
        return x


class CPPSampling(Sampling):
    def __init__(self, n_switches, n_controllers):
        super().__init__()
        self.n_switches = n_switches
        self.n_controllers = n_controllers

    def _do(self, problem, n_samples, **kwargs):
        logging.info("n_samples: " + str(n_samples))
        x = np.full((n_samples, self.n_switches + self.n_controllers), fill_value=0)

        for i in range(n_samples):
            controllers_positions = numpy.random.randint(0, self.n_switches, size=self.n_controllers)
            switches_assignments = numpy.random.randint(0, self.n_controllers, size=self.n_switches)

            x[i] = numpy.concatenate([controllers_positions,switches_assignments])

        return x


class CPPProblem(Problem):

    def __init__(self, n_switches, n_controllers, prop_delay_matrix):
        super().__init__(
            n_var=n_switches + n_controllers,
            n_obj=3,
            n_constr=0,
            xl=0.0,
            xu=1.0
        )
        self.prop_delay_matrix = prop_delay_matrix
        self.n_switches = n_switches
        self.n_controllers = n_controllers

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

    def _evaluate(self, x, out, *args, **kwargs):
        logging.info("x: " + str(len(x)))

        results = []

        for individual in x:
            results.append(
                np.array([self.f1(individual), self.f2(individual), self.f3(individual)])
            )

        out["F"] = np.array(results)

    # Average Switch to Controller Delay
    def f1(self, individual):
        controllers = individual[:self.n_controllers]
        switches = individual[self.n_controllers:]

        total_delay = 0
        for controller, controller_position in enumerate(controllers):
            for switch, switch_assignment in enumerate(switches):
                if switch_assignment == controller:
                    total_delay += self.prop_delay_matrix[controller_position][switch]

        return total_delay / len(switches)

    # Average Controller to Controller Delay
    def f2(self, individual):
        controllers = individual[:self.n_controllers]

        total_delay = 0
        for controller, controller_position in enumerate(controllers):
            for controller2, controller_position2 in enumerate(controllers):
                if controller != controller2:
                    total_delay += self.prop_delay_matrix[controller_position][controller_position2]

        return total_delay / (len(controllers) * len(controllers))

    # Maximal controller load imbalance
    def f3(self, individual):
        switches = individual[self.n_controllers:]
        occurrences = np.bincount(switches)
        return occurrences[occurrences.argmax()] - occurrences[occurrences.argmin()]


class CPP:
    def __init__(self, n_switches, n_controllers, network_delay_matrix):
        self.n_switches = n_switches
        self.n_controllers = n_controllers
        self.network_delay_matrix = network_delay_matrix

    def execute(self):
        problem = CPPProblem(self.n_switches, self.n_controllers, self.network_delay_matrix)
        sampling = CPPSampling(self.n_switches, self.n_controllers)
        crossover = CPPCrossover()
        mutation = CPPMutation()

        algorithm = NSGA2(
            pop_size=100,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation
        )

        res = minimize(
            problem,
            algorithm,
            ('n_gen', 200),
            seed=1,
            verbose=True
        )


def test_cpp():
    graph = nx.random_internet_as_graph(60)
    network_delay_matrix = dict(all_pairs_shortest_path_length(graph))
    cpp = CPP(60, 10, network_delay_matrix)
    cpp.execute()


if __name__ == "__main__":
    test_cpp()
