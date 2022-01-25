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

    N_OFFSPRINGS = 1

    def __init__(self, n_controllers, n_switches, weight):
        super().__init__(2, self.N_OFFSPRINGS)
        self.n_controllers = n_controllers
        self.n_switches = n_switches
        self.weight = weight

    def _do(self, problem, x, **kwargs):
        _, n_matings, n_var = x.shape

        # Output (n_offsprings, n_matings, n_var)
        y = np.empty((self.N_OFFSPRINGS, n_matings, n_var), dtype=object)

        for k in range(n_matings):
            # Parents
            a, b = x[0, k], x[1, k]

            r = np.random.Generator(np.random.PCG64()).normal()
            offspring = a + (b - a) * r * self.weight

            y[0, k] = self.ensure_bounds(offspring, problem.n_controllers, problem.n_switches)

        return y

    @staticmethod
    def ensure_bounds(offspring, n_controllers, n_switches):
        offspring[offspring < 0] = 0

        controllers = offspring[:n_controllers]
        switches = offspring[n_controllers:]

        controllers[controllers >= n_switches] = n_switches - 1
        switches[switches >= n_controllers] = n_controllers - 1

        return np.concatenate([controllers, switches])


class CPPSampling(Sampling):
    def __init__(self, n_controllers, n_switches):
        super().__init__()
        self.n_controllers = n_controllers
        self.n_switches = n_switches

    def _do(self, problem, n_samples, **kwargs):
        x = np.full((n_samples, self.n_switches + self.n_controllers), fill_value=0)

        for i in range(n_samples):
            controllers_positions = numpy.random.randint(0, self.n_switches, size=self.n_controllers)
            switches_assignments = numpy.random.randint(0, self.n_controllers, size=self.n_switches)

            x[i] = numpy.concatenate([controllers_positions, switches_assignments])

        print("Sampling size: " + str(len(x)))
        return x


class CPPProblem(Problem):

    def __init__(self, n_controllers, n_switches, prop_delay_matrix):
        super().__init__(
            n_var=n_switches + n_controllers,
            n_obj=3,
            n_constr=0,
            xl=0.0,
            xu=1.0
        )
        self.prop_delay_matrix = prop_delay_matrix
        self.n_controllers = n_controllers
        self.n_switches = n_switches

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

    def _evaluate(self, x, out, *args, **kwargs):
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
    def __init__(self, n_controllers, n_switches,  crossover_weight, network_delay_matrix):
        self.n_controllers = n_controllers
        self.n_switches = n_switches
        self.crossover_weight = crossover_weight
        self.network_delay_matrix = network_delay_matrix
        self.problem = None
        self.res = None

    def execute(self):
        self.problem = CPPProblem(self.n_controllers, self.n_switches,  self.network_delay_matrix)

        sampling = CPPSampling(self.n_controllers, self.n_switches)
        crossover = CPPCrossover(self.n_controllers, self.n_switches, self.crossover_weight)
        mutation = CPPMutation()

        algorithm = NSGA2(
            pop_size=600,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation
        )

        self.res = minimize(
            self.problem,
            algorithm,
            ('n_gen', 200),
            seed=1,
            save_history=True
        )


def test_crossover():
    crossover = CPPCrossover(3, 5, 0.8)
    controllers = [-1,5,3]
    switches = [0,2,0,3,1]

    r = crossover.ensure_bounds(np.array(controllers + switches), crossover.n_controllers, crossover.n_switches)

    print(r)


def test_cpp():
    graph = nx.random_internet_as_graph(60)
    network_delay_matrix = dict(all_pairs_shortest_path_length(graph))
    cpp = CPP(60, 10, 0.8, network_delay_matrix)
    cpp.execute()


if __name__ == "__main__":
    test_cpp()
