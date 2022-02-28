import time
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_mutation, get_sampling, get_crossover
from pymoo.optimize import minimize

from src.genetics.crossover import CPPCrossover
from src.genetics.problem import CPPProblem
from src.genetics.sampling import CPPSampling


class PsoMogaCpp:
    def __init__(self, n_controllers, n_switches, pop_size, n_gen, crossover_weight, network_delay_matrix):
        self.n_controllers = n_controllers
        self.n_switches = n_switches
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.crossover_weight = crossover_weight
        self.network_delay_matrix = network_delay_matrix
        self.problem = None
        self.res = None

    def execute(self):
        self.problem = CPPProblem(self.n_controllers, self.n_switches, self.network_delay_matrix)
        crossover = CPPCrossover(self.n_controllers, self.n_switches, self.crossover_weight)
        sampling = CPPSampling(self.n_controllers, self.n_switches)
        mutation = get_mutation("int_pm")

        algorithm = NSGA2(
            pop_size=self.pop_size,
            sampling=get_sampling("int_random"),
            crossover=get_crossover("int_sbx"),
            mutation=get_mutation("int_pm")
        )

        self.res = minimize(
            self.problem,
            algorithm,
            ('n_gen', self.n_gen),
            seed=1,
            save_history=True,
            verbose=True
        )


class ExhaustiveCpp:
    def __init__(self, n_controllers, n_switches, network_delay_matrix, execution_time):
        self.n_controllers = n_controllers
        self.n_switches = n_switches
        self.network_delay_matrix = network_delay_matrix
        self.time = execution_time
        self.results = []

    def execute(self):
        self.problem = CPPProblem(self.n_controllers, self.n_switches, self.network_delay_matrix)
        sampling = CPPSampling(self.n_controllers, self.n_switches)

        final_time = time.time() + self.time
        while time.time() < final_time:
            out = {}

            for individual in sampling._do(None, 1):
                self.problem._evaluate(individual, out)
                self.results.append(out["F"])