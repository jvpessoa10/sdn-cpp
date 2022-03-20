import time

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_mutation, get_sampling, get_crossover
from pymoo.optimize import minimize

from src.data.config import Config
from src.genetics.crossover import CPPCrossover
from src.genetics.mutation import PsoMutation
from src.genetics.problem import CPPProblem
from src.genetics.sampling import CPPSampling


class PsoMogaCpp:
    def __init__(self, config: Config):
        self.config = config
        self.problem = CPPProblem(
            self.config.n_controllers,
            self.config.network.n_switches(),
            self.config.network.delay_matrix()
        )
        self.res = None

    def execute(self):
        if self.config.use_generic_operators:
            sampling = get_sampling("int_random")
            crossover = get_crossover("int_sbx")
            mutation = get_mutation("int_pm")
        else:
            sampling = CPPSampling(self.config.n_controllers, self.config.network.n_switches())
            crossover = CPPCrossover(self.config.n_controllers, self.config.network.n_switches())
            mutation = PsoMutation()

        algorithm = NSGA2(
            pop_size=self.config.pop_size,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation
        )

        self.res = minimize(
            self.problem,
            algorithm,
            ('n_gen', self.config.n_generations),
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