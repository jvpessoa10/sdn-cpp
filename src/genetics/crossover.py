import numpy as np
from pymoo.core.crossover import Crossover


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

            r = np.random.random_sample()
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