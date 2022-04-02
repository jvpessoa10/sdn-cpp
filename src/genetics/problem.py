from multiprocessing.pool import ThreadPool
from pymoo.core.problem import ElementwiseProblem, starmap_parallelized_eval
import numpy as np


class CPPProblem(ElementwiseProblem):

    def __init__(self, n_controllers, n_switches, prop_delay_matrix):
        super().__init__(
            n_var=n_switches + n_controllers,
            n_obj=3,
            n_constr=1,
            xl=0.0,
            xu=[n_switches - 1] * n_controllers + [n_controllers - 1] * n_switches,
            func_eval=starmap_parallelized_eval,
            runner=ThreadPool(8).starmap
        )
        self.prop_delay_matrix = prop_delay_matrix
        self.n_controllers = n_controllers
        self.n_switches = n_switches

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.array([self.f1(x), self.f2(x), self.f3(x)])
        out["G"] = 0 if self.c1(x) else 1

    # Average Switch to Controller Delay
    def f1(self, individual):
        controllers = individual[:self.n_controllers]
        switches = individual[self.n_controllers:]

        total_delay = 0
        number_connections = 0
        for controller, controller_position in enumerate(controllers):
            for switch, switch_assignment in enumerate(switches):
                if switch_assignment == controller:
                    total_delay += self.prop_delay_matrix[controller_position][switch]
                    number_connections += 1

        return total_delay / number_connections

    # Average Controller to Controller Delay
    def f2(self, individual):
        controllers = individual[:self.n_controllers]

        total_delay = 0
        number_connections = 0
        for controller, controller_position in enumerate(controllers):
            for controller2, controller_position2 in enumerate(controllers[controller:]):
                if controller != controller2:
                    total_delay += self.prop_delay_matrix[controller_position][controller_position2]
                    number_connections += 1

        return total_delay / number_connections

    # Maximal controller load imbalance
    def f3(self, individual):
        switches = individual[self.n_controllers:]
        occurrences = np.bincount(switches)
        return occurrences[occurrences.argmax()] - occurrences[occurrences.argmin()]

    def c1(self, individual):
        controllers = individual[:self.n_controllers]

        return len(np.unique(controllers)) == len(controllers)
