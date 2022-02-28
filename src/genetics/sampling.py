import numpy as np
from pymoo.core.sampling import Sampling


class CPPSampling(Sampling):
    def __init__(self, n_controllers, n_switches):
        super().__init__()
        self.n_controllers = n_controllers
        self.n_switches = n_switches

    def _do(self, problem, n_samples, **kwargs):
        x = np.full((n_samples, self.n_switches + self.n_controllers), fill_value=0)

        for i in range(n_samples):
            controllers_positions = np.random.randint(0, self.n_switches, size=self.n_controllers)
            switches_assignments = np.random.randint(0, self.n_controllers, size=self.n_switches)

            x[i] = np.concatenate([controllers_positions, switches_assignments])

        print("Sampling size: " + str(len(x)))
        return x