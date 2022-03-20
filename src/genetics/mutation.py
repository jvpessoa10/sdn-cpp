import math

import numpy as np
from pymoo.core.mutation import Mutation


class PsoMutation(Mutation):
    def __init__(self):
        super().__init__()

    def _do(self, problem, x, **kwargs):
        fitness_matrix = kwargs['algorithm'].pop.get("F")
        global_best_solutions = fitness_matrix.argmin(axis=0)
        global_best_fitness = fitness_matrix.min(axis=0)

        for i in range(math.floor(len(x)/2)):
            fitness_i = fitness_matrix[i]

            biggest_accordance_obj = -1
            biggest_accordance = -1
            for f in range(len(fitness_i)):
                accordance_i = global_best_fitness[f] / fitness_i[f]

                if accordance_i > biggest_accordance:
                    biggest_accordance = accordance_i
                    biggest_accordance_obj = f

            global_best_i = global_best_solutions[biggest_accordance_obj]

            velocity_i = 0.5 * 2.5 * (global_best_i - x[i,0])

            x[i, 0] = np.floor(x[i, 0] + velocity_i).clip(min=0)

        return x
