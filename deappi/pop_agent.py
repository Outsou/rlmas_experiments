from creamas.core.agent import CreativeAgent

import aiomas
import numpy as np

from deap import base
from deap import creator
from deap import gp
from deap import tools


class PopAgent(CreativeAgent):
    def __init__(self, env, pop, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.pop = pop
        self.toolbox = self.create_toolbox()

    @staticmethod
    def create_toolbox():
        toolbox = base.Toolbox()
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("mutate", gp.mutInsert, pset=pset)
        toolbox.register("select", tools.selDoubleTournament, fitness_size=3, parsimony_size=1.4, fitness_first=True)
        toolbox.register("evaluate", evaluate)

        return toolbox

    def evolve_population(self, pop, NGEN):
        CXPB, MUTPB = 0.5, 0.2

        # Evaluate the entire population
        fitnesses = map(self.toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        for g in range(NGEN):
            # Select the next generation individuals
            offspring = self.toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.random() < CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if np.random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # The population is entirely replaced by the offspring
            pop[:] = offspring

        return pop

    @aiomas.expose
    async def act(self, *args, **kwargs):
        self.pop = self.evolve_population(self.pop, 10)
